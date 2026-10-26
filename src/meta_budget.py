"""Thread-safe dual-provider (Meta + Baseten) API budget guards.

Tracks, per provider:
  - Euro budget (``META_BUDGET_LIMIT_EUR`` / ``BASETEN_BUDGET_LIMIT_EUR``,
    both defaulting to 100.00 EUR; overridable via environment),
  - call counts (successful + failed),
  - token counts (prompt + completion),
  - actual HTTP round-trip latency in milliseconds.

Only the standard library is used so this module has zero third-party
dependencies.
"""

from __future__ import annotations

import os
import threading
from dataclasses import dataclass, field
from typing import Any, Dict, List, Optional


META_BUDGET_LIMIT_EUR: float = float(os.getenv("META_BUDGET_LIMIT_EUR", "100.00"))
BASETEN_BUDGET_LIMIT_EUR: float = float(
    os.getenv("BASETEN_BUDGET_LIMIT_EUR", "100.00")
)

#: Estimated price in EUR per 1000 tokens, per model. These are conservative
#: client-side estimates used to convert reported token usage into spend;
#: pass an explicit ``cost_eur`` to ``record`` when the provider reports an
#: exact charge instead.
PRICE_EUR_PER_KTOK: Dict[str, float] = {
    "muse-wonder-1.0": 0.004,
    "muse-spark-1.3": 0.001,
    "muse-spark-1.3-contributor": 0.001,
    "sam-3.1": 0.002,
    "muse-voice-transcribe-1.0": 0.001,
    "muse-image-1.0": 0.002,
    "meta-models/muse-spark-open-1026": 0.0005,
    "meta-models/muse-spark-open": 0.0005,
    "default": 0.001,
}

_MAX_LATENCY_SAMPLES = 4096


def estimate_cost_eur(
    prompt_tokens: int, completion_tokens: int, model: str = "default"
) -> float:
    """Estimate EUR cost for a call from token counts (see price table)."""
    rate = PRICE_EUR_PER_KTOK.get(model, PRICE_EUR_PER_KTOK["default"])
    return (prompt_tokens + completion_tokens) / 1000.0 * rate


@dataclass
class BudgetGuard:
    """Thread-safe spend/call/token/latency tracker for one API provider."""

    limit_eur: float = 100.00
    provider: str = "meta"
    spent_eur: float = 0.0
    calls: int = 0
    failed_calls: int = 0
    prompt_tokens: int = 0
    completion_tokens: int = 0
    _lock: threading.Lock = field(default_factory=threading.Lock, repr=False)
    _latencies_ms: List[float] = field(default_factory=list, repr=False)

    def record(
        self,
        cost_eur: float = 0.0,
        prompt_tokens: int = 0,
        completion_tokens: int = 0,
        latency_ms: float = 0.0,
        model: str = "default",
        failed: bool = False,
    ) -> float:
        """Record one API call.

        When ``cost_eur`` is 0 and token counts are given, the cost is
        estimated from :data:`PRICE_EUR_PER_KTOK`. Raises ``RuntimeError``
        *before* mutating state if the call would exceed the budget.
        Returns the remaining budget in EUR.
        """
        if cost_eur <= 0.0 and (prompt_tokens or completion_tokens):
            cost_eur = estimate_cost_eur(prompt_tokens, completion_tokens, model)
        with self._lock:
            if self.spent_eur + cost_eur > self.limit_eur:
                raise RuntimeError(
                    f"Budget limit exceeded [{self.provider}]: "
                    f"{self.spent_eur + cost_eur:.4f}EUR > {self.limit_eur:.2f}EUR"
                )
            self.spent_eur += cost_eur
            self.calls += 1
            if failed:
                self.failed_calls += 1
            self.prompt_tokens += max(0, prompt_tokens)
            self.completion_tokens += max(0, completion_tokens)
            if latency_ms:
                self._latencies_ms.append(float(latency_ms))
                if len(self._latencies_ms) > _MAX_LATENCY_SAMPLES:
                    del self._latencies_ms[
                        : len(self._latencies_ms) - _MAX_LATENCY_SAMPLES
                    ]
            return max(0.0, self.limit_eur - self.spent_eur)

    def record_spend(self, amount_eur: float) -> float:
        """Record a raw EUR spend (no tokens/latency attached)."""
        return self.record(cost_eur=amount_eur)

    def check(self, cost_eur: float = 0.0) -> None:
        """Raise ``RuntimeError`` if ``cost_eur`` would exceed the budget."""
        with self._lock:
            if self.spent_eur + cost_eur > self.limit_eur:
                raise RuntimeError(
                    f"Budget limit exceeded [{self.provider}]: "
                    f"{self.spent_eur + cost_eur:.4f}EUR > {self.limit_eur:.2f}EUR"
                )

    @property
    def remaining_eur(self) -> float:
        with self._lock:
            return max(0.0, self.limit_eur - self.spent_eur)

    @property
    def total_tokens(self) -> int:
        with self._lock:
            return self.prompt_tokens + self.completion_tokens

    @property
    def avg_latency_ms(self) -> float:
        with self._lock:
            if not self._latencies_ms:
                return 0.0
            return sum(self._latencies_ms) / len(self._latencies_ms)

    @property
    def last_latency_ms(self) -> float:
        with self._lock:
            return self._latencies_ms[-1] if self._latencies_ms else 0.0

    def reset(self) -> None:
        with self._lock:
            self.spent_eur = 0.0
            self.calls = 0
            self.failed_calls = 0
            self.prompt_tokens = 0
            self.completion_tokens = 0
            self._latencies_ms.clear()

    def summary(self) -> Dict[str, Any]:
        with self._lock:
            lat = list(self._latencies_ms)
        return {
            "provider": self.provider,
            "limit_eur": self.limit_eur,
            "spent_eur": self.spent_eur,
            "remaining_eur": max(0.0, self.limit_eur - self.spent_eur),
            "calls": self.calls,
            "failed_calls": self.failed_calls,
            "prompt_tokens": self.prompt_tokens,
            "completion_tokens": self.completion_tokens,
            "total_tokens": self.prompt_tokens + self.completion_tokens,
            "avg_latency_ms": (sum(lat) / len(lat)) if lat else 0.0,
            "last_latency_ms": lat[-1] if lat else 0.0,
        }


class DualBudgetGuard:
    """Thread-safe pair of :class:`BudgetGuard`\\ s (Meta + Baseten)."""

    def __init__(
        self,
        meta_limit_eur: Optional[float] = None,
        baseten_limit_eur: Optional[float] = None,
    ) -> None:
        self.meta = BudgetGuard(
            limit_eur=(
                META_BUDGET_LIMIT_EUR if meta_limit_eur is None else meta_limit_eur
            ),
            provider="meta",
        )
        self.baseten = BudgetGuard(
            limit_eur=(
                BASETEN_BUDGET_LIMIT_EUR
                if baseten_limit_eur is None
                else baseten_limit_eur
            ),
            provider="baseten",
        )

    def guard(self, provider: str) -> BudgetGuard:
        provider = provider.lower()
        if provider == "meta":
            return self.meta
        if provider == "baseten":
            return self.baseten
        raise ValueError(f"Unknown provider {provider!r} (expected meta/baseten)")

    def record(self, provider: str, **kwargs: Any) -> float:
        """Record one call against the named provider's guard."""
        return self.guard(provider).record(**kwargs)

    def summary(self) -> Dict[str, Any]:
        return {"meta": self.meta.summary(), "baseten": self.baseten.summary()}

    def reset(self) -> None:
        self.meta.reset()
        self.baseten.reset()
