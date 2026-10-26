"""Dual-provider Meta + Baseten AI client (OpenAI-compatible chat APIs).

Loads ``.env`` automatically from the workspace root on import. All HTTP
round trips go through ``httpx`` with real end-to-end latency measurement;
every call is recorded against a :class:`DualBudgetGuard` (calls, tokens,
EUR spend, latency) and raises before spending past the budget.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path
from typing import Any, Dict, List, Optional

try:
    from ..meta_budget import DualBudgetGuard, estimate_cost_eur
except ImportError:  # pragma: no cover - direct script execution fallback
    from meta_budget import DualBudgetGuard, estimate_cost_eur

try:
    import httpx
except ImportError:  # pragma: no cover
    httpx = None  # type: ignore[assignment]


# ---------------------------------------------------------------- .env loading

def _find_workspace_root(start: Path) -> Path:
    for parent in [start, *start.parents]:
        if (parent / ".env").is_file():
            return parent
    return Path.cwd()


def load_dotenv(path: Optional[Path] = None) -> Optional[Path]:
    """Minimal ``.env`` loader (stdlib only): ``KEY=VALUE`` lines, no override."""
    env_path = Path(path) if path else _find_workspace_root(Path(__file__)) / ".env"
    if not env_path.is_file():
        return None
    for raw in env_path.read_text(encoding="utf-8").splitlines():
        line = raw.strip()
        if not line or line.startswith("#") or "=" not in line:
            continue
        key, _, value = line.partition("=")
        key, value = key.strip(), value.strip()
        if len(value) >= 2 and value[0] == value[-1] and value[0] in "\"'":
            value = value[1:-1]
        if key and key not in os.environ:
            os.environ[key] = value
    return env_path


load_dotenv()


# ------------------------------------------------------------------ constants

META_API_BASE_URL = "https://api.meta.ai/v1"
BASETEN_API_BASE_URL = "https://inference.baseten.co/v1"

# Meta API models.
META_WONDER = "muse-wonder-1.0"
META_SPARK = "muse-spark-1.3"
META_SPARK_CONTRIBUTOR = "muse-spark-1.3-contributor"
META_SAM = "sam-3.1"
META_VOICE_TRANSCRIBE = "muse-voice-transcribe-1.0"
META_IMAGE = "muse-image-1.0"

# Baseten FP4 models.
BASETEN_SPARK_OPEN_1026 = "meta-models/muse-spark-open-1026"
BASETEN_SPARK_OPEN = "meta-models/muse-spark-open"


class BudgetExceeded(RuntimeError):
    """Raised when a call would exceed the provider's EUR budget."""


class DualAIClient:
    """OpenAI-compatible client spanning the Meta and Baseten providers."""

    def __init__(
        self,
        meta_api_key: Optional[str] = None,
        baseten_api_key: Optional[str] = None,
        meta_base_url: Optional[str] = None,
        baseten_base_url: Optional[str] = None,
        meta_budget_eur: Optional[float] = None,
        baseten_budget_eur: Optional[float] = None,
        timeout_s: float = 120.0,
    ) -> None:
        if httpx is None:
            raise ImportError("httpx is required by DualAIClient")
        self.meta_api_key = meta_api_key or os.getenv("META_API_KEY", "")
        self.baseten_api_key = baseten_api_key or os.getenv("BASETEN_API_KEY", "")
        self.meta_base_url = (
            meta_base_url or os.getenv("META_API_BASE_URL", META_API_BASE_URL)
        ).rstrip("/")
        self.baseten_base_url = (
            baseten_base_url
            or os.getenv("BASETEN_API_BASE_URL", BASETEN_API_BASE_URL)
        ).rstrip("/")
        self.budgets = DualBudgetGuard(
            meta_limit_eur=meta_budget_eur, baseten_limit_eur=baseten_budget_eur
        )
        self.timeout_s = timeout_s

    # ------------------------------------------------------------- internals

    def _credentials(self, provider: str) -> tuple[str, str]:
        if provider == "meta":
            return self.meta_base_url, self.meta_api_key
        if provider == "baseten":
            return self.baseten_base_url, self.baseten_api_key
        raise ValueError(f"Unknown provider {provider!r}")

    def chat(
        self,
        messages: List[Dict[str, str]],
        model: str,
        provider: str = "meta",
        temperature: float = 0.2,
        max_tokens: int = 2048,
        extra_body: Optional[Dict[str, Any]] = None,
    ) -> Dict[str, Any]:
        """POST one OpenAI-compatible chat-completion request.

        Returns the raw JSON body plus ``_provider``, ``_model`` and
        ``_latency_ms``. Records real tokens/latency against the budget
        guard (failures recorded with zero cost).
        """
        base_url, api_key = self._credentials(provider)
        if not api_key:
            raise RuntimeError(f"No API key configured for provider {provider!r}")
        self.budgets.guard(provider).check()
        payload: Dict[str, Any] = {
            "model": model,
            "messages": messages,
            "temperature": temperature,
            "max_tokens": max_tokens,
        }
        if extra_body:
            payload.update(extra_body)
        t0 = time.perf_counter()
        try:
            with httpx.Client(timeout=self.timeout_s) as http:
                resp = http.post(
                    f"{base_url}/chat/completions",
                    headers={
                        "Authorization": f"Bearer {api_key}",
                        "Content-Type": "application/json",
                    },
                    json=payload,
                )
                resp.raise_for_status()
                body = resp.json()
        except Exception:
            latency_ms = (time.perf_counter() - t0) * 1000.0
            self.budgets.record(provider, latency_ms=latency_ms, failed=True)
            raise
        latency_ms = (time.perf_counter() - t0) * 1000.0
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        self.budgets.record(
            provider,
            cost_eur=estimate_cost_eur(prompt_tokens, completion_tokens, model),
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            latency_ms=latency_ms,
            model=model,
        )
        if isinstance(body, dict):
            body["_provider"] = provider
            body["_model"] = model
            body["_latency_ms"] = latency_ms
        return body

    @staticmethod
    def _message_text(body: Dict[str, Any]) -> str:
        try:
            return body["choices"][0]["message"].get("content", "") or ""
        except (KeyError, IndexError, TypeError):
            return ""

    @staticmethod
    def _reasoning_text(body: Dict[str, Any]) -> str:
        try:
            msg = body["choices"][0]["message"]
        except (KeyError, IndexError, TypeError):
            return ""
        for key in ("reasoning_content", "reasoning"):
            val = msg.get(key)
            if isinstance(val, str) and val:
                return val
        return ""

    @staticmethod
    def _parse_json_object(text: str) -> Dict[str, Any]:
        candidate = text.strip()
        if candidate.startswith("```"):
            lines = candidate.splitlines()
            lines = lines[1:] if len(lines) > 1 else []
            if lines and lines[-1].strip().startswith("```"):
                lines = lines[:-1]
            candidate = "\n".join(lines).strip()
            if candidate.lower().startswith("json"):
                candidate = candidate[4:].strip()
        return json.loads(candidate)

    # ---------------------------------------------------------- public API

    @staticmethod
    def _sanitize_tree_impedance(tree: Dict[str, Any]) -> Dict[str, Any]:
        imp = tree.get("impedance")
        if not isinstance(imp, dict):
            imp = {}
        try:
            kd = float(imp.get("K_d", 3000.0))
        except Exception:
            kd = 3000.0
        try:
            bd = float(imp.get("B_d", 0.25))
        except Exception:
            bd = 0.25
        if kd < 800.0 or kd > 3000.0:
            kd = max(1500.0, min(3000.0, kd if kd > 0 else 3000.0))
        if bd <= 0.0 or bd >= 1.2:
            imp["B_d_cartesian"] = bd
            bd = 0.25
        imp["K_d"] = kd
        imp["B_d"] = bd
        tree["impedance"] = imp
        return tree

    def compile_control_tree(
        self,
        task_prompt: str,
        morphology: str,
        tpe_summary: dict,
        temperature: float = 0.2,
        max_tokens: int = 8192,
    ) -> dict:
        """Synthesize a hierarchical Control Tree via ``muse-wonder-1.0``.

        Falls back to Baseten ``meta-models/muse-spark-open-1026`` when the
        Meta call fails. Returns the parsed model JSON (stages, target
        coordinates, impedance ``K_d``/``B_d``, contact schedule) annotated
        with ``_provider`` / ``_model`` / ``_latency_ms`` / ``_fallback``.
        Raises if both providers fail — never fabricates a tree.
        """
        system = (
            "You are the System-2 Mission & Dynamic Control Tree Architect for a "
            "whole-body operational-space robot controller. Keep internal reasoning "
            "brief (<250 tokens). Given the task, the robot morphology and its "
            "Tree Positional Encoding (TPE) summary, synthesize a hierarchical "
            "Control Tree. Respond with a SINGLE valid JSON object and nothing "
            "else, with keys: 'stages' (list of {'name','objective','priority',"
            "'duration_s','target_xyz':[x,y,z] per-stage 3D waypoint in meters}),"
            " 'target_coordinates' ({'ee_target_xyz':[x,y,z],"
            "'goal_zone_xyz':[x,y,z]}), 'impedance' ({'K_d': float in [1800,3000], "
            "'B_d': float in [0.15,0.45]}), 'contact_schedule' (list of "
            "{'phase','support_legs','weight_w01','duration_s'})."
        )
        user = (
            f"TASK: {task_prompt}\nMORPHOLOGY: {morphology}\n"
            f"TPE_SUMMARY: {json.dumps(tpe_summary)}"
        )
        messages = [
            {"role": "system", "content": system},
            {"role": "user", "content": user},
        ]
        last_error: Optional[Exception] = None
        try:
            body = self.chat(
                messages,
                model=META_WONDER,
                provider="meta",
                temperature=temperature,
                max_tokens=max_tokens,
            )
            tree = self._parse_json_object(self._message_text(body))
            tree["_provider"] = "meta"
            tree["_model"] = META_WONDER
            tree["_latency_ms"] = body.get("_latency_ms", 0.0)
            tree["_fallback"] = False
            return tree
        except Exception as exc:  # Meta failed -> Baseten FP4 fallback
            last_error = exc
        try:
            body = self.chat(
                messages,
                model=BASETEN_SPARK_OPEN_1026,
                provider="baseten",
                temperature=temperature,
                max_tokens=max_tokens,
            )
        except Exception as exc:
            raise RuntimeError(
                f"Both Meta ({META_WONDER}) and Baseten "
                f"({BASETEN_SPARK_OPEN_1026}) control-tree calls failed; "
                f"Baseten error: {exc}; Meta error: {last_error}"
            ) from exc
        try:
            tree = self._parse_json_object(self._message_text(body))
        except Exception as exc:
            raise RuntimeError(
                f"Both Meta ({META_WONDER}) and Baseten "
                f"({BASETEN_SPARK_OPEN_1026}) control-tree calls failed; "
                f"last parse error: {exc}; Meta error: {last_error}"
            ) from exc
        tree["_provider"] = "baseten"
        tree["_model"] = BASETEN_SPARK_OPEN_1026
        tree["_latency_ms"] = body.get("_latency_ms", 0.0)
        tree["_fallback"] = True
        return tree

    def run_baseten_fp4_stress_burst(
        self,
        prompt: str,
        model: str = BASETEN_SPARK_OPEN_1026,
        temperature: float = 0.7,
        max_tokens: int = 512,
    ) -> dict:
        """Call Baseten FP4 ``meta-models/muse-spark-open-1026`` for stress telemetry.

        Returns real token decode speed, latency, reasoning content and text.
        """
        body = self.chat(
            [{"role": "user", "content": prompt}],
            model=model,
            provider="baseten",
            temperature=temperature,
            max_tokens=max_tokens,
        )
        usage = body.get("usage", {}) if isinstance(body, dict) else {}
        prompt_tokens = int(usage.get("prompt_tokens", 0) or 0)
        completion_tokens = int(usage.get("completion_tokens", 0) or 0)
        latency_ms = float(body.get("_latency_ms", 0.0) or 0.0)
        decode_tok_per_s = (
            completion_tokens / (latency_ms / 1000.0) if latency_ms > 0 else 0.0
        )
        return {
            "provider": "baseten",
            "model": model,
            "content": self._message_text(body),
            "reasoning_content": self._reasoning_text(body),
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "latency_ms": latency_ms,
            "decode_tok_per_s": decode_tok_per_s,
        }

    def status_summary(self) -> dict:
        """Live budget & call telemetry for both Meta and Baseten."""
        return self.budgets.summary()
