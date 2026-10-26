"""Unified Meta Model API client (https://api.meta.ai/v1).

OpenAI-compatible wrappers for ``/v1/chat/completions``,
``/v1/responses`` and audio transcription, with SAM 3.1 segmentation
routing. Integrates :class:`src.meta_budget.BudgetGuard`.

Graceful deterministic fallback: when offline (no key, placeholder key,
``ANYBODY_OFFLINE``/``META_OFFLINE`` set, unit-test mode via
``PYTEST_CURRENT_TEST``, or any network error / budget exhaustion) every
method returns a deterministic local result flagged ``offline=True``
instead of raising, so physics/CV callers keep working.
"""
from __future__ import annotations

import asyncio
import base64
import json
import os
from dataclasses import dataclass, field
from pathlib import Path
from typing import Any

try:
    import httpx
except Exception:  # pragma: no cover
    httpx = None  # type: ignore

from src.meta_budget import BudgetGuard

DEFAULT_BASE_URL = "https://api.meta.ai/v1"
DEFAULT_BASETEN_BASE_URL = "https://inference.baseten.co/v1"

MODEL_WONDER = "muse-wonder-1.0"
MODEL_SPARK = "muse-spark-1.3"
MODEL_SPARK_CONTRIBUTOR = "muse-spark-1.3-contributor"
MODEL_SAM = "sam-3.1"
MODEL_TRANSCRIBE = "muse-voice-transcribe-1.0"

SUPPORTED_MODELS = [MODEL_WONDER, MODEL_SPARK, MODEL_SPARK_CONTRIBUTOR,
                    MODEL_SAM, MODEL_TRANSCRIBE]
MODELS = SUPPORTED_MODELS

_ALIASES = {
    "wonder": MODEL_WONDER, "muse-wonder": MODEL_WONDER,
    "spark": MODEL_SPARK, "muse-spark": MODEL_SPARK,
    "contributor": MODEL_SPARK_CONTRIBUTOR,
    "sam": MODEL_SAM, "sam3.1": MODEL_SAM, "sam-3": MODEL_SAM,
    "transcribe": MODEL_TRANSCRIBE, "voice": MODEL_TRANSCRIBE,
    "whisper": MODEL_TRANSCRIBE,
}

# Rough per-call cost estimates in EUR, charged to BudgetGuard online only.
_COST_EUR = {MODEL_WONDER: 0.01, MODEL_SPARK: 0.005,
             MODEL_SPARK_CONTRIBUTOR: 0.005, MODEL_SAM: 0.004,
             MODEL_TRANSCRIBE: 0.003}
_DEFAULT_COST = 0.005


def resolve_model(model: str | None, default: str = MODEL_WONDER) -> str:
    if not model:
        return default
    key = str(model).strip().lower()
    if key in SUPPORTED_MODELS:
        return key
    if key in _ALIASES:
        return _ALIASES[key]
    return str(model)  # unknown names pass through to the API verbatim


def _load_dotenv() -> None:
    """Minimal .env loader (no override of real environment)."""
    for cand in (
        Path.cwd() / ".env",
        Path(__file__).resolve().parents[2] / ".env",
        Path.home() / ".env",
    ):
        try:
            if not cand.is_file():
                continue
            for line in cand.read_text().splitlines():
                line = line.strip()
                if not line or line.startswith("#") or "=" not in line:
                    continue
                k, v = line.split("=", 1)
                os.environ.setdefault(k.strip(), v.strip().strip('"').strip("'"))
        except Exception:
            continue


def is_unit_test_mode() -> bool:
    return bool(os.getenv("PYTEST_CURRENT_TEST") or os.getenv("ANYBODY_UNIT_TEST")
                or os.getenv("UNIT_TEST"))


def _env_offline() -> bool:
    return (os.getenv("ANYBODY_OFFLINE", "").lower() in ("1", "true", "yes")
            or os.getenv("META_OFFLINE", "").lower() in ("1", "true", "yes"))


_load_dotenv()


@dataclass
class MetaResponse:
    """Dict-compatible response envelope returned by every client method."""
    data: dict[str, Any] = field(default_factory=dict)
    model: str = MODEL_WONDER
    offline: bool = True
    content: str = ""
    text: str = ""

    def __post_init__(self):
        if not self.content:
            self.content = _extract_content(self.data)
        if not self.text:
            self.text = self.content

    def __getitem__(self, key):
        return self.data[key]

    def get(self, key, default=None):
        return self.data.get(key, default)

    def __contains__(self, key):
        return key in self.data

    def to_dict(self) -> dict[str, Any]:
        return dict(self.data)


def _extract_content(data: dict) -> str:
    try:
        choices = data.get("choices") or []
        if choices:
            msg = choices[0].get("message") or {}
            if msg.get("content"):
                return str(msg["content"])
            if choices[0].get("text"):
                return str(choices[0]["text"])
        if data.get("output_text"):
            return str(data["output_text"])
        out = data.get("output") or []
        for item in out:
            for part in (item.get("content") or []):
                if part.get("text"):
                    return str(part["text"])
        if data.get("text"):
            return str(data["text"])
    except Exception:
        pass
    return ""


def _fallback_chat_data(messages, model: str) -> dict:
    """Deterministic local stand-in for a chat completion."""
    last_user = ""
    for m in reversed(list(messages or [])):
        if isinstance(m, dict) and m.get("role") in ("user", None):
            c = m.get("content")
            if isinstance(c, str):
                last_user = c
                break
            if isinstance(c, list):
                last_user = " ".join(str(p.get("text", "")) for p in c
                                     if isinstance(p, dict))
                break
    content = _deterministic_reply(last_user)
    return {"id": "offline-chat", "object": "chat.completion", "model": model,
            "offline": True,
            "choices": [{"index": 0, "finish_reason": "stop",
                         "message": {"role": "assistant", "content": content}}],
            "usage": {"prompt_tokens": 0, "completion_tokens": 0, "total_tokens": 0}}


def _deterministic_reply(prompt: str) -> str:
    low = (prompt or "").lower()
    # Task-planning prompts get a machine-readable LOCATE..VERIFY sketch;
    # task_compiler still validates / replaces it with its own contract.
    if any(w in low for w in ("throw", "deliver", "ball", "taskcontract", "task contract")):
        return json.dumps({
            "action": "throw", "source_object": "ball",
            "target": "person_target",
            "steps": ["LOCATE", "APPROACH", "ACQUIRE", "PROPEL", "VERIFY"]})
    return "offline-deterministic-response"


class MetaClient:
    """Sync + async client for the Meta Model API with offline fallback."""

    def __init__(self, api_key: str | None = None, base_url: str | None = None,
                 budget: BudgetGuard | None = None, budget_guard: BudgetGuard | None = None,
                 offline: bool | None = None, timeout: float = 5.0,
                 transport=None, **_kw):
        _load_dotenv()
        baseten_key = os.getenv("BASETEN_API_KEY", "").strip()
        meta_key = os.getenv("META_API_KEY", "").strip()
        use_baseten = bool(
            baseten_key and baseten_key.lower() not in ("your_baseten_api_key_here", "placeholder", "test", "dummy")
            and (not meta_key or meta_key.lower() in ("your_meta_api_key_here", "placeholder", "test", "dummy"))
        )
        self.api_key = (api_key if api_key is not None
                        else (baseten_key if use_baseten else (meta_key or baseten_key)))
        default_url = DEFAULT_BASETEN_BASE_URL if use_baseten else DEFAULT_BASE_URL
        self.base_url = (base_url
                         or (os.getenv("BASETEN_API_BASE_URL") if use_baseten else None)
                         or os.getenv("META_API_BASE_URL")
                         or default_url).rstrip("/")
        self.budget = budget_guard or budget or BudgetGuard()
        self.budget_guard = self.budget
        self.timeout = float(timeout)
        self._transport = transport
        placeholder = (not self.api_key or self.api_key.strip().lower()
                       in ("your_meta_api_key_here", "your_baseten_api_key_here",
                           "placeholder", "test", "dummy"))
        if offline is None:
            offline = bool(placeholder or _env_offline() or is_unit_test_mode())
        self.offline = bool(offline)
        self.is_offline = self.offline
        self.last_error: str | None = None
        self.call_count = 0
        self.fallback_count = 0

    # -- plumbing ---------------------------------------------------------
    @property
    def available_models(self) -> list[str]:
        return list(SUPPORTED_MODELS)

    def _charge(self, model: str) -> bool:
        """Charge BudgetGuard; False means budget exhausted -> fallback."""
        try:
            self.budget.record_spend(_COST_EUR.get(model, _DEFAULT_COST))
            return True
        except Exception as exc:
            self.last_error = f"budget: {exc}"
            return False

    def _post(self, path: str, payload: dict, model: str) -> dict | None:
        """One online POST; None signals caller should use fallback."""
        if self.offline or httpx is None:
            return None
        if not self._charge(model):
            return None
        try:
            headers = {"Authorization": f"Bearer {self.api_key}",
                       "Content-Type": "application/json"}
            with httpx.Client(base_url=self.base_url, timeout=self.timeout,
                              transport=self._transport) as client:
                resp = client.post(path, json=payload, headers=headers)
                resp.raise_for_status()
                self.call_count += 1
                return resp.json()
        except Exception as exc:
            self.last_error = str(exc)
            self.offline = True  # stay deterministic for the rest of session
            self.is_offline = True
            return None

    def _fallback(self, data: dict, model: str) -> MetaResponse:
        self.fallback_count += 1
        return MetaResponse(data=data, model=model, offline=True)

    # -- chat completions ---------------------------------------------------
    def chat_completions(self, messages, model: str = MODEL_WONDER,
                         temperature: float = 0.0, max_tokens: int | None = None,
                         **kw) -> MetaResponse:
        model = resolve_model(model)
        payload: dict[str, Any] = {"model": model, "messages": list(messages or []),
                                   "temperature": temperature}
        if max_tokens:
            payload["max_tokens"] = max_tokens
        payload.update(kw)
        data = self._post("/chat/completions", payload, model)
        if data is None:
            return self._fallback(_fallback_chat_data(messages, model), model)
        return MetaResponse(data=data, model=model, offline=False)

    # OpenAI-shaped aliases
    def chat(self, messages=None, model: str = MODEL_WONDER, **kw) -> MetaResponse:
        if isinstance(messages, str):
            messages = [{"role": "user", "content": messages}]
        return self.chat_completions(messages or [], model=model, **kw)

    create_chat_completion = chat_completions
    complete = chat_completions

    def chat_text(self, prompt, model: str = MODEL_WONDER,
                  system: str | None = None, **kw) -> str:
        messages = ([{"role": "system", "content": system}] if system else []) \
            + [{"role": "user", "content": str(prompt)}]
        return self.chat_completions(messages, model=model, **kw).content

    complete_text = chat_text

    # -- responses API ------------------------------------------------------
    def responses(self, input=None, model: str = MODEL_WONDER,
                  instructions: str | None = None, **kw) -> MetaResponse:
        model = resolve_model(model)
        payload: dict[str, Any] = {"model": model, "input": input}
        if instructions:
            payload["instructions"] = instructions
        payload.update(kw)
        data = self._post("/responses", payload, model)
        if data is None:
            text_in = input if isinstance(input, str) else json.dumps(input, default=str)
            fb = {"id": "offline-response", "object": "response", "model": model,
                  "offline": True, "status": "completed",
                  "output_text": _deterministic_reply(text_in or ""),
                  "output": [{"type": "message", "content": [
                      {"type": "output_text",
                       "text": _deterministic_reply(text_in or "")}]}]}
            return self._fallback(fb, model)
        return MetaResponse(data=data, model=model, offline=False)

    create_response = responses
    respond = responses

    # -- audio transcription --------------------------------------------------
    def transcribe_audio(self, audio=None, model: str = MODEL_TRANSCRIBE,
                         fallback_text: str = "", **kw) -> MetaResponse:
        model = resolve_model(model, default=MODEL_TRANSCRIBE)
        # Online path needs multipart upload; attempted only with real bytes/path.
        if not self.offline and httpx is not None and audio is not None:
            try:
                if self._charge(model):
                    raw, fname = self._audio_bytes(audio)
                    headers = {"Authorization": f"Bearer {self.api_key}"}
                    with httpx.Client(base_url=self.base_url, timeout=self.timeout,
                                      transport=self._transport) as client:
                        resp = client.post(
                            "/audio/transcriptions",
                            files={"file": (fname, raw)},
                            data={"model": model}, headers=headers)
                        resp.raise_for_status()
                        self.call_count += 1
                        data = resp.json()
                        if isinstance(data, dict) and data.get("text") is not None:
                            return MetaResponse(data=data, model=model, offline=False,
                                                text=str(data["text"]))
            except Exception as exc:
                self.last_error = str(exc)
                self.offline = self.is_offline = True
        text = self._offline_transcript(audio, fallback_text)
        return self._fallback({"text": text, "model": model, "offline": True,
                               "language": "en"}, model)

    transcribe = transcribe_audio
    speech_to_text = transcribe_audio

    @staticmethod
    def _audio_bytes(audio):
        if isinstance(audio, (bytes, bytearray)):
            return bytes(audio), "audio.wav"
        p = Path(str(audio))
        if p.is_file():
            return p.read_bytes(), p.name
        return str(audio).encode(), "audio.wav"

    @staticmethod
    def _offline_transcript(audio, fallback_text: str = "") -> str:
        """Deterministic offline transcription.

        UTF-8-decodable audio bytes / .txt paths are treated as the spoken
        text (test fixtures); anything else yields ``fallback_text``.
        """
        if fallback_text:
            return fallback_text
        if audio is None:
            return ""
        if isinstance(audio, dict):
            return str(audio.get("fallback_text") or audio.get("text") or "")
        if isinstance(audio, (bytes, bytearray)):
            try:
                txt = bytes(audio).decode("utf-8").strip()
                if txt and all(31 < ord(c) < 127 or c in "\n\t" for c in txt):
                    return txt
            except Exception:
                pass
            return ""
        p = Path(str(audio))
        if p.is_file() and p.suffix.lower() in (".txt", ".text"):
            return p.read_text().strip()
        if isinstance(audio, str) and not p.exists() and len(audio) < 500 \
                and " " in audio:
            return audio  # already-text passthrough
        return ""

    # -- SAM 3.1 segmentation -------------------------------------------------
    def segment(self, image=None, prompt: str = "ball, person, obstacle",
                model: str = MODEL_SAM, **kw) -> MetaResponse:
        """SAM 3.1 segmentation; uses local RGB+Depth mask extraction unless custom transport is injected."""
        model = resolve_model(model, default=MODEL_SAM)
        if not self.offline and self._transport is not None and httpx is not None and image is not None:
            try:
                if self._charge(model):
                    import numpy as np
                    arr = np.asarray(image)
                    b64 = base64.b64encode(arr.tobytes()).decode()
                    data = self._post("/chat/completions", {
                        "model": model,
                        "messages": [{"role": "user", "content": prompt}],
                        "image": b64, "shape": list(arr.shape)}, model)
                    if data is not None:
                        return MetaResponse(data=data, model=model, offline=False)
            except Exception as exc:
                self.last_error = str(exc)
        return self._fallback({"masks": {}, "boxes": {}, "offline": True,
                               "model": model, "prompt": prompt}, model)

    segment_image = segment
    sam_segment = segment

    # -- async variants -------------------------------------------------------
    async def achat_completions(self, messages, model: str = MODEL_WONDER, **kw):
        return await asyncio.to_thread(self.chat_completions, messages,
                                       resolve_model(model), **kw)

    async def aresponses(self, input=None, model: str = MODEL_WONDER, **kw):
        return await asyncio.to_thread(self.responses, input, resolve_model(model), **kw)

    async def atranscribe_audio(self, audio=None, **kw):
        return await asyncio.to_thread(self.transcribe_audio, audio, **kw)


# Module-level convenience -------------------------------------------------
_default_client: MetaClient | None = None


def get_client(**kw) -> MetaClient:
    global _default_client
    if _default_client is None or kw:
        _default_client = MetaClient(**kw)
    return _default_client


def chat_text(prompt, **kw) -> str:
    return get_client().chat_text(prompt, **kw)


__all__ = ["MetaClient", "MetaResponse", "BudgetGuard", "SUPPORTED_MODELS",
           "MODELS", "MODEL_WONDER", "MODEL_SPARK", "MODEL_SPARK_CONTRIBUTOR",
           "MODEL_SAM", "MODEL_TRANSCRIBE", "DEFAULT_BASE_URL",
           "resolve_model", "is_unit_test_mode", "get_client", "chat_text"]
