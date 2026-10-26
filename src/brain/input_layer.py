"""Unified multimodal input ingestion.

Accepts vague text, spoken audio (via ``muse-voice-transcribe-1.0``) or
video frames (one-shot demonstration / webcam) and normalizes all of them
into a :class:`UnifiedInput` that the body-blind task compiler consumes.
All paths degrade deterministically offline.
"""
from __future__ import annotations

import hashlib
from dataclasses import dataclass, field, asdict
from pathlib import Path
from typing import Any

import numpy as np


@dataclass
class UnifiedInput:
    modality: str = "text"           # text | audio | video
    text: str = ""                   # normalized instruction text
    transcript: str = ""             # audio transcript (== text for audio)
    raw_input: Any = None
    frames: list = field(default_factory=list)
    frame_count: int = 0
    video_summary: str = ""
    offline: bool = True
    model: str = ""
    metadata: dict[str, Any] = field(default_factory=dict)

    def to_dict(self) -> dict[str, Any]:
        d = asdict(self)
        d.pop("frames", None)
        d.pop("raw_input", None)
        return d

    def __getitem__(self, key):
        return getattr(self, key)

    def get(self, key, default=None):
        return getattr(self, key, default)


def _summarize_frames(frames) -> str:
    """Deterministic CV summary of demonstration frames.

    Colour heuristics map the arena's objects to a spoken-style
    description: a red sphere is the ball, blue is the person.
    """
    arrs = [np.asarray(f) for f in (frames or []) if f is not None]
    arrs = [a for a in arrs if a.size]
    if not arrs:
        return ""
    try:
        stack = np.stack([a[..., :3].astype(float) for a in arrs
                          if a.ndim == 3 and a.shape[-1] >= 3])
    except Exception:
        return f"video with {len(arrs)} frames"
    mean = stack.mean(axis=(0, 1, 2))
    has_red = bool(((stack[..., 0] > 120) & (stack[..., 1] < 100)
                    & (stack[..., 2] < 100)).mean() > 0.001)
    has_blue = bool(((stack[..., 2] > 100) & (stack[..., 0] < 120)).mean() > 0.001)
    parts = []
    if has_red:
        parts.append("a red ball is visible")
    if has_blue:
        parts.append("a person is visible")
    if has_red and has_blue:
        return "throw the ball to the person (video demonstration: " \
            + " and ".join(parts) + ")"
    if parts:
        return "demonstration: " + " and ".join(parts)
    return f"video demonstration with {len(arrs)} frames (mean RGB {mean.round(1).tolist()})"


class InputLayer:
    """Normalize text / audio / video into :class:`UnifiedInput`."""

    def __init__(self, client=None, **kw):
        if client is None and not kw.get("no_client"):
            try:
                from src.api.meta_client import MetaClient
                client = MetaClient()
            except Exception:
                client = None
        self.client = client

    # -- text ---------------------------------------------------------------
    def ingest_text(self, text: str) -> UnifiedInput:
        text = str(text or "").strip()
        return UnifiedInput(modality="text", text=text, raw_input=text,
                            offline=bool(getattr(self.client, "offline", True)))

    from_text = ingest_text

    # -- audio --------------------------------------------------------------
    def ingest_audio(self, audio=None, fallback_text: str = "", **kw) -> UnifiedInput:
        transcript, offline = "", True
        if isinstance(audio, dict):
            fallback_text = fallback_text or str(
                audio.get("fallback_text") or audio.get("text") or "")
            audio = audio.get("audio", audio.get("path"))
        if self.client is not None:
            try:
                resp = self.client.transcribe_audio(
                    audio, fallback_text=fallback_text)
                transcript = resp.text or resp.get("text", "") or ""
                offline = bool(resp.offline)
            except Exception:
                transcript = fallback_text
        if not transcript:
            transcript = fallback_text
            # Last-resort local decode mirrors client offline behaviour.
            if not transcript and isinstance(audio, (bytes, bytearray)):
                try:
                    transcript = bytes(audio).decode("utf-8").strip()
                except Exception:
                    transcript = ""
            if not transcript and isinstance(audio, str) and Path(audio).is_file() \
                    and Path(audio).suffix.lower() in (".txt", ".text"):
                transcript = Path(audio).read_text().strip()
        return UnifiedInput(modality="audio", text=transcript,
                            transcript=transcript, raw_input=audio,
                            offline=offline, model="muse-voice-transcribe-1.0")

    from_audio = ingest_audio
    ingest_speech = ingest_audio

    # -- video ----------------------------------------------------------------
    def ingest_video(self, frames=None, text: str = "throw the ball to the person",
                     **kw) -> UnifiedInput:
        if frames is None:
            frames = kw.get("video") or kw.get("clip") or []
        if isinstance(frames, np.ndarray) and frames.ndim == 4:
            frame_list = [frames[i] for i in range(frames.shape[0])]
        elif isinstance(frames, np.ndarray):
            frame_list = [frames]
        elif isinstance(frames, (list, tuple)):
            frame_list = list(frames)
        else:
            frame_list = [frames] if frames is not None else []
        summary = _summarize_frames(frame_list)
        # An explicit caption / instruction wins; the CV summary enriches it.
        final_text = text or summary
        if summary and text and "video demonstration" in summary \
                and summary != text:
            final_text = text
        digest = hashlib.sha1(
            f"{len(frame_list)}:{final_text}".encode()).hexdigest()[:8]
        return UnifiedInput(modality="video", text=final_text,
                            raw_input=f"video:{digest}", frames=frame_list,
                            frame_count=len(frame_list), video_summary=summary,
                            offline=bool(getattr(self.client, "offline", True)),
                            metadata={"video_digest": digest})

    from_video = ingest_video
    ingest_frames = ingest_video

    # -- unified dispatch -------------------------------------------------------
    def ingest(self, data=None, **kw) -> UnifiedInput:
        modality = str(kw.pop("modality", "") or "").lower()
        if data is None:
            data = kw.pop("input", None) or kw.pop("text", None) \
                or kw.pop("audio", None) or kw.pop("frames", None) or ""
        if modality == "audio" or kw.get("is_audio"):
            return self.ingest_audio(data, **kw)
        if modality == "video" or kw.get("is_video"):
            return self.ingest_video(data, **kw)
        if isinstance(data, UnifiedInput):
            return data
        if isinstance(data, dict):
            if data.get("modality") == "audio" or "audio" in data:
                return self.ingest_audio(data, **kw)
            if data.get("modality") == "video" or "frames" in data:
                return self.ingest_video(data.get("frames") or data.get("video"),
                                         text=data.get("text", "throw the ball to the person"), **kw)
            return self.ingest_text(str(data.get("text") or data.get("transcript") or ""))
        if isinstance(data, (bytes, bytearray)):
            return self.ingest_audio(data, **kw)
        if isinstance(data, np.ndarray):
            return self.ingest_video(data, **kw)
        if isinstance(data, (list, tuple)) and data and isinstance(
                np.asarray(data[0]) if not isinstance(data[0], str) else "", np.ndarray) \
                and np.asarray(data[0]).size > 4:
            return self.ingest_video(list(data), **kw)
        return self.ingest_text(str(data))

    __call__ = ingest
    process = ingest
    normalize = ingest


def ingest(data=None, **kw) -> UnifiedInput:
    return InputLayer().ingest(data, **kw)


__all__ = ["InputLayer", "UnifiedInput", "ingest"]
