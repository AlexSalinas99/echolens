"""Minimal synchronous gpt-audio-1.5 call helper for the smoke test.

gpt-audio-1.5 is audio-native; in the paper it is evaluated only on the
``direct_audio`` condition. This helper mirrors that contract.
"""
from __future__ import annotations

import base64
import os
import time
from dataclasses import dataclass
from pathlib import Path


def is_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@dataclass
class CallResult:
    response_text: str
    model_version: str | None
    latency_s: float


def call_once(
    *,
    model_id: str,
    system_prompt: str,
    audio_path: Path,
    max_output_tokens: int = 600,
) -> CallResult:
    """One synchronous chat-completions call with input_audio (base64)."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY to call gpt-audio-1.5")

    fmt = Path(audio_path).suffix.lstrip(".").lower()
    if fmt not in {"wav", "mp3"}:
        raise ValueError(f"unsupported audio format: {fmt}")
    b64 = base64.b64encode(Path(audio_path).read_bytes()).decode()

    client = OpenAI(api_key=api_key)
    t0 = time.time()
    resp = client.chat.completions.create(
        model=model_id,
        modalities=["text"],
        max_tokens=max_output_tokens,
        messages=[
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": [
                {"type": "input_audio", "input_audio": {"data": b64, "format": fmt}},
            ]},
        ],
    )
    dt = time.time() - t0
    return CallResult(
        response_text=(resp.choices[0].message.content or "").strip(),
        model_version=resp.model,
        latency_s=round(dt, 3),
    )
