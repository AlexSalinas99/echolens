"""OpenAI Whisper-1 ASR helper — reproduces the ``external_transcript`` source.

The Whisper-1 transcripts shipped under
``data/model_outputs/transcripts/external/whisper-1/`` were produced by
calling ``client.audio.transcriptions.create(model='whisper-1', file=...)``
on each assembled-audio wav. This helper does one such call at a time.

We use the traditional Whisper-1 endpoint.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


WHISPER_MODEL_ID = "whisper-1"


def is_available() -> bool:
    return bool(os.environ.get("OPENAI_API_KEY"))


@dataclass
class TranscriptionResult:
    text: str
    latency_s: float
    model_id: str = WHISPER_MODEL_ID


def transcribe_once(*, audio_path: Path) -> TranscriptionResult:
    """One synchronous Whisper-1 call."""
    from openai import OpenAI

    api_key = os.environ.get("OPENAI_API_KEY")
    if not api_key:
        raise RuntimeError("set OPENAI_API_KEY to call Whisper-1")
    client = OpenAI(api_key=api_key)

    t0 = time.time()
    with open(audio_path, "rb") as f:
        # response_format='json'; the OpenAI
        # client returns a Transcription object whose .text field is the transcript.
        resp = client.audio.transcriptions.create(
            model=WHISPER_MODEL_ID,
            file=f,
            response_format="json",
        )
    dt = time.time() - t0
    text = resp if isinstance(resp, str) else getattr(resp, "text", "")
    return TranscriptionResult(text=str(text).strip(), latency_s=round(dt, 3))
