"""Minimal synchronous Gemini call helper for the smoke test.

Supports both auth paths used in practice:

- **Vertex AI (preferred)** — picked when ``GEMINI_VERTEX_PROJECT`` is set in
  the environment. Authenticates via Application Default Credentials (run
  ``gcloud auth application-default login`` once, or run inside a GCE/GKE
  pod with the right service account). This is how the released outputs in
  ``data/model_outputs/responses/gemini/`` were produced. The Vertex path
  is also what supports the Batch API at scale.
- **API key (fallback)** — picked when ``GOOGLE_API_KEY`` or ``GEMINI_API_KEY``
  is set and the Vertex env vars are not. Useful when you just want to run
  a single ad-hoc call (e.g. on a laptop) without a GCP project.

Everything here is one-call-deep on purpose.

Environment variables read:

- ``GEMINI_VERTEX_PROJECT``   GCP project for Vertex AI (e.g. ``your-gcp-project``)
- ``GEMINI_VERTEX_LOCATION``  Vertex location/region; defaults to ``global``
- ``GOOGLE_API_KEY`` / ``GEMINI_API_KEY``  ad-hoc API key for AI Studio
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path


def _auth_mode() -> str | None:
    """Return 'vertex' if Vertex is configured, 'api_key' if a key is set, else None."""
    if os.environ.get("GEMINI_VERTEX_PROJECT"):
        return "vertex"
    if os.environ.get("GOOGLE_API_KEY") or os.environ.get("GEMINI_API_KEY"):
        return "api_key"
    return None


def is_available() -> bool:
    """True iff either Vertex env vars OR an API key are present.

    For Vertex this only checks that the project env var is set; it does
    not verify that Application Default Credentials actually work. A real
    call below will surface ADC errors if creds are stale or missing.
    """
    return _auth_mode() is not None


def _build_client():
    """Construct a google-genai Client using whichever auth path is configured."""
    from google import genai

    mode = _auth_mode()
    if mode == "vertex":
        project = os.environ["GEMINI_VERTEX_PROJECT"]
        location = os.environ.get("GEMINI_VERTEX_LOCATION", "global")
        return genai.Client(vertexai=True, project=project, location=location), mode
    if mode == "api_key":
        key = os.environ.get("GOOGLE_API_KEY") or os.environ["GEMINI_API_KEY"]
        return genai.Client(api_key=key), mode
    raise RuntimeError(
        "Gemini not configured. Either set GEMINI_VERTEX_PROJECT (preferred, "
        "uses Application Default Credentials) or set GOOGLE_API_KEY / GEMINI_API_KEY."
    )


@dataclass
class CallResult:
    response_text: str
    model_version: str | None
    latency_s: float
    auth_mode: str   # 'vertex' or 'api_key'


def call_once(
    *,
    model_id: str,
    condition: str,
    system_prompt: str,
    audio_path: Path | None,
    text_input: str | None,
    max_output_tokens: int = 600,
    response_schema: dict | None = None,
    temperature: float = 0.0,
    top_p: float = 1.0,
) -> CallResult:
    """One synchronous Gemini call covering all four conditions.

    Parameters: ``temperature=0.0``, ``top_p=1.0``,
    ``candidate_count=1``, ``max_output_tokens`` from ``conditions.GENERATION_PARAMS``
    (600 for response conditions, 1200 for self_transcription), and a structured-
    output ``response_schema`` (with ``response_mime_type='application/json'``) so
    responses come back as JSON in the same shape as the released JSONLs.

    - ``direct_audio``: ``audio_path`` set, ``text_input`` None.
    - ``canonical_text`` / ``external_transcript``: ``text_input`` set,
      ``audio_path`` None.
    - ``self_transcript``: same shape as a text condition (the caller has
      already produced the self-transcript text upstream — for example
      with :func:`transcribe_audio_self` below).
    """
    from google.genai import types as gtypes

    client, mode = _build_client()
    gc_kwargs: dict = dict(
        max_output_tokens=max_output_tokens,
        temperature=temperature,
        top_p=top_p,
        candidate_count=1,
        system_instruction=system_prompt,
    )
    if response_schema is not None:
        gc_kwargs["response_mime_type"] = "application/json"
        gc_kwargs["response_schema"] = response_schema
    gc = gtypes.GenerateContentConfig(**gc_kwargs)

    parts: list = []
    if audio_path is not None:
        if text_input is not None:
            raise ValueError("Gemini smoke test passes either audio or text, not both.")
        parts.append(
            gtypes.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type="audio/wav")
        )
    elif text_input is not None:
        parts.append(gtypes.Part.from_text(text=text_input))
    else:
        raise ValueError("call_once: one of audio_path, text_input must be provided")
    contents = gtypes.Content(role="user", parts=parts)

    t0 = time.time()
    
    resp = client.models.generate_content(model=model_id, contents=[contents], config=gc)
    dt = time.time() - t0
    text = getattr(resp, "text", None) or ""
    version = getattr(resp, "model_version", None)
    return CallResult(
        response_text=text.strip(),
        model_version=version,
        latency_s=round(dt, 3),
        auth_mode=mode,
    )


# ---- self-transcription (drives the `self_transcript` condition) ---------

SELF_TRANSCRIPTION_INSTRUCTION = "Generate a transcript of the audio."


def transcribe_audio_self(
    *,
    model_id: str,
    audio_path: Path,
    max_output_tokens: int = 1200,
    response_schema: dict | None = None,
) -> CallResult:
    """Have Gemini transcribe its own audio input.

    The instruction *"Generate a transcript of
    the audio."* is sent as the user-content text alongside the audio bytes
    (not as a system instruction), with `max_output_tokens=1200` and the
    `SELF_TRANSCRIPTION_SCHEMA` for structured output. This call produces the
    strings stored under
    ``data/model_outputs/transcripts/self/gemini/<model_id>/`` that drive the
    ``self_transcript`` condition: the resulting text is fed back to Gemini
    in a second call as the user message.
    """
    from google.genai import types as gtypes

    client, mode = _build_client()
    gc_kwargs: dict = dict(
        max_output_tokens=max_output_tokens,
        temperature=0.0,
        top_p=1.0,
        candidate_count=1,
    )
    if response_schema is not None:
        gc_kwargs["response_mime_type"] = "application/json"
        gc_kwargs["response_schema"] = response_schema
    gc = gtypes.GenerateContentConfig(**gc_kwargs)

    contents = gtypes.Content(
        role="user",
        parts=[
            gtypes.Part.from_text(text=SELF_TRANSCRIPTION_INSTRUCTION),
            gtypes.Part.from_bytes(data=Path(audio_path).read_bytes(), mime_type="audio/wav"),
        ],
    )
    t0 = time.time()
    resp = client.models.generate_content(model=model_id, contents=[contents], config=gc)
    dt = time.time() - t0
    text = getattr(resp, "text", None) or ""
    version = getattr(resp, "model_version", None)
    return CallResult(
        response_text=text.strip(),
        model_version=version,
        latency_s=round(dt, 3),
        auth_mode=mode,
    )
