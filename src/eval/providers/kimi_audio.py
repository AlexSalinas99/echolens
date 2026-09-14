"""Minimal synchronous Kimi-Audio-7B-Instruct call helper.

Kimi-Audio is a local audio-language model, run on-GPU like Qwen2.5-Omni-7B.
Unlike Qwen it is **not** a standard `transformers` checkpoint: it is loaded
through Moonshot's own ``kimia_infer`` package (installed from the
`Kimi-Audio` GitHub repo), which wraps the tokenizer, model, and (optional)
audio detokenizer behind a single ``KimiAudio`` class.

The model is loaded lazily on the first call. Loading the 7B checkpoint takes
~30-60s and requires a GPU with ~16 GB free; the helper raises a clear error
if the package or a CUDA device is missing.

Install note (done once by the user; not a repo dependency):

    git clone https://github.com/MoonshotAI/Kimi-Audio.git
    cd Kimi-Audio && pip install -e .

Determinism: text output uses ``text_temperature=0.0`` (greedy). Kimi always
runs its audio branch internally; we request ``output_type="text"`` and
discard the waveform, and we skip loading the audio detokenizer to save VRAM.
"""
from __future__ import annotations

import os
import sys
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MODEL: Any = None
_LOADED_ID: str | None = None


def _ensure_importable() -> None:
    """Make `kimia_infer` importable.

    Kimi-Audio is normally installed via `pip install -e .` from the repo, but
    its build pins torch==2.6.0 and drags in a full CUDA toolchain. To avoid
    clobbering an existing torch, we instead put the cloned repo on sys.path
    (it is pure Python). Point ``KIMI_AUDIO_REPO`` at the clone; defaults to
    ``~/Kimi-Audio``.
    """
    try:
        import kimia_infer  # noqa: F401
        return
    except Exception:
        repo = Path(os.environ.get("KIMI_AUDIO_REPO", Path.home() / "Kimi-Audio"))
        if repo.exists() and str(repo) not in sys.path:
            sys.path.insert(0, str(repo))


def _stub_flash_attn_if_broken() -> None:
    """The audio *detokenizer* imports flash_attn at module load, and a
    flash_attn built for a different torch fails with an ABI (undefined-symbol)
    error. We only produce text output (``load_detokenizer=False``,
    ``output_type='text'``), so the detokenizer / flash_attn code never runs.
    Install a no-op stub so the unused import succeeds. If a working flash_attn
    is present we leave it untouched.
    """
    try:
        import flash_attn  # noqa: F401
        return
    except Exception:
        import importlib.machinery
        import types
        stub = types.ModuleType("flash_attn")
        # A valid spec so importlib.util.find_spec(...) (used by transformers'
        # availability check) doesn't raise "flash_attn.__spec__ is None". The
        # distribution itself must be *uninstalled* so importlib.metadata.version
        # fails and transformers decides flash-attn-2 is unavailable → the main
        # model falls back to SDPA attention instead of calling these stubs.
        stub.__spec__ = importlib.machinery.ModuleSpec("flash_attn", loader=None)
        stub.__version__ = "0.0.0-stub"

        def _unavailable(*_a, **_k):
            raise RuntimeError("flash_attn stub called — not available in this env "
                               "(only the unused Kimi audio detokenizer needs it).")

        for fn in ("flash_attn_varlen_func", "flash_attn_varlen_qkvpacked_func",
                   "flash_attn_func", "flash_attn_qkvpacked_func"):
            setattr(stub, fn, _unavailable)
        sys.modules["flash_attn"] = stub


def is_available() -> bool:
    """True iff the kimia_infer package (on path) + a CUDA device are available."""
    try:
        import torch
        _ensure_importable()
        _stub_flash_attn_if_broken()
        import kimia_infer  # noqa: F401
    except Exception:
        return False
    return getattr(torch, "cuda", None) is not None and torch.cuda.is_available()


@dataclass
class CallResult:
    response_text: str
    model_version: str | None
    latency_s: float


def _ensure_loaded(model_id: str) -> None:
    global _MODEL, _LOADED_ID
    if _MODEL is not None and _LOADED_ID == model_id:
        return
    _ensure_importable()
    _stub_flash_attn_if_broken()
    from kimia_infer.api.kimia import KimiAudio

    # load_detokenizer=False: we only want text output, so skip the audio
    # detokenizer (saves VRAM and load time).
    _MODEL = KimiAudio(model_path=model_id, load_detokenizer=False)
    _LOADED_ID = model_id


def call_once(
    *,
    model_id: str,
    system_prompt: str,
    audio_path: Path | None,
    text_input: str | None,
    max_output_tokens: int = 600,
) -> CallResult:
    """One synchronous Kimi-Audio call covering both audio and text inputs.

    Mirrors the Qwen helper's contract: pass ``audio_path`` for direct_audio,
    or ``text_input`` for the text conditions (canonical_text /
    external_transcript). The system prompt is sent as a leading text turn.
    """
    if not is_available():
        raise RuntimeError(
            "Kimi-Audio requires the `kimia_infer` package + a CUDA device "
            "(pip install -e . from the MoonshotAI/Kimi-Audio repo)."
        )

    _ensure_loaded(model_id)
    assert _MODEL is not None

    # Kimi messages are a flat list of typed turns. We front-load the system
    # prompt as a text turn (Kimi has no dedicated system role), then the
    # user's audio or text.
    messages: list[dict] = [
        {"role": "user", "message_type": "text", "content": system_prompt},
    ]
    if audio_path is not None:
        messages.append({"role": "user", "message_type": "audio", "content": str(audio_path)})
    if text_input is not None:
        messages.append({"role": "user", "message_type": "text", "content": text_input})
    if len(messages) == 1:
        raise ValueError("call_once: one of audio_path, text_input must be provided")

    sampling_params = {
        "audio_temperature": 0.8,   # unused for text output, but the API expects it
        "audio_top_k": 10,
        "text_temperature": 0.0,    # greedy text decoding
        "text_top_k": 5,
        "max_new_tokens": max_output_tokens,
    }

    t0 = time.time()
    _, text = _MODEL.generate(messages, **sampling_params, output_type="text")
    dt = time.time() - t0
    return CallResult(
        response_text=(text or "").strip(),
        model_version=_LOADED_ID,
        latency_s=round(dt, 3),
    )
