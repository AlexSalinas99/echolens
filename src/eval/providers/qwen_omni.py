"""Minimal synchronous Qwen2.5-Omni-7B call helper for the smoke test.

The model is loaded lazily on the first call. Loading the 7B checkpoint
takes ~30-60s and requires a GPU with ~16 GB free; the helper raises a
clear error if that's not available.
"""
from __future__ import annotations

import os
import time
from dataclasses import dataclass
from pathlib import Path
from typing import Any


_MODEL: Any = None
_PROCESSOR: Any = None
_LOADED_ID: str | None = None


def is_available() -> bool:
    """True iff transformers + a CUDA device are available."""
    try:
        import torch
        import transformers  # noqa: F401
    except Exception:
        return False
    return getattr(torch, "cuda", None) is not None and torch.cuda.is_available()


@dataclass
class CallResult:
    response_text: str
    model_version: str | None
    latency_s: float


def _ensure_loaded(model_id: str) -> None:
    global _MODEL, _PROCESSOR, _LOADED_ID
    if _MODEL is not None and _LOADED_ID == model_id:
        return
    import torch
    from transformers import Qwen2_5OmniForConditionalGeneration, Qwen2_5OmniProcessor

    processor = Qwen2_5OmniProcessor.from_pretrained(model_id)
    model = Qwen2_5OmniForConditionalGeneration.from_pretrained(
        model_id,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
    )
    # Skip waveform synthesis — we only want the text output.
    if hasattr(model, "disable_talker"):
        model.disable_talker()
    _MODEL, _PROCESSOR, _LOADED_ID = model, processor, model_id


def call_once(
    *,
    model_id: str,
    system_prompt: str,
    audio_path: Path | None,
    text_input: str | None,
    max_output_tokens: int = 600,
) -> CallResult:
    """One synchronous Qwen2.5-Omni call covering both audio and text inputs."""
    if not is_available():
        raise RuntimeError("Qwen2.5-Omni-7B requires transformers + a CUDA device")
    import numpy as np
    import torch

    _ensure_loaded(model_id)
    assert _MODEL is not None and _PROCESSOR is not None

    user_content: list[dict] = []
    if audio_path is not None:
        user_content.append({"type": "audio", "audio": str(audio_path)})
    if text_input is not None:
        user_content.append({"type": "text", "text": text_input})
    if not user_content:
        raise ValueError("call_once: one of audio_path, text_input must be provided")

    messages = [
        {"role": "system", "content": [{"type": "text", "text": system_prompt}]},
        {"role": "user", "content": user_content},
    ]
    text_in = _PROCESSOR.apply_chat_template(messages, add_generation_prompt=True, tokenize=False)

    proc_kwargs: dict[str, Any] = {"text": text_in, "return_tensors": "pt"}
    if audio_path is not None:
        import librosa
        audio_array, _sr = librosa.load(str(audio_path), sr=16000, mono=True)
        proc_kwargs["audio"] = [audio_array.astype(np.float32)]
        proc_kwargs["padding"] = True
    inputs = _PROCESSOR(**proc_kwargs).to(_MODEL.device)

    t0 = time.time()
    with torch.inference_mode():
        out = _MODEL.generate(
            **inputs,
            do_sample=False,
            thinker_max_new_tokens=max_output_tokens,
            return_audio=False,
            use_cache=True,
        )
    dt = time.time() - t0

    out_ids = out[0] if isinstance(out, tuple) else out
    if "input_ids" in inputs:
        out_ids = out_ids[:, inputs["input_ids"].shape[-1]:]
    text = _PROCESSOR.batch_decode(out_ids, skip_special_tokens=True)[0]
    return CallResult(response_text=text.strip(), model_version=_LOADED_ID, latency_s=round(dt, 3))
