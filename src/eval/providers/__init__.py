"""One synchronous call_once() per model family.

Each provider exposes:

- ``is_available()``  → bool : true iff the credentials / hardware it needs are present
- ``call_once(**kwargs)`` → CallResult

These are intentionally minimal — no retries, no batch APIs, no structured
output schemas. They demonstrate the wiring.
"""
from . import asr_openai, gemini, kimi_audio, openai_audio, qwen_omni  # noqa: F401
