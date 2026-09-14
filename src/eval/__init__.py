"""Audio-bias evaluation primitives — minimal, notebook-friendly.

Submodules:
    conditions     : the (model, condition) matrix + system prompts
    input_prep     : assemble audio / pick text input for a (participant, prompt, condition)
    providers/     : one synchronous call_once() helper per model family
    extraction     : layered response normalizer (Gemini-structured / rule / gpt-4o-mini)
"""
from . import conditions, disparity, extraction, input_prep, providers  # noqa: F401
