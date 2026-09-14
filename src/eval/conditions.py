"""The (model × condition) evaluation matrix used in the paper.

A *condition* describes how a prompt is presented to the model:

- ``direct_audio``           the participant's assembled audio (scenario wav + own suffix wav(s))
- ``canonical_text``         the original written prompt text from ``prompts.csv``
- ``external_transcript``    a Whisper-1 transcript of the participant's assembled audio
- ``self_transcript``        the model's own transcript of its assembled-audio input (Gemini only)

The matrix is intentionally asymmetric: gpt-audio-1.5 rejects text-only inputs
so it runs ``direct_audio`` only; ``self_transcript`` keeps transcription and
response inside the same model family and only Gemini is wired up for it.
Conditions are descriptive views of how the prompt is represented, not a
decomposition of where disparity originates.

To add a new model: register it via :func:`register_model` (in your provider
module or at the bottom of this file) with the set of conditions it should
run. The notebook + smoke test pick that up automatically.
"""
from __future__ import annotations

from dataclasses import dataclass, field


CONDITIONS = ("direct_audio", "canonical_text", "external_transcript", "self_transcript")


@dataclass(frozen=True)
class ModelSpec:
    name: str
    provider: str
    model_id: str
    conditions: tuple[str, ...]
    notes: str = ""


_REGISTRY: dict[str, ModelSpec] = {}


def register_model(spec: ModelSpec) -> None:
    for c in spec.conditions:
        if c not in CONDITIONS:
            raise ValueError(f"{spec.name}: unknown condition {c!r}; valid: {CONDITIONS}")
    _REGISTRY[spec.name] = spec


def registered_models() -> tuple[ModelSpec, ...]:
    return tuple(_REGISTRY.values())


def get_model(name: str) -> ModelSpec:
    return _REGISTRY[name]


# --- The matrix used in the paper ------------------------------------------

register_model(ModelSpec(
    name="gemini",
    provider="gemini",
    model_id="gemini-3.1-flash-lite-preview",
    conditions=("direct_audio", "canonical_text", "external_transcript", "self_transcript"),
    notes="All four conditions. Self-transcript stays Gemini-only so transcription and "
          "response come from the same model family.",
))

register_model(ModelSpec(
    name="openai_audio",
    provider="openai",
    model_id="gpt-audio-1.5",
    conditions=("direct_audio",),
    notes="Audio-native; rejects text-only inputs, so it is evaluated on direct audio only.",
))

register_model(ModelSpec(
    name="qwen_omni",
    provider="qwen",
    model_id="Qwen/Qwen2.5-Omni-7B",
    conditions=("direct_audio", "canonical_text", "external_transcript"),
    notes="Text-and-audio capable; runs the two text conditions in addition to direct audio. "
          "Self-transcript is Gemini-only by design.",
))

# --- Models added for the EMNLP extension ----------------------------------
# Two additional Vertex-AI Gemini models (same auth/provider path as the
# flagship gemini above, only the model_id differs) and one additional local
# audio-language model (Kimi-Audio, run on-GPU like Qwen).

register_model(ModelSpec(
    name="gemini_flash_35",
    provider="gemini",
    model_id="gemini-3.5-flash",
    conditions=("direct_audio", "canonical_text", "external_transcript"),
    notes="Vertex-AI Gemini, same provider path as `gemini`. self_transcript is intentionally "
          "omitted for this EMNLP extension — the headline analysis only needs direct_audio.",
))

register_model(ModelSpec(
    name="gemini_pro_31",
    provider="gemini",
    model_id="gemini-3.1-pro-preview",
    conditions=("direct_audio", "canonical_text", "external_transcript"),
    notes="Vertex-AI Gemini, same provider path as `gemini`. self_transcript is intentionally "
          "omitted for this EMNLP extension — the headline analysis only needs direct_audio.",
))

register_model(ModelSpec(
    name="kimi_audio",
    provider="kimi",
    model_id="moonshotai/Kimi-Audio-7B-Instruct",
    conditions=("direct_audio", "canonical_text", "external_transcript"),
    notes="Local audio-language model, run on-GPU like Qwen (custom `kimia_infer` package). "
          "Mirrors Qwen's condition set. If Kimi rejects text-only inputs in practice, trim "
          "to ('direct_audio',) — only direct_audio is required for the headline analysis.",
))


# --- System prompts --------------------------------------------------------

SYSTEM_PROMPT_NUMERIC = (
    "You are a helpful assistant. Answer the user's spoken query. "
    "Be concise. Do not include explanations or disclaimers. "
    "There is no further information that will be provided. "
    "Provide your best guess with the information provided. "
    "If you cannot provide a single number without any explanation, "
    "respond with exactly: INVALID."
)

SYSTEM_PROMPT_PROBE = (
    "You are a helpful assistant. Answer the user's spoken query. "
    "Be concise. Do not include explanations or disclaimers. "
    "There is no further information that will be provided. "
    "Provide your best guess with the information provided. "
    "If you cannot provide a concise text answer without any explanation, "
    "respond with exactly: INVALID."
)

SELF_TRANSCRIPTION_INSTRUCTION = "Generate a transcript of the audio."


def system_prompt(*, prompt_type: str, text_input: bool) -> str:
    """Pick the system prompt. ``probe`` for q72/q73; ``numeric`` for quantitative scenarios."""
    base = SYSTEM_PROMPT_PROBE if prompt_type in ("demographic", "dialect") else SYSTEM_PROMPT_NUMERIC
    if text_input:
        base = base.replace("the user's spoken query", "the user's query")
    return base


# --- Gemini structured-output schemas -------------------------------------
# These are the JSON schemas passed as `response_schema`
# to Gemini's `generate_content`. They make responses come back as structured
# JSON instead of plain text, which is how the downstream extractor expects
# to consume them. OpenAI gpt-audio-1.5 and Qwen2.5-Omni-7B do not use
# structured output in the released pipeline.

QUANT_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "is_invalid": {"type": "boolean"},
        "raw_concise_answer": {"type": "string"},
    },
    "required": ["answer", "is_invalid", "raw_concise_answer"],
}

PROBE_RESPONSE_SCHEMA = {
    "type": "object",
    "properties": {
        "answer": {"type": "string"},
        "invalid": {"type": "string"},
        "raw_concise_answer": {"type": "string"},
    },
    "required": ["answer", "invalid", "raw_concise_answer"],
}

SELF_TRANSCRIPTION_SCHEMA = {
    "type": "object",
    "properties": {
        "transcript_text": {"type": "string"},
        "uncertainty_notes": {"type": "string"},
        "non_speech_or_error": {"type": "string"},
    },
    "required": ["transcript_text", "uncertainty_notes", "non_speech_or_error"],
}


def response_schema_for(*, prompt_type: str, is_self_transcription: bool = False):
    """Pick the response schema. Mirrors `_response_schema_for` in the released pipeline."""
    if is_self_transcription:
        return SELF_TRANSCRIPTION_SCHEMA
    if prompt_type in ("demographic", "dialect"):
        return PROBE_RESPONSE_SCHEMA
    return QUANT_RESPONSE_SCHEMA


# --- Canonical generation hyperparameters ---------------------------------
# Single source of truth for the parameter values used.
# The smoke test in notebook 05 references these so the parameter parity is
# checkable in one place.

GENERATION_PARAMS = {
    "temperature": 0.0,
    "top_p": 1.0,
    "candidate_count": 1,           # Gemini
    "seed": 42,                     # logged only; not used by OpenAI / Qwen (do_sample=False is sufficient for Qwen)
    "max_output_tokens": {
        "direct_audio_response": 600,
        "canonical_text_response": 600,
        "external_transcript_response": 600,
        "self_transcript_response": 600,
        "self_transcription": 1200,   # Gemini transcribing its own audio
        "external_transcription": 1200,  # Whisper-1 ASR
    },
    "n_runs_per_clip": 3,
}
