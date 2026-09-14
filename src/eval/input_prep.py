"""Input construction for an (participant, prompt, condition) evaluation item.

Two responsibilities:

1. **Audio assembly** — concatenate the scenario wav with the participant's
   own suffix wavs (q70 = suffix_number, q71 = suffix_additional) when the
   prompt's flags require it. Demographic (q72) and dialect (q73) probes are
   always returned standalone. Concatenation is a literal raw-PCM frame
   append: no inserted silence, no crossfading, no amplitude normalization.

2. **Text selection** — for the text-input conditions, return the string the
   downstream model should receive:

   - ``canonical_text``        : the prompt's original written text from prompts.csv
   - ``external_transcript``   : the participant's Whisper-1 transcript (looked up from disk)
   - ``self_transcript``       : the model's own transcript (produced upstream by a
     separate Gemini call; here we just provide a lookup hook)
"""
from __future__ import annotations

import wave
from dataclasses import dataclass
from pathlib import Path

import pandas as pd


CANONICAL_SR = 16_000
CANONICAL_NCHANNELS = 1
CANONICAL_SAMPWIDTH = 2  # bytes; 16-bit PCM


# ---- Audio assembly -------------------------------------------------------

@dataclass(frozen=True)
class AssembledAudio:
    audio_path: Path
    suffix_components: tuple[str, ...]
    duration_s: float


def _check_canonical(p: Path) -> wave._wave_params:
    with wave.open(str(p), "rb") as r:
        params = r.getparams()
    if (params.framerate != CANONICAL_SR
            or params.nchannels != CANONICAL_NCHANNELS
            or params.sampwidth != CANONICAL_SAMPWIDTH):
        raise RuntimeError(
            f"{p} has non-canonical params {params}; expected "
            f"{CANONICAL_SR} Hz / {CANONICAL_NCHANNELS} ch / "
            f"{CANONICAL_SAMPWIDTH * 8}-bit PCM."
        )
    return params


def assemble_audio(
    *,
    participant_id: str,
    cohort: str,
    question_id: int,
    audio_root: Path,
    suffix_number: bool,
    suffix_additional: bool,
    cache_dir: Path | None = None,
) -> AssembledAudio:
    """Return the wav path to feed the model for this (participant, prompt).

    For q72 / q73 the standalone wav is returned regardless of the suffix
    flags. Otherwise the scenario wav is concatenated with q70 and/or q71
    as the per-prompt flags require, and the result is cached under
    ``cache_dir`` (created if absent).
    """
    pid_dir = audio_root / cohort / participant_id
    main_wav = pid_dir / f"{participant_id}_q{question_id:02d}.wav"
    if not main_wav.exists():
        raise FileNotFoundError(main_wav)

    # Probes are always standalone.
    if question_id in (72, 73):
        params = _check_canonical(main_wav)
        with wave.open(str(main_wav), "rb") as r:
            duration_s = r.getnframes() / params.framerate
        return AssembledAudio(audio_path=main_wav, suffix_components=(), duration_s=duration_s)

    components: list[str] = []
    suffix_paths: list[Path] = []
    if suffix_number:
        p = pid_dir / f"{participant_id}_q70.wav"
        if not p.exists():
            raise FileNotFoundError(f"suffix_number wav missing for {participant_id}: {p}")
        components.append("q70")
        suffix_paths.append(p)
    if suffix_additional:
        p = pid_dir / f"{participant_id}_q71.wav"
        if not p.exists():
            raise FileNotFoundError(f"suffix_additional wav missing for {participant_id}: {p}")
        components.append("q71")
        suffix_paths.append(p)

    # No suffixes needed → return scenario wav as-is.
    if not components:
        params = _check_canonical(main_wav)
        with wave.open(str(main_wav), "rb") as r:
            duration_s = r.getnframes() / params.framerate
        return AssembledAudio(audio_path=main_wav, suffix_components=(), duration_s=duration_s)

    # Concatenate. Atomic write under cache_dir (default: alongside main_wav).
    cache_dir = cache_dir or (audio_root.parent / "_audio_assembled")
    cache_dir.mkdir(parents=True, exist_ok=True)
    flags = "_".join(components)
    out = cache_dir / f"{participant_id}__q{question_id:02d}__{flags}.wav"

    if not out.exists():
        ref_params = _check_canonical(main_wav)
        all_frames: list[bytes] = []
        for p in [main_wav] + suffix_paths:
            _check_canonical(p)
            with wave.open(str(p), "rb") as r:
                all_frames.append(r.readframes(r.getnframes()))
        tmp = out.with_suffix(".tmp.wav")
        with wave.open(str(tmp), "wb") as w:
            w.setnchannels(ref_params.nchannels)
            w.setsampwidth(ref_params.sampwidth)
            w.setframerate(ref_params.framerate)
            for fr in all_frames:
                w.writeframes(fr)
        tmp.replace(out)

    with wave.open(str(out), "rb") as r:
        duration_s = r.getnframes() / r.getframerate()
    return AssembledAudio(audio_path=out, suffix_components=tuple(components), duration_s=duration_s)


# ---- Text selection -------------------------------------------------------

def canonical_text(prompts_df: pd.DataFrame, question_id: int) -> str:
    """The original written prompt text the participant was asked to read."""
    row = prompts_df.loc[prompts_df["question_id"] == question_id]
    if row.empty:
        raise KeyError(f"no prompt with question_id={question_id}")
    return str(row.iloc[0]["prompt_text"])


def external_transcript_text(
    *,
    transcripts_dir: Path,
    participant_id: str,
    question_id: int,
) -> str:
    """Load the Whisper-1 transcript for an assembled clip.

    transcripts_dir is the root of ``data/model_outputs/transcripts/external/whisper-1``.
    We try a couple of common naming conventions; if neither hits, raise so
    the caller can fall back to live ASR.
    """
    candidates = [
        transcripts_dir / f"{participant_id}_q{question_id:02d}.txt",
        transcripts_dir / participant_id / f"{participant_id}_q{question_id:02d}.txt",
    ]
    for c in candidates:
        if c.exists():
            return c.read_text().strip()
    raise FileNotFoundError(
        f"no whisper-1 transcript for {participant_id} q{question_id:02d}; tried {candidates}"
    )


def self_transcript_text(
    *,
    transcripts_dir: Path,
    participant_id: str,
    question_id: int,
) -> str:
    """Load the model's own self-transcript for an assembled clip.

    transcripts_dir is the root of ``data/model_outputs/transcripts/self/gemini/...``.
    """
    return external_transcript_text(
        transcripts_dir=transcripts_dir,
        participant_id=participant_id,
        question_id=question_id,
    )
