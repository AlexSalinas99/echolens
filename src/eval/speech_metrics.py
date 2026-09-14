"""Speech metrics: WER, VAD-based non-vocal fraction, WPM.

Single source of truth for the metrics consumed by notebook 10.

WER follows the OpenAI Whisper convention: hypothesis and reference are
passed through `whisper.normalizers.EnglishTextNormalizer` (lower-case,
strip punctuation, drop fillers, expand contractions, spell-out → digits)
and scored with `jiwer.process_words`. Empty-ref + empty-hyp ⇒ 0.0;
empty-ref + non-empty-hyp ⇒ NaN.

VAD uses Silero VAD (the de-facto open-source standard) on 16 kHz mono
audio. Speech timestamps give `speech_s`; `non_vocal_fraction = 1 -
speech_s / duration_s`.

WPM is computed ASR-agnostically as `canonical_word_count * 60 /
duration_s` — the prompt's word count (from `prompts.prompt_text`,
suffixes included) divided by the full clip duration. We use the full
duration rather than VAD-detected speech_s because Silero is tight about
what counts as speech (drawn-out vowels, breaths, micro-pauses get
excluded), which inflates the per-second word rate to physically
implausible numbers for slower speakers. Truncated audio (participant
recorded only part of the prompt) still produces inflated WPM.
"""
from __future__ import annotations

import json
import re
import warnings
from pathlib import Path
from typing import Iterable

import numpy as np
import pandas as pd


# --------------------------- WER -----------------------------------------

_NORMALIZER = None


def _normalizer():
    global _NORMALIZER
    if _NORMALIZER is None:
        from whisper.normalizers import EnglishTextNormalizer
        _NORMALIZER = EnglishTextNormalizer()
    return _NORMALIZER


def compute_wer(hyp: str, ref: str) -> dict:
    """OpenAI-Whisper-style WER on one (hyp, ref) pair.

    Returns dict with: wer, substitutions, deletions, insertions, errors,
    ref_words, hyp_words. NaN-valued if ref normalizes to empty but hyp does
    not; (0.0, 0, 0, 0, 0, 0, 0) if both normalize to empty.
    """
    import jiwer
    norm = _normalizer()
    h = norm(str(hyp))
    r = norm(str(ref))
    if not r.strip():
        if not h.strip():
            return dict(wer=0.0, substitutions=0, deletions=0, insertions=0,
                        errors=0, ref_words=0, hyp_words=0)
        return dict(wer=float("nan"), substitutions=None, deletions=None,
                    insertions=None, errors=None, ref_words=0,
                    hyp_words=len(h.split()))
    out = jiwer.process_words(r, h)
    return dict(
        wer=float(out.wer),
        substitutions=int(out.substitutions),
        deletions=int(out.deletions),
        insertions=int(out.insertions),
        errors=int(out.substitutions + out.deletions + out.insertions),
        ref_words=len(r.split()),
        hyp_words=len(h.split()),
    )


# ---------------------- Transcript ingest --------------------------------

ASR_DIRS = {
    "whisper-1": "data/model_outputs/transcripts/external/whisper-1",
    "gemini-self": "data/model_outputs/transcripts/self/gemini/gemini-3.1-flash-lite-preview",
}

ENGLISH_L1L2_MAP = {
    "I speak English fluently as a first language": "L1 English",
    "I speak English fluently as a second or additional language": "L2 English",
}


def read_jsonl_dir(d: str | Path) -> list[dict]:
    rows = []
    for f in sorted(Path(d).glob("*.jsonl")):
        with open(f) as fh:
            for line in fh:
                line = line.strip()
                if line:
                    rows.append(json.loads(line))
    return rows


def load_all_transcripts(repo_root: str | Path = ".") -> pd.DataFrame:
    """One row per (clip × ASR). Columns: clip_id, participant_id,
    question_id, prompt_type, asr, transcript_text."""
    parts = []
    for asr, d in ASR_DIRS.items():
        rows = read_jsonl_dir(Path(repo_root) / d)
        df = pd.DataFrame(rows)[
            ["clip_id", "participant_id", "question_id",
             "prompt_type", "transcript_text"]
        ].copy()
        df["asr"] = asr
        parts.append(df)
    return pd.concat(parts, ignore_index=True)


def build_universe(transcripts: pd.DataFrame, participants: pd.DataFrame,
                   prompts: pd.DataFrame) -> pd.DataFrame:
    """Restrict to strict_audit_eligible × scenario × non-empty transcript.

    Attaches canonical reference text (from prompts.prompt_text) and
    demographic columns used by the by-axis bootstrap.
    """
    par = participants.copy()
    par["english_l1l2"] = par["english_proficiency"].map(ENGLISH_L1L2_MAP)

    df = transcripts.merge(
        prompts[["question_id", "prompt_text", "scenario"]]
            .rename(columns={"prompt_text": "reference"}),
        on="question_id", how="left",
    ).merge(
        par[["participant_id", "race_simplified", "gender_simplified",
             "audit_eligible", "strict_audit_eligible", "recording_device",
             "english_l1l2", "age_bin", "english_accent_self_report",
             "accent_simplified", "primary_language", "cohort"]],
        on="participant_id", how="left",
    )

    universe = df[
        (df["prompt_type"] == "scenario")
        & (df["strict_audit_eligible"] == True)  # noqa: E712
        & df["transcript_text"].notna()
        & (df["transcript_text"].astype(str).str.strip() != "")
        & df["reference"].notna()
    ].copy().reset_index(drop=True)
    return universe


def compute_wer_table(universe: pd.DataFrame) -> pd.DataFrame:
    """Add WER columns to the universe DataFrame."""
    out_rows = [compute_wer(h, r) for h, r in
                zip(universe["transcript_text"], universe["reference"])]
    stats = pd.DataFrame(out_rows)
    return pd.concat([universe.reset_index(drop=True), stats], axis=1)


# ---------------------- VAD + WPM + filler -------------------------------

TARGET_SR = 16000

_WORD_RE = re.compile(r"\b[A-Za-z']+\b")


def word_count(text: str) -> int:
    if not text:
        return 0
    return len(_WORD_RE.findall(text))


def load_wav_mono_16k(path: str | Path) -> np.ndarray:
    """Load PCM/etc. → mono float32 at 16 kHz."""
    import soundfile as sf
    arr, sr = sf.read(str(path), dtype="float32", always_2d=False)
    if arr.ndim == 2:
        arr = arr.mean(axis=1)
    if sr != TARGET_SR:
        try:
            import resampy
            arr = resampy.resample(arr, sr, TARGET_SR)
        except ImportError:
            ratio = TARGET_SR / sr
            new_len = int(len(arr) * ratio)
            arr = np.interp(
                np.linspace(0, len(arr), new_len, endpoint=False),
                np.arange(len(arr)),
                arr,
            ).astype("float32")
    return arr


def audio_path_for(participant_id: str, question_id: int | float, cohort: str,
                   repo_root: str | Path = ".") -> Path:
    return (Path(repo_root) / "data" / "audio" / str(cohort)
            / participant_id / f"{participant_id}_q{int(question_id):02d}.wav")


def vad_speech_seconds(arr: np.ndarray, model) -> float:
    """Total speech seconds via Silero VAD on a 16 kHz mono float32 array."""
    import torch
    from silero_vad import get_speech_timestamps
    if len(arr) == 0:
        return 0.0
    ts = get_speech_timestamps(
        torch.from_numpy(arr), model,
        sampling_rate=TARGET_SR, return_seconds=True,
    )
    return float(sum(t["end"] - t["start"] for t in ts))


def compute_audio_metrics(
    clip_table: pd.DataFrame,
    repo_root: str | Path = ".",
    verbose: bool = True,
) -> pd.DataFrame:
    """VAD-only audio metrics — `duration_s`, `speech_s`,
    `non_vocal_fraction`. The expensive step (~25 min CPU for 12,843 clips).

    `clip_table` must carry `clip_id`, `participant_id`, `question_id`,
    `cohort` (used to locate the per-clip WAV under `data/audio/{cohort}/`).
    """
    import torch
    torch.set_num_threads(4)
    from silero_vad import load_silero_vad

    model = load_silero_vad()
    repo = Path(repo_root)

    out = []
    n = len(clip_table)
    for i, row in enumerate(clip_table.itertuples(index=False)):
        ap = audio_path_for(row.participant_id, row.question_id,
                            row.cohort, repo_root=repo)
        if not ap.exists():
            out.append(dict(
                clip_id=row.clip_id, participant_id=row.participant_id,
                question_id=row.question_id,
                audio_exists=False, duration_s=np.nan,
                speech_s=np.nan, non_vocal_fraction=np.nan,
            ))
            continue

        arr = load_wav_mono_16k(ap)
        dur = len(arr) / TARGET_SR if len(arr) else 0.0
        speech_s = vad_speech_seconds(arr, model) if dur > 0 else 0.0
        non_vocal = (1.0 - speech_s / dur) if dur > 0 else np.nan

        out.append(dict(
            clip_id=row.clip_id, participant_id=row.participant_id,
            question_id=row.question_id,
            audio_exists=True, duration_s=dur,
            speech_s=speech_s, non_vocal_fraction=non_vocal,
        ))

        if verbose and (i + 1) % 500 == 0:
            print(f"  {i+1:,} / {n:,}")

    return pd.DataFrame(out)


def compute_text_metrics(
    audio_metrics: pd.DataFrame,
    canonical_by_clip: dict[str, str],
) -> pd.DataFrame:
    """Per-clip text-derived metrics — cheap, no audio I/O.

    Returns `canonical_word_count` and `wpm` (canonical-word-count /
    full-duration; ASR-agnostic, see module docstring).
    """
    rows = []
    dur_by_clip = dict(zip(audio_metrics["clip_id"], audio_metrics["duration_s"]))
    for clip_id, dur in dur_by_clip.items():
        canonical = canonical_by_clip.get(clip_id, "") or ""
        wc_canonical = word_count(canonical)
        wpm = (wc_canonical * 60.0 / dur) if (pd.notna(dur) and dur > 0) else np.nan
        rows.append(dict(
            clip_id=clip_id,
            canonical_word_count=wc_canonical,
            wpm=wpm,
        ))
    return pd.DataFrame(rows)
