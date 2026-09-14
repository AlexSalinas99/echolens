#!/usr/bin/env python3
"""Bulk-build the assembled-audio cache under ``data/_audio_assembled/``.

Neither the raw recordings nor the assembled cache are distributed through this
repository; fetch ``data/audio/`` from the Hugging Face dataset first (see
``scripts/download_from_hf.py``). Notebook 04 demonstrates the concatenation on
one participant. The full ``direct_audio`` condition needs the scenario wav
concatenated with each participant's own suffix recordings (q70 / q71) for
every evaluable quantitative prompt. This script materialises that whole cache
by looping the strict-audit eval set through ``input_prep.assemble_audio``
(which is idempotent — it skips clips already on disk).

Probe prompts (q72 / q73) are standalone and are *not* assembled; the runner
feeds their raw wav directly.

Usage:
    python scripts/build_assembled_audio.py            # all strict_audit participants
    python scripts/build_assembled_audio.py --limit 5  # smoke test on a few
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval import input_prep

META       = REPO_ROOT / "data" / "metadata"
AUDIO_ROOT = REPO_ROOT / "data" / "audio"
CACHE_DIR  = REPO_ROOT / "data" / "_audio_assembled"


def evaluable_prompts(prompts: pd.DataFrame) -> pd.DataFrame:
    """55 quantitative scenarios + the two probes (q72, q73)."""
    return prompts[
        ((prompts["prompt_type"] == "scenario") & (prompts["is_quantitative"]))
        | (prompts["question_id"].isin([72, 73]))
    ].copy()


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--limit", type=int, default=None, help="only the first N participants")
    args = ap.parse_args()

    prompts      = pd.read_csv(META / "prompts.csv")
    participants = pd.read_csv(META / "participants.csv")
    splits       = pd.read_csv(META / "splits.csv")

    cohort_of = dict(zip(participants["participant_id"], participants["cohort"]))
    pids = sorted(splits.loc[splits["in_strict_audit"], "participant_id"])
    if args.limit:
        pids = pids[: args.limit]

    ev = evaluable_prompts(prompts)
    # Only scenario prompts get assembled; probes are standalone.
    ev = ev[~ev["question_id"].isin([72, 73])]

    built = skipped = missing = 0
    for pid in pids:
        cohort = cohort_of.get(pid)
        if cohort is None:
            print(f"⚠ no cohort for {pid}; skipping")
            continue
        for _, row in ev.iterrows():
            qid = int(row["question_id"])
            try:
                res = input_prep.assemble_audio(
                    participant_id=pid,
                    cohort=cohort,
                    question_id=qid,
                    audio_root=AUDIO_ROOT,
                    suffix_number=bool(row["suffix_number"]),
                    suffix_additional=bool(row["suffix_additional"]),
                    cache_dir=CACHE_DIR,
                )
                # assemble_audio returns the scenario wav as-is when no suffix
                # is needed; only count clips that actually landed in the cache.
                if res.audio_path.parent == CACHE_DIR:
                    built += 1
                else:
                    skipped += 1
            except FileNotFoundError as e:
                missing += 1
                print(f"  missing input for {pid} q{qid:02d}: {e}")

    print(f"\ndone: {built} assembled clips in cache, {skipped} standalone (no suffix), "
          f"{missing} missing-input skips")
    print(f"cache dir: {CACHE_DIR.relative_to(REPO_ROOT)}  "
          f"({len(list(CACHE_DIR.glob('*.wav')))} wavs total)")


if __name__ == "__main__":
    main()
