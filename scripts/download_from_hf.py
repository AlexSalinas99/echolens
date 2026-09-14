#!/usr/bin/env python3
"""Fetch the bulk EchoLens artifacts from Hugging Face into ``data/``.

This repository ships the code, the paper's rendered figures and tables, the
metadata manifests, and the aggregated results. The bulk artifacts — the audio
corpus and the raw model responses — live on the Hugging Face dataset instead:

    https://huggingface.co/datasets/alexsdl/EchoLens

The dataset layout mirrors this repository's ``data/`` tree, so downloaded files
land exactly where the notebooks expect them. Nothing needs to be moved.

The dataset is gated: accept the use policy on the dataset page once, then
authenticate with ``hf auth login`` or set ``HF_TOKEN``.

Usage:
    python scripts/download_from_hf.py --list
    python scripts/download_from_hf.py --parts responses,transcripts --dry-run
    python scripts/download_from_hf.py --parts audio
    python scripts/download_from_hf.py --parts all
"""
from __future__ import annotations

import argparse
import os
import shutil
import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parent.parent
HF_REPO = os.environ.get("ECHOLENS_HF_REPO", "alexsdl/EchoLens")

GB = 1024 ** 3
MB = 1024 ** 2

# name -> (allow_patterns, approx_bytes, description)
PARTS: dict[str, tuple[list[str], int, str]] = {
    "metadata": (
        ["data/metadata/**"],
        2 * MB,
        "participant / prompt / recording / split manifests (also shipped in git)",
    ),
    "clips": (
        ["data/clips/*.parquet"],
        6 * GB,
        "packed parquet shards with embedded audio — what load_dataset() reads",
    ),
    "audio": (
        ["data/audio/**"],
        6 * GB,
        "20,008 raw wav files, 16 kHz mono 16-bit PCM, one folder per participant",
    ),
    "responses": (
        ["data/model_outputs/responses/**"],
        1500 * MB,
        "raw model outputs, JSONL shards, 6 models x 4 conditions",
    ),
    "transcripts": (
        ["data/model_outputs/transcripts/**"],
        183 * MB,
        "ASR transcripts (whisper-1 external, Gemini self-transcript)",
    ),
    "extracted": (
        ["data/model_outputs/extracted_responses.parquet"],
        11 * MB,
        "canonical normalized response table (also shipped in git)",
    ),
    "llm_cache": (
        ["data/model_outputs/_llm_cache/**"],
        15 * MB,
        "gpt-4o-mini extraction cache — only needed by scripts/extend_llm_cache.py",
    ),
    "wer": (
        ["data/model_outputs/wer/**", "data/model_outputs/secondary_speech_metrics.csv"],
        16 * MB,
        "per-clip WER and speech metrics (the by-group rollups ship in git)",
    ),
}

# Presets that map to how people actually use this.
PRESETS = {
    "all": list(PARTS),
    "analysis": ["extracted", "wer"],
    "pipeline": ["responses", "transcripts", "extracted", "wer"],
}


def human(n: int) -> str:
    return f"{n / GB:.1f} GB" if n >= GB else f"{n / MB:.0f} MB"


def resolve(spec: str) -> list[str]:
    out: list[str] = []
    for tok in (t.strip() for t in spec.split(",") if t.strip()):
        if tok in PRESETS:
            out.extend(p for p in PRESETS[tok] if p not in out)
        elif tok in PARTS:
            if tok not in out:
                out.append(tok)
        else:
            sys.exit(
                f"Unknown part {tok!r}.\n"
                f"  parts:   {', '.join(PARTS)}\n"
                f"  presets: {', '.join(PRESETS)}"
            )
    return out


def print_catalogue() -> None:
    print(f"Dataset: https://huggingface.co/datasets/{HF_REPO}\n")
    width = max(len(k) for k in PARTS)
    for name, (_, size, desc) in PARTS.items():
        print(f"  {name:<{width}}  {human(size):>8}   {desc}")
    print("\nPresets:")
    for name, members in PRESETS.items():
        total = sum(PARTS[m][1] for m in members)
        print(f"  {name:<{width}}  {human(total):>8}   {', '.join(members)}")
    print(
        "\nNot downloadable: data/_audio_assembled/ (7.3 GB). It is derived from\n"
        "the raw audio — regenerate it with scripts/build_assembled_audio.py."
    )


def main() -> int:
    ap = argparse.ArgumentParser(
        description="Fetch bulk EchoLens artifacts from Hugging Face.",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    ap.add_argument("--parts", default="", help="comma-separated parts or presets")
    ap.add_argument("--list", action="store_true", help="show available parts and exit")
    ap.add_argument("--dest", type=Path, default=REPO_ROOT, help="repo root to populate")
    ap.add_argument("--dry-run", action="store_true", help="resolve and report, download nothing")
    ap.add_argument("--workers", type=int, default=8, help="parallel download workers")
    args = ap.parse_args()

    if args.list or not args.parts:
        print_catalogue()
        if not args.parts:
            print("\nNothing selected. Pass --parts, e.g. --parts pipeline")
        return 0

    parts = resolve(args.parts)
    patterns = [p for name in parts for p in PARTS[name][0]]
    total = sum(PARTS[name][1] for name in parts)

    print(f"Dataset:     {HF_REPO}")
    print(f"Destination: {args.dest}")
    print(f"Parts:       {', '.join(parts)}")
    print(f"Approx size: {human(total)}\n")
    for name in parts:
        print(f"  {name:<12} {human(PARTS[name][1]):>8}   {' '.join(PARTS[name][0])}")

    free = shutil.disk_usage(args.dest).free
    print(f"\nFree space at destination: {human(free)}")
    if free < total * 1.1:
        print("  WARNING: less than the download size plus 10% headroom.")

    if args.dry_run:
        print("\n--dry-run: nothing downloaded.")
        return 0

    try:
        from huggingface_hub import snapshot_download
    except ImportError:
        sys.exit("huggingface_hub is not installed.  pip install huggingface-hub")

    print("\nDownloading (resumable — re-run to continue after an interruption)...")
    snapshot_download(
        repo_id=HF_REPO,
        repo_type="dataset",
        local_dir=str(args.dest),
        allow_patterns=patterns,
        max_workers=args.workers,
    )
    print("\nDone.")
    if "audio" in parts:
        print(
            "\nTo rebuild the assembled-audio cache the direct_audio condition needs:\n"
            "    python scripts/build_assembled_audio.py"
        )
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
