#!/usr/bin/env python3
"""Extend the Layer-4 (gpt-4o-mini) extraction cache for new model outputs.

Notebook 06's Layer-4 step is *cache-only* in its main path: it merges LLM
labels from ``_llm_cache/extracted_responses_with_llm_labels.parquet`` and only
warns about task_ids missing from the cache. When new models are added, their
free-text / range / unit-bearing responses (e.g. gemini-3.1-pro's
"$90,000 to $120,000") fall through the deterministic rules to Layer 4 but have
no cache entry, so they end up unresolved.

This script closes that gap. It walks the raw responses, finds rows that the
rule layers can't resolve AND that aren't already cached, runs the *same*
gpt-4o-mini labeler notebook 06 uses (``eval.extraction.llm_extract_batch``),
and appends the results to the cache parquet. Re-running notebook 06 afterwards
then resolves every row deterministically from the (now-complete) cache.

Idempotent: only uncached pending rows are labeled. Needs ``OPENAI_API_KEY``.

    python scripts/extend_llm_cache.py --concurrency 16
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval import extraction, input_prep

MO         = REPO_ROOT / "data" / "model_outputs"
RESP_ROOT  = MO / "responses"
CACHE      = MO / "_llm_cache" / "extracted_responses_with_llm_labels.parquet"
PROMPTS    = REPO_ROOT / "data" / "metadata" / "prompts.csv"

CACHE_COLS = ["task_id", "provider", "model_id", "condition", "clip_id",
              "question_id", "prompt_type", "run_index", "raw_response",
              "error_type", "extraction_method", "extracted_kind",
              "extracted_value", "rule_name", "llm_label"]


def walk(resp_root: Path) -> list[dict]:
    rows = []
    for prov in sorted(p for p in resp_root.iterdir() if p.is_dir()):
        for model in sorted(p for p in prov.iterdir() if p.is_dir()):
            for cond in sorted(p for p in model.iterdir() if p.is_dir()):
                if "INCOMPATIBLE" in cond.name or "BUG" in cond.name:
                    continue
                for shard in sorted(cond.glob("*.jsonl")):
                    with shard.open() as fh:
                        for line in fh:
                            if line.strip():
                                rows.append(json.loads(line))
    return rows


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--concurrency", type=int, default=16)
    ap.add_argument("--model", default="gpt-4o-mini")
    args = ap.parse_args()

    if not os.environ.get("OPENAI_API_KEY"):
        sys.exit("set OPENAI_API_KEY")

    prompts = pd.read_csv(PROMPTS)
    q_text_cache: dict[int, str] = {}

    def question_text(qid) -> str:
        if qid is None or (isinstance(qid, float) and pd.isna(qid)):
            return ""
        qid = int(qid)
        if qid not in q_text_cache:
            try:
                q_text_cache[qid] = extraction.clean_question_for_labeler(
                    input_prep.canonical_text(prompts, qid))
            except KeyError:
                q_text_cache[qid] = ""
        return q_text_cache[qid]

    cache = pd.read_parquet(CACHE)
    cached_ids = set(cache["task_id"])
    print(f"cache: {len(cache):,} rows")

    raw_rows = walk(RESP_ROOT)
    print(f"walked {len(raw_rows):,} raw response rows")

    # Find rule-unresolved, non-error, not-yet-cached rows (dedup on task_id).
    pending: dict[str, dict] = {}
    for r in raw_rows:
        tid = r.get("task_id")
        if not tid or tid in cached_ids or tid in pending:
            continue
        res = extraction.extract_rules_only(
            raw_response=r.get("response_text") or "",
            error_type=r.get("error_type"),
            response_parsed_json=r.get("response_parsed_json"),
        )
        if res.method == "none" and res.kind != "error":
            pending[tid] = {
                "task_id": tid,
                "provider": r.get("model_provider"),
                "model_id": r.get("model_id"),
                "condition": r.get("condition"),
                "clip_id": r.get("clip_id"),
                "question_id": r.get("question_id"),
                "prompt_type": r.get("prompt_type"),
                "run_index": r.get("run_index"),
                "raw_response": res.raw_used,          # structured answer if gemini, else text
                "error_type": None,
                "question_text": question_text(r.get("question_id")),
            }

    todo = list(pending.values())
    print(f"uncached pending Layer-4 rows to label: {len(todo):,}")
    if not todo:
        print("nothing to do — cache already complete.")
        return
    print("by model_id:", pd.Series([r["model_id"] for r in todo]).value_counts().to_dict())

    from openai import OpenAI
    client = OpenAI(api_key=os.environ["OPENAI_API_KEY"])

    t0 = time.time()
    labeled = extraction.llm_extract_batch(todo, client=client, model=args.model,
                                           max_workers=args.concurrency)
    print(f"labeled {len(labeled):,} rows in {time.time()-t0:.0f}s")

    new_rows = []
    for r in labeled:
        new_rows.append({
            "task_id": r["task_id"], "provider": r["provider"], "model_id": r["model_id"],
            "condition": r["condition"], "clip_id": r["clip_id"], "question_id": r["question_id"],
            "prompt_type": r["prompt_type"], "run_index": r["run_index"],
            "raw_response": r["raw_response"], "error_type": None,
            "extraction_method": "llm", "extracted_kind": r["extracted_kind"],
            # store as string (or None) to match the cache's column type — the
            # nb06 merge stringifies extracted_value anyway.
            "extracted_value": None if r["extracted_value"] is None else str(r["extracted_value"]),
            "rule_name": None,
            "llm_label": r["llm_label"],
        })
    add = pd.DataFrame(new_rows)[CACHE_COLS]
    print("new label kind distribution:", add["extracted_kind"].value_counts(dropna=False).to_dict())

    merged = pd.concat([cache, add], ignore_index=True).drop_duplicates("task_id", keep="last")
    # guarantee a uniform column type for the parquet writer
    merged["extracted_value"] = merged["extracted_value"].apply(lambda v: None if v is None else str(v))
    bak = CACHE.with_suffix(".parquet.bak")
    if not bak.exists():
        cache.to_parquet(bak, index=False)
        print(f"backed up original cache → {bak.name}")
    merged.to_parquet(CACHE, index=False)
    print(f"cache: {len(cache):,} → {len(merged):,} rows written to {CACHE.name}")


if __name__ == "__main__":
    main()
