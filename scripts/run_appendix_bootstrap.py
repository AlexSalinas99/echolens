#!/usr/bin/env python3
"""Stand-alone runner for the direction-aware appendix bootstrap.

Mirrors the upstream state of notebook 07 (universe + imputation + 50% filter),
then runs ONLY the appendix bootstrap from § 7b — filter to direction_clear=1,
negate value_imputed for direction=positive prompts, re-run the same
participant-level cluster bootstrap that the headline uses, write
prompt_level_smds__direction_clear.csv and scenario_smds_with_ci__direction_clear.csv.

Skips the headline bootstrap (cell 19 of notebook 07), which already lives on
disk and doesn't need re-running.
"""
import sys
import time
from concurrent.futures import ProcessPoolExecutor, as_completed
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval import disparity

META = REPO_ROOT / "data" / "metadata"
MO   = REPO_ROOT / "data" / "model_outputs"
DISP = MO / "disparity"
DISP.mkdir(parents=True, exist_ok=True)


def _bootstrap_cell(model_id, condition, winsorize_pct, imp):
    import sys
    from pathlib import Path
    sys.path.insert(0, str(Path(__file__).resolve().parent.parent / "src"))
    from eval import disparity as _disp

    imp_w = _disp.winsorize_per_prompt(imp, winsorize_pct)
    sc = _disp.summarize_with_ci(
        imp_w, value_col="value_w",
        n_boot=_disp.N_BOOT, seed=_disp.BOOT_SEED, ci=_disp.CI,
    )
    sc.insert(0, "winsorize_pct", winsorize_pct)
    sc.insert(0, "condition",     condition)
    sc.insert(0, "model_id",      model_id)
    return sc


def build_imputed():
    """Reproduce nb 07 cells 2-11: universe + impute + 50% filter per (m, cond)."""
    print("Loading extracted_responses.parquet…")
    extr = pd.read_parquet(MO / "extracted_responses.parquet")
    extr = extr.drop_duplicates(subset="task_id", keep="first").reset_index(drop=True)

    participants = pd.read_csv(META / "participants.csv")
    prompts      = pd.read_csv(META / "prompts.csv")

    extr["participant_id"] = extr["clip_id"].astype(str).str.split("_").str[0]
    extr.loc[extr["clip_id"].isna(), "participant_id"] = None

    df = extr.merge(
        participants[["participant_id",
                       "race_simplified", "gender_simplified", "accent_simplified",
                       "strict_audit_eligible"]],
        on="participant_id", how="left",
    )

    evaluable_qids = set(prompts[
        ((prompts["prompt_type"] == "scenario") & prompts["is_quantitative"])
        | prompts["question_id"].isin([72, 73])
    ]["question_id"].astype(float))
    df["question_id"]     = pd.to_numeric(df["question_id"], errors="coerce")
    df["is_quantitative"] = df["question_id"].isin(evaluable_qids)
    df = df.merge(
        prompts[["question_id", "scenario", "prompt_type"]].rename(
            columns={"prompt_type": "pt_canonical"}),
        on="question_id", how="left",
    )

    def _to_num(v, k):
        if k != "number" or v is None or (isinstance(v, float) and pd.isna(v)):
            return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan

    df["value_num"]     = [_to_num(v, k) for v, k in zip(df["extracted_value"], df["extracted_kind"])]
    df["response_3cat"] = [disparity.classify_3cat(k, e)
                           for k, e in zip(df["extracted_kind"], df["error_type"])]

    universe = disparity.build_smd_universe(df, prompts)
    print(f"  universe rows: {len(universe):,}  participants: {universe['participant_id'].nunique()}")

    imputed = {}
    for cond in sorted(universe["condition"].unique()):
        for m in sorted(universe.loc[universe["condition"] == cond, "model_id"].unique()):
            imp_kept, _drop = disparity.impute_and_filter(universe, m, cond)
            imputed[(m, cond)] = imp_kept
    print(f"  imputed cells: {len(imputed)}")
    return imputed, prompts


def main():
    imputed, prompts = build_imputed()

    # Filter + sign-flip: direction_clear=1, with direction=positive negated.
    prompts["question_id"] = pd.to_numeric(prompts["question_id"], errors="coerce")
    clear_qids    = set(prompts.loc[prompts["direction_clear"] == 1, "question_id"].dropna())
    positive_qids = set(prompts.loc[(prompts["direction_clear"] == 1)
                                     & (prompts["direction"] == "positive"), "question_id"].dropna())
    print(f"\ndirection_clear=1 qids ({len(clear_qids)}): "
          f"{sorted(int(q) for q in clear_qids)}")
    print(f"positive-and-clear qids ({len(positive_qids)}): "
          f"{sorted(int(q) for q in positive_qids)}")

    imputed_app = {}
    for (m, cond), imp in imputed.items():
        if len(imp) == 0:
            imputed_app[(m, cond)] = imp
            continue
        sub = imp[imp["question_id"].isin(clear_qids)].copy()
        flip_mask = sub["question_id"].isin(positive_qids)
        sub.loc[flip_mask, "value_imputed"] = -sub.loc[flip_mask, "value_imputed"]
        imputed_app[(m, cond)] = sub

    # Per-prompt SMDs (both winsorize variants).
    print("\nComputing per-prompt SMDs over the appendix subset…")
    prompt_smds_rows = []
    for (m, cond), imp in imputed_app.items():
        if len(imp) == 0:
            continue
        audit_pids = sorted(imp["participant_id"].unique())
        weights_unit = {p: 1 for p in audit_pids}
        for w in disparity.WINSORIZE_MODES:
            imp_w = disparity.winsorize_per_prompt(imp, w)
            pi = disparity.per_iter_prompt_smds(imp_w, value_col="value_w", weights=weights_unit)
            ps = disparity.prompt_iter_avg_smds(pi)
            ps.insert(0, "winsorize_pct", w)
            ps.insert(0, "condition",     cond)
            ps.insert(0, "model_id",      m)
            prompt_smds_rows.append(ps)
    prompt_smds_app = pd.concat(prompt_smds_rows, ignore_index=True)
    out = DISP / "prompt_level_smds__direction_clear.csv"
    prompt_smds_app.to_csv(out, index=False)
    print(f"  wrote {out.relative_to(REPO_ROOT)}  ({len(prompt_smds_app):,} rows)")

    # Cluster bootstrap.
    jobs = [(m, cond, w, imp)
             for (m, cond), imp in imputed_app.items() if len(imp) > 0
             for w in disparity.WINSORIZE_MODES]
    print(f"\nBootstrapping {len(jobs)} (model × condition × winsorize) appendix cells "
          f"with B={disparity.N_BOOT}, seed={disparity.BOOT_SEED}, "
          f"up to {min(len(jobs), 12)} workers.")

    scenario_rows = []
    t0 = time.time()
    with ProcessPoolExecutor(max_workers=min(len(jobs), 12)) as ex:
        futures = {ex.submit(_bootstrap_cell, m, cond, w, imp): (m, cond, w)
                   for (m, cond, w, imp) in jobs}
        for n_done, fut in enumerate(as_completed(futures), 1):
            m, cond, w = futures[fut]
            sc = fut.result()
            scenario_rows.append(sc)
            wlabel = "raw" if w is None else f"wins{int(w * 100)}"
            elapsed = time.time() - t0
            print(f"  [{n_done:2d}/{len(jobs)}  {elapsed/60:5.1f} min wall] "
                  f"{m} / {cond} ({wlabel})")

    scenario_smds_app = (pd.concat(scenario_rows, ignore_index=True)
                            .sort_values(["model_id", "condition", "winsorize_pct",
                                          "scenario", "axis"])
                            .reset_index(drop=True))
    out = DISP / "scenario_smds_with_ci__direction_clear.csv"
    scenario_smds_app.to_csv(out, index=False)
    print(f"\nwrote {out.relative_to(REPO_ROOT)}  ({len(scenario_smds_app):,} rows)  "
          f"in {(time.time()-t0)/60:.1f} min wall")


if __name__ == "__main__":
    main()
