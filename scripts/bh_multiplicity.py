#!/usr/bin/env python3
"""Benjamini-Hochberg FDR correction on the headline scenario-level SMDs.

Scope: direct-audio condition, 1% winsorization, axes race + gender (no accent).
`--models orig3` (default) is the three original models (gemini-3.1-flash-lite-
preview, gpt-audio-1.5, Qwen2.5-Omni-7B) = 66 tests; `--models all6` adds the
three EMNLP-extension models (gemini-3.5-flash, gemini-3.1-pro-preview,
Kimi-Audio-7B-Instruct), which run direct-audio only.

The per-replicate B=2000 bootstrap draws are not persisted on disk — only the
summary CIs are. But the bootstrap is deterministic (fixed seed=42, B=2000), so
we regenerate the *exact* same draws by re-running the same pipeline
(build_smd_universe -> impute_and_filter -> winsorize_per_prompt(0.01) ->
hierarchical_bootstrap_iter_then_avg) and capturing the `boot` object that
summarize_with_ci normally discards. Step 0 below verifies the regenerated
point/CI match data/model_outputs/disparity/scenario_smds_with_ci.csv exactly,
which proves we are using the same cells / imputation / prompt-filtering.

Procedure:
  1. Two-sided bootstrap p-value per (model, axis, scenario) from the B draws,
     using the (r+1)/(B+1) convention (never exactly 0):
        p = min(1, 2 * min((#<=0 + 1)/(B+1), (#>=0 + 1)/(B+1)))
     B = number of finite draws for that cell (matches CI computation).
  2. BH FDR under two family definitions, both reported side by side:
       - `BH_q_pooled` — ONE global family over every estimable test in the run
         (66 for orig3). This is the headline correction reported in the paper.
       - `BH_q_family` — BH within each (model, axis) family separately (the
         <=11 scenarios in one model x one axis), kept as a sensitivity check.
     The pooled correction is the more conservative of the two.
Outputs: CSV + printed table + summary + CI-vs-p sanity check.
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import pandas as pd

REPO_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(REPO_ROOT / "src"))
from eval import disparity

MO      = REPO_ROOT / "data" / "model_outputs"
META    = REPO_ROOT / "data" / "metadata"
DISP    = MO / "disparity"

CONDITION   = "direct_audio_response"
WINS        = 0.01
ORIG_MODELS = ["gemini-3.1-flash-lite-preview", "gpt-audio-1.5", "Qwen/Qwen2.5-Omni-7B"]
NEW_MODELS  = ["gemini-3.5-flash", "gemini-3.1-pro-preview",
               "moonshotai/Kimi-Audio-7B-Instruct"]
MODEL_SETS  = {"orig3": ORIG_MODELS, "all6": ORIG_MODELS + NEW_MODELS}
AXES        = ["race", "gender"]              # no accent
B_TARGET    = disparity.N_BOOT                # 2000
SEED        = disparity.BOOT_SEED             # 42
ALPHA       = 0.05


# ---------------------------------------------------------------------------
# Rebuild the exact SMD universe (verbatim from notebook 07 cell 6).
# ---------------------------------------------------------------------------
def build_universe():
    extr = pd.read_parquet(MO / "extracted_responses.parquet")
    extr = extr.drop_duplicates(subset="task_id", keep="first").reset_index(drop=True)
    participants = pd.read_csv(META / "participants.csv")
    prompts      = pd.read_csv(META / "prompts.csv")

    extr["participant_id"] = extr["clip_id"].astype(str).str.split("_").str[0]
    extr.loc[extr["clip_id"].isna(), "participant_id"] = None
    df = extr.merge(
        participants[["participant_id", "race_simplified", "gender_simplified",
                      "accent_simplified", "strict_audit_eligible"]],
        on="participant_id", how="left")

    evaluable_qids = set(prompts[
        ((prompts["prompt_type"] == "scenario") & prompts["is_quantitative"])
        | prompts["question_id"].isin([72, 73])]["question_id"].astype(float))
    df["question_id"]     = pd.to_numeric(df["question_id"], errors="coerce")
    df["is_quantitative"] = df["question_id"].isin(evaluable_qids)
    df = df.merge(prompts[["question_id", "scenario", "prompt_type"]].rename(
        columns={"prompt_type": "pt_canonical"}), on="question_id", how="left")

    def _to_num(v, k):
        if k != "number" or v is None or (isinstance(v, float) and pd.isna(v)):
            return np.nan
        try:
            return float(v)
        except (TypeError, ValueError):
            return np.nan
    df["value_num"]     = [_to_num(v, k) for v, k in zip(df["extracted_value"], df["extracted_kind"])]
    df["response_3cat"] = [disparity.classify_3cat(k, e) for k, e in zip(df["extracted_kind"], df["error_type"])]
    return disparity.build_smd_universe(df, prompts), prompts


# ---------------------------------------------------------------------------
# Bootstrap p-value with the (r+1)/(B+1) convention.
# ---------------------------------------------------------------------------
def boot_pvalue(draws: np.ndarray) -> tuple[float, int]:
    vals = draws[np.isfinite(draws)]
    B = len(vals)
    if B < 2:
        return np.nan, B
    le = (np.sum(vals <= 0) + 1) / (B + 1)     # observed stat counts as one extra null draw
    ge = (np.sum(vals >= 0) + 1) / (B + 1)
    return float(min(1.0, 2.0 * min(le, ge))), B


# ---------------------------------------------------------------------------
# Benjamini-Hochberg within one family; returns q-values aligned to input order.
# ---------------------------------------------------------------------------
def bh_qvalues(pvals: np.ndarray) -> np.ndarray:
    p = np.asarray(pvals, dtype=float)
    m = len(p)
    order = np.argsort(p)
    ranked = p[order]
    q = ranked * m / (np.arange(1, m + 1))
    q = np.minimum.accumulate(q[::-1])[::-1]   # enforce monotonicity
    q = np.clip(q, 0, 1)
    out = np.empty(m)
    out[order] = q
    return out


def main():
    ap = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    ap.add_argument("--models", choices=sorted(MODEL_SETS), default="orig3",
                    help="orig3 (default, 66 tests) or all6 (adds the three "
                         "EMNLP-extension models, direct-audio only)")
    args = ap.parse_args()
    models  = MODEL_SETS[args.models]
    # out_csv is built after the BH step, once the family and pooled test
    # counts are known, so the filename and column names carry them.

    universe, prompts = build_universe()
    saved = pd.read_csv(DISP / "scenario_smds_with_ci.csv")
    saved = saved[(saved.condition == CONDITION) & (saved.winsorize_pct == WINS)]

    rows = []
    mism = []   # CI regeneration mismatches
    for m in models:
        imp, _ = disparity.impute_and_filter(universe, m, CONDITION)
        imp_w  = disparity.winsorize_per_prompt(imp, WINS)
        boot, point = disparity.hierarchical_bootstrap_iter_then_avg(
            imp_w, "value_w", n_boot=B_TARGET, seed=SEED)
        pt = point.set_index("scenario")
        for scen in sorted(point["scenario"].unique()):
            for ax in AXES:
                col   = f"smd_{ax}"
                draws = np.asarray(boot[scen][col], dtype=float)
                smd   = float(pt.loc[scen, col]) if pd.notna(pt.loc[scen, col]) else np.nan
                fin   = draws[np.isfinite(draws)]
                lo = hi = np.nan
                if len(fin) >= 2:
                    lo = float(np.percentile(fin, 2.5)); hi = float(np.percentile(fin, 97.5))
                p, Bfin = boot_pvalue(draws)
                # Step 0 verification: does regenerated point/CI match the saved CSV?
                srow = saved[(saved.model_id == m) & (saved.scenario == scen) & (saved.axis == ax)]
                if not srow.empty:
                    sr = srow.iloc[0]
                    for label, got, exp in [("point", smd, sr.point), ("ci_lo", lo, sr.ci_lo), ("ci_hi", hi, sr.ci_hi)]:
                        if not (pd.isna(got) and pd.isna(exp)) and not np.isclose(got, exp, atol=1e-9, rtol=0, equal_nan=True):
                            mism.append((m, scen, ax, label, got, exp))
                rows.append({"model_id": m, "axis": ax, "scenario": scen,
                             "SMD": smd, "ci_lo": lo, "ci_hi": hi,
                             "raw_p": p, "n_boot_finite": Bfin})

    res = pd.DataFrame(rows)

    # (a) Sensitivity: BH within each (model, axis) family separately.
    res["BH_q_family"] = np.nan
    for (m, ax), idx in res.groupby(["model_id", "axis"]).groups.items():
        est = res.loc[idx][lambda s: s.raw_p.notna()]
        if len(est):
            res.loc[est.index, "BH_q_family"] = bh_qvalues(est.raw_p.values)
    res["sig_family"] = res["BH_q_family"] <= ALPHA

    # (b) Headline: ONE global BH family over every estimable test in the run.
    res["BH_q_pooled"] = np.nan
    est_all = res[res.raw_p.notna()]
    n_pooled = len(est_all)
    if n_pooled:
        res.loc[est_all.index, "BH_q_pooled"] = bh_qvalues(est_all.raw_p.values)
    res["sig_pooled"] = res["BH_q_pooled"] <= ALPHA

    # Encode the family sizes in the column and file names. The headline
    # reported in the paper is the pooled family (66 tests for orig3: three
    # models x two axes x eleven scenarios); the per-(model, axis) family of
    # eleven is retained as the sensitivity analysis the paper refers to when
    # it argues the pooled correction is the more conservative choice.
    fam_sizes = (res[res.raw_p.notna()].groupby(["model_id", "axis"]).size().unique())
    n_family = int(fam_sizes[0]) if len(fam_sizes) == 1 else 0
    fam_tag = f"family{n_family}" if n_family else "family"
    res = res.rename(columns={"BH_q_family": f"BH_q_{fam_tag}",
                               "sig_family":  f"sig_{fam_tag}",
                               "BH_q_pooled": f"BH_q_pooled{n_pooled}",
                               "sig_pooled":  f"sig_pooled{n_pooled}"})
    q_fam_col,  sig_fam_col  = f"BH_q_{fam_tag}", f"sig_{fam_tag}"
    q_pool_col, sig_pool_col = f"BH_q_pooled{n_pooled}", f"sig_pooled{n_pooled}"
    out_csv = DISP / (f"bh_multiplicity__direct_audio__wins1__"
                       f"{args.models}__pooled{n_pooled}.csv")

    # ---- Step 0 report -----------------------------------------------------
    print("=" * 78)
    if mism:
        print(f"!! {len(mism)} regenerated point/CI values do NOT match the saved CSV:")
        for m, scen, ax, lab, got, exp in mism[:20]:
            print(f"   {m} / {scen} / {ax} / {lab}: regenerated={got:.6f} saved={exp:.6f}")
        print("   -> universe/seed mismatch; STOP and inspect before trusting p-values.")
    else:
        print("Step-0 check PASSED: regenerated point + 95% CI match "
              "scenario_smds_with_ci.csv exactly (atol=1e-9) for all cells.")
        print("   => same universe, imputation, prompt-filtering, and RNG as the headline.")
    print("=" * 78)

    # ---- CI-vs-p sanity check ---------------------------------------------
    res["ci_excludes_0"] = (res.ci_lo > 0) | (res.ci_hi < 0)
    disagree = res[res.ci_excludes_0 & ~(res.raw_p < 0.05) & res.raw_p.notna()]
    print("\nSanity check (CI-excludes-0  =>  raw_p < 0.05):")
    if disagree.empty:
        print("   PASSED: every scenario whose 95% CI excludes 0 has raw_p < 0.05.")
    else:
        print(f"   !! {len(disagree)} disagreement(s) — flagged, do not trust blindly:")
        for _, r in disagree.iterrows():
            print(f"      {r.model_id}/{r.scenario}/{r.axis}: CI=[{r.ci_lo:+.3f},{r.ci_hi:+.3f}] raw_p={r.raw_p:.4f}")

    # ---- full table + save -------------------------------------------------
    res = res.sort_values(["model_id", "axis", "raw_p"]).reset_index(drop=True)
    res.to_csv(out_csv, index=False)
    pd.set_option("display.width", 220, "display.max_rows", 200)
    show = res.copy()
    for c in ["SMD", "ci_lo", "ci_hi"]:
        show[c] = show[c].map(lambda x: f"{x:+.3f}" if pd.notna(x) else "—")
    for c in ["raw_p", q_fam_col, q_pool_col]:
        show[c] = show[c].map(lambda x: f"{x:.4f}" if pd.notna(x) else "—")
    print(f"\nFull results (wrote {out_csv.relative_to(REPO_ROOT)}):\n")
    print(show[["model_id", "axis", "scenario", "SMD", "ci_lo", "ci_hi", "raw_p",
                q_pool_col, sig_pool_col, q_fam_col, sig_fam_col]].to_string(index=False))

    # ---- summary of survivors ---------------------------------------------
    print("\n" + "=" * 78)
    print(f"HEADLINE — global BH over all {n_pooled} estimable tests "
          f"({len(models)} models x {len(AXES)} axes x <=11 scenarios), q <= {ALPHA}:")
    sig = res[res[sig_pool_col]]
    if sig.empty:
        print("   none.")
    else:
        for _, r in sig.sort_values(q_pool_col).iterrows():
            print(f"    {r.model_id:32s} {r.axis:6s} {r.scenario:32s} "
                  f"SMD={r.SMD:+.3f} [{r.ci_lo:+.3f},{r.ci_hi:+.3f}]  "
                  f"raw_p={r.raw_p:.4f}  q={r[q_pool_col]:.4f}")

    print(f"\nSENSITIVITY — BH within each (model x axis) family, q <= {ALPHA}:")
    sigf = res[res[sig_fam_col]]
    if sigf.empty:
        print("   none.")
    else:
        for _, r in sigf.sort_values(q_fam_col).iterrows():
            flag = "" if r[sig_pool_col] else "   <- survives family-wise BH only"
            print(f"    {r.model_id:32s} {r.axis:6s} {r.scenario:32s} "
                  f"q={r[q_fam_col]:.4f}{flag}")

    print(f"\nCounts: {int(res.ci_excludes_0.sum())} of {n_pooled} tests have a 95% CI "
          f"excluding 0; {int(res[sig_pool_col].sum())} survive the global BH correction; "
          f"{int(res[sig_fam_col].sum())} survive the per-family correction.")
    print("Per model x axis (pooled-significant / estimable):")
    for (m, ax), g in res[res.raw_p.notna()].groupby(["model_id", "axis"]):
        print(f"  {m:34s} {ax:6s}: {int(g[sig_pool_col].sum())}/{len(g)}")


if __name__ == "__main__":
    main()
