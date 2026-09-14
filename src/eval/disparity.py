"""SMD-based disparity metrics — ported from the original analysis notebook.

Pipeline (per (model, condition, winsorize_pct)):

1. **Universe construction.** Restrict to strict-audit-eligible participants × scenario-
   quantitative prompts × evaluation-eligible conditions. Drop probes
   (q72/q73), the BP tuple prompt (q25), and rows where the upstream
   recording was unavailable (`error_type == 'MissingAudio'`).
2. **Three-category collapse.** Each row gets a `response_3cat` ∈
   {`numeric`, `invalid`, `refusal`}. This is the SMD-analysis universe.
3. **Speaker-level median imputation.** For each (participant, prompt) cell
   with ≥1 valid numeric run, fill missing runs with the participant's
   median across their valid iterations. Drop cells with no valid run.
4. **Prompt filter (50 % imputability).** Keep a prompt only if at least
   `IMPUTABILITY_THRESHOLD = 0.50` of the speakers who saw it have ≥1 valid
   numeric response.
5. **Winsorization.** Optionally clip each prompt's values at the
   [`pct`, `1-pct`] quantiles before SMD computation. The published
   headline uses `pct=0.01`; the raw (`None`) variant is also produced.
   Quantile bounds are estimated once on the original sample and held
   fixed across bootstrap iterations.
6. **Per-(prompt, iter) SMD.** Frequency-weighted Cohen-style SMD inside
   each (prompt × run_index) cell, computed for each axis in `AXES` with
   sign conventions
   *race = Black − White*, *gender = Female − Male*, *accent = non-SAE − SAE*.
7. **Scenario aggregation (per-iter, then iter-average).** Aggregate by
   averaging across prompts within each iteration first, then averaging
   the (≤3) per-iter scenario means.
8. **Cluster bootstrap.** B = 2000 iterations; each iteration resamples
   strict-audit-eligible participants with replacement and re-runs steps 6 + 7
   using the resampled multiplicities as frequency weights. Percentile
   95 % CIs are computed from the bootstrap distribution per
   (scenario × axis).

Constants are exposed as module attributes so callers can override them
for sensitivity analyses or smoke tests. To add a new binary contrast,
append one row to `AXES` — every downstream function iterates over it.
"""
from __future__ import annotations

from dataclasses import dataclass

import numpy as np
import pandas as pd


# ---------- Constants (match the released pipeline) ------------------------

IMPUTABILITY_THRESHOLD = 0.50      # § 4: prompts with < 50 % imputable speakers dropped
WINSORIZE_MODES = (None, 0.01)     # § 5: raw + 1 % winsorize
N_BOOT = 2000                      # § 8: 2 000 cluster-bootstrap iterations
BOOT_SEED = 42                     # § 8: fixed RNG seed for reproducibility
CI = 0.95                          # § 8: 95 % percentile CI

PROBE_QIDS = (72, 73)              # demographic + dialect probes (excluded from SMD)
BP_QID = 25                        # blood-pressure tuple prompt (excluded from SMD)


# ---------- Axes (binary contrasts the SMD is computed across) -------------

@dataclass(frozen=True)
class Axis:
    """One binary contrast. `pos − neg` is the sign of `smd_<name>`."""
    name: str   # short name used in column suffixes: smd_<name>, n_iters_<name>
    col:  str   # column in the imputed dataframe carrying the group label
    pos:  str   # group whose mean enters the SMD numerator with +1
    neg:  str   # group whose mean enters the SMD numerator with −1


AXES: tuple[Axis, ...] = (
    Axis("race",   "race_simplified",   "Black",   "White"),
    Axis("gender", "gender_simplified", "Female",  "Male"),
    Axis("accent", "accent_simplified", "non-SAE", "SAE"),
)


def axis_columns() -> list[str]:
    """The set of demographic columns the imputed frame must carry through."""
    return [a.col for a in AXES]


# ---------- Step 2: three-category collapse --------------------------------

def classify_3cat(extracted_kind: str | None, error_type: str | None) -> str:
    """Map (extracted_kind, error_type) → {numeric, invalid, refusal, missing_data}.

    The SMD analysis filters out `missing_data` rows (the upstream recording was
    unavailable, so we never even asked the model). All non-`numeric` /
    non-`invalid` model outputs become `refusal`.
    """
    if error_type == "MissingAudio":
        return "missing_data"
    if extracted_kind == "number":
        return "numeric"
    if extracted_kind == "invalid":
        return "invalid"
    # refusal | error | category | tuple | uncertain → 'refusal'
    return "refusal"


def build_smd_universe(df: pd.DataFrame, prompts_df: pd.DataFrame) -> pd.DataFrame:
    """Filter `df` (clip-level, post-extraction) to the SMD analysis universe.

    Expects columns: model_id, condition, question_id, response_3cat,
    is_quantitative, prompt_type (or pt_canonical), strict_audit_eligible.
    Returns rows where `response_3cat ∈ {numeric, invalid, refusal}` and the
    prompt is a scenario-quantitative prompt (BP excluded, probes excluded).
    Eligibility matches the rest of the paper (notebooks 03/04/05/10 and
    `src/eval/speech_metrics.py`) — the strict 500-participant fold.
    """
    pt_col = "pt_canonical" if "pt_canonical" in df.columns else "prompt_type"
    return df[
        df["is_quantitative"]
        & (df[pt_col] == "scenario")
        & (df["question_id"] != BP_QID)
        & (~df["question_id"].isin(PROBE_QIDS))
        & df["strict_audit_eligible"].fillna(False).astype(bool)
        & df["response_3cat"].isin(["numeric", "invalid", "refusal"])
    ].copy()


# ---------- Validity tables (the complementary metric) ---------------------

def build_validity_table(universe_df: pd.DataFrame, condition: str) -> pd.DataFrame:
    """Per-(model, prompt, axis, group) numeric / invalid / refusal counts and rates.

    Iterates over `AXES`; the invalid+refusal columns are the complementary
    disparity dimension to the SMD (the SMD itself is defined only over numeric
    responses, so invalid / refusal asymmetries across groups are invisible to it).
    """
    sub = universe_df[universe_df["condition"] == condition].copy()
    if sub.empty:
        return pd.DataFrame()
    rows = []
    for axis in AXES:
        for g in (axis.pos, axis.neg):
            grp = sub[sub[axis.col] == g]
            agg = (grp.groupby(["model_id", "question_id", "scenario"])
                       .agg(n_total=("response_3cat", "size"),
                            n_numeric=("response_3cat", lambda s: (s == "numeric").sum()),
                            n_invalid=("response_3cat", lambda s: (s == "invalid").sum()),
                            n_refusal=("response_3cat", lambda s: (s == "refusal").sum()))
                       .reset_index())
            agg["axis"] = axis.name
            agg["group"] = g
            rows.append(agg)
    out = pd.concat(rows, ignore_index=True)
    out["pct_numeric"] = out["n_numeric"] / out["n_total"]
    out["pct_invalid"] = out["n_invalid"] / out["n_total"]
    out["pct_refusal"] = out["n_refusal"] / out["n_total"]
    out = out[["model_id", "question_id", "scenario", "axis", "group",
                "n_total", "n_numeric", "n_invalid", "n_refusal",
                "pct_numeric", "pct_invalid", "pct_refusal"]]
    return out.sort_values(["model_id", "question_id", "axis", "group"]).reset_index(drop=True)


# ---------- Step 3 + 4: speaker-level median imputation + prompt filter ----

def impute_and_filter(universe_df: pd.DataFrame, model_id: str, condition: str):
    """Apply speaker-level median imputation and the 50 % imputability prompt drop.

    Returns
    -------
    imp_kept : DataFrame
        One row per (participant, question_id, run_index) for the kept (model,
        condition, prompts) cells, with columns participant_id, race_simplified,
        gender_simplified, question_id, scenario, run_index, value_imputed,
        was_imputed.
    drop_report : dict
        Diagnostics: n_prompts_total, n_prompts_kept, n_prompts_dropped,
        pct_dropped, fraction_imputed_among_kept, dropped_qids.
    """
    sub = universe_df[
        (universe_df["model_id"] == model_id)
        & (universe_df["condition"] == condition)
    ].copy()
    if sub.empty:
        return sub, {"n_prompts_total": 0, "n_prompts_kept": 0, "n_prompts_dropped": 0,
                     "fraction_imputed_among_kept": np.nan, "dropped_qids": []}

    # (1) Per-(participant, prompt) median of valid values.
    valid = sub[sub["response_3cat"] == "numeric"][
        ["participant_id", "question_id", "value_num"]]
    medians = (valid.groupby(["participant_id", "question_id"])["value_num"]
                     .median().reset_index()
                     .rename(columns={"value_num": "median_valid"}))
    sub = sub.merge(medians, on=["participant_id", "question_id"], how="left")

    # (2) Imputation: original value if numeric, else the participant's median.
    sub["was_imputed"] = (sub["response_3cat"] != "numeric")
    sub["value_imputed"] = np.where(sub["response_3cat"] == "numeric",
                                      sub["value_num"],
                                      sub["median_valid"])

    # (3) Drop participant×prompt rows with no median (zero valid responses).
    imp = sub.dropna(subset=["value_imputed"]).copy()

    # (4) Imputability rate per prompt; drop if < threshold.
    speakers_total = (sub.groupby("question_id")["participant_id"]
                          .nunique().rename("n_total_spk"))
    speakers_imp   = (imp.groupby("question_id")["participant_id"]
                          .nunique().rename("n_imputable_spk"))
    rates = pd.concat([speakers_total, speakers_imp], axis=1).fillna(0)
    rates["imputable_rate"] = rates["n_imputable_spk"] / rates["n_total_spk"]
    rates["keep"] = rates["imputable_rate"] >= IMPUTABILITY_THRESHOLD
    keep_qids = set(rates.loc[rates["keep"]].index)

    imp_kept = imp[imp["question_id"].isin(keep_qids)].copy()
    imp_kept = imp_kept[["participant_id", *axis_columns(),
                          "question_id", "scenario", "run_index",
                          "value_imputed", "was_imputed"]]

    drop_report = {
        "n_prompts_total":   int(len(rates)),
        "n_prompts_kept":    int(rates["keep"].sum()),
        "n_prompts_dropped": int((~rates["keep"]).sum()),
        "pct_dropped":       round((~rates["keep"]).sum() / max(1, len(rates)) * 100, 2),
        "imputable_rate_min": float(rates["imputable_rate"].min()) if len(rates) else np.nan,
        "fraction_imputed_among_kept":
            float(imp_kept["was_imputed"].mean()) if len(imp_kept) else np.nan,
        "dropped_qids": sorted(int(q) for q in rates.loc[~rates["keep"]].index),
    }
    return imp_kept, drop_report


# ---------- Step 5: winsorization ------------------------------------------

def winsorize_per_prompt(imp_df: pd.DataFrame, pct: float | None) -> pd.DataFrame:
    """Clip `value_imputed` per question_id at the [pct, 1-pct] quantiles.

    Returns a DataFrame with an extra `value_w` column (= raw if pct is None).
    Quantile bounds are estimated once on `imp_df` and reused across bootstrap
    iterations.
    """
    out = imp_df.copy()
    if pct is None or pct <= 0:
        out["value_w"] = out["value_imputed"]
        return out
    bounds = (imp_df.groupby("question_id")["value_imputed"]
                      .agg(lo=lambda s: s.quantile(pct),
                           hi=lambda s: s.quantile(1 - pct))
                      .reset_index())
    out = out.merge(bounds, on="question_id", how="left")
    out["value_w"] = out["value_imputed"].clip(lower=out["lo"], upper=out["hi"])
    return out.drop(columns=["lo", "hi"])


# ---------- Step 6: SMD primitive ------------------------------------------

def smd_weighted(a: np.ndarray, b: np.ndarray, wa: np.ndarray, wb: np.ndarray) -> float:
    """Frequency-weighted Cohen-style SMD: (mean_a − mean_b) / pooled_sd.

    `wa`, `wb` are participant frequency weights from the cluster bootstrap
    (point estimates use weights = 1). Returns NaN if either group has effective
    sample size < 2 or the pooled SD is zero / non-finite.
    """
    n_a = float(wa.sum()); n_b = float(wb.sum())
    if n_a < 2 or n_b < 2:
        return np.nan
    mean_a = float(np.sum(a * wa) / n_a)
    mean_b = float(np.sum(b * wb) / n_b)
    var_a = float(np.sum(wa * (a - mean_a) ** 2) / (n_a - 1))
    var_b = float(np.sum(wb * (b - mean_b) ** 2) / (n_b - 1))
    denom = n_a + n_b - 2
    if denom <= 0:
        return np.nan
    sd = np.sqrt(((n_a - 1) * var_a + (n_b - 1) * var_b) / denom)
    if not (np.isfinite(sd) and sd > 0):
        return np.nan
    return float(mean_a - mean_b) / sd


def per_iter_prompt_smds(imp_df: pd.DataFrame, value_col: str, weights: dict) -> pd.DataFrame:
    """One SMD per (question_id, run_index) per axis in `AXES`, using `weights`.

    Sign conventions live on each `Axis`: positive = pos-group − neg-group.
    Returns rows with columns: question_id, scenario, run_index, smd_<axis> for each axis.
    """
    smd_cols = [f"smd_{a.name}" for a in AXES]
    imp = imp_df.assign(_w=imp_df["participant_id"]
                              .map(weights).fillna(0).astype(int))
    imp = imp[imp["_w"] > 0]
    if imp.empty:
        return pd.DataFrame(columns=["question_id", "scenario", "run_index", *smd_cols])
    rows = []
    for (qid, run), grp in imp.groupby(["question_id", "run_index"]):
        row = {"question_id": qid, "scenario": grp["scenario"].iloc[0], "run_index": run}
        for axis in AXES:
            A = grp[grp[axis.col] == axis.pos]
            B = grp[grp[axis.col] == axis.neg]
            row[f"smd_{axis.name}"] = smd_weighted(
                A[value_col].values, B[value_col].values,
                A["_w"].values,        B["_w"].values,
            )
        rows.append(row)
    return pd.DataFrame(rows)


# ---------- Step 7: scenario aggregation (per-iter, then iter-average) -----

def scenario_smds_iter_then_average(per_iter_df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate per-(prompt, iter) SMDs to scenario level for every axis in `AXES`.

    Algorithm:
      1) Per iter, mean across prompts within each scenario (NaN-skip).
      2) Per scenario, mean across the (≤3) per-iter scenario means.

    The 1-then-2 ordering matters when some (prompt × iter) cells are NaN
    (e.g., degenerate-variance prompts after imputation): the per-iter
    scenario mean covers a different prompt subset for that iter; those
    per-iter means then average together. Returns columns:
    scenario, smd_<axis> for each axis, n_prompts, n_iters_with_data_race
    (counted on the race axis only — see the note at the aggregation below).
    """
    smd_cols = [f"smd_{a.name}" for a in AXES]
    base_cols = ["scenario", *smd_cols, "n_prompts", "n_iters_with_data_race"]
    if per_iter_df is None or per_iter_df.empty:
        return pd.DataFrame(columns=base_cols)

    per_iter_agg = {f"scen_{c}": (c, "mean") for c in smd_cols}
    per_iter_scen = (per_iter_df.groupby(["scenario", "run_index"])
                                  .agg(**per_iter_agg)
                                  .reset_index())

    final_agg = {c: (f"scen_{c}", "mean") for c in smd_cols}
    # Counted on the FIRST axis (race) only, hence the column name. The same
    # value is emitted on every axis row; it is a data-availability diagnostic
    # for the race axis, not a per-axis count.
    final_agg["n_iters_with_data_race"] = (f"scen_{smd_cols[0]}",
                                         lambda s: int(s.notna().sum()))
    final = (per_iter_scen.groupby("scenario")
                            .agg(**final_agg)
                            .reset_index())

    nprompts = (per_iter_df[per_iter_df[smd_cols].notna().any(axis=1)]
                  .groupby("scenario")["question_id"]
                  .nunique().rename("n_prompts").reset_index())
    final = final.merge(nprompts, on="scenario", how="left")
    final["n_prompts"] = final["n_prompts"].fillna(0).astype(int)
    return final[base_cols]


def prompt_iter_avg_smds(per_iter_df: pd.DataFrame) -> pd.DataFrame:
    """Per-prompt iter-averaged SMDs (for the per-prompt CSV; not used for scenario aggregation).

    Wide format: one row per (question_id, scenario), with smd_<axis> and n_iters_<axis>
    columns for each axis in `AXES`.
    """
    smd_cols     = [f"smd_{a.name}" for a in AXES]
    niter_cols   = [f"n_iters_{a.name}" for a in AXES]
    cols_out     = ["question_id", "scenario", *smd_cols, *niter_cols]
    if per_iter_df is None or per_iter_df.empty:
        return pd.DataFrame(columns=cols_out)
    rows = []
    for qid, grp in per_iter_df.groupby("question_id"):
        row = {"question_id": qid, "scenario": grp["scenario"].iloc[0]}
        for axis in AXES:
            v = grp[f"smd_{axis.name}"].dropna()
            row[f"smd_{axis.name}"]      = float(v.mean()) if len(v) else np.nan
            row[f"n_iters_{axis.name}"]  = int(len(v))
        rows.append(row)
    return pd.DataFrame(rows)[cols_out].sort_values("question_id").reset_index(drop=True)


# ---------- Step 8: cluster bootstrap + CI ---------------------------------

def hierarchical_bootstrap_iter_then_avg(
    imp_df: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
):
    """Participant-level cluster bootstrap. Returns (boot, point).

    For each bootstrap draw:
      a) Sample participants with replacement → frequency weights w_p.
      b) Per (prompt, iter): frequency-weighted SMD for every axis in `AXES`.
      c) Per iter: scenario-mean across prompts (NaN-skip).
      d) Per scenario: mean across the (≤3) per-iter scenario means.

    boot : dict[scenario][smd_<axis>] → list of bootstrap scenario means
    point : DataFrame — the point-estimate scenario aggregation (weights=1)
    """
    smd_cols = [f"smd_{a.name}" for a in AXES]
    audit_pids = sorted(imp_df["participant_id"].unique())
    n = len(audit_pids)

    pi_point  = per_iter_prompt_smds(imp_df, value_col, {p: 1 for p in audit_pids})
    point     = scenario_smds_iter_then_average(pi_point)
    scenarios = sorted(point["scenario"].unique())
    boot = {s: {c: [] for c in smd_cols} for s in scenarios}

    rng = np.random.default_rng(seed)
    for _ in range(n_boot):
        sampled = rng.integers(0, n, size=n)
        counts = np.bincount(sampled, minlength=n)
        weights = {audit_pids[i]: int(counts[i]) for i in range(n) if counts[i] > 0}
        pi  = per_iter_prompt_smds(imp_df, value_col, weights)
        scen = scenario_smds_iter_then_average(pi)
        for _, r in scen.iterrows():
            for c in smd_cols:
                boot[r["scenario"]][c].append(r[c])
    return boot, point


def summarize_with_ci(
    imp_df: pd.DataFrame,
    value_col: str,
    *,
    n_boot: int = N_BOOT,
    seed: int = BOOT_SEED,
    ci: float = CI,
) -> pd.DataFrame:
    """Run the cluster bootstrap and return one row per (scenario, axis)
    with point estimate + percentile CI.
    """
    boot, point = hierarchical_bootstrap_iter_then_avg(
        imp_df, value_col, n_boot=n_boot, seed=seed)
    rows = []
    alpha = (1.0 - ci) / 2.0
    for _, r in point.iterrows():
        s = r["scenario"]
        for axis in AXES:
            col = f"smd_{axis.name}"
            vals = np.array(boot[s][col], dtype=float)
            vals = vals[np.isfinite(vals)]
            if len(vals) >= 2:
                lo = float(np.percentile(vals, alpha * 100))
                hi = float(np.percentile(vals, (1.0 - alpha) * 100))
            else:
                lo, hi = (np.nan, np.nan)
            rows.append({
                "scenario":          s,
                "axis":              axis.name,
                "point":             float(r[col]) if pd.notna(r[col]) else np.nan,
                "ci_lo":             lo,
                "ci_hi":             hi,
                "n_prompts":         int(r["n_prompts"]),
                "n_iters_with_data_race": int(r.get("n_iters_with_data_race", 0)),
                # Number of the N_BOOT draws that yielded a finite statistic.
                # Draws that resample nobody from a group give NaN and drop out.
                "n_boot_finite":     int(len(vals)),
            })
    return pd.DataFrame(rows)
