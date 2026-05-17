"""
feature_correlations_v2.py – Composite Policy Target + Δsentiment Features

Key additions vs. v1
─────────────────────────────────────────────────────────────────────────────
1. Composite policy target (MRO outside ZLB, Wu-Xia shadow rate inside ZLB)
   – During ZLB (2012-2021) the MRO was frozen at zero; shadow rate movements
     represent the true stance of accommodation. Stitching the two series
     roughly doubles n_changed from ~54 to ~110-130, giving more stable
     correlation estimates for the Full sample.

2. Δsentiment features (first differences of meeting-level sentiment)
   – Captures *acceleration* of tone rather than level.
   – By construction less collinear with level features (approximately
     orthogonal when sentiment is mean-reverting).
   – Motivated by exploration.py: contemporaneous OIS correlation was near
     zero, but the *lagged* differential has not been tested systematically.

3. Full_composite era uses composite target → richer Full-sample profile.
   Original Full (MRO-only, n_changed≈54) is retained for comparison.

Methodological note
─────────────────────────────────────────────────────────────────────────────
Pearson r against a {-1, 0, +1} target is point-biserial correlation —
technically valid but measures linear association with a discrete variable.
This table serves as *economic validation* (does sentiment co-move with
policy direction?) rather than as a direct feature-selection criterion.
Feature selection for the predictive model uses nested cross-validation
in predict_model_v3.py.
"""

import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Constants ─────────────────────────────────────────────────────────────────
WORD_LIMIT = 200
WORD_LIMIT_LABEL = 200
MAX_LAG = 15
MIN_OBS = 15  # raised from 8 — eliminates Hiking sparse-topic entries
MIN_FEAT_STD = 1e-8
MIN_CHUNK_COUNT = 5  # coverage filter (IS_OI, IS_FS excluded)
FIXED_LAGS = [1, 2, 5, 7, 11]

# ZLB shadow rate threshold: meeting-to-meeting changes smaller than this
# are treated as "no policy signal" (interpolation noise in the Wu-Xia model)
MIN_SHADOW_CHANGE = 0.10  # % — ≈ 10bp equivalent


ZLB_START = "2012-01-01"
ZLB_END = "2022-06-30"

ERAS = {
    "Full_composite": ("1999-01-01", "2025-12-31"),
    "Full": ("1999-01-01", "2025-12-31"),
    "Pre-ZLB": ("1999-01-01", "2011-12-31"),
    "ZLB": ("2012-01-01", "2022-06-30"),
    "Hiking": ("2022-07-01", "2025-12-31"),
}

ERA_TARGET = {
    "Full_composite": "composite_diff",  # stitched MRO + shadow
    "Full": "MRO_diff",
    "Pre-ZLB": "MRO_diff",
    "ZLB": "shadow_diff",  # first-difference of shadow rate
    "Hiking": "MRO_diff",
}

ERA_FILTER_CHANGED = {
    "Full_composite": True,  # filter where composite_diff != 0
    "Full": True,  # filter where MRO_diff != 0
    "Pre-ZLB": True,
    "ZLB": False,  # shadow_diff changes at every meeting; keep all
    "Hiking": True,
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load sentiment (unlabeled + labeled) and compute:
      – MRO_diff, shadow_diff, composite_diff  (policy targets)
      – Δsentiment features  (first differences)
      – Lagged versions of all sentiment columns
    """
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True, qa_options="just_answers", with_label=False),
        dict(IS_QA_division=False, qa_options="both_together", with_label=True),
        dict(IS_QA_division=True, qa_options="just_answers", with_label=True),
    ]
    df = None
    for cfg in configs:
        wl = WORD_LIMIT_LABEL if cfg["with_label"] else WORD_LIMIT
        d = return_data(
            market_data="ECB Money Market.xlsx",
            word_limit=wl,
            **cfg,
        )
        if df is None:
            df = d
        else:
            new_cols = [c for c in d.columns if c not in df.columns and c != "date"]
            df = pd.merge(df, d[["date"] + new_cols], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True).copy()
    df["date"] = pd.to_datetime(df["date"])

    # ── Policy targets ────────────────────────────────────────────────────────
    df["MRO_diff"] = df["MRO"].diff()

    shadow_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("shadow", "wu", "xia"))),
        None,
    )
    if shadow_col is None:
        raise KeyError(
            f"Wu-Xia shadow rate column not found. Columns: {list(df.columns)}"
        )
    df["shadow_diff"] = df[shadow_col].diff()

    # Composite: shadow_diff inside ZLB, MRO_diff outside
    # Shadow movements < MIN_SHADOW_CHANGE treated as zero (noise threshold)
    zlb_mask = (df["date"] >= ZLB_START) & (df["date"] <= ZLB_END)
    shadow_signal = df["shadow_diff"].where(
        df["shadow_diff"].abs() >= MIN_SHADOW_CHANGE, 0.0
    )
    df["composite_diff"] = np.where(zlb_mask, shadow_signal, df["MRO_diff"])

    # ── Sentiment columns ─────────────────────────────────────────────────────
    sent_cols = [
        c
        for c in df.columns
        if any(m in c for m in ("finbert", "roberta"))
        and c.endswith(("_mean", "_std", "_max", "_min"))
    ]

    # Δsentiment: first difference (communication surprise)
    diff_dict = {f"d_{col}": df[col].diff() for col in sent_cols}

    # Lags of both level and Δsentiment features
    all_base = sent_cols + list(diff_dict.keys())
    lag_dict = {
        f"{col}:lag{lag}": (
            diff_dict.get(col, df.get(col, pd.Series()))
            if col.startswith("d_")
            else df[col]
        ).shift(lag)
        for col in all_base
        for lag in range(1, MAX_LAG + 1)
        if col in df.columns or col in diff_dict
    }

    # Build lag columns properly
    lag_frames = {}
    for col in sent_cols:
        for lag in range(1, MAX_LAG + 1):
            lag_frames[f"{col}:lag{lag}"] = df[col].shift(lag)
    for col in sent_cols:
        dcol = f"d_{col}"
        dvals = df[col].diff()
        lag_frames[dcol] = dvals  # lag-0 diff
        for lag in range(1, MAX_LAG + 1):
            lag_frames[f"{dcol}:lag{lag}"] = dvals.shift(lag)

    df = pd.concat(
        [df, pd.DataFrame(lag_frames, index=df.index)],
        axis=1,
    ).copy()

    return df


def _get_feature_cols(df: pd.DataFrame) -> list[str]:
    """All sentiment level + Δsentiment features (base + lagged)."""
    return [
        c
        for c in df.columns
        if (
            any(m in c for m in ("finbert", "roberta"))
            and (
                c.endswith(("_mean", "_std", "_max", "_min"))
                or (
                    c.startswith("d_")
                    and any(
                        c.endswith(sfx)
                        for sfx in (
                            "_mean",
                            "_std",
                            "_max",
                            "_min",
                            *("_mean:lag" + str(i) for i in range(1, MAX_LAG + 1)),
                        )
                    )
                )
                or ":lag" in c
            )
        )
    ]


def _get_feature_cols_clean(df: pd.DataFrame) -> list[str]:
    """
    All level and Δsentiment columns (base + lagged),
    excluding obvious non-feature columns.
    """
    features = []
    for c in df.columns:
        if not any(m in c for m in ("finbert", "roberta")):
            continue
        # Level features
        base_ends = ("_mean", "_std", "_max", "_min")
        is_level = any(c.endswith(sfx) for sfx in base_ends)
        is_level_lag = ":lag" in c and any(
            c.split(":lag")[0].endswith(sfx) for sfx in base_ends
        )
        # Delta features
        is_delta = c.startswith("d_") and any(
            c.replace("d_", "").split(":lag")[0].endswith(sfx) for sfx in base_ends
        )
        if is_level or is_level_lag or is_delta:
            features.append(c)
    return features


# ── Coverage filter ───────────────────────────────────────────────────────────
def _build_coverage_mask(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """
    Exclude topic-restricted features whose underlying topic averages
    fewer than MIN_CHUNK_COUNT chunks per meeting (IS_OI, IS_FS).
    Δsentiment and unlabeled features are always kept.
    """
    valid = []
    for feat in feat_cols:
        base = feat.split(":lag")[0]
        if base.startswith("d_"):
            base = base[2:]  # strip d_ prefix to find count column
        count_col = "_".join(base.split("_")[:-1]) + "_count"
        if count_col in df.columns:
            if df[count_col].mean() < MIN_CHUNK_COUNT:
                continue
        valid.append(feat)
    return valid


# ── Label-overlap diagnostic ──────────────────────────────────────────────────
def compute_label_overlap(df: pd.DataFrame) -> pd.DataFrame:
    is_means = [
        c
        for c in df.columns
        if "IS" in c
        and c.endswith("_mean")
        and ":lag" not in c
        and not c.startswith("d_")
    ]
    if len(is_means) < 2:
        return pd.DataFrame()

    records = []
    for i, c1 in enumerate(is_means):
        for c2 in is_means[i + 1 :]:
            valid = df[[c1, c2]].dropna()
            if len(valid) < 20:
                continue
            r, _ = pearsonr(valid[c1], valid[c2])
            records.append(
                {"Feature A": c1, "Feature B": c2, "r": round(r, 3), "n": len(valid)}
            )

    overlap = (
        pd.DataFrame(records)
        .assign(abs_r=lambda x: x["r"].abs())
        .sort_values("abs_r", ascending=False)
        .drop(columns="abs_r")
    )
    print("\n=== Label-overlap diagnostic ===")
    print(overlap.head(10).to_string(index=False))
    return overlap


# ── Core correlation table ────────────────────────────────────────────────────
def compute_corr_table(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = _get_feature_cols_clean(df)
    feat_cols = _build_coverage_mask(df, feat_cols)
    print(f"  Features after coverage filter: {len(feat_cols)}")

    rows = []
    for era, (start, end) in ERAS.items():
        target_col = ERA_TARGET[era]
        era_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        if ERA_FILTER_CHANGED[era]:
            analysis_df = era_df[era_df[target_col] != 0].copy()
        else:
            analysis_df = era_df.copy()

        print(
            f"\n{era}: n_total={len(era_df)}, "
            f"n_analysis={len(analysis_df)}, target={target_col}"
        )

        for feat in feat_cols:
            if feat not in analysis_df.columns:
                continue
            valid = analysis_df[[feat, target_col]].dropna()
            if len(valid) < MIN_OBS:
                continue
            if (
                valid[feat].std() < MIN_FEAT_STD
                or valid[target_col].std() < MIN_FEAT_STD
            ):
                continue

            r, p = pearsonr(valid[feat], valid[target_col])
            lag = int(feat.split(":lag")[1]) if ":lag" in feat else 0
            base_feat = feat.split(":lag")[0] if ":lag" in feat else feat
            is_delta = base_feat.startswith("d_")

            rows.append(
                {
                    "Era": era,
                    "Feature": feat,
                    "Base": base_feat,
                    "Lag": lag,
                    "IsDelta": is_delta,
                    "r": round(r, 4),
                    "p": round(p, 4),
                    "sig": (
                        "***"
                        if p < 0.001
                        else "**" if p < 0.01 else "*" if p < 0.05 else ""
                    ),
                    "n": len(valid),
                    "target": target_col,
                }
            )

    return pd.DataFrame(rows)


# ── Pivot summary ─────────────────────────────────────────────────────────────
def pivot_summary(
    corr_df: pd.DataFrame,
    era_focus: str = "Full_composite",
    top_n: int = 25,
    require_full_coverage: bool = True,
    levels_only: bool = False,
    deltas_only: bool = False,
) -> pd.DataFrame:
    """
    One row per base feature at its peak-|r| lag in era_focus.
    Optionally filter to level-only or delta-only features.

    era_focus: which era to rank by (default: Full_composite)
    """
    full = corr_df[corr_df["Era"] == era_focus].copy()
    full["abs_r"] = full["r"].abs()

    if levels_only:
        full = full[~full["IsDelta"]]
    if deltas_only:
        full = full[full["IsDelta"]]

    best_per_base = (
        full.sort_values("abs_r", ascending=False)
        .drop_duplicates(subset="Base")
        .nlargest(top_n * 3, "abs_r")
    )

    era_cols = [e for e in ERAS if e != "Full"]  # show all eras in pivot
    subset = corr_df[corr_df["Feature"].isin(best_per_base["Feature"].tolist())]
    pivot = subset.pivot_table(
        index="Feature", columns="Era", values="r", aggfunc="first"
    ).reset_index()

    if require_full_coverage:
        check_cols = [c for c in ["Pre-ZLB", "ZLB", "Hiking"] if c in pivot.columns]
        pivot = pivot.dropna(subset=check_cols)

    if "Full_composite" in pivot.columns:
        pivot["_abs"] = pivot["Full_composite"].abs()
        pivot = pivot.sort_values("_abs", ascending=False).drop(columns="_abs")

    return pivot.head(top_n)


# ── Fixed-lag table ───────────────────────────────────────────────────────────
def fixed_lag_table(
    corr_df: pd.DataFrame,
    lags: list[int] = FIXED_LAGS,
    era_focus: str = "Full_composite",
    top_per_lag: int = 5,
) -> pd.DataFrame:
    """Apples-to-apples: all features at the same forecast horizons."""
    full_sub = (
        corr_df[(corr_df["Era"] == era_focus) & (corr_df["Lag"].isin(lags))]
        .copy()
        .assign(abs_r=lambda x: x["r"].abs())
        .sort_values(["Lag", "abs_r"], ascending=[True, False])
        .drop_duplicates(subset=["Lag", "Base"])
        .groupby("Lag")
        .head(top_per_lag)
        .reset_index(drop=True)
    )

    print(f"\n=== Fixed-lag table ({era_focus}, top {top_per_lag} per lag) ===")
    for lag in sorted(lags):
        block = full_sub[full_sub["Lag"] == lag][
            ["Feature", "IsDelta", "r", "sig", "n"]
        ]
        print(f"\n  --- Lag {lag} ---")
        print(block.to_string(index=False))

    return full_sub


# ── Lag profile ───────────────────────────────────────────────────────────────
def lag_profile(
    corr_df: pd.DataFrame,
    top_n: int = 6,
    era_focus: str = "Full_composite",
) -> pd.DataFrame:
    full = corr_df[corr_df["Era"] == era_focus].copy()
    full["abs_r"] = full["r"].abs()
    top_bases = full.groupby("Base")["abs_r"].max().nlargest(top_n).index.tolist()
    return corr_df[corr_df["Base"].isin(top_bases)].copy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        f"Loading data (word_limit_unlabeled={WORD_LIMIT}, "
        f"word_limit_labeled={WORD_LIMIT_LABEL})..."
    )
    df = load_data()
    n_lag = len([c for c in df.columns if ":lag" in c])
    n_delta = len([c for c in df.columns if c.startswith("d_")])
    print(f"  {len(df)} meetings, {n_lag} lag features, {n_delta} Δ features")

    print("\nComputing correlations...")
    corr_df = compute_corr_table(df)
    print(f"  {len(corr_df)} correlation rows computed")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    compute_label_overlap(df)

    # n_changed comparison: old vs new
    print("\n=== n_changed per era (composite vs MRO-only) ===")
    for era in ERAS:
        t = ERA_TARGET[era]
        start, end = ERAS[era]
        era_df = df[(df["date"] >= start) & (df["date"] <= end)]
        if ERA_FILTER_CHANGED[era]:
            n = (era_df[t] != 0).sum()
        else:
            n = len(era_df)
        print(f"  {era:20s}  target={t:20s}  n_analysis={n}")

    # ── Top 10 per era ────────────────────────────────────────────────────────
    print("\n=== TOP 10 per era (one per base, absolute r) ===")
    for era in ERAS:
        era_top = (
            corr_df[corr_df["Era"] == era]
            .assign(abs_r=lambda x: x["r"].abs())
            .sort_values("abs_r", ascending=False)
            .drop_duplicates(subset="Base")
            .head(10)[["Feature", "IsDelta", "Lag", "r", "sig", "n"]]
        )
        print(f"\n--- {era} ---")
        print(era_top.to_string(index=False))

    # ── Lag profile for finbert_mean ──────────────────────────────────────────
    print("\n=== Lag profile: finbert_mean (Full_composite) ===")
    fb_lag = corr_df[
        (corr_df["Base"] == "finbert_mean") & (corr_df["Era"] == "Full_composite")
    ][["Lag", "r", "sig", "n"]].sort_values("Lag")
    print(fb_lag.to_string(index=False))

    print("\n=== Lag profile: d_finbert_mean (Full_composite) ===")
    dfb_lag = corr_df[
        (corr_df["Base"] == "d_finbert_mean") & (corr_df["Era"] == "Full_composite")
    ][["Lag", "r", "sig", "n"]].sort_values("Lag")
    print(dfb_lag.to_string(index=False))

    # ── Pivots ────────────────────────────────────────────────────────────────
    piv_all = pivot_summary(corr_df, require_full_coverage=True)
    piv_levels = pivot_summary(corr_df, require_full_coverage=True, levels_only=True)
    piv_deltas = pivot_summary(corr_df, require_full_coverage=True, deltas_only=True)

    print("\n=== Pivot — all features (deduplicated, full coverage) ===")
    print(piv_all.to_string(index=False))

    print("\n=== Pivot — LEVEL features only ===")
    print(piv_levels.to_string(index=False))

    print("\n=== Pivot — DELTA (Δsentiment) features only ===")
    print(piv_deltas.to_string(index=False))

    fixed_lag_table(corr_df)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / "correlations"
    out.mkdir(exist_ok=True)
    corr_df.to_csv(out / "table_feature_v2.csv", index=False)
    piv_all.to_csv(out / "table_summary_v2.csv", index=False)
    piv_levels.to_csv(out / "table_summary_levels.csv", index=False)
    piv_deltas.to_csv(out / "table_summary_deltas.csv", index=False)
    lag_profile(corr_df).to_csv(out / "table_lag_profile_v2.csv", index=False)
    print(f"\nSaved all tables to {out}")
