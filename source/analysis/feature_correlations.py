"""
Table 3.6 (v3) – Feature Correlation with Monetary Policy Changes

Methodological fixes vs. v2
─────────────────────────────────────────────────────────────────────────────
1. ZLB target  →  Δ(Wu-Xia shadow rate), i.e. first-difference, NOT the level.
   The shadow rate trends monotonically from ~0 % to ~−8 % across 2012-2021.
   Correlating any sentiment feature against that level yields r ≈ −0.9 for
   free (pure co-trending). Using the first-difference removes the common
   trend and isolates meeting-to-meeting co-movement.

2. Pivot deduplication  →  one row per BASE feature (at its peak-|r| lag in
   the Full sample). Previously the same base feature (e.g. finbert_IS_OI_std)
   occupied 11 consecutive rows, which is pseudoreplication, not 11 findings.

3. Cross-era coverage filter  →  features that have NaN in ≥ 1 era column
   are excluded from the main summary table by default. Sparse topic-labeled
   features (especially IS_OI_*) have insufficient Hiking observations (n=13)
   and produce unstable or identical r values across consecutive lags.

4. Fixed-lag comparison table  →  separate output evaluating all features at
   the same forecast horizons {1, 2, 5, 7, 11}. Comparing feature A at lag 5
   vs. feature B at lag 4 is not a valid horse-race; this table enforces the
   same horizon for every entry.

5. Label-overlap diagnostic  →  topic-restricted features (IS_MP, IS_EP,
   IS_OI) are derived from chunks that may carry ≥ 2 topic labels above the
   τ = 0.5 threshold. The proxy metric is the pairwise Pearson r between
   IS sub-features: high r signals that the sub-features share most of their
   data and are not independent predictors.

Word-limit note
───────────────
Unlabeled features use WORD_LIMIT = 1 (single-sentence chunks; best for
direction prediction per the chunk-sensitivity analysis in Appendix A).
Labeled features use WORD_LIMIT_LABEL = 50 (topic classifier needs context).
Because these derive from different segmentations, labelled vs. unlabelled
comparisons are cross-granularity robustness checks, NOT direct head-to-head
comparisons; keep them in separate sections of the table.
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
WORD_LIMIT = 200  # unlabeled features: single-sentence chunks
WORD_LIMIT_LABEL = 200  # labeled features: needs context for topic classifier
MAX_LAG = 15  # meetings to look back
MIN_OBS = 8  # minimum valid observations for a correlation to count
FIXED_LAGS = [1, 2, 5, 7, 11]  # horizons for the apples-to-apples table
MIN_FEAT_STD = 1e-8  # variance guard — skip near-constant features


ERAS = {
    "Full": ("1999-01-01", "2024-12-31"),
    "Pre-ZLB": ("1999-01-01", "2011-12-31"),
    "ZLB": ("2012-01-01", "2022-06-30"),
    "Hiking": ("2022-07-01", "2025-12-31"),
}

# FIX 1: ZLB target is now the FIRST-DIFFERENCE of the Wu-Xia shadow rate.
ERA_TARGET = {
    "Full": "MRO_diff",
    "Pre-ZLB": "MRO_diff",
    "ZLB": "shadow_diff",  # <── was "Wu-Xia shadow rate" (level) → spurious
    "Hiking": "MRO_diff",
}

# Whether to restrict to changed-meeting rows for each era.
# Shadow rate changes at every meeting (continuous), so no filter is needed/possible.
ERA_FILTER_CHANGED = {
    "Full": True,
    "Pre-ZLB": True,
    "ZLB": False,  # shadow_diff is already a change; all meetings kept
    "Hiking": True,
}


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Merge four data configurations into one wide DataFrame:
      • unlabeled IS+QA combined
      • unlabeled IS/QA split
      • labeled  IS+QA combined
      • labeled  IS/QA split

    Also computes MRO_diff and shadow_diff targets plus all lagged features.
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

    df = (
        df.sort_values("date").reset_index(drop=True).copy()
    )  # defragment memory after repeated merges
    df["date"] = pd.to_datetime(df["date"])

    # ── Targets ───────────────────────────────────────────────────────────────
    df["MRO_diff"] = df["MRO"].diff()

    # FIX 1: locate Wu-Xia column and first-difference it
    shadow_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("shadow", "wu", "xia"))),
        None,
    )
    if shadow_col is None:
        raise KeyError(
            "Wu-Xia shadow rate column not found. "
            f"Available columns: {list(df.columns)}"
        )
    df["shadow_diff"] = df[shadow_col].diff()

    # ── Lagged sentiment features ─────────────────────────────────────────────
    sent_cols = [
        c
        for c in df.columns
        if any(m in c for m in ("finbert", "roberta"))
        and c.endswith(("_mean", "_std", "_max", "_min"))
    ]

    lag_dict = {
        f"{col}:lag{lag}": df[col].shift(lag)
        for col in sent_cols
        for lag in range(1, MAX_LAG + 1)
    }
    df = pd.concat([df, pd.DataFrame(lag_dict, index=df.index)], axis=1)

    return df


MIN_CHUNK_COUNT = 5  # minimum average chunks/meeting for a feature to be analyzed
# features below this are single-chunk measurements, not aggregates


def _build_coverage_mask(df: pd.DataFrame, feat_cols: list[str]) -> list[str]:
    """
    Return only those features whose corresponding _count column
    has a mean of >= MIN_CHUNK_COUNT chunks per meeting.

    For unlabeled features (no _count column), always keep them —
    they aggregate over ALL chunks in that section, so coverage is never thin.
    """
    valid = []
    for feat in feat_cols:
        base = feat.split(":lag")[0] if ":lag" in feat else feat
        # Derive count column name: finbert_IS_MP_mean → finbert_IS_MP_count
        count_col = "_".join(base.split("_")[:-1]) + "_count"
        if count_col in df.columns:
            if df[count_col].mean() < MIN_CHUNK_COUNT:
                continue  # thin topic — skip entirely
        valid.append(feat)
    return valid


def _get_feature_cols(df: pd.DataFrame, include_lags: bool = True) -> list[str]:
    base = [
        c
        for c in df.columns
        if any(m in c for m in ("finbert", "roberta"))
        and not any(c.endswith(f":lag{i}") for i in range(1, MAX_LAG + 1))
        and c.endswith(("_mean", "_std", "_max", "_min", "_count"))
    ]
    if not include_lags:
        return base
    return base + [c for c in df.columns if ":lag" in c]


# ── Label-overlap diagnostic ──────────────────────────────────────────────────
def compute_label_overlap(df: pd.DataFrame) -> pd.DataFrame:
    """
    Quantify multi-label chunk overlap via pairwise inter-correlation of
    IS topic-restricted mean features.

    High r between (e.g.) finbert_IS_MP_mean and finbert_IS_EP_mean signals
    that most IS chunks carry both MP and EP labels, so the two features are
    NOT independent predictors. This must be acknowledged in the thesis when
    interpreting topic-restricted correlation results.

    Returns a DataFrame sorted by |r| descending.
    """
    is_means = [
        c for c in df.columns if "IS" in c and c.endswith("_mean") and ":lag" not in c
    ]
    if len(is_means) < 2:
        print("  Not enough IS mean features for overlap analysis.")
        return pd.DataFrame()

    records = []
    for i, c1 in enumerate(is_means):
        for c2 in is_means[i + 1 :]:
            valid = df[[c1, c2]].dropna()
            if len(valid) < 20:
                continue
            r, _ = pearsonr(valid[c1], valid[c2])
            records.append(
                {
                    "Feature A": c1,
                    "Feature B": c2,
                    "r": round(r, 3),
                    "n": len(valid),
                }
            )

    overlap = (
        pd.DataFrame(records)
        .assign(abs_r=lambda x: x["r"].abs())
        .sort_values("abs_r", ascending=False)
        .drop(columns="abs_r")
    )

    print("\n=== Label-overlap diagnostic (IS feature pairwise r) ===")
    print("  r > 0.9 → near-identical data; topic adds no independent signal.")
    print(overlap.head(15).to_string(index=False))
    return overlap


# ── Core correlation table ────────────────────────────────────────────────────
def compute_corr_table(df: pd.DataFrame) -> pd.DataFrame:
    """
    Compute Pearson r between every (feature × lag) and the era-appropriate
    monetary-policy target, restricted to the relevant meeting subset.

    Returns a long-format DataFrame with one row per (Era, Feature) pair.
    """
    feat_cols = _get_feature_cols(df, include_lags=True)
    print(f"  Features before coverage filter: {len(feat_cols)}")
    feat_cols = _build_coverage_mask(df, feat_cols)
    print(f"  Features after coverage filter: {len(feat_cols)}")
    rows = []

    for era, (start, end) in ERAS.items():
        target_col = ERA_TARGET[era]
        era_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        if ERA_FILTER_CHANGED[era]:
            analysis_df = era_df[era_df["MRO_diff"] != 0].copy()
        else:
            analysis_df = era_df.copy()

        print(
            f"\n{era}: n_total={len(era_df)}, "
            f"n_analysis={len(analysis_df)}, "
            f"target={target_col}"
        )

        for feat in feat_cols:
            if feat not in analysis_df.columns:
                continue
            valid = analysis_df[[feat, target_col]].dropna()
            if len(valid) < MIN_OBS:
                continue
            # Variance guard: skip near-constant features
            if (
                valid[feat].std() < MIN_FEAT_STD
                or valid[target_col].std() < MIN_FEAT_STD
            ):
                continue

            r, p = pearsonr(valid[feat], valid[target_col])

            lag = int(feat.split(":lag")[1]) if ":lag" in feat else 0
            base_feat = feat.split(":lag")[0] if ":lag" in feat else feat

            rows.append(
                {
                    "Era": era,
                    "Feature": feat,
                    "Base": base_feat,
                    "Lag": lag,
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


# ── Pivot summary (FIX 2 + FIX 3) ────────────────────────────────────────────
def pivot_summary(
    corr_df: pd.DataFrame,
    top_n: int = 25,
    require_full_coverage: bool = True,
) -> pd.DataFrame:
    """
    Summary table with ONE ROW PER BASE FEATURE at its peak-|r| lag in the
    Full sample. Optionally excludes features with NaN in any era column.

    Parameters
    ----------
    require_full_coverage : bool
        If True (default), drop rows where any era column is NaN. This removes
        sparse topic features (e.g. IS_OI_*) that are unreliable in the Hiking
        era (n ≈ 13). Set False to see the full ranking including sparse features.
    """
    full = corr_df[corr_df["Era"] == "Full"].copy()
    full["abs_r"] = full["r"].abs()

    # FIX 2: one row per base feature → take the row with highest |r| in Full
    best_per_base = (
        full.sort_values("abs_r", ascending=False)
        .drop_duplicates(subset="Base")  # <── deduplication
        .nlargest(top_n * 3, "abs_r")  # oversample before coverage filter
    )

    subset = corr_df[corr_df["Feature"].isin(best_per_base["Feature"].tolist())]
    pivot = subset.pivot_table(
        index="Feature", columns="Era", values="r", aggfunc="first"
    ).reset_index()

    # FIX 3: coverage filter
    if require_full_coverage:
        era_cols = [c for c in ERAS if c in pivot.columns]
        pivot = pivot.dropna(subset=era_cols)

    if "Full" in pivot.columns:
        pivot["_abs"] = pivot["Full"].abs()
        pivot = pivot.sort_values("_abs", ascending=False).drop(columns="_abs")

    return pivot.head(top_n)


# ── Fixed-lag comparison table (FIX 4) ───────────────────────────────────────
def fixed_lag_table(
    corr_df: pd.DataFrame,
    lags: list[int] = FIXED_LAGS,
    top_per_lag: int = 5,
) -> pd.DataFrame:
    """
    Apples-to-apples comparison: evaluate all features at the SAME forecast
    horizons rather than each feature's personal best lag.

    Deduplicates within each lag by base feature to avoid pseudoreplication.

    Returns a DataFrame with columns [Lag, Feature, Base, r, sig, n, Full_r].
    """
    full_subset = (
        corr_df[(corr_df["Era"] == "Full") & (corr_df["Lag"].isin(lags))]
        .copy()
        .assign(abs_r=lambda x: x["r"].abs())
        .sort_values(["Lag", "abs_r"], ascending=[True, False])
        .drop_duplicates(subset=["Lag", "Base"])  # one base feature per lag
        .groupby("Lag")
        .head(top_per_lag)
        .reset_index(drop=True)
    )

    print(f"\n=== Fixed-lag table (Full sample, top {top_per_lag} per lag) ===")
    print("  All features evaluated at the SAME horizon — no personal-best lag.")
    for lag in sorted(lags):
        block = full_subset[full_subset["Lag"] == lag][["Feature", "r", "sig", "n"]]
        print(f"\n  --- Lag {lag} ---")
        print(block.to_string(index=False))

    return full_subset


# ── Lag profile ───────────────────────────────────────────────────────────────
def lag_profile(corr_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """
    For the top-N base features (by peak |r| in Full), return their full
    lag profile across all eras. Used for Figure 3.12 (lag-profile plot).
    """
    full = corr_df[corr_df["Era"] == "Full"].copy()
    full["abs_r"] = full["r"].abs()
    top_bases = full.groupby("Base")["abs_r"].max().nlargest(top_n).index.tolist()
    return corr_df[corr_df["Base"].isin(top_bases)].copy()


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(
        f"Loading data  "
        f"(word_limit_unlabeled={WORD_LIMIT}, "
        f"word_limit_labeled={WORD_LIMIT_LABEL})..."
    )
    df = load_data()
    n_lag = len([c for c in df.columns if ":lag" in c])
    print(f"  {len(df)} meetings, {n_lag} lag features")

    print("\nComputing correlations...")
    corr_df = compute_corr_table(df)
    print(f"  {len(corr_df)} correlation rows computed")

    # ── Diagnostics ───────────────────────────────────────────────────────────
    compute_label_overlap(df)

    # Top 10 per era — deduplicated by base feature
    print("\n=== TOP 10 per era (one entry per base feature, absolute r) ===")
    for era in ERAS:
        era_top = (
            corr_df[corr_df["Era"] == era]
            .assign(abs_r=lambda x: x["r"].abs())
            .sort_values("abs_r", ascending=False)
            .drop_duplicates(subset="Base")  # <── no pseudoreplication
            .head(10)[["Feature", "Lag", "r", "sig", "n"]]
        )
        print(f"\n--- {era} ---")
        print(era_top.to_string(index=False))

    # Lag profile for finbert_mean (headline feature)
    print("\n=== Lag profile: finbert_mean (Full sample) ===")
    fb_lag = corr_df[(corr_df["Base"] == "finbert_mean") & (corr_df["Era"] == "Full")][
        ["Lag", "r", "sig", "n"]
    ].sort_values("Lag")
    print(fb_lag.to_string(index=False))

    # ── Main tables ───────────────────────────────────────────────────────────
    # Full-coverage pivot (for thesis Table 3.3 / Figure 3.13)
    piv_clean = pivot_summary(corr_df, top_n=25, require_full_coverage=True)
    print("\n=== Pivot — deduplicated, full cross-era coverage ===")
    print(piv_clean.to_string(index=False))

    # Unconstrained pivot (for inspection — includes sparse features)
    piv_all = pivot_summary(corr_df, top_n=25, require_full_coverage=False)
    print("\n=== Pivot — deduplicated, sparse features included ===")
    print(piv_all.to_string(index=False))

    # Fixed-lag comparison
    fixed_lag_table(corr_df, lags=FIXED_LAGS)

    # ── Save ──────────────────────────────────────────────────────────────────
    out = OUTPUT_DIR / "correlations"
    out.mkdir(exist_ok=True)
    corr_df.to_csv(out / "table_feature.csv", index=False)
    piv_clean.to_csv(out / "table_summary.csv", index=False)
    piv_all.to_csv(out / "table_summary_all.csv", index=False)
    lag_profile(corr_df).to_csv(out / "table_lag_profile.csv", index=False)
    print(f"\nSaved all tables to {out}")
