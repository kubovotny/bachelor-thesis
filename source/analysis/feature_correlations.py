"""
Table 3.6 (revised) – Feature Correlation with MRO Changes

Key changes vs. previous version:
  1. word_limit=1 (single sentences — best predictive performance per Section 2.X)
  2. Target = MRO.diff() filtered to changed meetings only (MRO_diff != 0)
     This removes ~80% hold meetings that dilute the signal.
  3. Lagged sentiment features (lag 1–15 meetings) — key finding from
     exploratory notebook: finbert_mean lagged 11 meetings → r = 0.506
  4. Era-specific targets:
       ZLB (2012–2021) → Wu-Xia shadow rate (MRO stuck at 0)
       All other eras  → MRO_diff

Output: pivot table (features × eras) + lag profile for top features.
"""

import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

WORD_LIMIT  = 1          # single sentences — best for direction prediction
MAX_LAG     = 15         # meetings
MIN_CHANGED = 8          # minimum changed meetings per era for valid correlation

ERAS = {
    "Full":     ("1999-01-01", "2024-12-31"),
    "Pre-ZLB":  ("1999-01-01", "2011-12-31"),
    "ZLB":      ("2012-01-01", "2021-12-31"),
    "Hiking":   ("2022-01-01", "2024-12-31"),
}

# ZLB uses shadow rate as target (MRO was stuck at 0)
ERA_TARGET = {
    "Full":    "MRO_diff",
    "Pre-ZLB": "MRO_diff",
    "ZLB":     "Wu-Xia shadow rate",
    "Hiking":  "MRO_diff",
}


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True,  qa_options="just_answers",  with_label=False),
        dict(IS_QA_division=False, qa_options="both_together", with_label=True),
        dict(IS_QA_division=True,  qa_options="just_answers",  with_label=True),
    ]
    df = None
    for cfg in configs:
        d = return_data(
            market_data="ECB Money Market.xlsx",
            word_limit=WORD_LIMIT if not cfg["with_label"] else 50,
            **cfg,
        )
        if df is None:
            df = d
        else:
            new = [c for c in d.columns if c not in df.columns and c != "date"]
            df  = pd.merge(df, d[["date"] + new], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Targets
    df["MRO_diff"]  = df["MRO"].diff()

    # Add lagged sentiment features
    sent_cols = [c for c in df.columns
                 if any(m in c for m in ["finbert", "roberta"])
                 and c.endswith(("_mean", "_std", "_max", "_min"))]

    for col in sent_cols:
        for lag in range(1, MAX_LAG + 1):
            df[f"{col}:lag{lag}"] = df[col].shift(lag)

    return df


def get_feature_cols(df: pd.DataFrame, include_lags: bool = True) -> list:
    base = [c for c in df.columns
            if any(m in c for m in ["finbert", "roberta"])
            and not c.endswith(tuple(f":lag{i}" for i in range(1, MAX_LAG + 1)))
            and c.endswith(("_mean", "_std", "_max", "_min"))]
    if not include_lags:
        return base
    lags = [c for c in df.columns if ":lag" in c]
    return base + lags


# ── Correlation computation ───────────────────────────────────────────────────
def compute_corr_table(df: pd.DataFrame) -> pd.DataFrame:
    feat_cols = get_feature_cols(df, include_lags=True)
    rows = []

    for era, (start, end) in ERAS.items():
        target_col = ERA_TARGET[era]
        era_df = df[(df["date"] >= start) & (df["date"] <= end)].copy()

        # Filter to changed meetings for MRO_diff target
        if target_col == "MRO_diff":
            analysis_df = era_df[era_df["MRO_diff"] != 0].copy()
        else:
            analysis_df = era_df.copy()

        print(f"\n{era}: n_total={len(era_df)}, n_changed={len(analysis_df)}, target={target_col}")

        for feat in feat_cols:
            if feat not in analysis_df.columns:
                continue
            valid = analysis_df[[feat, target_col]].dropna()
            if len(valid) < MIN_CHANGED:
                continue

            r, p = pearsonr(valid[feat], valid[target_col])

            # Parse lag
            lag = int(feat.split(":lag")[1]) if ":lag" in feat else 0
            base_feat = feat.split(":lag")[0] if ":lag" in feat else feat

            rows.append({
                "Era":       era,
                "Feature":   feat,
                "Base":      base_feat,
                "Lag":       lag,
                "r":         round(r, 4),
                "p":         round(p, 4),
                "sig":       "***" if p < 0.001 else "**" if p < 0.01
                             else "*" if p < 0.05 else "",
                "n":         len(valid),
                "target":    target_col,
            })

    result = pd.DataFrame(rows)
    return result


# ── Lag profile for top base features ────────────────────────────────────────
def lag_profile(corr_df: pd.DataFrame, top_n: int = 6) -> pd.DataFrame:
    """For each era, show how correlation varies with lag for top features."""
    # Top base features by |r| at any lag in Full sample
    full = corr_df[corr_df["Era"] == "Full"].copy()
    full["abs_r"] = full["r"].abs()
    top_bases = (full.groupby("Base")["abs_r"].max()
                     .nlargest(top_n).index.tolist())

    profile = corr_df[corr_df["Base"].isin(top_bases)].copy()
    return profile


# ── Pivot summary ─────────────────────────────────────────────────────────────
def pivot_summary(corr_df: pd.DataFrame, top_n: int = 25) -> pd.DataFrame:
    full = corr_df[corr_df["Era"] == "Full"].copy()
    full["abs_r"] = full["r"].abs()
    top_feats = full.nlargest(top_n, "abs_r")["Feature"].tolist()

    subset = corr_df[corr_df["Feature"].isin(top_feats)]
    pivot  = subset.pivot_table(
        index="Feature", columns="Era", values="r", aggfunc="first"
    ).reset_index()

    if "Full" in pivot.columns:
        pivot["_abs"] = pivot["Full"].abs()
        pivot = pivot.sort_values("_abs", ascending=False).drop(columns="_abs")

    return pivot.head(top_n)


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print(f"Loading data (word_limit={WORD_LIMIT})...")
    df = load_data()
    print(f"  {len(df)} meetings, "
          f"{len([c for c in df.columns if ':lag' in c])} lag features")

    print("\nComputing correlations...")
    corr_df = compute_corr_table(df)
    print(f"  {len(corr_df)} correlation rows computed")

    # Show top findings
    print("\n=== TOP 10 per era (absolute r, changed meetings only) ===")
    for era in ERAS:
        era_top = (corr_df[corr_df["Era"] == era]
                   .assign(abs_r=lambda x: x["r"].abs())
                   .nlargest(10, "abs_r")
                   [["Feature", "Lag", "r", "sig", "n"]])
        print(f"\n--- {era} ---")
        print(era_top.to_string(index=False))

    # Lag profile for key finding
    print("\n=== Lag profile: finbert_mean (Full sample) ===")
    fb_lag = corr_df[
        (corr_df["Base"] == "finbert_mean") &
        (corr_df["Era"] == "Full")
    ][["Lag", "r", "sig", "n"]].sort_values("Lag")
    print(fb_lag.to_string(index=False))

    piv = pivot_summary(corr_df)
    print("\n=== Pivot (top 25 by Full |r|) ===")
    print(piv.to_string(index=False))

    # Save
    corr_df.to_csv(OUTPUT_DIR / "correlations/table_feature.csv", index=False)
    piv.to_csv(OUTPUT_DIR / "correlations/table_summary.csv", index=False)
    lag_profile(corr_df).to_csv(OUTPUT_DIR / "correlations/table_lag_profile.csv", index=False)
    print("\nSaved CSVs (_v2 suffix)")