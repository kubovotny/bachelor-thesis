"""
Section 3.9 – Exploring the Trading Value: Delta Sentiment and Volatility

Key insight: OIS_1Y is already a CHANGE (market reaction during press conference).
Correlating it with sentiment LEVEL finds nothing. Two better approaches:

  A) Delta sentiment: Δsent = sent_t − sent_{t-1}
     "Did the ECB become MORE hawkish than last time?"
     This is the communication SURPRISE, analogous to PC1 for rates.

  B) Sentiment → |OIS| (market volatility, not direction)
     "Does uncertain/extreme sentiment cause bigger market moves?"
     Evidence from Section 3.6: roberta_std correlated -0.55 with ΔMRO in Hiking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from pathlib import Path
from scipy import stats

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

SENT_FEATURES = [
    ("finbert_IS_mean",  "ΔFB IS mean",   "#2c6fad"),
    ("finbert_QA_mean",  "ΔFB QA mean",   "#27ae60"),
    ("finbert_mean",     "ΔFB overall",   "#8e44ad"),
    ("roberta_IS_mean",  "ΔRB IS mean",   "#c0392b"),
    ("finbert_IS_std",   "FB IS std",     "#e67e22"),
    ("roberta_std",      "RB overall std","#1a4a7a"),
]


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data():
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True,  qa_options="just_answers",  with_label=False),
    ]
    df = None
    for cfg in configs:
        d = return_data(
            market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
            word_limit=150, **cfg,
        )
        if df is None:
            df = d
        else:
            new = [c for c in d.columns if c not in df.columns and c != "date"]
            df  = pd.merge(df, d[["date"] + new], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Delta sentiment (first difference per feature)
    sent_cols = [c for c in df.columns
                 if any(m in c for m in ["finbert", "roberta"])
                 and c.endswith(("_mean", "_std", "_max", "_min"))]
    for col in sent_cols:
        df[f"d_{col}"] = df[col].diff()

    # OIS volatility proxy
    ois_cols = [c for c in df.columns if c.startswith("OIS_")]
    if ois_cols:
        df["OIS_abs"] = df[ois_cols].abs().mean(axis=1)

    return df


# ── Figure 3.9a: Δsentiment vs OIS_1Y  +  std vs |OIS| ───────────────────────
def plot_fig_3_9a(df: pd.DataFrame, save: bool = True):
    fig, axes = plt.subplots(2, 3, figsize=(14, 8),
                             gridspec_kw={"hspace": 0.42, "wspace": 0.30})

    # Top row: delta sentiment vs OIS_1Y
    # Bottom row: std sentiment vs |OIS|
    top_pairs = [
        (f"d_{f}", lbl, c, "OIS_1Y", "OIS 1Y change (bps)")
        for f, lbl, c in SENT_FEATURES[:3]
    ]
    bot_pairs = [
        (f if "std" in f else f"d_{f}",
         lbl if "std" in f else lbl, c,
         "OIS_abs", "|OIS| mean abs change")
        for f, lbl, c in SENT_FEATURES[3:]
    ]

    for row, pairs in enumerate([top_pairs, bot_pairs]):
        for col, (feat, label, color, target, ylabel) in enumerate(pairs):
            ax = axes[row][col]
            if feat not in df.columns or target not in df.columns:
                ax.text(0.5, 0.5, "N/A", ha="center", va="center",
                        transform=ax.transAxes)
                continue

            mask = df[feat].notna() & df[target].notna()
            x, y = df.loc[mask, feat].values, df.loc[mask, target].values
            r, p  = stats.pearsonr(x, y)
            sig   = "***" if p<0.001 else "**" if p<0.01 \
                    else "*" if p<0.05 else "n.s."

            dates = df.loc[mask, "date"]
            ec = np.where(dates < "2012-01-01", "#3498db",
                 np.where(dates < "2022-01-01", "#e67e22", "#e74c3c"))
            ax.scatter(x, y, c=ec, alpha=0.5, s=22, edgecolors="none")
            ax.axhline(0, color="#ddd", linewidth=0.7)
            ax.axvline(0, color="#ddd", linewidth=0.7)

            slope, intercept, *_ = stats.linregress(x, y)
            xfit = np.linspace(x.min(), x.max(), 100)
            ax.plot(xfit, slope * xfit + intercept,
                    color=color, linewidth=1.8, alpha=0.85)

            ax.text(0.05, 0.95, f"r = {r:.3f} {sig}\nn = {mask.sum()}",
                    transform=ax.transAxes, fontsize=8.5, va="top",
                    bbox=dict(boxstyle="round,pad=0.3", facecolor="white",
                              edgecolor="#ccc", alpha=0.9))
            ax.set_xlabel(label, fontsize=9)
            ax.set_ylabel(ylabel, fontsize=9)
            prefix = "Δ" if row == 0 else ""
            ax.set_title(f"{prefix}{label}\nvs. {ylabel.split('(')[0].strip()}",
                         fontsize=9.5, pad=5)
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.autoscale_view()
            # Alebo manuálne nastavte limity pre 'std' grafy:
            if "std" in feat:
                ax.set_xlim(df[feat].min() * 0.9, df[feat].max() * 1.1)

    # Row labels
    axes[0][0].annotate("Δsentiment vs\nOIS 1Y", xy=(-0.25, 0.5),
                        xycoords="axes fraction", fontsize=8.5, ha="right",
                        va="center", rotation=90, color="#555")
    axes[1][0].annotate("Sent. std vs\n|OIS| volatility", xy=(-0.25, 0.5),
                        xycoords="axes fraction", fontsize=8.5, ha="right",
                        va="center", rotation=90, color="#555")

    from matplotlib.patches import Patch
    fig.legend(
        handles=[Patch(color="#3498db", label="Pre-ZLB (1999–2011)"),
                 Patch(color="#e67e22", label="ZLB (2012–2021)"),
                 Patch(color="#e74c3c", label="Hiking (2022–2024)")],
        loc="lower center", ncol=3, fontsize=9,
        framealpha=0.9, bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Communication Surprise (Δsentiment) and Market Volatility (|OIS|)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "model/fig_delta_sentiment.pdf",
                    bbox_inches="tight")
        print("Saved fig_3_9a")
    return fig


# ── Figure 3.9b: Rolling correlation (count-based, robust) ───────────────────
def plot_fig_3_9b(df: pd.DataFrame, save: bool = True):
    WINDOW = 20   # ~2.5 years at 8 meetings/year

    pairs = [
        ("d_finbert_IS_mean", "ΔFinBERT IS mean", "#2c6fad", "OIS_1Y"),
        ("d_finbert_QA_mean", "ΔFinBERT QA mean", "#27ae60", "OIS_1Y"),
        ("finbert_IS_std",    "FinBERT IS std",    "#e67e22", "OIS_abs"),
        ("roberta_std",       "RoBERTa overall std","#c0392b","OIS_abs"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 7),
                             gridspec_kw={"hspace": 0.35})

    titles = [
        "Rolling Correlation: Δsentiment vs. OIS 1Y (direction)",
        "Rolling Correlation: Sentiment std vs. |OIS| (volatility)",
    ]

    for ax_idx, (ax, title) in enumerate(zip(axes, titles)):
        ax.axhline(0, color="#555", linewidth=1.0, linestyle="--", alpha=0.6)

        for feat, label, color, target in pairs:
            if feat_target_row(ax_idx, feat, target):
                continue
            if feat not in df.columns or target not in df.columns:
                continue
            tmp = df[["date", feat, target]].dropna().reset_index(drop=True)
            roll_r = tmp[feat].rolling(WINDOW, min_periods=10).corr(tmp[target])
            ax.plot(tmp["date"], roll_r.values,
                    color=color, linewidth=1.8, label=label, alpha=0.85)

        # Era shading
        for s, e, c, lbl in [
            ("1999-01-01","2007-12-31","#3498db","Pre-GFC"),
            ("2008-01-01","2011-12-31","#e67e22","GFC"),
            ("2012-01-01","2021-12-31","#27ae60","ZLB"),
            ("2022-01-01","2024-12-31","#e74c3c","Hiking"),
        ]:
            ax.axvspan(pd.Timestamp(s), pd.Timestamp(e), alpha=0.06, color=c)
            mid = pd.Timestamp(s) + (pd.Timestamp(e) - pd.Timestamp(s)) / 2
            ax.text(mid, 0.92, lbl, ha="center", fontsize=7.5,
                    color=c, alpha=0.7, transform=ax.get_xaxis_transform())

        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel("Pearson r (rolling 20-meeting)", fontsize=10)
        ax.set_title(title, fontsize=10, pad=6)
        ax.legend(fontsize=9, framealpha=0.92, loc="lower left")
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    fig.suptitle(
        "Time-Varying Sentiment–Market Relationship\n"
        "(20-meeting rolling window ≈ 2.5 years)",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()
    if save:
        fig.savefig(OUTPUT_DIR / "model/fig_rolling_correlation.pdf",
                    bbox_inches="tight")
        print("Saved fig_3_9b")
    return fig


def feat_target_row(row_idx, feat, target):
    """Filter: row 0 = direction pairs, row 1 = volatility pairs."""
    if row_idx == 0:
        return "std" in feat or "abs" in target
    else:
        return "std" not in feat and "abs" not in target


# ── Print correlations ────────────────────────────────────────────────────────
def print_correlations(df: pd.DataFrame):
    for target, label in [("OIS_1Y", "OIS_1Y (direction)"),
                           ("OIS_abs", "|OIS| (volatility)")]:
        print(f"\n=== Correlations vs {label} ===")
        for feat_raw, feat_label, _ in SENT_FEATURES:
            for prefix, plabel in [("d_", "Δ"), ("", "")]:
                feat = prefix + feat_raw
                if feat not in df.columns:
                    continue
                mask = df[feat].notna() & df[target].notna()
                r, p = stats.pearsonr(df.loc[mask, feat], df.loc[mask, target])
                sig  = "***" if p<0.001 else "**" if p<0.01 \
                       else "*" if p<0.05 else "n.s."
                print(f"  {plabel}{feat_label:22s}: r={r:+.4f} {sig}")
            break  # only delta for direction, only level for std


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    print(f"  {len(df)} meetings, {len([c for c in df.columns if c.startswith('d_')])} delta features")
    print_correlations(df)
    plot_fig_3_9a(df)
    plot_fig_3_9b(df)
    plt.show()