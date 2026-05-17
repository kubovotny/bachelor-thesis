"""
Figures 3.7a, 3.7b – Divergence Analysis and Market Uncertainty

3.7a: IS–QA Divergence = |sent_IS − sent_QA|
      Correlates with OIS changes, STOXX50, and PC1 shocks.
      Tests whether communication mismatch → market uncertainty.

3.7b: Question–Answer Gap = |sent_Q − sent_A|
      Journalist sentiment vs. ECB official sentiment.
      Tests whether adversarial Q&A → market surprise.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from scipy import stats

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Load data ─────────────────────────────────────────────────────────────────
def load_divergence_data() -> pd.DataFrame:
    """IS vs QA sentiment split + all market variables."""
    sent = return_data(
        market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
        word_limit=150,
        IS_QA_division=True,
        qa_options="just_answers",
        with_label=False,
    )
    sent = sent.sort_values("date").reset_index(drop=True)

    # IS–QA divergence (absolute)
    sent["IS_QA_div"] = (sent["roberta_IS_mean"] - sent["roberta_QA_mean"]).abs()

    # OIS uncertainty: std across maturities of OIS changes
    ois_cols = [c for c in sent.columns if c.startswith("OIS_")]
    if ois_cols:
        sent["OIS_uncertainty"] = sent[ois_cols].abs().mean(axis=1)
        sent["OIS_1Y_abs"] = sent["OIS_1Y"].abs() if "OIS_1Y" in sent.columns else np.nan

    return sent


def load_qa_gap_data() -> pd.DataFrame:
    """Journalist questions vs ECB answers sentiment split."""
    sent = return_data(
        market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
        word_limit=150,
        IS_QA_division=True,
        qa_options="both_divided",
        with_label=False,
    )
    sent = sent.sort_values("date").reset_index(drop=True)

    # Find question and answer columns
    # With both_divided: label_formatter adds _question or _answer suffix
    q_col = next((c for c in sent.columns
                  if "roberta" in c and "question" in c and "mean" in c), None)
    a_col = next((c for c in sent.columns
                  if "roberta" in c and "answer" in c and "mean" in c), None)

    if q_col and a_col:
        sent["Q_A_gap"] = (sent[q_col] - sent[a_col]).abs()
        sent["Q_sent"]  = sent[q_col]
        sent["A_sent"]  = sent[a_col]
    else:
        print(f"Available columns: {[c for c in sent.columns if 'roberta' in c]}")
        raise ValueError("Could not find question/answer columns")

    ois_cols = [c for c in sent.columns if c.startswith("OIS_")]
    if ois_cols:
        sent["OIS_uncertainty"] = sent[ois_cols].abs().mean(axis=1)

    return sent


# ── Scatter helper ────────────────────────────────────────────────────────────
def scatter_with_reg(ax, x, y, xlabel, ylabel, title, color="#2c6fad", reg=True):
    """Scatter + OLS regression line + correlation stats."""
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    if len(xv) < 10:
        ax.text(0.5, 0.5, "Insufficient data", ha="center", va="center",
                transform=ax.transAxes)
        return

    r, p = stats.pearsonr(xv, yv)
    slope, intercept, *_ = stats.linregress(xv, yv)

    ax.scatter(xv, yv, alpha=0.45, s=20, color=color, edgecolors="none")
    if reg:
        xfit = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(xfit, slope * xfit + intercept,
                color=color, linewidth=1.8, alpha=0.9)

    sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
    ax.text(0.05, 0.95, f"r = {r:.3f} {sig}\nn = {mask.sum()}",
            transform=ax.transAxes, fontsize=9.5, va="top",
            bbox=dict(boxstyle="round,pad=0.4", facecolor="white",
                      edgecolor="#ccc", alpha=0.9))

    ax.set_xlabel(xlabel, fontsize=10.5)
    ax.set_ylabel(ylabel, fontsize=10.5)
    ax.set_title(title, fontsize=10.5, pad=5)
    ax.grid(alpha=0.22, linewidth=0.6)


# ── Figure 3.7a — IS–QA Divergence ───────────────────────────────────────────
def plot_fig_3_7a(df=None, save: bool = True):
    if df is None:
        df = load_divergence_data()

    market_targets = [
        ("OIS_uncertainty", "|ΔOIS| mean across maturities",  "#2c6fad"),
        ("STOXX50_x",         "STOXX50 return (%)",              "#27ae60"),
        ("pc1",             "PC1 monetary surprise",           "#8e44ad"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"wspace": 0.32})

    for ax, (col, xlabel, color) in zip(axes, market_targets):
        if col not in df.columns:
            ax.text(0.5, 0.5, f"'{col}' not in data",
                    ha="center", va="center", transform=ax.transAxes)
            continue
        scatter_with_reg(
            ax=ax,
            x=df[col],
            y=df["IS_QA_div"],
            xlabel=xlabel,
            ylabel="|IS − QA| Sentiment",
            title=f"IS–QA Divergence\nvs. {xlabel.split('(')[0].strip()}",
            color=color,
            reg=False
        )

    fig.suptitle(
        "IS–QA Communication Divergence vs. Market Variables (RoBERTa)",
        fontsize=12, y=1.01,
    )
    if save:
        path = OUTPUT_DIR / "is_qa/fig_divergence.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.7b — Q–A Sentiment Gap ──────────────────────────────────────────
def plot_fig_3_7b(df=None, save: bool = True):
    if df is None:
        df = load_qa_gap_data()

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2),
                             gridspec_kw={"wspace": 0.32})

    # Panel A: Q vs A time series
    ax1 = axes[0]
    ax1.plot(df["date"], df["Q_sent"],
             color="#e74c3c", linewidth=1.3, alpha=0.75, label="Questions (journalists)")
    ax1.plot(df["date"], df["A_sent"],
             color="#2c6fad", linewidth=1.3, alpha=0.75, label="Answers (ECB)")
    ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax1.set_xlabel("Date", fontsize=10.5)
    ax1.set_ylabel("RoBERTa Sentiment", fontsize=10.5)
    ax1.set_title("Journalist vs. ECB Sentiment\nover Time", fontsize=10.5, pad=5)
    ax1.legend(fontsize=9, loc="lower left", framealpha=0.9)
    ax1.grid(alpha=0.22, linewidth=0.6)
    ax1.xaxis.set_major_locator(mdates.YearLocator(8))
    ax1.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    # Panel B: Q-A gap vs OIS uncertainty
    scatter_with_reg(
        ax=axes[1],
        x=df["OIS_uncertainty"],
        y=df["Q_A_gap"],
        xlabel="|ΔOIS| mean across maturities",
        ylabel="|Q − A| Sentiment Gap",
        title="Q–A Gap vs.\nOIS Market Uncertainty",
        color="#e74c3c",
    )

    # Panel C: Q-A gap vs PC1
    if "pc1" in df.columns:
        scatter_with_reg(
            ax=axes[2],
            x=df["pc1"],
            y=df["Q_A_gap"],
            xlabel="PC1 monetary surprise",
            ylabel="|Q − A| Sentiment Gap",
            title="Q–A Gap vs.\nPC1 Policy Surprise",
            color="#8e44ad",
        )
    else:
        axes[2].text(0.5, 0.5, "PC1 not available",
                     ha="center", va="center", transform=axes[2].transAxes)

    fig.suptitle(
        "Question–Answer Sentiment Gap: Journalist vs. ECB (RoBERTa)",
        fontsize=12, y=1.01,
    )
    if save:
        path = OUTPUT_DIR / "is_qa/gap.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Print correlation summary ─────────────────────────────────────────────────
def print_correlation_summary():
    print("\n=== 3.7.1 IS–QA Divergence correlations ===")
    df = load_divergence_data()
    for col, label in [("OIS_uncertainty", "OIS uncertainty"),
                       ("STOXX50_x", "STOXX50"),
                       ("pc1", "PC1")]:
        if col not in df.columns:
            continue
        mask = df[col].notna() & df["IS_QA_div"].notna()
        r, p = stats.pearsonr(df.loc[mask, col], df.loc[mask, "IS_QA_div"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:25s}: r={r:.4f} {sig}  (n={mask.sum()})")

    print("\n=== 3.7.2 Q–A Gap correlations ===")
    df2 = load_qa_gap_data()
    for col, label in [("OIS_uncertainty", "OIS uncertainty"),
                       ("STOXX50_x", "STOXX50"),
                       ("pc1", "PC1")]:
        if col not in df2.columns:
            continue
        mask = df2[col].notna() & df2["Q_A_gap"].notna()
        r, p = stats.pearsonr(df2.loc[mask, col], df2.loc[mask, "Q_A_gap"])
        sig = "***" if p < 0.001 else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
        print(f"  {label:25s}: r={r:.4f} {sig}  (n={mask.sum()})")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print_correlation_summary()
    df1 = load_divergence_data()
    df2 = load_qa_gap_data()
    plot_fig_3_7a(df1)
    plot_fig_3_7b(df2)
    plt.show()