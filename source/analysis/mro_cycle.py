"""
Figures 3.6a, 3.6b – Sentiment and the Monetary Policy Cycle

3.6a: RoBERTa sentiment (6M MA) overlaid with MRO rate and Wu-Xia shadow rate.
      Shows whether sentiment leads or lags official rate changes.

3.6b: Asymmetry analysis — does sentiment signal tightening earlier than easing
      (or vice versa)? Compares sentiment dynamics in hiking vs. cutting regimes.
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/mro"
OUTPUT_DIR.mkdir(exist_ok=True)

ROLLING = "180D"   # 6-month centered MA

# ── Load data ─────────────────────────────────────────────────────────────────
def load_data():
    df = return_data(
        market_data="ECB Money Market.xlsx",
        word_limit=150,
        IS_QA_division=False,
        qa_options="both_together",
        with_label=False,
    )
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # 6M centered rolling average for sentiment
    df = df.set_index("date")
    fwd = df["finbert_mean"].rolling(ROLLING, min_periods=1).mean()
    bwd = df["finbert_mean"][::-1].rolling(ROLLING, min_periods=1).mean()[::-1]
    df["roberta_roll"] = (fwd + bwd) / 2
    df = df.reset_index()
    return df


# ── Identify MRO regimes ──────────────────────────────────────────────────────
def add_regimes(df):
    """Label each meeting as hiking, cutting, or hold based on MRO change."""
    df = df.copy()
    df["mro_change"] = df["MRO"].diff()
    df["regime"] = "hold"
    df.loc[df["mro_change"] > 0, "regime"] = "hiking"
    df.loc[df["mro_change"] < 0, "regime"] = "cutting"
    return df


# ── Figure 3.6a: Sentiment vs MRO + Shadow Rate ───────────────────────────────
def plot_fig_3_6a(df=None, save: bool = True):
    if df is None:
        df = load_data()

    fig, ax = plt.subplots(figsize=(13, 7),
                             sharex=True, gridspec_kw={"hspace": 0.08})

    # ── Panel A: Sentiment + MRO ──────────────────────────────────────────────
    axr = ax.twinx()

    ax.fill_between(df["date"], df["finbert_mean"],
                     alpha=0.28, color="#c0392b")
    ax.plot(df["date"], df["roberta_roll"],
             color="#c0392b", linewidth=2.0, label="RoBERTa (6M MA)", zorder=4)
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylabel("RoBERTa Sentiment", fontsize=10, color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")

    axr.plot(df["date"], df["MRO announced"] - df["MRO"],
              color="#2c6fad", linewidth=2.0, linestyle="-",
              label="MRO rate", zorder=3)
    axr.set_ylabel("MRO Rate (%)", fontsize=10, color="#2c6fad")
    axr.tick_params(axis="y", labelcolor="#2c6fad")
    axr.set_ylim(-1, 1)

    # Combined legend
    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.set_title("(A) Sentiment vs. MRO Rate", fontsize=10, pad=6)
    ax.grid(alpha=0.20, linewidth=0.6)


    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylabel("RoBERTa Sentiment", fontsize=10, color="#c0392b")
    ax.tick_params(axis="y", labelcolor="#c0392b")
    ax.set_ylim(-0.25, 0.35)

    axr.plot(df["date"], df["Wu-Xia shadow rate"].diff(),
              color="#27ae60", linewidth=2.0, linestyle="-",
              label="Wu-Xia Shadow Rate", zorder=3)
    axr.set_ylabel("Shadow Rate (%)", fontsize=10, color="#27ae60")
    axr.tick_params(axis="y", labelcolor="#27ae60")

    lines1, labels1 = ax.get_legend_handles_labels()
    lines2, labels2 = axr.get_legend_handles_labels()
    ax.legend(lines1 + lines2, labels1 + labels2,
               loc="upper right", fontsize=9, framealpha=0.9)
    ax.set_title("(B) Sentiment vs. Wu-Xia Shadow Rate", fontsize=10, pad=6)
    ax.grid(alpha=0.20, linewidth=0.6)

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date", fontsize=10)

    fig.suptitle("Sentiment and the Monetary Policy Cycle — RoBERTa",
                 fontsize=11, y=0.99)
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_sentiment_vs_rates.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.6b: Asymmetry — hiking vs cutting ────────────────────────────────
def plot_fig_3_6b(df=None, save: bool = True):
    if df is None:
        df = load_data()

    df = add_regimes(df)

    # Compute lag correlations: does sentiment LEAD MRO changes?
    max_lag = 6  # meetings
    lags = range(-max_lag, max_lag + 1)

    def lag_corr(series_a, series_b, lag):
        """Correlation of series_a with series_b shifted by lag meetings."""
        if lag > 0:
            return series_a.iloc[lag:].corr(series_b.iloc[:-lag])
        elif lag < 0:
            return series_a.iloc[:lag].corr(series_b.iloc[-lag:])
        return series_a.corr(series_b)

    corrs = [lag_corr(df["roberta_roll"], df["mro_change"], lag) for lag in lags]
    shadow_corrs = [lag_corr(df["roberta_roll"], df["Wu-Xia shadow rate"], lag) for lag in lags]

    fig, axes = plt.subplots(1, 2, figsize=(13, 4.5),
                             gridspec_kw={"wspace": 0.30})

    # ── Panel A: Lag correlation ──────────────────────────────────────────────
    ax1 = axes[0]
    ax1.plot(list(lags), corrs, marker="o", markersize=5,
             linewidth=1.8, color="#2c6fad", label="vs. MRO")
    ax1.plot(list(lags), shadow_corrs, marker="s", markersize=5,
             linewidth=1.8, color="#27ae60", linestyle="--",
             label="vs. Shadow Rate")
    ax1.axvline(0, color="#888", linewidth=0.8, linestyle=":")
    ax1.axhline(0, color="#888", linewidth=0.8, linestyle=":")
    ax1.set_xlabel("Lag (meetings, negative = sentiment leads)", fontsize=10)
    ax1.set_ylabel("Pearson Correlation", fontsize=10)
    ax1.set_title("(A) Lead–Lag Correlation:\nSentiment vs. Policy Rates", fontsize=10, pad=6)
    ax1.legend(fontsize=9, framealpha=0.9)
    ax1.grid(alpha=0.25, linewidth=0.6)

    # Annotate peak
    peak_lag = list(lags)[int(np.argmax(np.abs(corrs)))]
    peak_val = corrs[int(np.argmax(np.abs(corrs)))]
    ax1.annotate(f"Peak: lag={peak_lag}\nr={peak_val:.3f}",
                 xy=(peak_lag, peak_val),
                 xytext=(peak_lag + 1.5, peak_val - 0.05),
                 fontsize=8,
                 arrowprops=dict(arrowstyle="->", color="#555", lw=1),
                 color="#555")

    # ── Panel B: Sentiment in hiking vs cutting meetings ──────────────────────
    ax2 = axes[1]

    regime_data = {
        "Hiking\n(MRO ↑)": df[df["regime"] == "hiking"]["finbert_mean"].dropna(),
        "Hold\n(MRO =)": df[df["regime"] == "hold"]["finbert_mean"].dropna(),
        "Cutting\n(MRO ↓)": df[df["regime"] == "cutting"]["finbert_mean"].dropna(),
    }
    colors_regime = ["#e74c3c", "#95a5a6", "#3498db"]

    bp = ax2.boxplot(
        list(regime_data.values()),
        labels=list(regime_data.keys()),
        patch_artist=True, widths=0.55,
        medianprops=dict(color="black", linewidth=1.5),
        boxprops=dict(linewidth=1.2),
        whiskerprops=dict(linewidth=1.2),
        capprops=dict(linewidth=1.2),
    )
    for patch, color in zip(bp["boxes"], colors_regime):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    for i, data in enumerate(regime_data.values()):
        ax2.scatter([i+1], [data.mean()], marker="D", s=55,
                    color="white", edgecolors="black", linewidths=1.5, zorder=5)
        ax2.text(i+1, data.mean() + 0.02, f"{data.mean():.3f}",
                 ha="center", fontsize=8, color="#333")

    ax2.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("RoBERTa Sentiment", fontsize=10)
    ax2.set_title("(B) Sentiment Distribution\nby MRO Regime (◆ = mean)", fontsize=10, pad=6)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)

    # Print n for each regime
    for i, (label, data) in enumerate(regime_data.items()):
        ax2.text(i+1, -0.85, f"n={len(data)}", ha="center", fontsize=8, color="#666")

    fig.suptitle("Asymmetry of Sentiment across the Monetary Policy Cycle — RoBERTa",
                 fontsize=11, y=0.99)
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_sentiment_asymmetry.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    plot_fig_3_6a(df)
    plot_fig_3_6b(df)
    plt.show()