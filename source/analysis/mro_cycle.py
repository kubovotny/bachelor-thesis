"""
Figures 3.10, 3.11 – Sentiment and the Monetary Policy Cycle

Fig 3.10 (plot_fig_3_6a):
    BOTH models (FinBERT solid, RoBERTa dashed) 6M MA on left axis.
    MRO rate LEVEL on inner right axis; Wu-Xia shadow rate LEVEL on
    outer right axis (offset spine). ZLB era shaded.

Fig 3.11 (plot_fig_3_6b):
    (A) Lead-lag: BOTH models vs. MRO CHANGES (changed meetings only).
        FinBERT overall mean + RoBERTa IS mean (IS-restricted, avoids
        journalist-question contamination of the overall RoBERTa mean).
    (B) Grouped box plots: BOTH models side by side per MRO regime.

Both figures are now model-consistent (FinBERT + RoBERTa throughout).
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
import numpy as np
from pathlib import Path
from scipy.stats import pearsonr

from ..data.model_data import return_data
from .. import OUTPUT

plt.rcParams.update(
    {
        "font.size": 11,
        "axes.titlesize": 12,
        "axes.labelsize": 11,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

OUTPUT_DIR = Path(OUTPUT) / "results/mro"
OUTPUT_DIR.mkdir(exist_ok=True)

ROLLING = "180D"  # 6-month centered MA
MAX_LAG = 8  # meetings for lead-lag profile
MIN_CHANGED = 8  # minimum changed-meeting observations per lag

ZLB_START = pd.Timestamp("2012-01-01")
ZLB_END = pd.Timestamp("2022-07-01")

# Consistent model colors with the rest of the thesis
COLORS = {
    "finbert": "#8a7c00",  # blue  — solid lines
    "roberta": "#c0392b",  # red   — dashed lines
}


# ── Load data ─────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    # Overall sentiment (both sections combined)
    df = return_data(
        market_data="ECB Money Market.xlsx",
        word_limit=200,
        IS_QA_division=False,
        qa_options="both_together",
        with_label=False,
    )

    # IS-separated data — needed for roberta_IS_mean in the lead-lag
    df_is = return_data(
        market_data="ECB Money Market.xlsx",
        word_limit=150,
        IS_QA_division=True,
        qa_options="just_answers",
        with_label=False,
    )
    is_cols = [c for c in df_is.columns if c not in df.columns and c != "date"]
    df = pd.merge(df, df_is[["date"] + is_cols], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Centered 6M rolling average for both models (overall mean)
    df = df.set_index("date")
    for model in ("finbert", "roberta"):
        col = f"{model}_mean"
        fwd = df[col].rolling(ROLLING, min_periods=1).mean()
        bwd = df[col][::-1].rolling(ROLLING, min_periods=1).mean()[::-1]
        df[f"{model}_roll"] = (fwd + bwd) / 2
    df = df.reset_index()

    df["mro_change"] = df["MRO"].diff()
    return df


# ── Regime labels ─────────────────────────────────────────────────────────────
def add_regimes(df: pd.DataFrame) -> pd.DataFrame:
    """Label each meeting as hiking / cutting / hold based on MRO change."""
    df = df.copy()
    df["regime"] = "hold"
    df.loc[df["mro_change"] > 0, "regime"] = "hiking"
    df.loc[df["mro_change"] < 0, "regime"] = "cutting"
    return df


# ── Figure 3.10: Both models vs MRO + Shadow Rate ────────────────────────────
def plot_fig_3_6a(df: pd.DataFrame | None = None, save: bool = True):
    """
    Left axis : FinBERT 6M MA (solid blue) + RoBERTa 6M MA (dashed red).
    Inner right axis : MRO rate LEVEL — step function, range [−1, 6%].
    Outer right axis : Wu-Xia shadow rate LEVEL, range [−10, 4%], spine offset.
    ZLB era (2012–2021) shaded grey. Era labels at fixed transform position.
    """
    if df is None:
        df = load_data()

    shadow_col = next(
        (c for c in df.columns if any(k in c.lower() for k in ("shadow", "wu", "xia"))),
        None,
    )
    if shadow_col is None:
        raise KeyError(f"Shadow rate column not found. Available: {list(df.columns)}")

    fig, ax = plt.subplots(figsize=(13, 5.2))
    fig.subplots_adjust(right=0.86)  # leave room for two right-axis labels

    # ── Inner right axis: MRO ─────────────────────────────────────────────────
    axr1 = ax.twinx()
    axr1.set_ylim(-10, 10)
    axr1.set_ylabel("Rate (%)", color="#2980b9")
    axr1.tick_params(
        axis="y",
        labelcolor="#2980b9",
    )

    # ── ZLB shading + era labels (fixed transform — no ylim dependency) ───────
    ax.axvspan(ZLB_START, ZLB_END, color="#d5d5d5", alpha=0.40, zorder=0)
    for x, lbl in [
        (pd.Timestamp("2004-06-01"), "Pre-ZLB"),
        (pd.Timestamp("2016-06-01"), "ZLB"),
        (pd.Timestamp("2023-01-01"), "Hiking"),
    ]:
        ax.text(
            x,
            0.97,
            lbl,
            transform=ax.get_xaxis_transform(),
            ha="center",
            va="top",
            color="#888",
            fontstyle="italic",
        )

    # ── FinBERT 6M MA — solid blue ────────────────────────────────────────────
    ax.plot(
        df["date"],
        df["finbert_roll"],
        color=COLORS["finbert"],
        linewidth=2.0,
        linestyle="-",
        label="FinBERT (6M MA)",
        zorder=5,
    )

    # ── RoBERTa 6M MA — dashed red ────────────────────────────────────────────
    ax.plot(
        df["date"],
        df["roberta_roll"],
        color=COLORS["roberta"],
        linewidth=2.0,
        linestyle="--",
        label="CentralBankRoBERTa (6M MA)",
        zorder=5,
    )

    ax.axhline(0, color="#bbb", linewidth=0.8, linestyle="--", zorder=1)
    ax.set_ylabel(
        "Sentiment Score",
    )
    ax.set_ylim(-0.55, 0.40)  # wide enough for RoBERTa (−0.5) and FinBERT (+0.3)
    ax.tick_params(
        axis="y",
    )

    # ── MRO step function ─────────────────────────────────────────────────────
    axr1.step(
        df["date"],
        df["MRO"],
        where="post",
        color="#2980b9",
        linewidth=2.0,
        label="MRO rate",
        zorder=3,
    )

    # ── Shadow rate level ─────────────────────────────────────────────────────
    axr1.step(
        df["date"],
        df[shadow_col],
        color="#27ae60",
        linewidth=1.4,
        linestyle=":",
        label="Wu-Xia Shadow Rate",
        alpha=0.90,
        zorder=2,
    )
    axr1.axhline(0, color="#27ae60", linewidth=0.5, linestyle=":", alpha=0.4)

    # ── Combined legend ───────────────────────────────────────────────────────
    lines = ax.get_legend_handles_labels()[0] + axr1.get_legend_handles_labels()[0]
    labels = ax.get_legend_handles_labels()[1] + axr1.get_legend_handles_labels()[1]
    ax.legend(lines, labels, loc="upper right", framealpha=0.93, ncol=2)

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel(
        "Date",
    )
    ax.grid(alpha=0.18, linewidth=0.6)

    fig.suptitle(
        "Sentiment and the Monetary Policy Cycle\n"
        "FinBERT (solid) vs. CentralBankRoBERTa (dashed)  |  6M centred MA",
        y=1.01,
    )
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_sentiment_vs_rates.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.11: Both models — lead-lag + grouped regime boxplots ─────────────
def plot_fig_3_6b(df: pd.DataFrame | None = None, save: bool = True):
    """
    Panel A — Lead-lag: FinBERT overall + RoBERTa IS mean vs. ΔMRO.
              RoBERTa IS is used (not overall) to avoid journalist-question
              contamination that makes the overall RoBERTa mean sign-negative.
    Panel B — Grouped box plots: both models side by side per MRO regime.
    """
    if df is None:
        df = load_data()
    df = add_regimes(df)

    changed = df[df["mro_change"] != 0].copy().reset_index(drop=True)
    lags = list(range(-MAX_LAG, MAX_LAG + 1))

    def lag_corr(series: pd.Series, lag: int) -> float:
        t = changed["mro_change"]
        if lag > 0:
            a = series.iloc[lag:].reset_index(drop=True)
            b = t.iloc[:-lag].reset_index(drop=True)
        elif lag < 0:
            a = series.iloc[:lag].reset_index(drop=True)
            b = t.iloc[-lag:].reset_index(drop=True)
        else:
            a, b = series, t
        valid = pd.concat([a, b], axis=1).dropna()
        if len(valid) < MIN_CHANGED:
            return np.nan
        r, _ = pearsonr(valid.iloc[:, 0], valid.iloc[:, 1])
        return r

    rb_col = (
        "roberta_IS_mean" if "roberta_IS_mean" in changed.columns else "roberta_mean"
    )
    # ── Layout ────────────────────────────────────────────────────────────────
    fig, ax = plt.subplots(figsize=(7, 4.8))

    regime_specs = [
        ("hiking", "Hiking\n(MRO ↑)"),
        ("hold", "Hold\n(MRO =)"),
        ("cutting", "Cutting\n(MRO ↓)"),
    ]
    regime_colors = ["#e74c3c", "#95a5a6", "#3498db"]

    n_regimes = len(regime_specs)
    group_gap = 3.0  # distance between regime centers
    box_width = 0.65
    offset = 0.38  # half-gap between FB and RB within one regime

    centers = [i * group_gap for i in range(n_regimes)]
    fb_pos = [c - offset for c in centers]
    rb_pos = [c + offset for c in centers]

    for positions, feat, color, hatch, model_lbl in [
        (fb_pos, "finbert_IS_mean", COLORS["finbert"], "", "FinBERT IS"),
        (
            rb_pos,
            rb_col,
            COLORS["roberta"],
            "//",
            "RoBERTa IS" if rb_col == "roberta_IS_mean" else "RoBERTa",
        ),
    ]:
        data_list = [df[df["regime"] == r][feat].dropna() for r, _ in regime_specs]
        bp = ax.boxplot(
            data_list,
            positions=positions,
            widths=box_width,
            patch_artist=True,
            manage_ticks=False,
            medianprops=dict(color="black", linewidth=1.4),
            boxprops=dict(linewidth=1.1),
            whiskerprops=dict(linewidth=1.1),
            capprops=dict(linewidth=1.1),
            flierprops=dict(
                marker="o", markersize=3.5, alpha=0.45, markerfacecolor=color
            ),
        )
        for i, (patch, rcolor) in enumerate(zip(bp["boxes"], regime_colors)):
            patch.set_facecolor(rcolor)
            patch.set_alpha(0.55)
            patch.set_hatch(hatch)

        # Mean diamond + value label
        for i, (pos, data) in enumerate(zip(positions, data_list)):
            mu = data.mean()
            ax.scatter(
                [pos],
                [mu],
                marker="D",
                s=45,
                color="white",
                edgecolors="black",
                linewidths=1.3,
                zorder=6,
            )
            ax.text(pos, mu + 0.018, f"{mu:.3f}", ha="center", color="#111")

        # Invisible proxy for legend
        # Invisible proxy for legend
        from matplotlib.patches import Rectangle

        ax.add_patch(
            Rectangle(
                (0, 0),
                0,
                0,
                facecolor=COLORS["finbert" if "fin" in feat else "roberta"],
                alpha=0.55,
                hatch=hatch,
                label=model_lbl,
            )
        )

    ax.set_xticks(centers)
    ax.set_xticklabels(
        [lbl for _, lbl in regime_specs],
    )
    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_ylabel(
        "Sentiment Score",
    )
    ax.set_title(
        "Sentiment Distribution by MRO Regime\n(◆ = mean; solid=FinBERT, hatch=RoBERTa IS)",
        pad=6,
    )
    ax.legend(loc="upper right", framealpha=0.92)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)

    # n labels below each regime group
    for i, (r, _) in enumerate(regime_specs):
        n = len(df[df["regime"] == r])
        ax.text(
            centers[i],
            ax.get_ylim()[0] + 0.01,
            f"n={n}",
            ha="center",
            color="#666",
            transform=ax.transData,
        )

    if save:
        path = OUTPUT_DIR / "fig_distribution.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    plot_fig_3_6a(df)
    plot_fig_3_6b(df)
    plt.show()
