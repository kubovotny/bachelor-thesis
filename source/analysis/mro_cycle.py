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

OUTPUT_DIR = Path(OUTPUT) / "results/mro"
OUTPUT_DIR.mkdir(exist_ok=True)

ROLLING = "180D"  # 6-month centered MA
MAX_LAG = 8  # meetings for lead-lag profile
MIN_CHANGED = 8  # minimum changed-meeting observations per lag

ZLB_START = pd.Timestamp("2012-01-01")
ZLB_END = pd.Timestamp("2021-12-31")

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
        word_limit=150,
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
    axr1.set_ylim(-1, 6)
    axr1.set_ylabel("MRO Rate (%)", fontsize=9, color="#2980b9")
    axr1.tick_params(axis="y", labelcolor="#2980b9", labelsize=8)

    # ── Outer right axis: shadow rate (spine pushed outward) ──────────────────
    axr2 = ax.twinx()
    axr2.spines["right"].set_position(("axes", 1.09))
    axr2.set_ylim(-10, 4)
    axr2.set_ylabel("Shadow Rate (%)", fontsize=9, color="#27ae60")
    axr2.tick_params(axis="y", labelcolor="#27ae60", labelsize=8)

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
            fontsize=8.5,
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
    ax.set_ylabel("Sentiment Score", fontsize=10)
    ax.set_ylim(-0.55, 0.40)  # wide enough for RoBERTa (−0.5) and FinBERT (+0.3)
    ax.tick_params(axis="y", labelsize=8)

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
    axr2.step(
        df["date"],
        df[shadow_col],
        color="#27ae60",
        linewidth=1.4,
        linestyle=":",
        label="Wu-Xia Shadow Rate",
        alpha=0.90,
        zorder=2,
    )
    axr2.axhline(0, color="#27ae60", linewidth=0.5, linestyle=":", alpha=0.4)

    # ── Combined legend ───────────────────────────────────────────────────────
    lines = (
        ax.get_legend_handles_labels()[0]
        + axr1.get_legend_handles_labels()[0]
        + axr2.get_legend_handles_labels()[0]
    )
    labels = (
        ax.get_legend_handles_labels()[1]
        + axr1.get_legend_handles_labels()[1]
        + axr2.get_legend_handles_labels()[1]
    )
    ax.legend(lines, labels, loc="upper right", fontsize=8.5, framealpha=0.93, ncol=2)

    # ── X-axis ────────────────────────────────────────────────────────────────
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlabel("Date", fontsize=10)
    ax.grid(alpha=0.18, linewidth=0.6)

    fig.suptitle(
        "Sentiment and the Monetary Policy Cycle\n"
        "FinBERT (solid) vs. CentralBankRoBERTa (dashed)  |  6M centred MA",
        fontsize=11,
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

    # FinBERT overall and RoBERTa IS (IS-restricted avoids journalist drag)
    corrs_fb = [lag_corr(changed["finbert_mean"], lag) for lag in lags]
    rb_col = (
        "roberta_IS_mean" if "roberta_IS_mean" in changed.columns else "roberta_mean"
    )
    corrs_rb = [lag_corr(changed[rb_col], lag) for lag in lags]

    # ── Layout ────────────────────────────────────────────────────────────────
    fig, axes = plt.subplots(1, 2, figsize=(13, 4.8), gridspec_kw={"wspace": 0.32})

    # ── Panel A: Lead-lag ─────────────────────────────────────────────────────
    ax1 = axes[0]
    ax1.axvline(0, color="#888", linewidth=0.8, linestyle=":")
    ax1.axhline(0, color="#888", linewidth=0.8, linestyle=":")

    for corrs, model, color, ls, lbl in [
        (corrs_fb, "finbert", COLORS["finbert"], "-", "FinBERT overall"),
        (
            corrs_rb,
            "roberta",
            COLORS["roberta"],
            "--",
            f"RoBERTa {'IS' if rb_col == 'roberta_IS_mean' else 'overall'}",
        ),
    ]:
        ax1.plot(
            lags,
            corrs,
            marker="o",
            markersize=4,
            linewidth=1.8,
            color=color,
            linestyle=ls,
            label=lbl,
        )
        valid = [(l, c) for l, c in zip(lags, corrs) if not np.isnan(c)]
        if valid:
            pl, pv = max(valid, key=lambda x: abs(x[1]))
            ax1.scatter(
                [pl],
                [pv],
                color=color,
                s=70,
                zorder=6,
                edgecolors="white",
                linewidths=1.2,
            )
            va = "bottom" if pv > 0 else "top"
            ax1.annotate(
                f"lag={pl}, r={pv:.2f}",
                xy=(pl, pv),
                xytext=(pl + 0.8, pv + (0.05 if pv > 0 else -0.05)),
                fontsize=7.5,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
            )
        predictive = [(l, c) for l, c in valid if l < 0]
        if predictive:
            pl, pv = max(predictive, key=lambda x: x[1])  # max positive r
            ax1.annotate(
                f"lag={pl}, r={pv:.2f}",
                xy=(pl, pv),
                xytext=(pl + 0.8, pv + (0.05 if pv > 0 else -0.05)),
                fontsize=7.5,
                color=color,
                arrowprops=dict(arrowstyle="->", color=color, lw=0.9),
            )

    ax1.set_xlabel("Lag (meetings, negative = sentiment leads)", fontsize=10)
    ax1.set_ylabel("Pearson Correlation", fontsize=10)
    ax1.set_title(
        "(A) Lead–Lag Correlation:\nSentiment vs. ΔMRO (changed meetings only)",
        fontsize=10,
        pad=6,
    )
    ax1.legend(fontsize=9, framealpha=0.92, loc="upper right")
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_ylim(-0.70, 0.80)

    # ── Panel B: Grouped boxplots ─────────────────────────────────────────────
    ax2 = axes[1]

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
        (fb_pos, "finbert_mean", COLORS["finbert"], "", "FinBERT"),
        (
            rb_pos,
            rb_col,
            COLORS["roberta"],
            "//",
            "RoBERTa IS" if rb_col == "roberta_IS_mean" else "RoBERTa",
        ),
    ]:
        data_list = [df[df["regime"] == r][feat].dropna() for r, _ in regime_specs]
        bp = ax2.boxplot(
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
            ax2.scatter(
                [pos],
                [mu],
                marker="D",
                s=45,
                color="white",
                edgecolors="black",
                linewidths=1.3,
                zorder=6,
            )
            ax2.text(
                pos, mu + 0.018, f"{mu:.3f}", ha="center", fontsize=7.5, color="#111"
            )

        # Invisible proxy for legend
        # Invisible proxy for legend
        from matplotlib.patches import Rectangle

        ax2.add_patch(
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

    ax2.set_xticks(centers)
    ax2.set_xticklabels([lbl for _, lbl in regime_specs], fontsize=9)
    ax2.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax2.set_ylabel("Sentiment Score", fontsize=10)
    ax2.set_title(
        "(B) Sentiment Distribution by MRO Regime\n(◆ = mean; solid=FinBERT, hatch=RoBERTa IS)",
        fontsize=10,
        pad=6,
    )
    ax2.legend(fontsize=8.5, loc="upper right", framealpha=0.92)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)

    # n labels below each regime group
    for i, (r, _) in enumerate(regime_specs):
        n = len(df[df["regime"] == r])
        ax2.text(
            centers[i],
            ax2.get_ylim()[0] + 0.01,
            f"n={n}",
            ha="center",
            fontsize=7.5,
            color="#666",
            transform=ax2.transData,
        )

    fig.suptitle(
        "Asymmetry of Sentiment across the Monetary Policy Cycle\n"
        "FinBERT (solid) vs. CentralBankRoBERTa IS (hatched)",
        fontsize=11,
        y=1.01,
    )

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
