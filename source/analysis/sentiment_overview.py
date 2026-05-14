"""
Figure 3.1  –  Aggregate Sentiment Trends and Model Comparison
Figure 3.2  –  Sanity Check: Sentiment validated against historical ECB events

Usage (run from the project root):
    python -m thesis.figures.fig_3_1_sentiment_overview
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import matplotlib.patches as mpatches
import pandas as pd
import numpy as np
from pathlib import Path

# ── adjust this import to match your package structure ──────────────────────
from ..data.model_data import return_data
from .. import OUTPUT
# ────────────────────────────────────────────────────────────────────────────

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Matplotlib style ─────────────────────────────────────────────────────────
plt.rcParams.update({
    "font.family":      "serif",
    "font.size":        10,
    "axes.titlesize":   11,
    "axes.labelsize":   10,
    "legend.fontsize":  9,
    "xtick.labelsize":  9,
    "ytick.labelsize":  9,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

COLORS = {
    "finbert":  "#2c6fad",   # blue
    "roberta":  "#c0392b",   # red
    "finbert_roll": "#1a4a7a",
    "roberta_roll": "#8b1a10",
    "zero":     "#888888",
    "shade":    0.12,        # alpha for std band
}

ROLLING_WINDOW = "180D"   # 6-month time-based window — consistent across all meeting frequencies

# ── Historical events for Figure 3.2 ────────────────────────────────────────
EVENTS = [
    ("2008-10-08", "GFC\nrate cut",       "below"),
    ("2012-07-26", '"Whatever\nit takes"', "above"),
    ("2015-01-22", "QE\nlaunched",         "below"),
    ("2019-09-12", "Rates\nneg. again",    "above"),
    ("2020-03-12", "COVID\nemergency",     "below"),
    ("2022-07-21", "First\nhike (+50bp)",  "above"),
    ("2023-09-14", "Peak rate\n(4.50%)",   "below"),
]


# ── Data loading ─────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    df = return_data(
        market_data="ECB Money Market.xlsx",   # contains MRO — not needed for 3.1 but kept for later reuse
        IS_QA_division=False,
        qa_options="both_together",
        with_label=False,
    )
    df = df.sort_values("date").reset_index(drop=True)

    # rolling mean — time-based so it stays consistent across meeting-frequency changes
    # (ECB: ~12/year before 2015, ~8/year from 2015 onward)
    df = df.set_index("date")
    for model in ("finbert", "roberta"):
        # forward-looking pass
        fwd = df[f"{model}_mean"].rolling(ROLLING_WINDOW, min_periods=1).mean()
        # backward-looking pass (reversed)
        bwd = df[f"{model}_mean"][::-1].rolling(ROLLING_WINDOW, min_periods=1).mean()[::-1]
        # centered = average of the two
        df[f"{model}_roll"] = (fwd + bwd) / 2
    df = df.reset_index()
    return df


# ── Core plot helper ─────────────────────────────────────────────────────────
def _draw_sentiment_panel(ax: plt.Axes, df: pd.DataFrame, show_roll: bool = True):
    """Draw FinBERT and RoBERTa sentiment onto *ax*."""
    ax.axhline(0, color=COLORS["zero"], linewidth=0.8, linestyle="--", zorder=1)

    for model, label in [("finbert", "FinBERT"), ("roberta", "CentralBankRoBERTa")]:
        col_m  = f"{model}_mean"
        col_s  = f"{model}_std"
        col_r  = f"{model}_roll"
        c      = COLORS[model]

        # ±1 std band
        ax.fill_between(
            df["date"],
            df[col_m] - df[col_s],
            df[col_m] + df[col_s],
            color=c, alpha=COLORS["shade"], linewidth=0, zorder=2,
        )

        # raw mean (thin, faded)
        ax.plot(
            df["date"], df[col_m],
            color=c, alpha=0.35, linewidth=0.8, zorder=3,
        )

        # rolling mean (bold)
        if show_roll:
            ax.plot(
                df["date"], df[col_r],
                color=COLORS[f"{model}_roll"], linewidth=1.8,
                label=f"{label} (6-month centered MA)", zorder=4,
            )
        else:
            ax.plot(
                df["date"], df[col_m],
                color=c, linewidth=1.6, label=label, zorder=4,
            )

    ax.set_ylabel("Net Sentiment Score")
    ax.set_ylim(-1.05, 1.05)
    ax.yaxis.set_major_locator(plt.MultipleLocator(0.25))
    ax.xaxis.set_major_locator(mdates.YearLocator(4))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    ax.set_xlim(df["date"].min(), df["date"].max())


# ── Figure 3.1 ────────────────────────────────────────────────────────────────
def plot_fig_3_1(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 3.8))

    _draw_sentiment_panel(ax, df, show_roll=True)

    ax.set_title(
        "Aggregate ECB Press Conference Sentiment (1999–2024)\n"
        "FinBERT vs. CentralBankRoBERTa — full transcript, 6-month centered average",
        pad=8,
    )
    ax.legend(loc="upper left", framealpha=0.85)

    # Shade the std-band entries manually in legend
    fb_patch = mpatches.Patch(color=COLORS["finbert"], alpha=0.25, label="FinBERT ±1 SD")
    rb_patch = mpatches.Patch(color=COLORS["roberta"], alpha=0.25, label="RoBERTa ±1 SD")
    handles, labels = ax.get_legend_handles_labels()
    ax.legend(handles + [fb_patch, rb_patch],
              labels  + ["FinBERT ±1 SD", "RoBERTa ±1 SD"],
              loc="upper left", framealpha=0.85, ncol=2)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig_3_1_aggregate_sentiment.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig, ax


# ── Figure 3.2 ────────────────────────────────────────────────────────────────
def plot_fig_3_2(df: pd.DataFrame, save: bool = True):
    fig, ax = plt.subplots(figsize=(10, 4.2))

    _draw_sentiment_panel(ax, df, show_roll=True)

    ax.set_title(
        "ECB Sentiment and Key Historical Policy Events (1999–2024)",
        pad=8,
    )

    # ── annotate events ──────────────────────────────────────────────────────
    y_above =  0.78
    y_below = -0.78
    for date_str, label, position in EVENTS:
        xd = pd.Timestamp(date_str)
        y  = y_above if position == "above" else y_below
        va = "bottom"  if position == "above" else "top"

        ax.axvline(xd, color="#555555", linewidth=0.9, linestyle=":", alpha=0.7, zorder=2)
        ax.annotate(
            label,
            xy=(xd, 0),
            xytext=(xd, y),
            ha="center", va=va,
            fontsize=7.5,
            color="#333333",
            arrowprops=dict(arrowstyle="-", color="#999999", lw=0.7),
        )

    handles, labels_l = ax.get_legend_handles_labels()
    fb_patch = mpatches.Patch(color=COLORS["finbert"], alpha=0.25, label="FinBERT ±1 SD")
    rb_patch = mpatches.Patch(color=COLORS["roberta"], alpha=0.25, label="RoBERTa ±1 SD")
    ax.legend(handles + [fb_patch, rb_patch],
              labels_l + ["FinBERT ±1 SD", "RoBERTa ±1 SD"],
              loc="upper left", framealpha=0.85, ncol=2)

    fig.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig_3_2_sentiment_events.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig, ax


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()

    print("Columns:", [c for c in df.columns if "finbert" in c or "roberta" in c])
    print(f"Date range: {df['date'].min().date()} → {df['date'].max().date()}")
    print(f"N meetings: {len(df)}")

    plot_fig_3_1(df, False)
    plot_fig_3_2(df, False)
    plt.show()