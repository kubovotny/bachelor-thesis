"""
Figure 3.5 – Thematic Sentiment Decomposition

Shows how RoBERTa sentiment varies across the four macroeconomic topics:
  - MP: Monetary Policy and Inflation
  - EP: Economic Performance
  - FS: Fiscal and Structural
  - OI: Other / Irrelevant

Reveals that the ECB is systematically more hawkish when discussing MP (inflation)
and more dovish when discussing EP (growth, unemployment).
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path

from ..data.sentiment import return_sentiment_agg_pivot
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/topic"
OUTPUT_DIR.mkdir(exist_ok=True)

# Topic labels mapping (short codes used in column names)
TOPICS = {
    "MP": "Monetary Policy & Inflation",
    "EP": "Economic Performance",
    "FS": "Fiscal & Structural",
    "OI": "Other / Irrelevant",
}

TOPIC_COLORS = {
    "MP": "#e74c3c",  # red - inflation/tightening
    "EP": "#3498db",  # blue - growth/employment
    "FS": "#95a5a6",  # grey - structural/fiscal
    "OI": "#ecf0f1",  # light grey - noise
}

# ── Load data ─────────────────────────────────────────────────────────────────
def load_thematic_data():
    """Load RoBERTa sentiment split by topic."""
    df = return_sentiment_agg_pivot(
        word_limit=200,
        IS_QA_division=False,  # whole conference
        qa_options="just_answers",
        with_label=True,  # ← topic labels
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Descriptive stats per topic ──────────────────────────────────────────────
def compute_topic_stats(df):
    """Mean, std, min, max per topic."""
    stats = []
    for code, label in TOPICS.items():
        col = f"roberta_{code}_mean"
        if col not in df.columns:
            continue
        s = df[col].dropna()
        stats.append({
            "Topic": label,
            "Code": code,
            "Mean": s.mean(),
            "Std": s.std(),
            "Min": s.min(),
            "Max": s.max(),
        })
    return pd.DataFrame(stats)


# ── Figure 3.5a: Time series per topic ───────────────────────────────────────
def plot_fig_3_5a(df=None, save: bool = True):
    if df is None:
        df = load_thematic_data()

    fig, ax = plt.subplots(figsize=(13, 4.2))

    for code, label in TOPICS.items():
        col = f"roberta_{code}_mean"
        if col not in df.columns or code == "OI":
            continue
        ax.plot(df["date"], df[col],
                label=label, color=TOPIC_COLORS[code],
                linewidth=1.6, alpha=0.85)

    ax.axhline(0, color="#888", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_ylabel("RoBERTa Sentiment", fontsize=11)
    ax.set_title("Thematic Sentiment over Time — RoBERTa", fontsize=12, pad=8)
    ax.legend(loc="upper left", fontsize=10, ncol=3, framealpha=0.92)
    ax.grid(alpha=0.25, linewidth=0.6)
    ax.set_ylim(-1, 1)

    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig_timeseries.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.5b: Box plots + mean table side by side ─────────────────────────
def plot_fig_3_5b(df=None, save: bool = True):
    if df is None:
        df = load_thematic_data()

    fig, ax = plt.subplots(figsize=(7, 4))

    # ── Left: box plots ───────────────────────────────────────────────────────
    data_for_box, labels_for_box, colors_for_box = [], [], []
    for code, label in TOPICS.items():
        col = f"roberta_{code}_mean"
        if col not in df.columns or code == "OI":
            continue
        data_for_box.append(df[col].dropna())
        labels_for_box.append(label)
        colors_for_box.append(TOPIC_COLORS[code])

    bp = ax.boxplot(data_for_box, tick_labels=labels_for_box,
                     patch_artist=True, widths=0.55,
                     medianprops=dict(color="black", linewidth=1.5),
                     boxprops=dict(linewidth=1.2),
                     whiskerprops=dict(linewidth=1.2),
                     capprops=dict(linewidth=1.2))
    for patch, color in zip(bp["boxes"], colors_for_box):
        patch.set_facecolor(color)
        patch.set_alpha(0.7)

    for i, data in enumerate(data_for_box):
        ax.scatter([i+1], [data.mean()], marker="D", s=55,
                    color="white", edgecolors="black", linewidths=1.5, zorder=5)

    ax.axhline(0, color="#888", linewidth=0.9, linestyle="--", zorder=1)
    ax.set_ylabel("RoBERTa Sentiment", fontsize=11)
    ax.set_title("Distribution by Topic (◆ = mean)", fontsize=11, pad=6)
    ax.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax.set_ylim(-1.05, 0.95)
    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig_boxplot.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Export table ──────────────────────────────────────────────────────────────
def export_table(save: bool = True):
    df = load_thematic_data()
    stats = compute_topic_stats(df)
    
    print("\n=== Sentiment Statistics by Topic (RoBERTa) ===")
    print(stats.to_string(index=False, float_format="%.4f"))
    
    if save:
        stats.to_csv(OUTPUT_DIR / "table_sentiment.csv", index=False)
        print("\nSaved table to CSV")
    
    return stats


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_thematic_data()
    plot_fig_3_5a(df)
    plot_fig_3_5b(df)
    export_table()
    plt.show()