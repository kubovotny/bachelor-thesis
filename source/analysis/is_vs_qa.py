"""
Figure 3.4 – Introductory Statement vs. Q&A Sentiment Dynamics

Compares the prepared Introductory Statement (IS) with the spontaneous
Question & Answer (QA) session. Shows that:
- IS is smoother (lower volatility) — carefully drafted consensus
- QA is more reactive (higher volatility) — responds to journalist pressure
- Both correlate but QA can diverge during uncertain periods
"""

import matplotlib.pyplot as plt
import matplotlib.dates as mdates
import pandas as pd
from pathlib import Path
from scipy.stats import pearsonr

from ..data.sentiment import return_sentiment_agg_pivot
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/is_qa"
OUTPUT_DIR.mkdir(exist_ok=True)
plt.rcParams.update({
    "font.size":        13,
    "axes.titlesize":   14,
    "axes.labelsize":   13,
    "legend.fontsize":  11,
    "xtick.labelsize":  11,
    "ytick.labelsize":  11,
    "axes.spines.top":  False,
    "axes.spines.right":False,
    "figure.dpi":       150,
})

# ── Load data ─────────────────────────────────────────────────────────────────
def load_is_qa_data():
    """Load sentiment with IS/QA split."""
    df = return_sentiment_agg_pivot(
        word_limit=200,
        IS_QA_division=True,
        qa_options="just_answers",  # only answers, not questions
        with_label=False,
    )
    df = df.sort_values("date").reset_index(drop=True)
    return df


# ── Descriptive statistics ────────────────────────────────────────────────────
def compute_descriptives(df):
    """Generate descriptive stats table for IS vs QA."""
    stats = []
    for model in ["finbert", "roberta"]:
        for section in ["IS", "QA"]:
            col = f"{model}_{section}_mean"
            count_col = f"{model}_{section}_count"
            if col not in df.columns:
                continue
            s = df[col]
            stats.append(
                {
                    "Model": model.upper(),
                    "Section": section,
                    "Mean": s.mean(),
                    "Std": s.std(),
                    "Min": s.min(),
                    "Max": s.max(),
                    "IQR": s.quantile(0.75) - s.quantile(0.25),
                    "Count": s.dropna().count(),
                }
            )

    stats_df = pd.DataFrame(stats)
    return stats_df


def compute_correlations(df):
    """IS vs QA correlation per model."""
    corrs = []
    for model in ["finbert", "roberta"]:
        is_col = f"{model}_IS_mean"
        qa_col = f"{model}_QA_mean"
        if is_col in df.columns and qa_col in df.columns:
            r, p = pearsonr(df[is_col].dropna(), df[qa_col].dropna())
            corrs.append(
                {
                    "Model": model.upper(),
                    "IS-QA Correlation": r,
                    "p-value": p,
                }
            )
    return pd.DataFrame(corrs)


# ── Main figure ───────────────────────────────────────────────────────────────
def plot_fig_3_timeseries(save: bool = True):
    df = load_is_qa_data()

    # 1. Figure: Time Series (FinBERT a RoBERTa vedľa seba)
    # figsize (14, 5) je vhodná pre jeden riadok grafov
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))

    for col_idx, model in enumerate(["finbert", "roberta"]):
        ax = axes[col_idx]

        is_col = f"{model}_IS_mean"
        qa_col = f"{model}_QA_mean"

        # Plot IS (solid) and QA (dashed) - presne podľa pôvodných parametrov
        ax.plot(
            df["date"],
            df[is_col],
            color="#2c6fad",
            linewidth=1.6,
            label="Introductory Statement",
            alpha=0.9,
        )
        ax.plot(
            df["date"],
            df[qa_col],
            color="#c0392b",
            linewidth=1.3,
            linestyle="--",
            label="Q&A",
            alpha=0.8,
        )

        ax.axhline(0, color="#888", linewidth=0.8, linestyle=":", zorder=1)
        ax.set_ylabel("Net Sentiment Score")
        ax.set_title(f"{model.upper()} — IS vs. Q&A", pad=6)

        # Legend umiestnenie podľa modelu
        ax.legend(loc="upper left", framealpha=0.9)
        ax.grid(alpha=0.25, linewidth=0.6)

        # Model-specific y-limits
        if model == "finbert":
            ax.set_ylim(-0.7, 0.8)
        else:
            ax.set_ylim(-0.7, 0.8)

        # Formátovanie osi X
        ax.xaxis.set_major_locator(mdates.YearLocator(5))
        ax.xaxis.set_minor_locator(mdates.YearLocator(1))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
        ax.set_xlabel("Date")

    fig.suptitle(
        "Prepared vs. Spontaneous Communication: Sentiment Dynamics (Time Series)",
        y=1.02,
    )
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_IS_vs_QA.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


def plot_fig_4_scatter(save: bool = True):
    df = load_is_qa_data().dropna()

    # 2. Figure: Scatter Plots (FinBERT a RoBERTa vedľa seba)
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))

    for col_idx, model in enumerate(["finbert", "roberta"]):
        ax = axes[col_idx]

        is_col = f"{model}_IS_mean"
        qa_col = f"{model}_QA_mean"

        # Scatter s farebnou škálou podľa času
        scatter = ax.scatter(
            df[is_col],
            df[qa_col],
            c=mdates.date2num(df["date"]),
            s=28,
            alpha=0.65,
            cmap="viridis",
            edgecolors="white",
            linewidths=0.3,
        )

        # 45-stupňová čiara
        lims = [-0.4, 0.75] if model == "finbert" else [-0.7, 0.35]
        ax.plot(
            lims,
            lims,
            "k--",
            alpha=0.4,
            linewidth=1.2,
            zorder=1,
            label="Perfect agreement",
        )

        # Výpočet korelácie
        r, p = pearsonr(df[is_col].dropna(), df[qa_col].dropna())
        ax.text(
            0.05,
            0.95,
            f"r = {r:.3f}\np < 0.001" if p < 0.001 else f"r = {r:.3f}\np = {p:.3f}",
            transform=ax.transAxes,
            va="top",
            bbox=dict(
                boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9
            ),
        )

        ax.set_xlabel("IS Sentiment")
        ax.set_ylabel("Q&A Sentiment")
        ax.set_title(f"{model.upper()} Scatter", pad=4)
        ax.grid(alpha=0.25, linewidth=0.6)
        ax.legend(loc="lower right", framealpha=0.9)
        ax.set_aspect("equal")

        # Colorbar len pri druhom grafe (vpravo)
        if col_idx == 1:
            cbar = plt.colorbar(scatter, ax=ax, pad=0.02, aspect=20)
            cbar.set_label("Date")
            cbar.ax.tick_params(labelsize=9)
            cbar_ticks = cbar.get_ticks()
            cbar.set_ticks(cbar_ticks[::2])
            cbar.set_ticklabels([mdates.num2date(t).year for t in cbar_ticks[::2]])

    fig.suptitle(
        "Prepared vs. Spontaneous Communication: Sentiment Dynamics (Correlation)",
        y=1.02,
    )
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_Scatter_IS_vs_QA.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Export tables ─────────────────────────────────────────────────────────────
def export_tables(save: bool = True):
    df = load_is_qa_data()

    desc = compute_descriptives(df)
    corr = compute_correlations(df.dropna())

    print("\n=== Descriptive Statistics: IS vs QA ===")
    print(desc.to_string(index=False, float_format="%.4f"))

    print("\n=== IS–QA Correlation ===")
    print(corr.to_string(index=False, float_format="%.4f"))

    if save:
        desc.to_csv(OUTPUT_DIR / "table_descriptives_IS_QA.csv", index=False)
        corr.to_csv(OUTPUT_DIR / "table_correlation_IS_QA.csv", index=False)
        print("\nSaved tables to CSV")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_fig_3_timeseries()
    plot_fig_4_scatter()
    export_tables(False)
    plt.show()
