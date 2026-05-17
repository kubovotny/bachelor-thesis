"""
Appendix Figure – Sensitivity of Sentiment Time Series to Chunk Size

Shows that while the aggregate trend is preserved across all word_limit values,
shorter chunks introduce more high-frequency noise (volatility) and longer chunks
may miss localized sentiment shifts within a meeting.
"""

import matplotlib.pyplot as plt
import pandas as pd
import numpy as np
from pathlib import Path

from ..data.model_data import return_data
from ..data.queries import CHUNK_LIMITS
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "appendix"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Load all limits for one model ─────────────────────────────────────────────
def load_all_limits(model_col: str = "roberta_mean"):
    """Load sentiment time series for all chunk sizes."""
    data = {}
    for limit in CHUNK_LIMITS:
        df = return_data(
            market_data="shocks_ecb_mpd_me_d.csv",
            word_limit=limit,
            IS_QA_division=False,
            qa_options="both_together",
            with_label=False,
        )
        df = df.sort_values("date").reset_index(drop=True)
        data[limit] = df[["date", model_col]].copy()
        data[limit].rename(columns={model_col: f"limit_{limit}"}, inplace=True)
    
    # Merge all on date
    merged = data[1].copy()
    for limit in CHUNK_LIMITS[1:]:
        merged = pd.merge(merged, data[limit], on="date", how="outer")
    
    return merged.sort_values("date").reset_index(drop=True)


# ── Plot ──────────────────────────────────────────────────────────────────────
def plot_limit_comparison(save: bool = True):
    df = load_all_limits("roberta_mean")
    
    # Select a few representative limits to avoid clutter
    selected = [1, 50, 200, 350]
    
    fig, axes = plt.subplots(2, 1, figsize=(12, 6), sharex=True)
    
    # Panel A: All selected limits overlaid
    ax1 = axes[0]
    colors = {"1": "#e74c3c", "50": "#f39c12", "200": "#27ae60", "350": "#3498db"}
    alphas = {"1": 0.4, "50": 0.6, "200": 1.0, "350": 0.8}
    lws    = {"1": 0.9, "50": 1.2, "200": 2.0, "350": 1.5}
    
    for limit in selected:
        col = f"limit_{limit}"
        ax1.plot(
            df["date"], df[col],
            label=f"{limit} words",
            color=colors[str(limit)],
            alpha=alphas[str(limit)],
            linewidth=lws[str(limit)],
        )
    
    ax1.axhline(0, color="#888", linewidth=0.8, linestyle="--", zorder=1)
    ax1.set_ylabel("RoBERTa mean sentiment", fontsize=11)
    ax1.set_title("(A) Sentiment Time Series across Chunk Sizes", fontsize=11, pad=6)
    ax1.legend(loc="lower left", fontsize=9.5, ncol=4, framealpha=0.9)
    ax1.grid(alpha=0.25, linewidth=0.6)
    ax1.set_ylim(-0.7, 0.4)
    
    # Panel B: Difference from 200-word baseline (to highlight divergence)
    ax2 = axes[1]
    baseline = df["limit_200"]
    for limit in [1, 50, 350]:
        diff = df[f"limit_{limit}"] - baseline
        ax2.plot(
            df["date"], diff,
            label=f"{limit} − 200 words",
            color=colors[str(limit)],
            alpha=0.7,
            linewidth=1.3,
        )
    
    ax2.axhline(0, color="#888", linewidth=1.0, linestyle="-", zorder=1,
                label="200-word baseline")
    ax2.set_xlabel("Date", fontsize=11)
    ax2.set_ylabel("Δ Sentiment (vs. 200 words)", fontsize=11)
    ax2.set_title("(B) Deviation from 200-word Baseline", fontsize=11, pad=6)
    ax2.legend(loc="lower left", fontsize=9.5, ncol=4, framealpha=0.9)
    ax2.grid(alpha=0.25, linewidth=0.6)
    ax2.set_ylim(-0.25, 0.4)
    
    # Format x-axis
    import matplotlib.dates as mdates
    ax2.xaxis.set_major_locator(mdates.YearLocator(4))
    ax2.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax2.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))
    
    fig.suptitle(
        "Sensitivity of Sentiment Extraction to Chunk Size (RoBERTa)",
        fontsize=12, y=0.995,
    )
    plt.tight_layout()
    
    if save:
        path = OUTPUT_DIR / "fig_appendix_chunk_sensitivity.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    
    return fig


# ── Statistics ────────────────────────────────────────────────────────────────
def compute_divergence_stats():
    """Quantify how much time series diverge across limits."""
    df = load_all_limits("roberta_mean")
    baseline = df["limit_200"]
    
    print("\n=== Divergence from 200-word baseline ===")
    for limit in [1, 50, 200, 350]:
        diff = df[f"limit_{limit}"] - baseline
        mae  = diff.abs().mean()
        rmse = np.sqrt((diff**2).mean())
        maxd = diff.abs().max()
        print(f"  {limit:3d} words: MAE={mae:.4f}, RMSE={rmse:.4f}, Max={maxd:.4f}")


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_limit_comparison()
    compute_divergence_stats()
    plt.show()