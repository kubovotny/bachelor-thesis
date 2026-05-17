"""
Figure 3.8 – Intra-Conference Sentiment Dynamics

Shows how sentiment evolves across the course of a press conference,
using chunk_percentile = chunk_id / max(chunk_id) as position proxy.

Two views:
  3.8a – Separately for IS and QA (within-section position)
  3.8b – Combined: IS first (0–0.5), QA second (0.5–1.0)

Key question: Does the ECB start hawkish and soften? Or vice versa?
Does the Q&A become more adversarial toward the end?
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from pathlib import Path

from ..data.connection import connect_to_db
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/is_qa"
OUTPUT_DIR.mkdir(exist_ok=True)

MODEL = "roberta"


# ── Load chunks with percentile ───────────────────────────────────────────────
def load_chunks_with_percentile(model: str = MODEL) -> pd.DataFrame:
    """
    Load all chunks with:
      - percentile_within: position within IS or QA separately (chunk_id / max per date+part)
      - percentile_global: position across whole conference (IS first, QA second)
    """
    conn, _ = connect_to_db()
    sql = """
    SELECT
        DATE(st.date)  AS date,
        ch.part        AS part,
        ch.chunk_id    AS chunk_id,
        ch.is_question AS is_question,
        se.score       AS score,
        ch.chunk_id * 1.0 / MAX(ch.chunk_id) OVER (
            PARTITION BY st.date, ch.part
        ) AS percentile_within
    FROM sentiments se
    JOIN chunks      ch ON ch.rowid      = se.chunk_rowid
    JOIN statements  st ON st.rowid      = ch.statement_id
    JOIN sentiment_models sm ON sm.rowid = se.model_id
    WHERE ch.chunk_limit = 150
      AND sm.name        = ?
    ORDER BY st.date, ch.part, ch.chunk_id;
    """
    df = pd.read_sql(sql, conn, params=(model,), parse_dates=["date"])
    conn.close()

    df["is_question"] = df["is_question"].astype(bool)
    df["part_label"]  = df["part"].map({0: "IS", 1: "QA"})

    # Global percentile: IS maps to [0, 0.5), QA maps to [0.5, 1.0]
    df["percentile_global"] = df.apply(
        lambda r: r["percentile_within"] * 0.5
                  if r["part"] == 0
                  else 0.5 + r["percentile_within"] * 0.5,
        axis=1,
    )
    return df


# ── Binned profile helper ─────────────────────────────────────────────────────
def binned_profile(x: np.ndarray, y: np.ndarray,
                   n_bins: int = 20) -> tuple:
    """Return (bin_centers, mean_y, se_y) using quantile bins."""
    edges = np.linspace(0, 1, n_bins + 1)
    centers, means, ses = [], [], []
    for i in range(n_bins):
        mask = (x >= edges[i]) & (x <= edges[i + 1])
        if mask.sum() > 5:
            centers.append((edges[i] + edges[i + 1]) / 2)
            means.append(y[mask].mean())
            ses.append(y[mask].std() / np.sqrt(mask.sum()))
    return np.array(centers), np.array(means), np.array(ses)


# ── Figure 3.8a – Both models on same axes, 1×2 ──────────────────────────────
def plot_fig_3_8a(save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(13, 5.5),
                             gridspec_kw={"wspace": 0.28})

    model_styles = [
        ("finbert", "FinBERT",            "#2c6fad", "#c0392b", "#e67e22", "-",  2.2),
        ("roberta", "CentralBankRoBERTa", "#1a4a7a", "#8b1a10", "#c0650a", "--", 1.8),
    ]

    # ── IS panel ─────────────────────────────────────────────────────────────
    ax = axes[0]
    for model, label, c_is, _, _, ls, lw in model_styles:
        df   = load_chunks_with_percentile(model)
        is_df = df[df["part"] == 0]
        cx, my, se = binned_profile(
            is_df["percentile_within"].values, is_df["score"].values, n_bins=20)
        ax.fill_between(cx, my - se, my + se, alpha=0.18, color=c_is)
        ax.plot(cx, my, color=c_is, linewidth=lw, linestyle=ls,
                label=f"{label}  (μ={my.mean():.3f})")

    ax.axhline(0, color="#ddd", linewidth=0.7)
    ax.set_xlabel("Relative position within IS (0 = start, 1 = end)", fontsize=9.5)
    ax.set_ylabel("Sentiment", fontsize=10)
    ax.set_title("Introductory Statement\nSentiment Profile", fontsize=10, pad=6)
    ax.legend(fontsize=9, framealpha=0.92)
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.set_xlim(-0.02, 1.02)

    # ── QA panel ─────────────────────────────────────────────────────────────
    ax = axes[1]
    for model, label, _, c_ans, c_q, ls, lw in model_styles:
        df    = load_chunks_with_percentile(model)
        qa_df = df[df["part"] == 1]
        for mask, role, color in [
            (~qa_df["is_question"], "Answers", c_ans),
            (qa_df["is_question"],  "Questions", c_q),
        ]:
            sub = qa_df[mask]
            cx, my, se = binned_profile(
                sub["percentile_within"].values, sub["score"].values, n_bins=20)
            ax.fill_between(cx, my - se, my + se, alpha=0.15, color=color)
            ax.plot(cx, my, color=color, linewidth=lw, linestyle=ls,
                    label=f"{label}  [{role}]  (μ={my.mean():.3f})")

    ax.axhline(0, color="#ddd", linewidth=0.7)
    ax.set_xlabel("Relative position within Q&A (0 = start, 1 = end)", fontsize=9.5)
    ax.set_ylabel("Sentiment", fontsize=10)
    ax.set_title("Q&A Session\nSentiment Profile", fontsize=10, pad=6)
    ax.legend(fontsize=8.5, framealpha=0.92, ncol=1, loc="upper left")
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_ylim(-0.75,0.4)

    fig.suptitle(
        "Intra-Section Sentiment Dynamics\n"
        "FinBERT (solid) vs. CentralBankRoBERTa (dashed)  |  ±SE shaded",
        fontsize=11, y=1.01,
    )
    if save:
        path = OUTPUT_DIR / "fig_intra_section_both.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.8b – Both models, single panel ───────────────────────────────────
def plot_fig_3_8b(save: bool = True):
    fig, ax = plt.subplots(figsize=(13, 5.5))

    model_styles = [
        ("finbert", "FinBERT",            "#2c6fad", "#c0392b", "#e67e22", "-",  2.2),
        ("roberta", "CentralBankRoBERTa", "#1a4a7a", "#8b1a10", "#c0650a", "--", 1.8),
    ]

    for model, label, c_is, c_ans, c_q, ls, lw in model_styles:
        df = load_chunks_with_percentile(model)

        for part_val, mask_fn, role, color in [
            (0, lambda d: d,                     "IS",       c_is),
            (1, lambda d: d[~d["is_question"]], "answers",  c_ans),
            (1, lambda d: d[d["is_question"]],  "questions", c_q),
        ]:
            sub = mask_fn(df[df["part"] == part_val])
            cx, my, se = binned_profile(
                sub["percentile_global"].values, sub["score"].values, n_bins=12)
            ax.fill_between(cx, my - se, my + se, alpha=0.12, color=color)
            ax.plot(cx, my, color=color, linewidth=lw, linestyle=ls,
                    label=f"{label}  [{role}]")

    ax.axvline(0.5, color="#555", linewidth=1.2, linestyle=":", alpha=0.6)
    ax.axhline(0, color="#ddd", linewidth=0.7)

    # Section labels
    ax.annotate("", xy=(0.02, 1.04), xytext=(0.48, 1.04), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="#2c6fad", lw=1.2))
    ax.text(0.25, 1.055, "Introductory Statement", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=9, color="#2c6fad")
    ax.annotate("", xy=(0.52, 1.04), xytext=(0.98, 1.04), xycoords="axes fraction",
                arrowprops=dict(arrowstyle="<->", color="#c0392b", lw=1.2))
    ax.text(0.75, 1.055, "Q&A Session", ha="center", va="bottom",
            transform=ax.transAxes, fontsize=9, color="#c0392b")

    ax.set_xlabel(
        "Relative position  (0 = IS start,  0.5 = IS end / QA start,  1 = QA end)",
        fontsize=9.5)
    ax.set_ylabel("Sentiment", fontsize=10)
    ax.legend(fontsize=8.5, framealpha=0.92, ncol=2, loc="upper right")
    ax.grid(alpha=0.20, linewidth=0.6)
    ax.set_xlim(-0.02, 1.02)
    ax.set_title(
        "Full Press Conference Sentiment Profile\n"
        "FinBERT (solid) vs. CentralBankRoBERTa (dashed)",
        fontsize=10, pad=14)

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig_global_profile_both.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig

# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_fig_3_8a()
    plot_fig_3_8b()
    plt.show()