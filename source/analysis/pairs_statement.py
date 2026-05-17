"""
Q→A Pair Matching from Database

Analogous to chunker.py::q_to_a_merged(), but reads from the SQLite database
instead of CSV files. Pairs each journalist question with the ECB answer(s)
that follow it, then computes sentiment gap per pair.

The pairing logic:
  - Within each statement, chunks are ordered by chunk_id
  - is_question=True  → journalist question block
  - is_question=False → ECB answer block(s) that follow
  - A new Q-A pair starts at each is_question=True chunk
  - We take the mean sentiment of all answer chunks in the block

Output: DataFrame with one row per Q-A pair, containing:
  - date, statement_id, pair_id
  - q_score (mean sentiment of question chunks)
  - a_score (mean sentiment of answer chunks)
  - gap = |q_score - a_score|
"""

import pandas as pd
import numpy as np
from pathlib import Path
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/is_qa"
OUTPUT_DIR.mkdir(exist_ok=True)


# ── Load chunk-level QA sentiment from DB ─────────────────────────────────────
def load_qa_chunks(model: str = "roberta") -> pd.DataFrame:
    """
    Load QA chunks in correct chronological order (date → chunk_id).
    Uses custom SQL to include chunk_id for proper pairing.
    """
    from ..data.connection import connect_to_db

    conn, _ = connect_to_db()
    sql = """
    SELECT
        DATE(st.date)   AS date,
        ch.chunk_id     AS chunk_id,
        ch.is_question  AS is_question,
        ch.chunk        AS chunk,
        se.score        AS score
    FROM sentiments se
    JOIN chunks      ch ON ch.rowid      = se.chunk_rowid
    JOIN statements  st ON st.rowid      = ch.statement_id
    JOIN sentiment_models sm ON sm.rowid = se.model_id
    WHERE ch.chunk_limit = 200
      AND ch.part        = 1
      AND sm.name        = ?
    ORDER BY st.date, ch.chunk_id;
    """
    df = pd.read_sql(sql, conn, params=(model,), parse_dates=["date"])
    conn.close()

    df["is_question"] = df["is_question"].astype(bool)
    print(f"Loaded {len(df)} QA chunks for model={model}")
    print(
        f"  Questions: {df['is_question'].sum()}, Answers: {(~df['is_question']).sum()}"
    )
    print(df.date.nunique())
    return df


# ── Pair Q → A within each statement ─────────────────────────────────────────
def build_qa_pairs(df: pd.DataFrame) -> pd.DataFrame:
    """
    For each statement (date), walk through QA chunks in chunk_id order
    and pair consecutive question blocks with the answer blocks that follow.
    """
    pairs = []

    for date, stmt_df in df.groupby("date"):
        stmt_df = stmt_df.sort_values("chunk_id").reset_index(drop=True)

        pair_id = 0
        q_scores = []
        a_scores = []

        for _, row in stmt_df.iterrows():
            is_q = bool(row["is_question"])

            if is_q:
                # New question — flush previous pair if complete
                if q_scores and a_scores:
                    pairs.append(
                        {
                            "date": date,
                            "pair_id": pair_id,
                            "q_score": float(np.mean(q_scores)),
                            "a_score": float(np.mean(a_scores)),
                            "q_n": len(q_scores),
                            "a_n": len(a_scores),
                        }
                    )
                    pair_id += 1
                    a_scores = []
                q_scores.append(float(row["score"]))
            else:
                # Answer chunk — only collect if we already have a question
                if q_scores:
                    a_scores.append(float(row["score"]))

        # Flush last pair
        if q_scores and a_scores:
            pairs.append(
                {
                    "date": date,
                    "pair_id": pair_id,
                    "q_score": float(np.mean(q_scores)),
                    "a_score": float(np.mean(a_scores)),
                    "q_n": len(q_scores),
                    "a_n": len(a_scores),
                }
            )

    result = pd.DataFrame(pairs)
    if result.empty:
        print("WARNING: No pairs found! Check is_question values:")
        print(df["is_question"].value_counts())
        print("First 10 rows:")
        print(df.head(10)[["date", "chunk_id", "is_question", "score"]])
    else:
        result["gap"] = (result["q_score"] - result["a_score"]).abs()
    return result


# ── Aggregate per meeting ─────────────────────────────────────────────────────
def aggregate_per_meeting(pairs_df: pd.DataFrame) -> pd.DataFrame:
    """Mean gap, mean q_score, mean a_score per meeting date."""
    agg = (
        pairs_df.groupby("date")
        .agg(
            mean_gap=("gap", "mean"),
            mean_q=("q_score", "mean"),
            mean_a=("a_score", "mean"),
            n_pairs=("pair_id", "count"),
            max_gap=("gap", "max"),
        )
        .reset_index()
    )
    return agg


# ── Merge with market data ────────────────────────────────────────────────────
def load_with_market(pairs_agg: pd.DataFrame) -> pd.DataFrame:
    from ..data.model_data import return_data

    market = return_data(
        market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
        word_limit=150,
        IS_QA_division=False,
        qa_options="both_together",
        with_label=False,
    )
    ois_cols = [c for c in market.columns if c.startswith("OIS_")]
    market["OIS_uncertainty"] = (
        market[ois_cols].abs().mean(axis=1) if ois_cols else np.nan
    )

    merged = pd.merge(
        pairs_agg,
        market[["date", "OIS_uncertainty", "STOXX50", "pc1"]],
        on="date",
        how="inner",
    )
    return merged


# ── Figure 3.7b_1 — Q vs A Time Series with rolling MA ────────────────────────
def plot_fig_3_7b_timeseries(pairs_agg, save: bool = True):
    fig, ax = plt.subplots(figsize=(13, 4.2))

    # Raw
    ax.plot(
        pairs_agg["date"],
        pairs_agg["mean_q"],
        color="#e74c3c",
        linewidth=0.7,
        alpha=0.3,
    )
    ax.plot(
        pairs_agg["date"],
        pairs_agg["mean_a"],
        color="#2c6fad",
        linewidth=0.7,
        alpha=0.3,
    )

    # 6M rolling MA
    pairs_agg = pairs_agg.set_index("date")
    for col, color, label in [
        ("mean_q", "#e74c3c", "Questions — journalists (6M MA)"),
        ("mean_a", "#2c6fad", "Answers — ECB officials (6M MA)"),
    ]:
        fwd = pairs_agg[col].rolling("180D", min_periods=1).mean()
        bwd = pairs_agg[col][::-1].rolling("180D", min_periods=1).mean()[::-1]
        roll = (fwd + bwd) / 2
        ax.plot(roll.index, roll.values, color=color, linewidth=2.0, label=label)
    pairs_agg = pairs_agg.reset_index()

    ax.axhline(0, color="#888", linewidth=0.8, linestyle="--")
    ax.set_xlabel("Date", fontsize=11)
    ax.set_ylabel("RoBERTa Sentiment", fontsize=11)
    ax.legend(fontsize=10, loc="lower left", framealpha=0.92)
    ax.grid(alpha=0.22, linewidth=0.6)
    ax.xaxis.set_major_locator(mdates.YearLocator(5))
    ax.xaxis.set_minor_locator(mdates.YearLocator(1))
    ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

    plt.tight_layout()
    if save:
        path = OUTPUT_DIR / "fig_QA_timeseries.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.7b_2 — Signed gap histogram + era box plots ─────────────────────
def plot_fig_3_7b_distribution(pairs, save: bool = True):
    fig, axes = plt.subplots(1, 2, figsize=(12, 4.2), gridspec_kw={"wspace": 0.28})

    # ── Left: Signed gap histogram ────────────────────────────────────────────
    ax1 = axes[0]
    signed_gap = pairs["q_score"] - pairs["a_score"]

    ax1.hist(
        signed_gap,
        bins=50,
        color="#6375b4",
        alpha=0.65,
        edgecolor="white",
        linewidth=0.5,
        density=True,
    )
    ax1.axvline(0, color="#888", linewidth=1.2, linestyle="--")
    ax1.axvline(
        signed_gap.mean(),
        color="#e74c3c",
        linewidth=2.0,
        label=f"Mean = {signed_gap.mean():.3f}",
    )
    ax1.axvline(
        signed_gap.median(),
        color="#2c6fad",
        linewidth=2.0,
        linestyle=":",
        label=f"Median = {signed_gap.median():.3f}",
    )

    # Annotate direction
    pct_neg = (signed_gap < 0).mean() * 100
    pct_pos = (signed_gap > 0).mean() * 100
    ax1.text(
        0.03,
        0.95,
        f"Q more dovish\nthan A: {pct_neg:.0f}%",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        color="#c0392b",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ddd"),
    )
    ax1.text(
        0.97,
        0.95,
        f"Q more hawkish\nthan A: {pct_pos:.0f}%",
        transform=ax1.transAxes,
        fontsize=9,
        va="top",
        ha="right",
        color="#0915b6",
        bbox=dict(facecolor="white", alpha=0.8, edgecolor="#ddd"),
    )

    ax1.set_xlabel("Signed Gap  (Q − A sentiment)", fontsize=11)
    ax1.set_ylabel("Density", fontsize=11)
    ax1.set_title(
        "Distribution of Q−A Sentiment Gap\n(negative = journalist more dovish)",
        fontsize=11,
        pad=6,
    )
    ax1.legend(fontsize=9.5, framealpha=0.9)
    ax1.grid(alpha=0.22, linewidth=0.6)

    # ── Right: Box plots per era (horizontal) ─────────────────────────────────
    ax2 = axes[1]
    eras = {
        "Hiking\n2022–2024": ("2022-01-01", "2024-12-31"),
        "ZLB\n2012–2021": ("2012-01-01", "2021-12-31"),
        "Pre-ZLB\n1999–2011": ("1999-01-01", "2011-12-31"),
    }
    era_data = []
    era_labels = []
    for label, (s, e) in eras.items():
        mask = (pairs["date"] >= s) & (pairs["date"] <= e)
        era_data.append(pairs.loc[mask, "gap"].values)
        era_labels.append(label)

    bp = ax2.boxplot(
        era_data,
        tick_labels=era_labels,
        vert=False,
        patch_artist=True,
        widths=0.5,
        medianprops=dict(color="black", linewidth=1.5),
    )
    colors = ["#e74c3c", "#e67e22", "#3498db"]
    for patch, c in zip(bp["boxes"], colors):
        patch.set_facecolor(c)
        patch.set_alpha(0.6)
    for i, data in enumerate(era_data):
        ax2.scatter(
            [np.mean(data)],
            [i + 1],
            marker="D",
            s=55,
            color="white",
            edgecolors="black",
            linewidths=1.5,
            zorder=5,
        )

    ax2.axvline(0, color="#888", linewidth=0.8, linestyle="--")
    ax2.set_xlabel("|Q − A| Sentiment Gap (per pair)", fontsize=11)
    ax2.set_title("Q–A Gap by Era\n(◆ = mean)", fontsize=11, pad=6)
    ax2.grid(axis="x", alpha=0.22, linewidth=0.6)

    if save:
        path = OUTPUT_DIR / "fig_QA_distribution.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    print("Loading QA chunks from DB...")
    chunks = load_qa_chunks("roberta")

    print("Building Q→A pairs...")
    pairs = build_qa_pairs(chunks)
    print(f"  {len(pairs)} pairs across {pairs['date'].nunique()} meetings")
    print(f"  Mean gap: {pairs['gap'].mean():.4f}, Max: {pairs['gap'].max():.4f}")

    pairs_agg = aggregate_per_meeting(pairs)

    plot_fig_3_7b_timeseries(pairs_agg)
    plot_fig_3_7b_distribution(pairs)

    # Save pair-level data
    pairs.to_csv(OUTPUT_DIR / "table_qa_pairs.csv", index=False)
    plt.show()
