"""
Inter-Model Topic Agreement: BART-large-mnli vs DeBERTa-v3

Computes Cohen's kappa between the two zero-shot NLI backbones on the
top-1 topic label for each chunk.  Run after both models have finished
labelling at limits 50, 200, 350.

Usage
-----
    python -m source.analysis.model_agreement

Output
------
Prints a summary table (kappa + raw agreement per limit) and saves:
    output/results/fig_model_agreement.pdf   — confusion-matrix heat-map
    output/results/table_model_agreement.csv — machine-readable summary
"""

from __future__ import annotations

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from pathlib import Path
from sklearn.metrics import cohen_kappa_score, confusion_matrix

from ..data.connection import connect_to_db
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

# Labels in display order
LABEL_ORDER = [
    "MONETARY_POLICY_AND_INFLATION",
    "ECONOMIC_PERFORMANCE",
    "FISCAL_AND_STRUCTURAL",
    "OTHER_IRRELEVANT",
]
LABEL_SHORT = ["MP", "EP", "FS", "OI"]

LIMITS_TO_COMPARE = [1, 50, 200, 350]


# ── Data loading ──────────────────────────────────────────────────────────────

def _load_top1_labels(limit: int) -> pd.DataFrame:
    """
    For every chunk at *limit*, return the single highest-probability label
    assigned by each model.  Chunks labelled by only one model are dropped
    (inner join), so kappa is computed on a common set.

    Returns a DataFrame with columns:
        chunk_rowid | part | bart_label | deberta_label
    """
    conn, _ = connect_to_db()
    sql = """
        SELECT
            t.chunk_rowid,
            CASE ch.part WHEN 0 THEN 'IS' ELSE 'QA' END AS part,
            tm.name  AS model,
            tl.name  AS label,
            t.prob
        FROM   topics t
        JOIN   topic_labels tl ON tl.rowid  = t.label_rowid
        JOIN   topic_models tm ON tm.rowid  = t.model_id
        JOIN   chunks       ch ON ch.rowid  = t.chunk_rowid
        WHERE  ch.chunk_limit = ?
        ORDER  BY t.chunk_rowid, tm.name, t.prob DESC
    """
    df = pd.read_sql(sql, conn, params=(limit,))
    conn.close()

    # Top-1 per (chunk, model) = first row after sort by prob DESC
    top1 = (
        df.sort_values("prob", ascending=False)
          .groupby(["chunk_rowid", "part", "model"], sort=False)
          .first()
          .reset_index()
    )

    bart    = top1[top1["model"] == "facebook"][["chunk_rowid", "part", "label"]].rename(columns={"label": "bart_label"})
    deberta = top1[top1["model"] == "moritz"  ][["chunk_rowid", "label"]].rename(columns={"label": "deberta_label"})

    merged = bart.merge(deberta, on="chunk_rowid")   # inner join = common chunks only
    return merged


# ── Metrics ───────────────────────────────────────────────────────────────────

def _compute_kappa(df: pd.DataFrame) -> dict:
    """Return kappa, raw agreement and per-label precision/recall dict."""
    kappa = cohen_kappa_score(df["bart_label"], df["deberta_label"],
                              labels=LABEL_ORDER)
    raw_agreement = (df["bart_label"] == df["deberta_label"]).mean()
    n = len(df)
    return {"kappa": kappa, "raw_agreement": raw_agreement, "n_chunks": n}


# ── Plot ──────────────────────────────────────────────────────────────────────

def plot_confusion_heatmap(
    merged_200: pd.DataFrame,
    summary: pd.DataFrame,
    save: bool = True,
) -> plt.Figure:
    """
    Left panel  – normalised confusion matrix at limit=200 (row = BART, col = DeBERTa).
    Right panel – kappa vs chunk limit bar chart.
    """
    fig, axes = plt.subplots(
        1, 2,
        figsize=(13, 5),
        gridspec_kw={"width_ratios": [1.6, 1], "wspace": 0.38},
    )

    # ── Left: confusion matrix ────────────────────────────────────────────────
    ax = axes[0]
    cm = confusion_matrix(
        merged_200["bart_label"],
        merged_200["deberta_label"],
        labels=LABEL_ORDER,
        normalize="true",          # row-normalised: "given BART said X, DeBERTa said …"
    )

    im = ax.imshow(cm, cmap="Blues", vmin=0, vmax=1, aspect="auto")
    fig.colorbar(im, ax=ax, fraction=0.046, pad=0.04, label="Row-normalised proportion")

    for i in range(len(LABEL_ORDER)):
        for j in range(len(LABEL_ORDER)):
            ax.text(
                j, i, f"{cm[i,j]:.2f}",
                ha="center", va="center", fontsize=9,
                color="white" if cm[i, j] > 0.55 else "#222",
            )

    ax.set_xticks(range(len(LABEL_SHORT)))
    ax.set_xticklabels(LABEL_SHORT, fontsize=9)
    ax.set_yticks(range(len(LABEL_SHORT)))
    ax.set_yticklabels(LABEL_SHORT, fontsize=9)
    ax.set_xlabel("DeBERTa-v3 label", fontsize=10)
    ax.set_ylabel("BART-large label", fontsize=10)
    ax.set_title(
        f"(A) Top-1 label agreement at chunk limit = 200\n"
        f"$\\kappa$ = {summary.loc[summary['limit']==200,'kappa'].values[0]:.3f}  |  "
        f"raw agreement = {summary.loc[summary['limit']==200,'raw_agreement'].values[0]:.1%}  |  "
        f"$n$ = {summary.loc[summary['limit']==200,'n_chunks'].values[0]:,}",
        fontsize=9.5, pad=8,
    )

    # ── Right: kappa by limit ─────────────────────────────────────────────────
    ax2 = axes[1]
    x = np.arange(len(summary))
    bars = ax2.bar(
        x, summary["kappa"],
        color=["#2c6fad" if lim == 200 else "#a8c8e8" for lim in summary["limit"]],
        width=0.55, edgecolor="white",
    )
    ax2.bar_label(bars, fmt="%.3f", padding=3, fontsize=9)

    # Benchmark lines
    for y_val, label, ls in [
        (0.80, "Almost perfect (0.80)", ":"),
        (0.60, "Substantial (0.60)", "--"),
        (0.40, "Moderate (0.40)", "-."),
    ]:
        ax2.axhline(y_val, color="#888", linewidth=0.9, linestyle=ls, alpha=0.7)
        ax2.text(len(summary) - 0.2, y_val + 0.01, label, fontsize=7,
                 color="#666", ha="right", va="bottom")

    ax2.set_xticks(x)
    ax2.set_xticklabels([f"limit={lim}" for lim in summary["limit"]], fontsize=9)
    ax2.set_ylabel("Cohen's $\\kappa$", fontsize=10)
    ax2.set_ylim(0, 1.0)
    ax2.set_title("(B) Kappa across chunk sizes", fontsize=9.5, pad=8)
    ax2.grid(axis="y", alpha=0.25, linewidth=0.6)
    ax2.tick_params(axis="y", labelsize=8)

    fig.suptitle(
        "BART-large-mnli vs DeBERTa-v3: Top-1 Topic Label Agreement",
        fontsize=11, y=1.01,
    )
    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_model_agreement.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Per-section breakdown (IS vs QA) ─────────────────────────────────────────

def _section_breakdown(merged: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for part in ["IS", "QA"]:
        sub = merged[merged["part"] == part]
        if len(sub) < 20:
            continue
        k = cohen_kappa_score(sub["bart_label"], sub["deberta_label"],
                              labels=LABEL_ORDER)
        rows.append({
            "section": part,
            "kappa": round(k, 4),
            "raw_agreement": round((sub["bart_label"] == sub["deberta_label"]).mean(), 4),
            "n": len(sub),
        })
    return pd.DataFrame(rows)


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    summary_rows = []
    merged_200 = None

    for limit in LIMITS_TO_COMPARE:
        try:
            merged = _load_top1_labels(limit)
        except Exception as e:
            print(f"  limit={limit}: could not load data ({e}), skipping.")
            continue

        if merged.empty:
            print(f"  limit={limit}: no overlapping chunks (DeBERTa not run yet?).")
            continue

        metrics = _compute_kappa(merged)
        summary_rows.append({"limit": limit, **metrics})

        print(f"\n── limit = {limit} ({'%d' % metrics['n_chunks']} chunks) ──")
        print(f"   Cohen's κ     : {metrics['kappa']:.4f}")
        print(f"   Raw agreement : {metrics['raw_agreement']:.1%}")

        # Section breakdown
        breakdown = _section_breakdown(merged)
        if not breakdown.empty:
            print(breakdown.to_string(index=False))

        if limit == 200:
            merged_200 = merged

    if not summary_rows:
        print("No data loaded — check that DeBERTa has finished and data is in DB.")
    else:
        summary = pd.DataFrame(summary_rows)
        print("\n\n=== SUMMARY ===")
        print(summary.to_string(index=False))

        # Save CSV
        summary.to_csv(OUTPUT_DIR / "table_model_agreement.csv", index=False)
        print(f"\nSaved → {OUTPUT_DIR / 'table_model_agreement.csv'}")

        # Plot — uses limit=200 confusion matrix
        if merged_200 is not None:
            plot_confusion_heatmap(merged_200, summary)
            plt.show()
        else:
            print("limit=200 not available — skipping plot.")