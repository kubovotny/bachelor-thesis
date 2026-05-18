"""
Figures 3.7a, 3.7b – Divergence Analysis and Market Uncertainty

3.7a: IS–QA Divergence = |sent_IS − sent_QA|
      Correlates with OIS changes, STOXX50, and PC1 shocks.
      Tests whether communication mismatch → market uncertainty.

3.7b: Question–Answer Gap = |sent_Q − sent_A|
      Journalist sentiment vs. ECB official sentiment.
      Tests whether adversarial Q&A → market surprise.

Multiplicity note (§3.2.3)
---------------------------
Six Pearson tests are run across the two figures.  At family α = 0.05 the
expected number of false positives under the global null is 6 × 0.05 = 0.30,
and the probability of at least one spurious "significant" result is ≈ 26 %.
Holm–Bonferroni correction is therefore applied to the full family of six
p-values via :func:`compute_holm_table`; adjusted p-values are reported in
the scatter annotations and in the console summary.
"""

from __future__ import annotations

from pathlib import Path
from typing import Optional

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy import stats
from statsmodels.stats.multitest import multipletests

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

plt.rcParams.update(
    {
        "font.size": 13,
        "axes.titlesize": 14,
        "axes.labelsize": 13,
        "legend.fontsize": 11,
        "xtick.labelsize": 11,
        "ytick.labelsize": 11,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)

# ── Test family specification ──────────────────────────────────────────────────
# All six Pearson tests in §3.2.3 declared in one place.
# Changing order here propagates automatically to compute_holm_table,
# print_correlation_summary, and the scatter annotations.
_TEST_SPECS: list[tuple[str, str, str, str, str]] = [
    # (df_key,  y_col,        x_col,           y_label,      x_label )
    ("div", "IS_QA_div", "OIS_uncertainty", "IS–QA Div.", "|ΔOIS|"),
    ("div", "IS_QA_div", "STOXX50_x",       "IS–QA Div.", "STOXX50"),
    ("div", "IS_QA_div", "pc1",             "IS–QA Div.", "PC1"),
    ("gap", "Q_A_gap",   "OIS_uncertainty", "Q–A Gap",    "|ΔOIS|"),
    ("gap", "Q_A_gap",   "STOXX50_x",       "Q–A Gap",    "STOXX50"),
    ("gap", "Q_A_gap",   "pc1",             "Q–A Gap",    "PC1"),
]


# ── Private helpers ────────────────────────────────────────────────────────────

def _sig_stars(p: float) -> str:
    """Return APA-style significance stars for *p* (or '' for NaN)."""
    if pd.isna(p):
        return ""
    if p < 0.001:
        return "***"
    if p < 0.01:
        return "**"
    if p < 0.05:
        return "*"
    return "n.s."


def _compute_corr(x: pd.Series, y: pd.Series) -> tuple[float, float, int]:
    """
    Compute Pearson r, two-tailed p-value, and valid-observation count.

    Parameters
    ----------
    x, y : pd.Series
        Input series; NaNs are dropped pairwise.

    Returns
    -------
    tuple[float, float, int]
        ``(r, p, n)``.  Returns ``(nan, nan, n)`` when ``n < 10``.
    """
    mask = x.notna() & y.notna()
    n = int(mask.sum())
    if n < 10:
        return np.nan, np.nan, n
    r, p = stats.pearsonr(x[mask].values, y[mask].values)
    return float(r), float(p), n


def _lookup_p_adj(
    holm_table: Optional[pd.DataFrame],
    y_label: str,
    x_label: str,
) -> Optional[float]:
    """
    Return the Holm-adjusted p-value for a given test pair, or *None*.

    Parameters
    ----------
    holm_table : pd.DataFrame or None
        Output of :func:`compute_holm_table`.
    y_label, x_label : str
        Labels matching the ``y`` and ``x`` columns of *holm_table*.
    """
    if holm_table is None:
        return None
    row = holm_table[
        (holm_table["y"] == y_label) & (holm_table["x"] == x_label)
    ]
    if row.empty or pd.isna(row["p_adj"].iloc[0]):
        return None
    return float(row["p_adj"].iloc[0])


# ── Load data ──────────────────────────────────────────────────────────────────

def load_divergence_data() -> pd.DataFrame:
    """
    Load IS-statement vs. Q&A sentiment split with market variables.

    Derived columns added:
    - ``IS_QA_div`` : absolute IS–QA sentiment divergence.
    - ``OIS_uncertainty`` : mean absolute OIS change across all maturities.
    """
    sent = return_data(
        market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
        word_limit=150,
        IS_QA_division=True,
        qa_options="just_answers",
        with_label=False,
    )
    sent = sent.sort_values("date").reset_index(drop=True)
    sent["IS_QA_div"] = (sent["roberta_IS_mean"] - sent["roberta_QA_mean"]).abs()

    ois_cols = [c for c in sent.columns if c.startswith("OIS_")]
    if ois_cols:
        sent["OIS_uncertainty"] = sent[ois_cols].abs().mean(axis=1)

    return sent


def load_qa_gap_data() -> pd.DataFrame:
    """
    Load journalist-question vs. ECB-answer sentiment split with market variables.

    Derived columns added:
    - ``Q_A_gap`` : absolute question–answer sentiment gap.
    - ``Q_sent``, ``A_sent`` : raw question and answer sentiment scores.
    - ``OIS_uncertainty`` : mean absolute OIS change across all maturities.

    Raises
    ------
    ValueError
        If the expected RoBERTa question/answer columns are not found.
    """
    sent = return_data(
        market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
        word_limit=150,
        IS_QA_division=True,
        qa_options="both_divided",
        with_label=False,
    )
    sent = sent.sort_values("date").reset_index(drop=True)

    q_col = next(
        (c for c in sent.columns if "roberta" in c and "question" in c and "mean" in c),
        None,
    )
    a_col = next(
        (c for c in sent.columns if "roberta" in c and "answer" in c and "mean" in c),
        None,
    )
    if q_col is None or a_col is None:
        available = [c for c in sent.columns if "roberta" in c]
        raise ValueError(
            f"Could not find question/answer RoBERTa columns. "
            f"Available: {available}"
        )

    sent["Q_A_gap"] = (sent[q_col] - sent[a_col]).abs()
    sent["Q_sent"] = sent[q_col]
    sent["A_sent"] = sent[a_col]

    ois_cols = [c for c in sent.columns if c.startswith("OIS_")]
    if ois_cols:
        sent["OIS_uncertainty"] = sent[ois_cols].abs().mean(axis=1)

    return sent


# ── Holm–Bonferroni correction across the full test family ─────────────────────

def compute_holm_table(
    df_div: pd.DataFrame,
    df_gap: pd.DataFrame,
) -> pd.DataFrame:
    """
    Run all six Pearson correlations from §3.2.3 and apply Holm–Bonferroni.

    Parameters
    ----------
    df_div : pd.DataFrame
        Output of :func:`load_divergence_data`.
    df_gap : pd.DataFrame
        Output of :func:`load_qa_gap_data`.

    Returns
    -------
    pd.DataFrame
        One row per test with columns:
        ``y``, ``x``, ``r``, ``n``, ``p_raw``, ``sig_raw``,
        ``p_adj`` (Holm), ``sig_adj``, ``reject``.

    Notes
    -----
    Holm–Bonferroni is strictly more powerful than Bonferroni while still
    controlling the family-wise error rate (FWER) at α = 0.05.
    """
    dfs = {"div": df_div, "gap": df_gap}
    records: list[dict] = []

    for df_key, y_col, x_col, y_label, x_label in _TEST_SPECS:
        df = dfs[df_key]
        if x_col not in df.columns or y_col not in df.columns:
            records.append(
                dict(y=y_label, x=x_label, r=np.nan, n=0, p_raw=np.nan)
            )
            continue
        r, p, n = _compute_corr(df[x_col], df[y_col])
        records.append(dict(y=y_label, x=x_label, r=r, n=n, p_raw=p))

    results = pd.DataFrame(records)

    valid = results["p_raw"].notna()
    if valid.any():
        reject, p_adj, _, _ = multipletests(
            results.loc[valid, "p_raw"], alpha=0.05, method="holm"
        )
        results.loc[valid, "p_adj"] = p_adj
        results.loc[valid, "reject"] = reject
    else:
        results["p_adj"] = np.nan
        results["reject"] = False

    results["sig_raw"] = results["p_raw"].map(_sig_stars)
    results["sig_adj"] = results["p_adj"].map(_sig_stars)

    return results


# ── Scatter helper ────────────────────────────────────────────────────────────

def scatter_with_reg(
    ax,
    x: pd.Series,
    y: pd.Series,
    xlabel: str,
    ylabel: str,
    title: str,
    color: str = "#2c6fad",
    reg: bool = True,
    p_adj: Optional[float] = None,
) -> None:
    """
    Scatter plot with optional OLS regression line and Pearson correlation annotation.

    Parameters
    ----------
    ax : matplotlib.axes.Axes
    x, y : pd.Series
        Data series; NaNs are dropped pairwise.
    xlabel, ylabel, title : str
    color : str
        Hex colour for markers and regression line.
    reg : bool
        Draw the OLS regression line when ``True``.
    p_adj : float, optional
        Holm-adjusted p-value.  When supplied, significance stars reflect
        *p_adj*; both raw and adjusted p-values are shown in the annotation
        for full transparency.
    """
    mask = x.notna() & y.notna()
    xv, yv = x[mask].values, y[mask].values
    n = int(mask.sum())

    if n < 10:
        ax.text(
            0.5, 0.5, "Insufficient data",
            ha="center", va="center", transform=ax.transAxes,
        )
        return

    r, p = stats.pearsonr(xv, yv)
    slope, intercept, *_ = stats.linregress(xv, yv)

    ax.scatter(xv, yv, alpha=0.45, s=20, color=color, edgecolors="none")
    if reg:
        xfit = np.linspace(xv.min(), xv.max(), 200)
        ax.plot(xfit, slope * xfit + intercept, color=color, linewidth=1.8, alpha=0.9)

    # Significance stars driven by adjusted p when available; raw p otherwise.
    p_for_stars = p_adj if p_adj is not None else p
    sig = _sig_stars(p_for_stars)

    # Show both p-values when a correction was applied.
    if p_adj is not None:
        annotation = (
            f"r = {r:.3f} {sig}\n"
            f"p = {p:.3f} | p_adj = {p_adj:.3f}\n"
            f"n = {n}"
        )
    else:
        annotation = f"r = {r:.3f} {sig}\nn = {n}"

    ax.text(
        0.05, 0.95, annotation,
        transform=ax.transAxes, va="top",
        bbox=dict(boxstyle="round,pad=0.4", facecolor="white", edgecolor="#ccc", alpha=0.9),
    )
    ax.set_xlabel(xlabel)
    ax.set_ylabel(ylabel)
    ax.set_title(title, pad=5)
    ax.grid(alpha=0.22, linewidth=0.6)


# ── Figure 3.7a — IS–QA Divergence ───────────────────────────────────────────

def plot_fig_3_7a(
    df: Optional[pd.DataFrame] = None,
    holm_table: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot IS–QA communication divergence against three market variables.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded output of :func:`load_divergence_data`.
        Loaded automatically when *None*.
    holm_table : pd.DataFrame, optional
        Pre-computed output of :func:`compute_holm_table`.
        When supplied, scatter annotations show Holm-adjusted p-values.
    save : bool
        Persist figure to ``OUTPUT_DIR/is_qa/fig_divergence.pdf``.
    """
    if df is None:
        df = load_divergence_data()

    # (df_col, x-axis display label, marker colour, x_label key in holm_table)
    market_targets = [
        ("OIS_uncertainty", "|ΔOIS| mean across maturities", "#2c6fad", "|ΔOIS|"),
        ("STOXX50_x",       "STOXX50 return (%)",            "#27ae60", "STOXX50"),
        ("pc1",             "PC1 monetary surprise",          "#8e44ad", "PC1"),
    ]

    fig, axes = plt.subplots(1, 3, figsize=(14, 4.2), gridspec_kw={"wspace": 0.32})

    for ax, (col, xlabel, color, x_key) in zip(axes, market_targets):
        if col not in df.columns:
            ax.text(
                0.5, 0.5, f"'{col}' not in data",
                ha="center", va="center", transform=ax.transAxes,
            )
            continue
        scatter_with_reg(
            ax=ax,
            x=df[col],
            y=df["IS_QA_div"],
            xlabel=xlabel,
            ylabel="|IS − QA| Sentiment",
            title=f"IS–QA Divergence\nvs. {xlabel.split('(')[0].strip()}",
            color=color,
            reg=False,
            p_adj=_lookup_p_adj(holm_table, "IS–QA Div.", x_key),
        )

    fig.suptitle(
        "IS–QA Communication Divergence vs. Market Variables (RoBERTa)",
        y=1.05,
    )
    if save:
        path = OUTPUT_DIR / "is_qa/fig_divergence.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Figure 3.7b — Q–A Sentiment Gap ──────────────────────────────────────────

def plot_fig_3_7b(
    df: Optional[pd.DataFrame] = None,
    holm_table: Optional[pd.DataFrame] = None,
    save: bool = True,
) -> plt.Figure:
    """
    Plot journalist–ECB Q–A sentiment gap against market variables.

    Parameters
    ----------
    df : pd.DataFrame, optional
        Pre-loaded output of :func:`load_qa_gap_data`.
        Loaded automatically when *None*.
    holm_table : pd.DataFrame, optional
        Pre-computed output of :func:`compute_holm_table`.
    save : bool
        Persist figure to ``OUTPUT_DIR/is_qa/fig_gap.pdf``.
    """
    if df is None:
        df = load_qa_gap_data()

    fig, axes = plt.subplots(1, 2, figsize=(14, 4.2), gridspec_kw={"wspace": 0.32})

    scatter_with_reg(
        ax=axes[0],
        x=df["OIS_uncertainty"],
        y=df["Q_A_gap"],
        xlabel="|ΔOIS| mean across maturities",
        ylabel="|Q − A| Sentiment Gap",
        title="Q–A Gap vs.\nOIS Market Uncertainty",
        color="#e74c3c",
        p_adj=_lookup_p_adj(holm_table, "Q–A Gap", "|ΔOIS|"),
    )

    if "pc1" in df.columns:
        scatter_with_reg(
            ax=axes[1],
            x=df["pc1"],
            y=df["Q_A_gap"],
            xlabel="PC1 monetary surprise",
            ylabel="|Q − A| Sentiment Gap",
            title="Q–A Gap vs.\nPC1 Policy Surprise",
            color="#8e44ad",
            p_adj=_lookup_p_adj(holm_table, "Q–A Gap", "PC1"),
        )
    else:
        axes[1].text(
            0.5, 0.5, "PC1 not available",
            ha="center", va="center", transform=axes[1].transAxes,
        )

    fig.suptitle(
        "Question–Answer Sentiment Gap: Journalist vs. ECB (RoBERTa)",
        y=1.05,
    )
    if save:
        path = OUTPUT_DIR / "is_qa/fig_gap.pdf"
        path.parent.mkdir(parents=True, exist_ok=True)
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Correlation summary with Holm correction ──────────────────────────────────

def print_correlation_summary(holm_table: pd.DataFrame) -> None:
    """
    Print a formatted table of all six Pearson correlations with
    raw and Holm-adjusted p-values.

    Parameters
    ----------
    holm_table : pd.DataFrame
        Output of :func:`compute_holm_table`.
    """
    col_w = 76
    header = (
        f"{'Outcome':<14} {'Predictor':<10} {'r':>7} {'n':>5}  "
        f"{'p_raw':>7}  {'sig':>4}  {'p_adj':>7}  {'sig_adj':>7}  {'H₀':>12}"
    )

    print(f"\n{'═' * col_w}")
    print(
        " Holm–Bonferroni Correction across all §3.2.3 Pearson tests  "
        "(family α = 0.05)"
    )
    print(f"{'═' * col_w}")
    print(header)
    print(f"{'─' * col_w}")

    prev_y: Optional[str] = None
    for _, row in holm_table.iterrows():
        if row["y"] != prev_y and prev_y is not None:
            print(f"{'─' * col_w}")
        prev_y = str(row["y"])
        reject_str = "✓ reject H₀" if row.get("reject", False) else "  retain H₀"
        print(
            f"{row['y']:<14} {row['x']:<10} "
            f"{row['r']:>7.4f} {int(row['n']):>5}  "
            f"{row['p_raw']:>7.4f}  {row['sig_raw']:>4}  "
            f"{row['p_adj']:>7.4f}  {row['sig_adj']:>7}  "
            f"{reject_str:>12}"
        )

    print(f"{'═' * col_w}")
    print(
        " Bonferroni threshold: 0.05 / 6 ≈ 0.0083.  "
        "Stars in scatter plots reflect p_adj."
    )
    print(f"{'═' * col_w}\n")


# ── Entry point ───────────────────────────────────────────────────────────────

if __name__ == "__main__":
    df1 = load_divergence_data()
    df2 = load_qa_gap_data()
    # Compute the full Holm table once; reuse for printing and plotting.
    holm = compute_holm_table(df1, df2)
    print_correlation_summary(holm)
    plot_fig_3_7a(df1, holm_table=holm)
    plot_fig_3_7b(df2, holm_table=holm)
    plt.show()