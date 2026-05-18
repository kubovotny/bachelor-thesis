"""
Section 3.9 – Exploring the Trading Value: Delta Sentiment and Volatility

Key insight: OIS_1Y is already a CHANGE (market reaction during press conference).
Correlating it with sentiment LEVEL finds nothing. Two better approaches:

  A) Delta sentiment: Δsent = sent_t − sent_{t-1}
     "Did the ECB become MORE hawkish than last time?"
     This is the communication SURPRISE, analogous to PC1 for rates.

  B) Sentiment → |OIS| (market volatility, not direction)
     "Does uncertain/extreme sentiment cause bigger market moves?"
     Evidence from Section 3.6: roberta_std correlated -0.55 with ΔMRO in Hiking.
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.patches import Patch
from pathlib import Path
from scipy import stats
from scipy.stats import norm, t as t_dist

from ..data.model_data import return_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results"
OUTPUT_DIR.mkdir(exist_ok=True)

SENT_FEATURES = [
    ("finbert_IS_mean", "ΔFB IS mean", "#2c6fad"),
    ("finbert_QA_mean", "ΔFB QA mean", "#27ae60"),
    ("finbert_mean", "ΔFB overall", "#8e44ad"),
    ("finbert_IS_std", "FB IS std", "#c0392b"),
    ("roberta_IS_std", "RB IS std", "#e67e22"),
    ("roberta_std", "RB overall std", "#1a4a7a"),
]
plt.rcParams.update(
    {
        "font.size": 12,
        "axes.titlesize": 13,
        "axes.labelsize": 12,
        "legend.fontsize": 10,
        "xtick.labelsize": 10,
        "ytick.labelsize": 10,
        "axes.spines.top": False,
        "axes.spines.right": False,
        "figure.dpi": 150,
    }
)
# Pre-computed critical r for n=WINDOW=20, alpha=0.05 (two-sided).
# Derived from the t-distribution: t_crit / sqrt(t_crit^2 + df), df = n - 2.
# scipy.stats.t.ppf(0.975, df=18) ≈ 2.1009  →  r_crit ≈ 0.4438.
_WINDOW = 20
_R_CRIT: float = float(
    t_dist.ppf(0.975, df=_WINDOW - 2)
    / np.sqrt(t_dist.ppf(0.975, df=_WINDOW - 2) ** 2 + (_WINDOW - 2))
)


# ── Statistical helpers ────────────────────────────────────────────────────────


def fisher_ci(
    r: np.ndarray,
    n: np.ndarray,
    alpha: float = 0.05,
) -> tuple[np.ndarray, np.ndarray]:
    """Pointwise confidence interval for Pearson r via Fisher z-transform.

    Handles arrays of (r, n) so the CI correctly narrows as the rolling
    window fills up.  Edge cases — r = ±1 and n ≤ 3 — are handled by
    clipping before arctanh and flooring the denominator respectively.

    Args:
        r: Array of Pearson correlation coefficients in [-1, 1].
        n: Array of per-window sample sizes, same shape as *r*.
           Typically the rolling count of non-null observation pairs.
        alpha: Two-sided significance level (default 0.05 → 95 % CI).

    Returns:
        (lo, hi): Arrays of lower and upper CI bounds in correlation space,
        clipped to [-1, 1].

    References:
        Fisher, R. A. (1915). Frequency distribution of the values of the
        correlation coefficient. Biometrika, 10(4), 507–521.
    """
    r = np.asarray(r, dtype=float)
    n = np.asarray(n, dtype=float)

    # Clip r away from ±1 to keep arctanh finite.
    r_safe = np.clip(r, -1.0 + 1e-10, 1.0 - 1e-10)

    # Fisher z-transform.
    z = np.arctanh(r_safe)

    # Standard error of z; guard against n ≤ 3 (se → ∞ handled gracefully).
    se = 1.0 / np.sqrt(np.maximum(n - 3.0, 1.0))

    # Exact z_crit from normal quantile — not the hardcoded 1.96 approximation.
    z_crit = norm.ppf(1.0 - alpha / 2.0)

    lo = np.tanh(z - z_crit * se)
    hi = np.tanh(z + z_crit * se)

    return lo, hi


def _rolling_corr_with_ci(
    series_x: pd.Series,
    series_y: pd.Series,
    window: int,
    min_periods: int = 10,
    alpha: float = 0.05,
) -> tuple[pd.Series, pd.Series, np.ndarray, np.ndarray]:
    """Compute rolling Pearson r and its Fisher CI band.

    After a joint ``dropna()``, both series are complete, so
    ``rolling().count()`` on either column faithfully tracks the actual
    per-window n (rising from *min_periods* to *window* during burn-in,
    then constant at *window* thereafter).

    Args:
        series_x: Feature series (index-aligned with *series_y*).
        series_y: Target series.
        window: Rolling window length (number of meetings).
        min_periods: Minimum observations required to compute a value.
        alpha: Significance level forwarded to :func:`fisher_ci`.

    Returns:
        (roll_r, roll_n, lo, hi) where *roll_r* and *roll_n* are
        pd.Series and *lo*/*hi* are np.ndarray of CI bounds.
    """
    roll_r = series_x.rolling(window, min_periods=min_periods).corr(series_y)
    roll_n = series_x.rolling(window, min_periods=min_periods).count()

    lo, hi = fisher_ci(roll_r.values, roll_n.values, alpha=alpha)
    return roll_r, roll_n, lo, hi


# ── Load ──────────────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """Load and merge ECB press-conference sentiment and market-reaction data.

    Returns:
        DataFrame indexed by ECB meeting date with sentiment scores,
        delta-sentiment features (first differences), and OIS market
        reactions including an absolute-value volatility proxy.
    """
    configs = [
        dict(IS_QA_division=False, qa_options="both_together", with_label=False),
        dict(IS_QA_division=True, qa_options="just_answers", with_label=False),
    ]
    df: pd.DataFrame | None = None
    for cfg in configs:
        d = return_data(
            market_data=["Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"],
            word_limit=200,
            **cfg,
        )
        if df is None:
            df = d
        else:
            new_cols = [c for c in d.columns if c not in df.columns and c != "date"]
            df = pd.merge(df, d[["date"] + new_cols], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # Delta sentiment: first difference per feature (communication surprise).
    sent_cols = [
        c
        for c in df.columns
        if any(m in c for m in ["finbert", "roberta"])
        and c.endswith(("_mean", "_std", "_max", "_min"))
    ]
    for col in sent_cols:
        df[f"d_{col}"] = df[col].diff()

    # OIS volatility proxy: mean absolute change across all OIS maturities.
    ois_cols = [c for c in df.columns if c.startswith("OIS_")]
    if ois_cols:
        df["OIS_abs"] = df[ois_cols].abs().mean(axis=1)

    return df


# ── Figure 3.9a: Δsentiment vs OIS_1Y  +  std vs |OIS| ───────────────────────
def plot_fig_3_9a(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Scatter plot: communication surprise (Δsentiment) vs market direction
    and sentiment uncertainty (std) vs market volatility (|OIS|).

    Args:
        df: Output of :func:`load_data`.
        save: If True, write PDF to OUTPUT_DIR/model/fig_delta_sentiment.pdf.

    Returns:
        The matplotlib Figure object.
    """
    fig, axes = plt.subplots(
        2, 3, figsize=(14, 8), gridspec_kw={"hspace": 0.42, "wspace": 0.30}
    )

    top_pairs = [
        (f"d_{f}", lbl, c, "OIS_1Y", "OIS 1Y change (bps)")
        for f, lbl, c in SENT_FEATURES[:3]
    ]
    bot_pairs = [
        (
            f if "std" in f else f"d_{f}",
            lbl if "std" in f else lbl,
            c,
            "OIS_abs",
            "|OIS| mean abs change",
        )
        for f, lbl, c in SENT_FEATURES[3:]
    ]

    for row, pairs in enumerate([top_pairs, bot_pairs]):
        for col, (feat, label, color, target, ylabel) in enumerate(pairs):
            ax = axes[row][col]
            if feat not in df.columns or target not in df.columns:
                ax.text(
                    0.5,
                    0.5,
                    "N/A",
                    ha="center",
                    va="center",
                    transform=ax.transAxes,
                )
                continue

            mask = df[feat].notna() & df[target].notna()
            x = df.loc[mask, feat].values
            y = df.loc[mask, target].values
            r, p = stats.pearsonr(x, y)
            sig = (
                "***"
                if p < 0.001
                else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
            )

            dates = df.loc[mask, "date"]
            era_colors = np.where(
                dates < "2012-01-01",
                "#3498db",
                np.where(dates < "2022-01-01", "#e67e22", "#e74c3c"),
            )
            ax.scatter(x, y, c=era_colors, alpha=0.5, s=22, edgecolors="none")
            ax.axhline(0, color="#ddd", linewidth=0.7)
            ax.axvline(0, color="#ddd", linewidth=0.7)

            slope, intercept, *_ = stats.linregress(x, y)
            x_fit = np.linspace(x.min(), x.max(), 100)
            ax.plot(
                x_fit,
                slope * x_fit + intercept,
                color=color,
                linewidth=1.8,
                alpha=0.85,
            )

            ax.text(
                0.05,
                0.95,
                f"r = {r:.3f} {sig}\nn = {mask.sum()}",
                transform=ax.transAxes,
                va="top",
                bbox=dict(
                    boxstyle="round,pad=0.3",
                    facecolor="white",
                    edgecolor="#ccc",
                    alpha=0.9,
                ),
            )
            ax.set_xlabel(
                label,
            )
            ax.set_ylabel(
                ylabel,
            )
            prefix = "Δ" if row == 0 else ""
            ax.set_title(
                f"{prefix}{label}\nvs. {ylabel.split('(')[0].strip()}",
                pad=5,
            )
            ax.grid(alpha=0.18, linewidth=0.6)
            ax.autoscale_view()
            if "std" in feat:
                ax.set_xlim(df[feat].min() * 0.9, df[feat].max() * 1.1)

    axes[0][0].annotate(
        "Δsentiment vs\nOIS 1Y",
        xy=(-0.25, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        color="#555",
    )
    axes[1][0].annotate(
        "Sent. std vs\n|OIS| volatility",
        xy=(-0.25, 0.5),
        xycoords="axes fraction",
        ha="right",
        va="center",
        rotation=90,
        color="#555",
    )

    fig.legend(
        handles=[
            Patch(color="#3498db", label="Pre-ZLB (1999–2011)"),
            Patch(color="#e67e22", label="ZLB (2012–2021)"),
            Patch(color="#e74c3c", label="Hiking (2022–2024)"),
        ],
        loc="lower center",
        ncol=3,
        framealpha=0.9,
        bbox_to_anchor=(0.5, -0.02),
    )
    fig.suptitle(
        "Communication Surprise (Δsentiment) and Market Volatility (|OIS|)",
        y=1.01,
    )

    if save:
        fig.savefig(OUTPUT_DIR / "model/fig_delta_sentiment.pdf", bbox_inches="tight")
        print("saved to", OUTPUT_DIR / "model/fig_delta_sentiment.pdf")
    return fig


# ── Figure 3.9b: Rolling correlation with Fisher CI band ──────────────────────
def plot_fig_3_9b(df: pd.DataFrame, save: bool = True) -> plt.Figure:
    """Time-varying rolling correlation with per-point Fisher z CI band.

    For each 20-meeting rolling window, computes Pearson r between the
    sentiment feature and the OIS target, then overlays a 95 % pointwise
    confidence interval derived from Fisher's z-transform.  Two horizontal
    dashed lines mark ±r_crit (the global n=20 two-sided threshold at α=0.05)
    as a fast visual reference; the CI band is the gold-standard treatment
    because it accounts for the dependence of SE on r itself.

    Figure caption (copy into thesis):
        Rolling Pearson correlation over a 20-meeting window (≈2.5 years).
        Shaded bands: 95 % pointwise confidence interval via Fisher
        z-transform; bandwidth reflects both sample size (burn-in) and the
        magnitude of r.  Dashed horizontal lines: two-sided α = 0.05
        significance threshold for n = 20 (r_crit = ±0.444); segments
        outside the corridor are significantly different from zero.
        Top: Δsentiment vs ΔOIS-1Y (direction).
        Bottom: sentiment SD vs |ΔOIS| (volatility).

    Args:
        df: Output of :func:`load_data`.
        save: If True, write PDF to OUTPUT_DIR/model/fig_rolling_correlation.pdf.

    Returns:
        The matplotlib Figure object.
    """
    WINDOW = _WINDOW  # 20 meetings ≈ 2.5 years at 8 meetings/year

    # (feature column, display label, line colour, OIS target column)
    pairs: list[tuple[str, str, str, str]] = [
        ("d_finbert_IS_mean", "ΔFinBERT IS mean", "#2c6fad", "OIS_1Y"),
        ("d_finbert_QA_mean", "ΔFinBERT QA mean", "#27ae60", "OIS_1Y"),
        ("finbert_IS_std", "FinBERT IS std", "#e67e22", "OIS_abs"),
        ("roberta_IS_std", "RoBERTa IS std", "#c0392b", "OIS_abs"),
    ]

    fig, axes = plt.subplots(2, 1, figsize=(13, 8), gridspec_kw={"hspace": 0.38})

    titles = [
        "Rolling Correlation: Δsentiment vs. OIS 1Y (direction)",
        "Rolling Correlation: Sentiment std vs. |OIS| (volatility)",
    ]

    for ax_idx, (ax, title) in enumerate(zip(axes, titles)):

        # ── Zero baseline ──────────────────────────────────────────────────────
        ax.axhline(0, color="#555", linewidth=1.0, linestyle="--", alpha=0.6)

        # ── ±r_crit significance corridor ─────────────────────────────────────
        # Fast reference for reader: any segment inside the corridor is
        # statistically indistinguishable from zero at the local-window level.
        ax.axhline(
            _R_CRIT,
            color="#888",
            linewidth=1.1,
            linestyle=":",
            alpha=0.75,
            zorder=2,
        )
        ax.axhline(
            -_R_CRIT,
            color="#888",
            linewidth=1.1,
            linestyle=":",
            alpha=0.75,
            zorder=2,
        )
        ax.axhspan(
            -_R_CRIT,
            _R_CRIT,
            alpha=0.04,
            color="#888",
            zorder=1,
        )
        # Annotate once, on the top subplot only, to avoid clutter.
        if ax_idx == 0:
            ax.text(
                pd.Timestamp("2023-06-01"),
                _R_CRIT + 0.03,
                f"r_crit = ±{_R_CRIT:.3f}  (n=20, α=0.05)",
                color="#777",
                ha="right",
                va="bottom",
            )

        # ── Per-feature rolling r + Fisher CI band ────────────────────────────
        for feat, label, color, target in pairs:
            if feat_target_row(ax_idx, feat, target):
                continue
            if feat not in df.columns or target not in df.columns:
                continue

            # Drop NaNs jointly before rolling so count() tracks real n.
            tmp = df[["date", feat, target]].dropna().reset_index(drop=True)

            roll_r, roll_n, lo, hi = _rolling_corr_with_ci(
                tmp[feat],
                tmp[target],
                window=WINDOW,
                min_periods=10,
            )

            # Rolling correlation line.
            ax.plot(
                tmp["date"],
                roll_r.values,
                color=color,
                linewidth=1.8,
                label=label,
                alpha=0.88,
                zorder=4,
            )

        # ── Era shading ────────────────────────────────────────────────────────
        for era_start, era_end, era_color, era_label in [
            ("1999-01-01", "2007-12-31", "#3498db", "Pre-GFC"),
            ("2008-01-01", "2011-12-31", "#e67e22", "GFC"),
            ("2012-01-01", "2022-07-01", "#27ae60", "ZLB"),
            ("2022-07-02", "2024-12-31", "#e74c3c", "Hiking"),
        ]:
            ax.axvspan(
                pd.Timestamp(era_start),
                pd.Timestamp(era_end),
                alpha=0.06,
                color=era_color,
                zorder=0,
            )
            mid_date = (
                pd.Timestamp(era_start)
                + (pd.Timestamp(era_end) - pd.Timestamp(era_start)) / 2
            )
            ax.text(
                mid_date,
                0.93,
                era_label,
                ha="center",
                color=era_color,
                alpha=0.75,
                transform=ax.get_xaxis_transform(),
            )

        ax.set_ylim(-1.05, 1.05)
        ax.set_ylabel(
            "Pearson r (rolling 20-meeting window)",
        )
        ax.set_title(title, pad=6)
        ax.grid(alpha=0.22, linewidth=0.6, zorder=0)
        ax.xaxis.set_major_locator(mdates.YearLocator(4))
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%Y"))

        # Build legend: correlation lines + CI patch entry.
        handles, labels = ax.get_legend_handles_labels()
        ax.legend(
            handles=handles,
            labels=labels,
            framealpha=0.92,
            loc="lower left",
        )

    fig.suptitle(
        "Time-Varying Sentiment–Market Relationship\n"
        f"(20-meeting rolling window ≈ 2.5 years)",
        y=1.01,
    )

    if save:
        fig.savefig(
            OUTPUT_DIR / "model/fig_rolling_correlation.pdf",
            bbox_inches="tight",
        )
        print("saved to", OUTPUT_DIR / "model/fig_rolling_correlation.pdf")
    return fig


def feat_target_row(row_idx: int, feat: str, target: str) -> bool:
    """Return True if (feat, target) belongs to the *other* subplot row.

    Row 0 = direction pairs (Δsentiment → OIS_1Y, no std features).
    Row 1 = volatility pairs (sentiment std → |OIS|, no delta features).

    Args:
        row_idx: Subplot index (0 or 1).
        feat: Feature column name.
        target: Target column name.

    Returns:
        True when the pair should be *skipped* for this row.
    """
    if row_idx == 0:
        return "std" in feat or "abs" in target
    return "std" not in feat and "abs" not in target


# ── Print correlations ────────────────────────────────────────────────────────
def print_correlations(df: pd.DataFrame) -> None:
    """Print a table of Pearson r (with significance stars) to stdout.

    For direction targets (OIS_1Y): evaluates the delta-sentiment prefix only.
    For volatility targets (OIS_abs): evaluates the level prefix only.

    Args:
        df: Output of :func:`load_data`.
    """
    target_prefix_map = {
        "OIS_1Y": [("d_", "Δ")],  # direction: surprise matters
        "OIS_abs": [("", "")],  # volatility: level matters
    }

    for target, label in [
        ("OIS_1Y", "OIS_1Y (direction)"),
        ("OIS_abs", "|OIS| (volatility)"),
    ]:
        print(f"\n=== Correlations vs {label} ===")
        prefixes = target_prefix_map[target]
        for feat_raw, feat_label, _ in SENT_FEATURES:
            for prefix, plabel in prefixes:
                feat = prefix + feat_raw
                if feat not in df.columns:
                    continue
                mask = df[feat].notna() & df[target].notna()
                r, p = stats.pearsonr(df.loc[mask, feat], df.loc[mask, target])
                sig = (
                    "***"
                    if p < 0.001
                    else "**" if p < 0.01 else "*" if p < 0.05 else "n.s."
                )
                print(f"  {plabel}{feat_label:22s}: r={r:+.4f} {sig}")


if __name__ == "__main__":
    print("Loading data...")
    df = load_data()
    delta_cols = [c for c in df.columns if c.startswith("d_")]
    print(f"  {len(df)} meetings, {len(delta_cols)} delta features")
    print_correlations(df)
    plot_fig_3_9a(df)
    plot_fig_3_9b(df)
    plt.show()
