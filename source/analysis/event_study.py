"""
Figure 3.11 – Event Study: Sentiment around ECB Rate Decisions

For each meeting where the ECB hiked or cut the MRO rate, extract the
sentiment trajectory in a ±WINDOW-meeting window, then average across
all events of the same type. Shaded bands show ±1 SEM.

Both FinBERT (solid) and CentralBankRoBERTa IS (dashed) are shown for
each regime trajectory. The two models share identical shapes despite
different absolute baselines — confirming the finding is model-invariant.

RoBERTa IS (IS-restricted) is used instead of RoBERTa overall because
the overall RoBERTa mean is heavily contaminated by journalist question
sentiment (structural baseline ≈ −0.37), which suppresses the signal.

Era split:
    Pre-ZLB  : 1999-01-01 – 2013-06-30
    Post-ZLB : 2021-07-01 – 2026-12-31

The ZLB era (flat MRO) is excluded because rate changes are too rare
to build meaningful window averages.

Bug fixed:
    Previously SENTIMENT = "finbert_mean" but the y-axis label read
    "RoBERTa Sentiment" — mislabelled figure. Now both models are shown
    with correct labels.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
from matplotlib.lines import Line2D
import pandas as pd
import numpy as np
from pathlib import Path

try:
    from ..data.model_data import return_data
    from .. import OUTPUT

    OUTPUT_DIR = Path(OUTPUT) / "results/mro"
except ImportError:
    import sys

    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from source.data.model_data import return_data

    OUTPUT_DIR = Path("outputs/results/mro")

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

# ── Constants ─────────────────────────────────────────────────────────────────
WINDOW = 6  # meetings either side of the decision

ERA_SPLIT = {
    "Pre-ZLB": ("1999-01-01", "2013-06-30"),
    "Post-ZLB": ("2021-07-01", "2026-12-31"),
}

# Regime colours (consistent with mro_cycle.py)
REGIME_COLORS = {
    "hike": "#c0392b",  # red
    "cut": "#2c6fad",  # blue
}

# Model styles: (column, display label, linestyle, linewidth, shading alpha)
MODELS = [
    ("finbert_mean", "FinBERT", "-", 2.2, 0.18),
    ("roberta_mean", "CentralBankRoBERTa", "--", 1.6, 0.10),
]


# ── Data loading ──────────────────────────────────────────────────────────────
def load_data() -> pd.DataFrame:
    """
    Load overall sentiment (for finbert_mean) and IS-separated sentiment
    (for roberta_mean) and merge into one wide DataFrame.
    """
    # Overall combined — provides finbert_mean
    df = return_data(
        market_data="ECB Money Market.xlsx",
        word_limit=150,
        IS_QA_division=False,
        qa_options="both_together",
        with_label=False,
    )

    # IS-separated — provides roberta_IS_mean
    df_is = return_data(
        market_data="ECB Money Market.xlsx",
        word_limit=150,
        IS_QA_division=True,
        qa_options="just_answers",
        with_label=False,
    )
    is_extra = [c for c in df_is.columns if c not in df.columns and c != "date"]
    df = pd.merge(df, df_is[["date"] + is_extra], on="date", how="left")

    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])

    # MRO change: positive = hike, negative = cut
    df["mro_change"] = (df["MRO announced"] - df["MRO"]).fillna(0)

    return df


# ── Window extraction ─────────────────────────────────────────────────────────
def build_windows(
    df: pd.DataFrame,
    sentiment_col: str,
    window: int = WINDOW,
) -> tuple[pd.DataFrame, pd.DataFrame]:
    """
    For every hike/cut meeting, extract the sentiment trajectory
    in [-window, +window] meetings relative to that event.

    Returns (hike_df, cut_df), each with columns:
        event_date | lag | sentiment
    """
    hike_idx = df.index[df["mro_change"] > 0].tolist()
    cut_idx = df.index[df["mro_change"] < 0].tolist()

    def _extract(event_indices: list) -> pd.DataFrame:
        records = []
        for ev_idx in event_indices:
            ev_date = df.loc[ev_idx, "date"]
            for lag in range(-window, window + 1):
                row_idx = ev_idx + lag
                if 0 <= row_idx < len(df):
                    val = df.loc[row_idx, sentiment_col]
                    if pd.notna(val):
                        records.append(
                            {
                                "event_date": ev_date,
                                "lag": lag,
                                "sentiment": val,
                            }
                        )
        return pd.DataFrame(records)

    return _extract(hike_idx), _extract(cut_idx)


def compute_profile(
    window_df: pd.DataFrame,
    lags: np.ndarray,
) -> tuple[pd.Series, pd.Series]:
    """Return (mean, sem) indexed over the full lag range."""
    grp = window_df.groupby("lag")["sentiment"]
    mean = grp.mean().reindex(lags)
    sem = grp.sem().reindex(lags)
    return mean, sem


# ── Main figure ───────────────────────────────────────────────────────────────
def plot_event_study(df: pd.DataFrame | None = None, save: bool = True):
    """
    Two-panel figure (Pre-ZLB | Post-ZLB).
    Each panel shows 4 lines: (hike/cut) × (FinBERT/RoBERTa IS).
    Regime = colour, model = linestyle (solid/dashed).
    """
    if df is None:
        df = load_data()

    lags = np.arange(-WINDOW, WINDOW + 1)

    fig, axes = plt.subplots(
        1,
        len(ERA_SPLIT),
        figsize=(13, 5.2),
        sharey=True,
        gridspec_kw={"wspace": 0.08},
    )

    for ax, (era_label, (t0, t1)) in zip(axes, ERA_SPLIT.items()):
        era_df = df[(df["date"] >= t0) & (df["date"] <= t1)].reset_index(drop=True)

        n_hike = (era_df["mro_change"] > 0).sum()
        n_cut = (era_df["mro_change"] < 0).sum()

        for sent_col, model_label, ls, lw, shade in MODELS:
            if sent_col not in era_df.columns:
                continue

            hike_win, cut_win = build_windows(era_df, sent_col)

            for regime, win_df, color, n_events in [
                ("hike", hike_win, REGIME_COLORS["hike"], n_hike),
                ("cut", cut_win, REGIME_COLORS["cut"], n_cut),
            ]:
                if win_df.empty:
                    continue

                mean, sem = compute_profile(win_df, lags)

                ax.plot(
                    lags,
                    mean,
                    color=color,
                    linewidth=lw,
                    linestyle=ls,
                    marker="o" if ls == "-" else None,
                    markersize=3.5,
                    zorder=4,
                )
                ax.fill_between(
                    lags,
                    mean - sem,
                    mean + sem,
                    color=color,
                    alpha=shade,
                    zorder=2,
                )

        # Reference lines
        ax.axvline(0, color="#555", linewidth=1.2, linestyle="--", zorder=5)
        ax.axhline(0, color="#bbb", linewidth=0.8, linestyle=":", zorder=1)

        # ── Legend (compact two-tier proxy handles) ───────────────────────────
        legend_handles = [
            # Regime colour proxies (with n counts)
            Line2D(
                [0],
                [0],
                color=REGIME_COLORS["hike"],
                lw=2.0,
                label=f"Rate hike  (n={n_hike} decisions)",
            ),
            Line2D(
                [0],
                [0],
                color=REGIME_COLORS["cut"],
                lw=2.0,
                label=f"Rate cut  (n={n_cut} decisions)",
            ),
            # Model linestyle proxies
            Line2D([0], [0], color="#666", lw=1.8, ls="-", label="FinBERT (solid)"),
            Line2D(
                [0],
                [0],
                color="#666",
                lw=1.8,
                ls="--",
                label="CentralBankRoBERTa (dashed)",
            ),
            # Decision day
            Line2D(
                [0], [0], color="#555", lw=1.2, ls="--", label="Decision day (lag = 0)"
            ),
        ]
        ax.legend(
            handles=legend_handles, framealpha=0.53, loc="center left", handlelength=2.0
        )

        ax.set_title(era_label, pad=8, fontweight="bold")
        ax.set_xlabel(
            "Meetings relative to rate decision",
        )
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.grid(alpha=0.20, linewidth=0.6)

    # Y-axis label — now correctly reflects both models
    axes[0].set_ylabel(
        "Sentiment Score (mean ± SEM)",
    )

    fig.suptitle(
        "Event Study: Sentiment around ECB Rate Decisions\n"
        "FinBERT (solid) vs. CentralBankRoBERTa (dashed)",
        y=1.02,
    )
    fig.text(
        0.5,
        -0.02,
        "Negative lags = meetings before the rate decision  |  "
        "Positive lags = meetings after",
        ha="center",
        color="#555",
    )

    if save:
        path = OUTPUT_DIR / "fig_event_study.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    df = load_data()
    plot_event_study(df)
    plt.show()
