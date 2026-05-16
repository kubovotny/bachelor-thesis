"""
Figure 3.6b (revised) – Event Study: Sentiment around MRO Rate Changes

For each meeting where the ECB hiked or cut the MRO rate, we extract
sentiment in a window of ±6 meetings around that decision, then average
across all hike / cut events. Shaded bands show ±1 SEM.

Split by era:
    Pre-ZLB  : 1999-01-01 – 2013-06-30
    Post-ZLB : 2021-07-01 – present   (hiking cycle is here)

The ZLB era (flat MRO) is excluded because rate changes are too rare
to build a meaningful window average.
"""

import matplotlib.pyplot as plt
import matplotlib.ticker as mticker
import pandas as pd
import numpy as np
from pathlib import Path

# ── reproducible relative import shim (run standalone or as module) ──────────
try:
    from ..data.model_data import return_data
    from .. import OUTPUT
    OUTPUT_DIR = Path(OUTPUT) / "results/mro"
except ImportError:
    # standalone: adjust path as needed
    import sys
    sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
    from ..data.model_data import return_data
    OUTPUT_DIR = Path("outputs/results/mro")

OUTPUT_DIR.mkdir(parents=True, exist_ok=True)

WINDOW      = 6          # meetings before / after decision
SENTIMENT   = "finbert_mean"   # column name in the merged dataframe
ERA_SPLIT   = {
    "Pre-ZLB":  ("1999-01-01", "2013-06-30"),
    "Post-ZLB": ("2021-07-01", "2026-12-31"),
}
COLORS = {
    "hike": "#c0392b",   # red
    "cut":  "#2c6fad",   # blue
}


# ── helpers ──────────────────────────────────────────────────────────────────

def load_data() -> pd.DataFrame:
    df = return_data(
        market_data="ECB Money Market.xlsx",
        word_limit=150,
        IS_QA_division=False,
        qa_options="both_together",
        with_label=False,
    )
    df = df.sort_values("date").reset_index(drop=True)
    df["date"] = pd.to_datetime(df["date"])
    df["mro_change"] = (df["MRO announced"] - df["MRO"]).fillna(0)
    return df


def build_windows(df: pd.DataFrame, window: int = WINDOW):
    """
    For every meeting where |mro_change| > 0, extract the sentiment
    trajectory in [-window, +window] meetings relative to the event.

    Returns two DataFrames (hikes, cuts), each with one row per event-meeting,
    columns: [event_date, relative_meeting, sentiment].
    """
    events_hike = df.index[df["mro_change"] > 0].tolist()
    events_cut  = df.index[df["mro_change"] < 0].tolist()

    def _extract(event_indices):
        records = []
        for ev_idx in event_indices:
            ev_date = df.loc[ev_idx, "date"]
            for lag in range(-window, window + 1):
                row_idx = ev_idx + lag
                if 0 <= row_idx < len(df):
                    records.append({
                        "event_date": ev_date,
                        "lag":        lag,
                        "sentiment":  df.loc[row_idx, SENTIMENT],
                    })
        return pd.DataFrame(records)

    return _extract(events_hike), _extract(events_cut)


def compute_profile(window_df: pd.DataFrame):
    """Mean ± SEM per lag step."""
    grp = window_df.groupby("lag")["sentiment"]
    mean = grp.mean()
    sem  = grp.sem()
    n    = grp.count()
    return mean, sem, n


# ── main plot ─────────────────────────────────────────────────────────────────

def plot_event_study(df=None, save: bool = True):
    if df is None:
        df = load_data()

    n_eras = len(ERA_SPLIT)
    fig, axes = plt.subplots(
        1, n_eras,
        figsize=(13, 5),
        sharey=True,
        gridspec_kw={"wspace": 0.10},
    )

    lags = np.arange(-WINDOW, WINDOW + 1)

    for ax, (era_label, (t0, t1)) in zip(axes, ERA_SPLIT.items()):
        era_df = df[(df["date"] >= t0) & (df["date"] <= t1)].reset_index(drop=True)

        hike_win, cut_win = build_windows(era_df)

        for regime, win_df, color in [
            ("Rate hike", hike_win, COLORS["hike"]),
            ("Rate cut",  cut_win,  COLORS["cut"]),
        ]:
            if win_df.empty:
                continue
            mean, sem, n_events = compute_profile(win_df)

            # align to full lag range (some lags may be missing near edges)
            mean = mean.reindex(lags)
            sem  = sem.reindex(lags)

            n_unique = win_df["event_date"].nunique()

            ax.plot(lags, mean,
                    color=color, linewidth=2.2, marker="o", markersize=4,
                    label=f"{regime}  (n={n_unique} decisions)", zorder=4)
            ax.fill_between(lags, mean - sem, mean + sem,
                            color=color, alpha=0.18, zorder=2)

        # reference lines
        ax.axvline(0, color="#555", linewidth=1.2, linestyle="--",
                   label="Decision day (lag = 0)", zorder=5)
        ax.axhline(0, color="#aaa", linewidth=0.8, linestyle=":", zorder=1)

        ax.set_title(era_label, fontsize=11, pad=8, fontweight="bold")
        ax.set_xlabel("Meetings relative to rate decision", fontsize=10)
        ax.xaxis.set_major_locator(mticker.MultipleLocator(2))
        ax.grid(alpha=0.22, linewidth=0.6)
        ax.legend(fontsize=9, framealpha=0.92, loc="upper left")

    axes[0].set_ylabel("RoBERTa Sentiment (mean ± SEM)", fontsize=10)

    fig.suptitle(
        "Event Study: Sentiment around ECB Rate Decisions",
        fontsize=12, y=1.01,
    )

    # shared annotation
    fig.text(
        0.5, -0.03,
        "Negative lags = meetings before the rate decision  |  "
        "Positive lags = meetings after",
        ha="center", fontsize=9, color="#555",
    )

    plt.tight_layout()

    if save:
        path = OUTPUT_DIR / "fig_event_study.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


if __name__ == "__main__":
    df = load_data()
    plot_event_study(df)
    plt.show()