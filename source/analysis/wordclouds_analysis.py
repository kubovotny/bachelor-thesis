"""
Figure 3.3  –  Era-Distinctive Vocabulary across Monetary Policy Regimes

Each panel shows words that are DISPROPORTIONATELY frequent in that era
compared to the rest of the sample (relative-frequency weighting).
Word colour encodes the dominant sentiment polarity of that word in the era:
  teal  = word appears more in hawkish chunks (score > +THRESHOLD)
  red   = word appears more in dovish  chunks (score < -THRESHOLD)
  grey  = roughly balanced

This highlights era-specific vocabulary (e.g. "pandemic", "pepp" in 2020–21;
"ukraine", "energy" in 2022–24) rather than generic ECB boilerplate.
"""

import re
from pathlib import Path
from collections import Counter
import matplotlib.pyplot as plt
import matplotlib.gridspec as gridspec
import pandas as pd
from wordcloud import WordCloud, STOPWORDS

from ..data.sentiment import return_sentiment_chunk_data
from .. import OUTPUT

OUTPUT_DIR = Path(OUTPUT) / "results/words_clouds"
OUTPUT_DIR.mkdir(exist_ok=True)

# ── Config ────────────────────────────────────────────────────────────────────
ERAS = {
    "Pre-GFC\n1999–2007": ("1999-01-01", "2007-12-31"),
    "GFC & Recovery\n2008–2013": ("2008-01-01", "2013-12-31"),
    "ZLB & QE\n2014–2019": ("2014-01-01", "2019-12-31"),
    "COVID\n2020–2021": ("2020-01-01", "2021-12-31"),
    "Hiking Cycle\n2022–2024": ("2022-01-01", "2024-12-31"),
}

MODEL = "roberta"
THRESHOLD = 0.15
MIN_COUNT = 20  # raised: filters rare one-off words that get artificially boosted

CUSTOM_STOPWORDS = STOPWORDS | {
    # generic filler
    "will",
    "also",
    "said",
    "well",
    "one",
    "two",
    "think",
    "know",
    "would",
    "could",
    "may",
    "must",
    "need",
    "like",
    "make",
    "go",
    "come",
    "see",
    "say",
    "mean",
    "want",
    "looking",
    "continue",
    "question",
    "answer",
    "president",
    "vice",
    "mr",
    "ms",
    "thank",
    "sure",
    "right",
    "first",
    "second",
    "third",
    "let",
    "just",
    "now",
    "still",
    "already",
    "much",
    "many",
    "way",
    "part",
    "thing",
    "point",
    "terms",
    "kind",
    "fact",
    "sense",
    # ECB boilerplate
    "governing",
    "council",
    "ecb",
    "european",
    "central",
    "bank",
    "euro",
    "area",
    "eurozone",
    "member",
    "states",
    # names of ECB presidents / officials that skew per-era
    "draghi",
    "lagarde",
    "trichet",
    "duisenberg",
    "weidmann",
    "schnabel",
    "fazio",
    "solbes",
    "willem",
    "wim",
    "wiliem",
    "reynders",
    "rompuy",
    "juncker",
    "barroso",
    "monti",
    "pedro",
    "madrid",
    "singapore",
    "greenspan",
    "welteke",
    "noyer",
    "issing",
    "christian",
    "otmar",
    "liaison",
    "residents",
    "homework",
    "printing",
    "coins",
    # country names that appear due to membership events, not substance
    "latvian",
    "latvia",
    "estonian",
    "estonia",
    "croatian",
    "croatia",
    "slovak",
    "slovenian",
    "lithuanian",
}


# ── Helpers ───────────────────────────────────────────────────────────────────
def _clean(text: str) -> str:
    text = text.lower()
    text = re.sub(r"[^a-z\s]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _count(texts: pd.Series) -> Counter:
    c: Counter = Counter()
    for t in texts.dropna():
        words = _clean(str(t)).split()
        c.update(w for w in words if w not in CUSTOM_STOPWORDS and len(w) > 3)
    return c


def load_data(model: str = MODEL) -> pd.DataFrame:
    df = return_sentiment_chunk_data(limit_version=150, with_label=False)
    df = df[df["sentiment_model"] == model].copy()
    df["date"] = pd.to_datetime(df["date"])
    return df


# ── Relative frequency weighting ─────────────────────────────────────────────
def relative_freq(
    era_count: Counter, background_count: Counter, n_era: int, n_bg: int
) -> dict:
    """
    Score = (freq_in_era / n_era) / (freq_in_background / n_bg + epsilon)
    Words with score >> 1 are distinctively frequent in this era.
    """
    epsilon = 1e-6
    scores = {}
    for word, count in era_count.items():
        if count < MIN_COUNT:
            continue
        p_era = count / max(n_era, 1)
        p_bg = background_count.get(word, 0) / max(n_bg, 1)
        scores[word] = p_era / (p_bg + epsilon)
    return scores


# ── Polarity colour per word ──────────────────────────────────────────────────
def polarity_colors(era_df: pd.DataFrame, bg_df: pd.DataFrame) -> dict:
    """
    Color = RELATIVE polarity in this era vs. full sample.
    A word is teal (hawkish) if it appears proportionally MORE in hawkish
    chunks in this era than it does in the full sample average.
    This corrects for RoBERTa's systematic dovish bias.
    """

    def hawk_ratio(df):
        hawk = _count(df[df["score"] > THRESHOLD]["chunk"])
        dove = _count(df[df["score"] < -THRESHOLD]["chunk"])
        all_words = set(hawk) | set(dove)
        return {
            w: (hawk.get(w, 0) - dove.get(w, 0))
            / max(hawk.get(w, 0) + dove.get(w, 0), 1)
            for w in all_words
        }

    era_ratio = hawk_ratio(era_df)
    bg_ratio = hawk_ratio(bg_df)

    colors = {}
    for w, er in era_ratio.items():
        bg = bg_ratio.get(w, 0)
        delta = er - bg  # positive = more hawkish in this era than average
        if delta > 0.1:
            colors[w] = f"hsl(170,60%,40%)"  # teal = relatively hawkish
        elif delta < -0.1:
            colors[w] = f"hsl(5,65%,42%)"  # red  = relatively dovish
        else:
            colors[w] = "hsl(0,0%,52%)"  # grey = consistent with average
    return colors


def make_color_func(word_colors: dict):
    def color_func(word, **kw):
        return word_colors.get(word, "hsl(0,0%,55%)")

    return color_func


# ── Main figure ───────────────────────────────────────────────────────────────
def plot_fig_3_3(save: bool = True) -> plt.Figure:
    """Two-row comparison: RoBERTa (top) vs FinBERT (bottom), all 5 eras."""
    models = [("roberta", "CentralBankRoBERTa"), ("finbert", "FinBERT")]
    era_list = list(ERAS.items())
    n_eras = len(era_list)

    fig = plt.figure(figsize=(14, 7.2), constrained_layout=False)
    fig.subplots_adjust(bottom=0.08, top=0.94, hspace=0.35, wspace=0.04)
    gs = gridspec.GridSpec(2, n_eras, figure=fig, hspace=0.35, wspace=0.04)

    for row, (model, model_label) in enumerate(models):
        print(f"Processing {model}...")
        df = load_data(model)

        for col, (era_label, (start, end)) in enumerate(era_list):
            era_df = df[(df["date"] >= start) & (df["date"] <= end)]
            era_cnt = _count(era_df["chunk"])
            bg_cnt = _count(df["chunk"])
            n_era = len(era_df)
            n_bg = len(df)

            rel = relative_freq(era_cnt, bg_cnt, n_era, n_bg)
            wcols = polarity_colors(era_df, df)

            ax = fig.add_subplot(gs[row, col])
            ax.axis("off")

            if rel:
                cloud = WordCloud(
                    width=420,
                    height=280,
                    background_color="white",
                    color_func=make_color_func(wcols),
                    max_words=50,
                    prefer_horizontal=0.8,
                    collocations=False,
                    min_font_size=7,
                ).generate_from_frequencies(rel)
                ax.imshow(cloud, interpolation="bilinear", aspect="auto")

            # Era labels on top row only
            if row == 0:
                ax.set_title(
                    era_label, fontsize=8.5, fontfamily="serif", pad=4, color="#222222"
                )

            # Model labels on left column only
            if col == 0:
                ax.set_ylabel(
                    model_label,
                    fontsize=9,
                    fontfamily="serif",
                    rotation=90,
                    labelpad=6,
                    va="center",
                    color="#333333",
                )
                ax.yaxis.set_label_position("left")

            # border
            for spine in ["top", "bottom", "left", "right"]:
                ax.spines[spine].set_visible(True)
                ax.spines[spine].set_color("#dddddd")
                ax.spines[spine].set_linewidth(0.6)
            ax.set_xticks([])
            ax.set_yticks([])

    # legend
    from matplotlib.patches import Patch

    legend_elements = [
        Patch(facecolor="#2b9b8a", label="Relatively hawkish in this era"),
        Patch(facecolor="#b94a3d", label="Relatively dovish in this era"),
        Patch(facecolor="#888888", label="Balanced / consistent with average"),
    ]
    fig.legend(
        handles=legend_elements,
        loc="lower center",
        ncol=3,
        fontsize=8.5,
        framealpha=0.9,
        bbox_to_anchor=(0.5, 0.01),
    )

    if save:
        path = OUTPUT_DIR / "fig_comparison.pdf"
        fig.savefig(path, bbox_inches="tight")
        print(f"Saved → {path}")
    return fig


# ── Entry point ───────────────────────────────────────────────────────────────
if __name__ == "__main__":
    plot_fig_3_3()
    plt.show()
