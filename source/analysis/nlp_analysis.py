import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from ..nlp.chunker import paragraphs_qa, paragraphs_intro, chunk_intro, chunk_qa
from .. import OUTPUT
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
qa_par = paragraphs_qa("scraped_v2")
qa_par["part"] = "qa"
is_par = paragraphs_intro("scraped_v2")
is_par["part"] = "is"
paragraphs = (
    pd.concat([qa_par, is_par])
    .sort_values(["date", "part"])
    .drop(columns=["is_question", "part"])
)

paragraphs_lengths = paragraphs.text.str.split().str.len()
paragraphs["length"] = paragraphs_lengths
print("========= PARAGRAPHS =========")
print(paragraphs.describe())

qa_chunk = chunk_qa("scraped_v2",200)
qa_chunk["part"] = "qa"
is_chunk = chunk_intro("scraped_v2",200)
is_chunk["part"] = "is"
print("========= CHUNKS =========")
chunks = (
    pd.concat([qa_chunk, is_chunk])
    .sort_values(["date", "part"])
    .drop(columns=["is_question", "part", "qa_len"])
)
chunk_lengths = chunks.chunk.str.split().str.len()
chunks["length"] = chunk_lengths
print(chunks.describe())

# 1. Spravíme si kópie dát, aby sme si nepokazili originály
original_clipped = np.clip(paragraphs_lengths, a_min=None, a_max=250)
processed_clipped = np.clip(chunk_lengths, a_min=None, a_max=250)

# 2. Nastavíme intervaly (bins) - posledný končí presne na 250
moje_intervaly = np.arange(0, 260, 10)

plt.figure(figsize=(10, 6))

# 3. Vykreslíme to
sns.histplot(
    original_clipped,
    bins=moje_intervaly,
    color="grey",
    alpha=1,
    label="Original Paragraphs",
)
sns.histplot(
    processed_clipped,
    bins=moje_intervaly,
    color="#4e79a7",
    alpha=0.5,
    label="Processed Segments",
    edgecolor="black",
)

# 4. Úprava osi X, aby bolo jasné, že posledný bin je "špeciálny"
plt.xticks(
    list(range(0, 250, 50)) + [250], [str(x) for x in range(0, 250, 50)] + ["250+"]
)

plt.axvline(30, color="red", linestyle="--", label="Min threshold (30 words)")
plt.axvline(200, color="darkgreen", linestyle="--", label="Max threshold (200 words)")

plt.xlabel(
    "Word count per segment",
)
plt.ylabel(
    "Frequency",
)
plt.title(
    "Distribution of segment lengths after processing",
)
plt.legend()
plt.grid(axis="y", alpha=0.5)
plt.savefig(f'{OUTPUT}/data/segment_distribution.pdf', dpi=300, bbox_inches='tight')
plt.show()
