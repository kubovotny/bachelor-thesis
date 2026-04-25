import pandas as pd
import matplotlib.pyplot as plt
from ..data.sentiment import return_sentiment_chunk_data

df = return_sentiment_chunk_data("roberta", with_label=False)

# Predpokladám, že dáta máš v df
df["year"] = df["date"].dt.year
print(df.columns)

# 1. Spočítame počet chunkov pre každú tému v danom roku
topic_counts = df.groupby(["year", "label"]).size().unstack(fill_value=0)

# 2. Normalizujeme na percentá (aby každý rok mal 100 %)
topic_pct = topic_counts.div(topic_counts.sum(axis=1), axis=0) * 100

# 3. Vykreslenie grafu
plt.figure(figsize=(15, 7))
topic_pct.plot(kind="area", stacked=True, alpha=0.8, colormap="tab20", figsize=(15, 7))

plt.title(
    "Evolúcia naratívu ECB (1998 - 2025): Podiel tém na tlačových konferenciách",
    fontsize=16,
)
plt.xlabel("Rok", fontsize=12)
plt.ylabel("Podiel tém (%)", fontsize=12)
plt.legend(title="Téma (Label)", bbox_to_anchor=(1.05, 1), loc="upper left")

# Označenie kríz pre kontext
plt.axvspan(2008, 2009, color="gray", alpha=0.3, label="Finančná kríza")
plt.axvspan(2020, 2021, color="green", alpha=0.2, label="COVID-19")

plt.margins(x=0)
plt.tight_layout()
plt.savefig("ecb_topic_evolution.png", dpi=300)
plt.show()