import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from ..data.sentiment import return_sentiment_agg_data

data1=return_sentiment_agg_data("finbert",with_label=False)
data1 = data1.rename(columns={c: f"finbert_{c}" for c in data1.columns if c != "date"})
data2=return_sentiment_agg_data("roberta",with_label=False)
data2 = data2.rename(columns={c: f"roberta_{c}" for c in data2.columns if c != "date"})
df = pd.merge(data1, data2, on="date")

# 1. Príprava dát
df = df.sort_values('date')
part1 = "IS"
part2 = "QA"
# Výpočet 12-mesačného kĺzavého priemeru (cca 8-12 zasadaní ECB)
# Použijeme QA stĺpce, tie bývajú zaujímavejšie
df[f'roberta_rolling_{part1}'] = df[f'roberta_mean_{part1}'].rolling(window=10, center=True).mean()
df[f'finbert_rolling_{part1}'] = df[f'finbert_mean_{part1}'].rolling(window=10, center=True).mean()

df[f'roberta_rolling_{part2}'] = df[f'roberta_mean_{part2}'].rolling(window=10, center=True).mean()
df[f'finbert_rolling_{part2}'] = df[f'finbert_mean_{part2}'].rolling(window=10, center=True).mean()

# 2. Graf
plt.figure(figsize=(14, 7))

# RoBERTa
plt.plot(df['date'], df[f'roberta_mean_{part1}'], color='red', alpha=0.15, label='_nolegend_')
plt.plot(df['date'], df[f'roberta_rolling_{part1}'], color='red', linewidth=2, label=f'RoBERTa (Rolling Mean {part1})')

plt.plot(df['date'], df[f'roberta_mean_{part2}'], color='brown', alpha=0.15, label='_nolegend_')
plt.plot(df['date'], df[f'roberta_rolling_{part2}'], color='brown', linewidth=2, label=f'RoBERTa (Rolling Mean {part2})')

# FinBERT
plt.plot(df['date'], df[f'finbert_mean_{part1}'], color='blue', alpha=0.15, label='_nolegend_')
plt.plot(df['date'], df[f'finbert_rolling_{part1}'], color='blue', linewidth=2, label=f'FinBERT (Rolling Mean {part1})')
# FinBERT
plt.plot(df['date'], df[f'finbert_mean_{part2}'], color='darkblue', alpha=0.15, label='_nolegend_')
plt.plot(df['date'], df[f'finbert_rolling_{part2}'], color='darkblue', linewidth=2, label=f'FinBERT (Rolling Mean {part2})')



# 3. Zvýraznenie kľúčových momentov (Validácia)
plt.axvspan('2008-09-01', '2009-12-31', color='gray', alpha=0.2, label='Finančná kríza')
plt.axvspan('2020-03-01', '2020-12-31', color='green', alpha=0.1, label='COVID-19')
plt.axvspan('2022-07-01', '2023-12-31', color='orange', alpha=0.1, label='Inflačné šoky (Hikes)')

# Estetika
plt.title(f"Porovnanie NLP modelov: Sentiment ECB v čase", fontsize=15)
plt.axhline(0, color='black', linestyle='--', alpha=0.5)
plt.ylabel("Sentiment Score")
plt.legend(loc='lower left')
plt.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("ecb_sentiment_timeline.png", dpi=300)
plt.show()


# Predpokladám, že dáta máš v 'df' a dátumy sú zoradené
# Výpočet nových metrík
df['sentiment_spread'] = df['roberta_mean_QA'] - df['roberta_mean_IS']
df['spread_rolling'] = df['sentiment_spread'].rolling(window=10, center=True).mean()

df['roberta_uncertainty_rolling'] = df['roberta_std_QA'].rolling(window=10, center=True).mean()
df['finbert_uncertainty_rolling'] = df['finbert_std_QA'].rolling(window=10, center=True).mean()

# Nastavenie grafu (2 podgrafy pod sebou)
fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(14, 10), sharex=True)

# --- 1. PODGRAF: Divergencia (Spread) ---
ax1.axhline(0, color='black', linestyle='--', alpha=0.5)
ax1.plot(df['date'], df['sentiment_spread'], color='purple', alpha=0.15, label='_nolegend_')
ax1.plot(df['date'], df['spread_rolling'], color='purple', linewidth=2, label='Rozdiel (QA - IS)')
ax1.fill_between(df['date'], 0, df['spread_rolling'], where=(df['spread_rolling'] > 0), color='red', alpha=0.1, label='QA jastrabejšie ako IS')
ax1.fill_between(df['date'], 0, df['spread_rolling'], where=(df['spread_rolling'] < 0), color='blue', alpha=0.1, label='QA holubičejšie ako IS')

ax1.set_title("Rozdiel v sentimente: Odpovede na otázky (Q&A) vs. Úvodný prejav (IS)", fontsize=14)
ax1.set_ylabel("Rozdiel (Spread)")
ax1.legend(loc='upper left')

# --- 2. PODGRAF: Neistota (Standard Deviation) ---
ax2.plot(df['date'], df['roberta_std_QA'], color='orange', alpha=0.15, label='_nolegend_')
ax2.plot(df['date'], df['finbert_std_QA'], color='gray', alpha=0.15, label='_nolegend_')
ax2.plot(df['date'], df['roberta_uncertainty_rolling'], color='orange', linewidth=2, label='Neistota (Kĺzavý priemer Std. Odchýlky QA) - roberta')
ax2.plot(df['date'], df['finbert_uncertainty_rolling'], color='gray', linewidth=2, label='Neistota (Kĺzavý priemer Std. Odchýlky QA) - finbert')

ax2.set_title("Index komunikačnej neistoty ECB (Zmiešané signály v Q&A)", fontsize=14)
ax2.set_ylabel("Smerodajná odchýlka (Std)")
ax2.set_xlabel("Rok")
ax2.legend(loc='upper left')

# Označenie krízy (napr. COVID) pre oba grafy
for ax in [ax1, ax2]:
    ax.axvspan('2020-03-01', '2020-12-31', color='gray', alpha=0.1)
    ax.axvspan('2008-09-01', '2009-12-31', color='gray', alpha=0.2, label='Finančná kríza')
    ax.grid(axis='y', alpha=0.3)

plt.tight_layout()
plt.savefig("ecb_communication_dynamics.png", dpi=300)
plt.show()