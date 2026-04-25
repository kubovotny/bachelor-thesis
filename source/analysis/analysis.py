from ..data.market import return_market_data
from ..data.sentiment import return_sentiment_agg_data
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns


df_market = return_market_data("Dataset_EA-MPD.xlsx")
df_sentiment = return_sentiment_agg_data("roberta")
# The Final Merge! Spojíme to podľa dátumu
final_dataset = pd.merge(df_sentiment, df_market[['date', 'OIS_1M','OIS_3M', 'OIS_1Y']], on='date', how='inner')

df = final_dataset.sort_values('date')
def bootstrap_auc(y, x, n_boot=1000):
    aucs = []
    for _ in range(n_boot):
        y_res, x_res = resample(y, x)
        if len(np.unique(y_res)) < 2:
            continue
        aucs.append(roc_auc_score(y_res, x_res))
    return np.percentile(aucs, [5, 50, 95])
"""plt.figure(figsize=(14, 6))

# Vypočítame 6-mesačný kĺzavý priemer (rolling mean) pre vyhladenie trendu
# Predpokladáme, že tlačovky sú cca 8-krát do roka, takže window=4 je cca pol roka
# Namiesto starého kódu daj tento:
plt.plot(df['date'], df['std_IS_MP'].rolling(window=15).mean(), label='Monetary Policy (MP) Trend', color='firebrick', linewidth=2)
plt.plot(df['date'], df['std_QA_MP'].rolling(window=15).mean(), label='Economic Performance (EP) Trend', color='steelblue', linewidth=2)
# Pridáme čiaru pre "Neutrálny" sentiment

plt.axhline(y=0, color='black', linestyle='--', alpha=0.5)

plt.title('Vývoj sentimentu ECB v čase (6-mesačný kĺzavý priemer)', fontsize=14)
plt.xlabel('Rok', fontsize=12)
plt.ylabel('Dovish (-) <---> Hawkish (+)', fontsize=12)
plt.legend()
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.xticks([f"{year}-01-01" for year in range(1998,2026,2)], labels=range(1998,2026,2))

plt.show()

plt.figure(figsize=(10, 8))

# Vyberieme len stĺpce, ktoré nás reálne zaujímajú (vyhodíme FS a OI, lebo to je balast)
cols_of_interest = ['mean_QA_MP', 'max_QA_MP', 'min_QA_MP', 'std_QA_MP', 
                    'mean_IS_MP', 'max_IS_MP', 'min_IS_MP', 'std_IS_MP',
                    "diff_sentiment",
                    'OIS_1M', 'OIS_1Y']

# Vypočítame korelačnú maticu
corr_matrix = df[cols_of_interest].corr()

# Nakreslíme Heatmapu
sns.heatmap(corr_matrix, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f", linewidths=.5)

plt.title('Korelačná matica: Sentiment vs. Reakcia Trhu (OIS)', fontsize=14)
plt.tight_layout()
plt.show()

plt.figure(figsize=(10, 6))

# Vykreslíme body
sns.regplot(x=df['max_QA_MP'], y=df['OIS_1M'], 
            scatter_kws={'alpha':0.5, 'color': 'gray'}, 
            line_kws={'EA-MPDcolor': 'red', 'linewidth': 2})

plt.axhline(y=0, color='black', linestyle='--', alpha=0.3)
plt.axvline(x=0, color='black', linestyle='--', alpha=0.3)

plt.title('Vplyv najsilnejšieho MP signálu (Max Hawkish) na 1-mesačný OIS', fontsize=14)
plt.xlabel('Maximálny MP Sentiment (Dovish -> Hawkish)', fontsize=12)
plt.ylabel('Zmena OIS_1M (v bázických bodoch)', fontsize=12)
plt.grid(True, alpha=0.3)
plt.tight_layout()
plt.show()


# 1. Definujeme si tvoj prah (Threshold) - napríklad všetko medzi -0.3 a +0.3 považujeme za nudný šum
threshold = 0.3

# 2. Vyfiltrujeme dataset. Necháme len extrémne Hawkish alebo extrémne Dovish tlačovky
# Využijeme max_QA_MP pre Hawkish signály a min_QA_MP pre Dovish signály
df_filtered = df[(df['max_IS_MP'] > threshold) | (df['min_IS_MP'] < -threshold)].copy()

print(f"Pôvodný počet tlačoviek: {len(df)}")
print(f"Počet tlačoviek po vyhodení šumu: {len(df_filtered)}")

# 3. Pozrieme sa na novú, očistenú koreláciu
cols_of_interest = ['max_QA_MP', 'min_QA_MP', "max_IS_MP","min_IS_MP",'OIS_1M', 'OIS_1Y']
corr_matrix_filtered = df_filtered[cols_of_interest].corr("spearman")

# 4. Vykreslenie
plt.figure(figsize=(8, 6))
sns.heatmap(corr_matrix_filtered, annot=True, cmap='coolwarm', vmin=-1, vmax=1, fmt=".2f")
plt.title(f'Očistená Korelácia (Threshold > {threshold})', fontsize=14)
plt.tight_layout()
plt.show()"""

from sklearn.metrics import roc_curve, auc, roc_auc_score
import numpy as np
from sklearn.utils import resample


# 1. Krok: Definovanie "Chvostov" pre trh (Náš Target / Cieľ)
# Vypočítame 80. percentil (všetko nad toto je silný rast sadzieb)
target_col = 'OIS_1M'
hawkish_threshold = df[target_col].quantile(0.8) 

# Vytvoríme binárnu premennú: 1 ak OIS vyskočilo vysoko, 0 inak
df['Target_Hawkish_Tail'] = (df[target_col] > hawkish_threshold).astype(int)

# 2. Krok: Výber nášho prediktora (Sentimentu)
# Použijeme napríklad MAX sentiment z Intro (lebo to zvykne trhy najviac prekvapiť)
# Ak máš spojený stĺpec, použi napr. df['max_MP']
fig, axes = plt.subplots(8,4, figsize=(20,12))
results = {"sentiment column":[],"roc-auc":[]}
for i, column in enumerate(df_sentiment.columns[1:-1]):
    predictor = df[column] # Nahraď tvojím najsilnejším stĺpcom
    ax = axes[i%8][i//8]
    # 3. Krok: Výpočet ROC a AUC
    fpr, tpr, thresholds = roc_curve(df['Target_Hawkish_Tail'], predictor)
    ci = bootstrap_auc(df['Target_Hawkish_Tail'], predictor)
    print(column, target_col, ci)
    roc_auc = auc(fpr, tpr)
    results["sentiment column"].append(column)
    results["roc-auc"].append(roc_auc)
    ax.plot(fpr, tpr, color='firebrick', lw=2, label=f'ROC Krivka (AUC = {roc_auc:.2f})')


    ax.plot(fpr, tpr, color='firebrick', lw=2, label=f'ROC Krivka (AUC = {roc_auc:.2f})')
    ax.plot([0, 1], [0, 1], color='gray', lw=1, linestyle='--') # Čiara náhody

    ax.set_title(f"{column}", fontsize=14)
    ax.set_xlabel('FP', fontsize=12)
    ax.set_ylabel('TP', fontsize=12)
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
import re
df2 = pd.DataFrame(results)
# Tvoja úprava dát (zostáva nezmenená)
df2["roc-auc"] = df2["roc-auc"] - 0.5
df2["column"] = df2["sentiment column"].apply(lambda x: re.sub(re.compile("(min|max|std|mean)_"),"",x))
df2["measure"] = df2["sentiment column"].apply(lambda x: re.sub(re.compile("IS|QA|EP|_|MP|FS|OI"),"",x))
df2.drop(columns="sentiment column", inplace=True)

# 1. KROK: Preklopenie dát do matice (Pivot)
# Index = riadky (column), Columns = stĺpce (measure), Values = čísla v bunkách (roc-auc)
heatmap_data = df2.pivot_table(index="column", columns="measure", values="roc-auc")

# 2. KROK: Vykreslenie heatmapy
plt.figure(figsize=(10, 8))  # Nastavenie veľkosti grafu
sns.heatmap(heatmap_data, annot=True, cmap="coolwarm", fmt=".3f", linewidths=.5)

plt.title("ROC-AUC odchýlka od 0.5 (Sila signálu)")
plt.tight_layout()
plt.show()



