import pandas as pd
from .. import STATEMENTS_DIR, MARKET_DIR
import seaborn as sns
import matplotlib.pyplot as plt

FILENAME = "Dataset_EA-MPD.xlsx"
market_data = pd.read_csv(f"{MARKET_DIR}/shocks_ecb_mpd_me_d.csv")

def clean_date(date):
    if type(date) is not str:
        return date
    if "/" in date:
        date =  "-".join(date.split("/")[::-1])
    return pd.to_datetime(date)


market_data["date"] = market_data.date.apply(clean_date)
market_data = market_data.set_index("date")
market_data = market_data.dropna(how="all")

sentiment_data = pd.read_csv(f"{STATEMENTS_DIR}/finbert_sentiment.csv")
sentiment_data["date"] = sentiment_data["date"].apply(clean_date)
sentiment_data = sentiment_data.set_index("date").query("not score.isna()")

no_data_from_market = set(sentiment_data.index)- set(market_data.index)
# print(*sorted(no_data_from_market))

sentiment_data["dayofweek"] = sentiment_data.index.strftime("%w")


# ax = sns.barplot(data=data_press.groupby("dayofweek")["qa"].count().reset_index(),x="dayofweek",y="qa")
# for container in ax.containers:
#     ax.bar_label(container)
# plt.show()



# print(data)
df = pd.merge(sentiment_data, market_data, on='date', how='inner')
THRESHOLD = 0.5

df_filtered = df[df['z_score'].abs() > THRESHOLD].copy()
print(f"Počet konferencií pred filtrom: {len(df)}")
print(f"Počet konferencií so 'silným' sentimentom: {len(df_filtered)}")

korelacie = df_filtered[['z_score', 'pc1', 'MP_pm', 'STOXX50']].corr(method="spearman")
print("\nKorelačná matica:")
print(korelacie['z_score'])

from scipy.stats import pearsonr

def plot_relationship(data, x_col, y_col, title):
    # Vypočítame koreláciu a p-hodnotu
    corr, p_value = pearsonr(data[x_col], data[y_col])
    
    plt.figure(figsize=(8, 6))
    # sns.regplot vykreslí body a automaticky preloží lineárnu regresnú priamku
    sns.regplot(data=data, x=x_col, y=y_col, scatter_kws={'alpha':0.6}, line_kws={'color':'red'})
    
    plt.title(f"{title}\nPearson r: {corr:.3f} | p-value: {p_value:.3f}")
    plt.xlabel("Sentiment Score (-1 Negatívny do 1 Pozitívny)")
    plt.ylabel(f"Zmena {y_col} počas konferencie")
    plt.grid(True, linestyle='--', alpha=0.7)
    plt.axhline(0, color='black', linewidth=1) # Os Y=0 (nulová zmena trhu)
    plt.axvline(0, color='black', linewidth=1) # Os X=0 (neutrálny sentiment)
    plt.show()

# Vykreslíme vzťah pre pc1
# pc1 - surprise in the "policy indicator",
# ie the 1st principal component of the Monetary Event-window changes in overnight index swaps (OIS)
# with maturities 1-, 3-, 6-months and 1-year (Identifiers: OIS1M, OIS3M, OIS6M, OIS1Y)
plot_relationship(df_filtered, 'z_score', 'pc1', "Vplyv sentimentu na krátkodobé sadzby (pc1)")

# Vykreslíme vzťah pre Akcie (STOXX50)
plot_relationship(df_filtered, 'z_score', 'STOXX50', "Vplyv sentimentu na európske akcie (STOXX50)")