# diagnose_sentiment.py
import pandas as pd
from ..data.queries import return_sentiment

df = return_sentiment(word_limit=200, with_label=False)
df["year"] = df["date"].dt.year
df["era"] = pd.cut(df["year"], bins=[1997, 2007, 2013, 2019, 2021, 2026],
                   labels=["Pre-GFC", "Easing", "ZLB", "COVID", "Hiking"])

# 1. Per-era distribution by model. Verify the flip story.
print(df.groupby(["era", "sentiment_model"])["score"]
        .agg(["count", "mean", "std",
              lambda s: (s > 0).mean()]).round(3))

# 2. Per-conference correlation between FinBERT and RoBERTa.
wide = df.groupby(["date", "sentiment_model"])["score"].mean().unstack()
wide["year"] = wide.index.year
print(wide.groupby(pd.cut(wide["year"], bins=[1997,2013,2026], labels=["Pre-ZLB","Post-ZLB"]))
        [["finbert","roberta"]].corr())