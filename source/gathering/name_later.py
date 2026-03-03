import pandas as pd

df = pd.read_csv("statements.csv", sep="|", encoding="utf-8")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print(*df[df["qa"].isna()]["url"].to_list(), sep="\n")