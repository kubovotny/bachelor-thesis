import pandas as pd
from classifier import (
    clean_paragraph,
    process_long_paragraph,
    label_paragraph,
    wise_label_choose,
)
import matplotlib.pyplot as plt
import seaborn as sns
import numpy as np
import time

start_time = time.time()
STATEMENTS_DIR = "~/Documents/School/bachelor-thesis/source/statements"
POSTFIX = "_" "2022u"

data = pd.read_csv(f"{STATEMENTS_DIR}/sample_{POSTFIX}.csv", sep="|")
data["statement_id"] = data.iloc[:, 0]
data.drop(columns=["Unnamed: 0", "url"], inplace=True)
data.set_index("statement_id", inplace=True)

loading_time = time.time() - start_time

data["paragraph"] = data["press"].str.split("\t")
data["date"] = pd.to_datetime(data["date"])

df = data.explode("paragraph").drop(columns=["qa", "press"])
df["paragraph"] = df["paragraph"].apply(clean_paragraph)
df = df.query("paragraph != ''")

df["chunk"] = df["paragraph"].apply(process_long_paragraph)

df_w_chunked = df.explode("chunk").drop(columns=["paragraph"])
df_w_chunked["chunk_id"] = df_w_chunked.groupby("statement_id").cumcount()
df_w_chunked["chunk_percentile"] = (
    df_w_chunked["chunk_id"] / df_w_chunked.groupby("statement_id")["chunk_id"].max()
)

preparing_time = time.time() - start_time
# print(df_w_chunked["chunk"].str.split().str.len().describe())
# indices = list(df_w_chunked["statement_id"])
df_w_chunked["labels"] = label_paragraph(list(df_w_chunked["chunk"]))
df_w_chunked["label"] = df_w_chunked["labels"].apply(lambda x: wise_label_choose(**x))
labelling_time = time.time() - start_time
# # for i, row in df_w_chunked.iterrows():
# #     print(i, row["chunk"])
# #     print(row["label"])
# #     print()


if __name__ == "__main__":
    print(df["paragraph"].str.split().str.len().describe())
    print(10 * "*" + "TIMING" + 10 * "*")
    print("LOADING TIME:\t", loading_time)
    print("PREPARING_TIME:\t", preparing_time - loading_time)
    print("LABELLING_TIME:\t", labelling_time - preparing_time)
    print("FULL_TIME:\t\t", time.time() - start_time)
    df_w_chunked.drop(columns=["chunk_percentile","labels"]).to_csv(
        f"{STATEMENTS_DIR}/labeled_chunks{POSTFIX}.csv"
    )
    sns.lineplot(data=df_w_chunked, x="chunk_percentile", y="label", hue="date")
    plt.show()
