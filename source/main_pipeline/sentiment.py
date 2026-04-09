from chunker import chunk_it, STATEMENTS_DIR
from functions import get_sentiment, calculate_sentiment
import pandas as pd

FILENAME = "scraped"

df_w_chunked = chunk_it(FILENAME)


df_w_chunked["sentiment"] = get_sentiment(list(df_w_chunked["chunk"]))
df_w_chunked["score"] = df_w_chunked["sentiment"].apply(calculate_sentiment)

new_df = {"year":[],"score":[]}
df_w_chunked.groupby("date")["score"].mean().to_csv(f"{STATEMENTS_DIR}/finbert_sentiment.csv")