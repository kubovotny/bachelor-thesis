from chunker import chunk_press, STATEMENTS_DIR
from classifier import get_sentiment, calculate_sentiment, get_topic

FILENAME = "scraped"

df_w_chunked = chunk_press(FILENAME)

MODEL_SELECTION = "roberta"

# df_w_chunked["topic"] = get_topic(list(df_w_chunked["chunk"]))
df_w_chunked["sentiment"] = get_sentiment(list(df_w_chunked["chunk"]), MODEL_SELECTION)
# for i, row in df_w_chunked.iterrows():
#     print(row["sentiment"])
df_w_chunked["score"] = df_w_chunked["sentiment"].apply(calculate_sentiment)

new_df = {"year": [], "score": []}
df_w_chunked.groupby("date")["score"].mean().to_csv(
    f"{STATEMENTS_DIR}/sentiment/{MODEL_SELECTION}_intro.csv"
)
