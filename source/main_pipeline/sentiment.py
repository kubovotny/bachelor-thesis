from chunker import chunk_press, STATEMENTS_DIR
from classifier import get_sentiment, calculate_sentiment, get_topic

FILENAME = "scraped"

df_w_chunked = chunk_press(FILENAME)


# df_w_chunked["topic"] = get_topic(list(df_w_chunked["chunk"]))
# df_w_chunked["sentiment"] = get_sentiment(list(df_w_chunked["chunk"]))
# df_w_chunked["score"] = df_w_chunked["sentiment"].apply(calculate_sentiment)

# new_df = {"year": [], "score": []}
# df_w_chunked.groupby("date")["score"].mean().to_csv(
#     f"{STATEMENTS_DIR}/finbert_sentiment.csv"
# )
