from chunker import STATEMENTS_DIR
from classifier import get_sentiment, calculate_sentiment
from chunker import chunk_qa

FILENAME = "scraped_v2"
df_qa_chunked = chunk_qa("scraped_v2")
print(df_qa_chunked)

# df_qa_chunked["sentiment"] = get_sentiment(list(df_qa_chunked["chunk"]))
# df_qa_chunked["score"] = df_qa_chunked["sentiment"].apply(calculate_sentiment)

# # new_df = {"year":[],"score_q":[],"score_a":[]}
# df_qa_chunked.groupby(["date","is_question"])["score"].mean().to_csv(f"{STATEMENTS_DIR}/finbert_sentiment_qa.csv")
