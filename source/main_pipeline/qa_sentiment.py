from chunker import STATEMENTS_DIR
from classifier import get_sentiment, calculate_sentiment
from chunker import chunk_qa

FILENAME = "scraped_v2"
MODEL_SELECTION = "roberta"

df_qa_chunked = chunk_qa("scraped_v2")
# print(df_qa_chunked)

df_qa_chunked["sentiment"] = get_sentiment(list(df_qa_chunked["chunk"]))
df_qa_chunked["score"] = df_qa_chunked["sentiment"].apply(calculate_sentiment)


df_qa_chunked.groupby(["date", "is_question"]).agg(
    {"score": ["min", "mean", "max", "sd"]}
).to_csv(f"{STATEMENTS_DIR}/sentiment/{MODEL_SELECTION}_qa.csv")


# TOPIC MODELLING:
# 2 témy:
# 1. MP: úrokové sadzby, menová politika, inflácia
# 2. EP: nezamestnanosť, economic outlook


# Školiteť je preč: 22-23.4; 29.4-2.5
