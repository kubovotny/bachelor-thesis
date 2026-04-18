from classifier import label_paragraph, get_sentiment
from chunker import chunk_qa, STATEMENTS_DIR
import pandas as pd

data = chunk_qa("scraped_v2").drop(columns="press")
data["chunk_len"] = data["chunk"].str.split().str.len().apply(int)
print(
    data.groupby(["chunk_len", "is_question"])["chunk"]
    .count()[:20]
    .groupby("is_question")
    .cumsum()
)
print(data["chunk"])
df = data.query("chunk_len <=5")[["date", "chunk"]]
df.to_csv(
    f"{STATEMENTS_DIR}/qa_short_answers.csv", sep="|"
)

print(df)