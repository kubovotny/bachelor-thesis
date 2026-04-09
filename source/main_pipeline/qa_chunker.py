import pandas as pd
from chunker import STATEMENTS_DIR
from functions import qa_splitter, get_sentiment, calculate_sentiment
from chunker import process_long_paragraph, make_percentile
from typing import List

FILENAME = "scraped_v2"
data: pd.DataFrame = pd.read_csv(f"{STATEMENTS_DIR}/{FILENAME}.csv", sep="|")
data.drop(columns=["Unnamed: 0", "url", "press"], inplace=True)
data.index.name = "statement_id"


data["qa_paragraphs"] = data["qa"].str.split("\t")


def qa_multiple_proccesser(paragraphs: List):
    if type(paragraphs) is float:
        return paragraphs
    qa_proccessed = []
    for qa_paragraph in paragraphs:
        qa_proccessed.append(qa_splitter(qa_paragraph))
    return qa_proccessed


data["QA_processed"] = data["qa_paragraphs"].apply(qa_multiple_proccesser)
data.drop(columns=["qa", "qa_paragraphs"], inplace=True)
data.dropna(inplace=True)

# check_edge = 280
# for i,row in data.iterrows():
#     if i < check_edge:
#         continue
#     print(row["date"])
#     for QA_p in row["QA_processed"]:
#         print(10 * "\t" if not QA_p["is_question"] else "", QA_p["text"][:50])
#     if i >= check_edge + 10:
#         break

df_with_qa = data.explode("QA_processed")
df_with_qa["is_question"] = df_with_qa["QA_processed"].apply(
    lambda x: x["is_question"]
)
df_with_qa["text"] = df_with_qa["QA_processed"].apply(
    lambda x: x["text"]
)
df_with_qa.drop(columns="QA_processed", inplace=True)
# df_with_qa["len"] = df_with_qa.text.str.split(" ").str.len()

# print(df_with_qa.query("len > 200"))

df_with_qa["chunk"] = df_with_qa["text"].apply(process_long_paragraph)
df_qa_chunked = make_percentile(df_with_qa.explode("chunk").drop(columns=["text"]))

print(df_qa_chunked)
df_qa_chunked["sentiment"] = get_sentiment(list(df_qa_chunked["chunk"]))
df_qa_chunked["score"] = df_qa_chunked["sentiment"].apply(calculate_sentiment)

# new_df = {"year":[],"score_q":[],"score_a":[]}
df_qa_chunked.groupby(["date","is_question"])["score"].mean().to_csv(f"{STATEMENTS_DIR}/finbert_sentiment_qa.csv")
