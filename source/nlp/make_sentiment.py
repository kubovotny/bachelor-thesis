import time
from .classifier import get_sentiment, calculate_sentiment
from typing import Literal, List, Dict
import pandas as pd
from ..data.connection import (
    insert_sentiments,
    return_chunks,
    return_sentiment,
    return_limits,
)

MODEL_SELECTION = ["finbert", "roberta"]
QA_COLUMNS: Dict[str, str] = {
    "statement_id": "int",
    "is_question": "bool",
    "chunk": "str",
    "chunk_id": "int",
    "chunk_percentile": "float64",
    "label": "str",
    "prob": "float64",
    "score": "float64",
}

INTRO_COLUMNS: Dict[str, str] = {
    "statement_id": "int",
    "chunk": "str",
    "chunk_id": "int",
    "chunk_percentile": "float64",
    "label": "str",
    "prob": "float64",
    "score": "float64",
}


def return_sentiment_agg(with_topic: bool = True):
    data = return_sentiment(with_topic)
    print(data)
    grouping_columns: List["str"] = ["date", "part"]
    if with_topic:
        grouping_columns.append("topic")
    grouping_columns += ["is_question", "sentiment_model"]
    data_agg = data.groupby(grouping_columns).agg(
        {"score": ["min", "mean", "max", "std"]}
    )
    data_agg.columns = data_agg.columns.droplevel(0)
    data_agg = data_agg.reset_index().set_index("date")
    data_agg.columns.name = None
    return data_agg


def chunk_sentiment_maker(
    model: Literal["finbert", "roberta"],
    limit: Literal[50, 100, 150, 200, 250, 300, 350] | None = None,
) -> None | pd.DataFrame:
    if limit is not None:
        start = time.time()
        chunked_df = return_chunks(limit)
        chunked_df["sentiment"] = get_sentiment(list(chunked_df["chunk"]), model)
        chunked_df["score"] = chunked_df["sentiment"].apply(calculate_sentiment)
        chunked_df["model_id"] = 1 if model == "finbert" else 2
        insert_sentiments(df=chunked_df)
        print(f"LIMIT {limit} - TIME: {time.time() - start}")
        return chunked_df
    limits = return_limits()
    for limit in limits:
        chunk_sentiment_maker(model, limit)


if __name__ == "__main__":
    # WHOLE MAKING
    for model in MODEL_SELECTION:
        chunk_sentiment_maker(model)
    # print(return_sentiment_agg(False))

# TOPIC MODELLING:
# 2 témy:
# 1. MP: úrokové sadzby, menová politika, inflácia
# 2. EP: nezamestnanosť, economic outlook


# Školiteľ je preč: 22-23.4; 29.4-2.5
