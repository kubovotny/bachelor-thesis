import time
from .classifier import get_sentiment, calculate_sentiment
from typing import Literal, List, Dict
import pandas as pd
from ..data.queries import (
    insert_sentiments,
    return_chunks,
    return_limits,
    CHUNK_LIMIT_TYPE,
    clear_sentiment_for_limit_model,
)

MODEL_SELECTION: List[Literal["finbert", "roberta"]] = ["finbert", "roberta"]
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


def chunk_sentiment_maker(
    model: Literal["finbert", "roberta"],
    word_limit: CHUNK_LIMIT_TYPE | None = None,
    sample_size: int | None = None,
    save_to_db: bool = False,
    apply_divisor=False,
) -> None | pd.DataFrame:
    if word_limit is not None:
        start = time.time()
        chunked_df = return_chunks(word_limit)
        if isinstance(sample_size, int):
            chunked_df = chunked_df.sample(sample_size, random_state=42)
        chunked_df["sentiment"] = pd.Series(
            get_sentiment(list(chunked_df["chunk"]), model)
        ).to_list()
        chunked_df["score"] = chunked_df["sentiment"].apply(
            calculate_sentiment, apply_divisor=apply_divisor
        )
        chunked_df["model_id"] = 1 if model == "finbert" else 2
        if save_to_db:
            clear_sentiment_for_limit_model(word_limit, model)
            insert_sentiments(df=chunked_df)
        print(f"LIMIT {word_limit} - TIME: {time.time() - start}")
        return chunked_df
    limits = return_limits()
    for word_limit in limits:
        chunk_sentiment_maker(model, word_limit)


if __name__ == "__main__":
    # WHOLE MAKING
    # for model in MODEL_SELECTION:
    print(
        chunk_sentiment_maker(
            MODEL_SELECTION[0], word_limit=100, save_to_db=False, sample_size=10
        )
    )
    # print(return_sentiment_agg(False))

# TOPIC MODELLING:
# 2 témy:
# 1. MP: úrokové sadzby, menová politika, inflácia
# 2. EP: nezamestnanosť, economic outlook


# Školiteľ je preč: 22-23.4; 29.4-2.5
