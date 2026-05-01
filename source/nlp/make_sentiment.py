from .. import STATEMENTS_DIR
from .classifier import get_sentiment, calculate_sentiment
from typing import Literal, List, Dict
import pandas as pd
from ..data.connection import insert_sentiments

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


def day_sentiment_maker(
    model: Literal["finbert", "roberta"],
    part: Literal["intro", "qa"] = "intro",
    with_labels: bool = True,
    source_data: pd.DataFrame | None = None,
):
    grouping_columns: List["str"] = ["date"]
    ending: str
    if with_labels:
        ending = "_labeled"
        grouping_columns.append("label")
    else:
        ending = ""
    if part == "qa":
        grouping_columns.append("is_question")

    if source_data is None:
        source_data = pd.read_csv(
            f"{STATEMENTS_DIR}/sentiment/{model}/chunk_{part}_labeled.psv",
            sep="|",
            dtype=QA_COLUMNS if part == "qa" else INTRO_COLUMNS,
            parse_dates=["date"],
        )

    data = source_data.groupby(grouping_columns).agg(
        {"score": ["min", "mean", "max", "std"]}
    )
    data.columns = data.columns.droplevel(0)
    data = data.reset_index().set_index("date")
    data.columns.name = None
    data.to_csv(f"{STATEMENTS_DIR}/sentiment/{model}/agg_{part}{ending}.csv")


def chunk_sentiment_maker(
    model_selection: (
        Literal["finbert", "roberta"] | List[Literal["finbert", "roberta"]]
    ),
    part: Literal["intro", "qa"] = "intro",
    chunked_df: pd.DataFrame | None = None,
):
    if chunked_df is None:
        chunked_df = pd.read_csv(
            f"{STATEMENTS_DIR}/labeled_{part}.psv",
            sep="|",
            index_col="statement_id",
            parse_dates=["date"],
        )

    if type(model_selection) is list:
        for model in model_selection:
            chunked_df = chunk_sentiment_maker(model, part, chunked_df)
            chunked_df.rename(
                columns={"sentiment": f"{model}_sentiment", "score": f"{model}_score"}
            )
        return chunked_df

    model: Literal["finbert", "roberta"] = model_selection
    chunked_df["sentiment"] = get_sentiment(list(chunked_df["chunk"]), model)
    chunked_df["score"] = chunked_df["sentiment"].apply(calculate_sentiment)

    insert_sentiments(chunked_df)
    day_sentiment_maker(model, part, source_data=chunked_df)
    return chunked_df


if __name__ == "__main__":
    # WHOLE MAKING
    # for part in ["intro", "qa"]:
    #     chunk_sentiment_maker(MODEL_SELECTION)
    for model in MODEL_SELECTION:
        day_sentiment_maker(model, "qa", False)

# TOPIC MODELLING:
# 2 témy:
# 1. MP: úrokové sadzby, menová politika, inflácia
# 2. EP: nezamestnanosť, economic outlook


# Školiteľ je preč: 22-23.4; 29.4-2.5
