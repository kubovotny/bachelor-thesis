from chunker import STATEMENTS_DIR
from classifier import get_sentiment, calculate_sentiment
from typing import Literal, List
import pandas as pd


MODEL_SELECTION = ["finbert", "roberta"]


def day_sentiment_maker(
    model: Literal["finbert", "roberta"],
    part: Literal["intro", "qa"] = "intro",
    with_labels: bool = True,
    source_data: pd.DataFrame | None = None,
):
    grouping_columns = ["date"]
    if with_labels:
        ending = "_labeled"
        grouping_columns.append("label")
    else:
        ending = ""
    if part == "qa":
        grouping_columns.append("is_question")

    if source_data is None:
        source_data = pd.read_csv(
            f"{STATEMENTS_DIR}/sentiment/{model}/chunk_{part}_labeled.psv", sep="|"
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
            f"{STATEMENTS_DIR}/labeled_{part}.psv", sep="|", index_col="statement_id"
        )

    if type(model_selection) is list:
        for model in model_selection:
            chunked_df = chunk_sentiment_maker(model, part, chunked_df)
        return chunked_df

    model = model_selection
    chunked_df["sentiment"] = get_sentiment(list(chunked_df["chunk"]), model)
    chunked_df["score"] = chunked_df["sentiment"].apply(calculate_sentiment)

    chunked_df.drop(columns=["sentiment"]).to_csv(
        f"{STATEMENTS_DIR}/sentiment/{model}/chunk_{part}.psv", sep="|"
    )
    day_sentiment_maker(model, part, source_data=chunked_df)
    return chunked_df


if __name__ == "__main__":
    # WHOLE MAKING
    # for part in ["intro", "qa"]:
    #     chunk_sentiment_maker(MODEL_SELECTION)
    for model in MODEL_SELECTION:
        day_sentiment_maker(model,"qa",False)

# TOPIC MODELLING:
# 2 témy:
# 1. MP: úrokové sadzby, menová politika, inflácia
# 2. EP: nezamestnanosť, economic outlook


# Školiteľ je preč: 22-23.4; 29.4-2.5
