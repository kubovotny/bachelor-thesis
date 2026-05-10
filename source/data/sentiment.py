import pandas as pd
from typing import Literal, Dict, List
from ..data.connection import return_sentiment

INTRO_COLUMNS: Dict[str, str] = {
    "label": "str",
    "min": "float64",
    "mean": "float64",
    "max": "float64",
    "std": "float64",
}
QA_COLUMNS: Dict[str, str] = {
    "label": "str",
    "is_question": "bool",
    "min": "float64",
    "mean": "float64",
    "max": "float64",
    "std": "float64",
}


def label_formatter(
    part: Literal["QA", "IS"],
    model: Literal["finbert", "roberta"],
    x: (
        Literal[
            "MONETARY_POLICY_AND_INFLATION",
            "ECONOMIC_PERFORMANCE",
            "FISCAL_AND_STRUCTURAL",
            "OTHER_IRRELEVANT",
        ]
        | None
    ) = None,
) -> str:
    if x is None:
        return f"{model}_{part}"
    return {
        "MONETARY_POLICY_AND_INFLATION": f"{model}_{part}_MP",
        "ECONOMIC_PERFORMANCE": f"{model}_{part}_EP",
        "FISCAL_AND_STRUCTURAL": f"{model}_{part}_FS",
        "OTHER_IRRELEVANT": f"{model}_{part}_OI",
    }[x]


def return_sentiment_agg(
    with_label: bool = True,
    word_limit: Literal[50, 100, 150, 200, 250, 300, 350] = 200,
):
    data = return_sentiment(word_limit, with_label)
    grouping_columns: List["str"] = ["date", "part"]
    if with_label:
        grouping_columns.append("topic")
    grouping_columns += ["is_question", "sentiment_model"]
    data_agg = data.groupby(grouping_columns).agg(
        {"score": ["min", "mean", "max", "std"]}
    )
    data_agg.columns = data_agg.columns.droplevel(0)
    data_agg = data_agg.reset_index()
    data_agg.columns.name = None
    return data_agg


def return_sentiment_agg_pivot(
    just_answers: bool = True,
    with_label: bool = True,
    word_limit: Literal[50, 100, 150, 200, 250, 300, 350] = 200,
) -> pd.DataFrame:
    agg_data = return_sentiment_agg(with_label, word_limit)
    if with_label:
        agg_data["label"] = agg_data[["topic", "part", "sentiment_model"]].apply(
            (lambda x: label_formatter(x["part"], x["sentiment_model"], x["topic"])),
            axis=1,
        )
    else:
        agg_data["label"] = agg_data[["part", "sentiment_model"]].apply(
            (lambda x: label_formatter(x["part"], x["sentiment_model"])),
            axis=1,
        )
    if just_answers:
        agg_data = agg_data[agg_data["is_question"] == False]
        agg_data.drop(columns=["is_question"], inplace=True)
    # Preklopíme tabuľku (pivot)

    df_pivot: pd.DataFrame = agg_data.pivot(
        index="date", columns="label", values=["max", "mean", "min", "std"]
    )
    # Zlúčime viacúrovňové názvy stĺpcov do jedného (napr. 'mean_QA_MP')
    df_pivot.columns = [f"{col[1]}_{col[0]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_pivot.fillna(df_pivot.mean(), inplace=True)
    return df_pivot


def return_sentiment_chunk_data(
    limit_version: Literal[50, 100, 150, 200, 250, 300, 350] = 200,
    with_label: bool = True,
) -> pd.DataFrame:
    return return_sentiment(limit_version, with_label)


if __name__ == "__main__":
    print(return_sentiment_agg_pivot(with_label=True))
