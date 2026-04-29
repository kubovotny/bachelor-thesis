import pandas as pd
from typing import Literal, Dict
from .. import STATEMENTS_DIR

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
    x: Literal[
        "MONETARY_POLICY_AND_INFLATION",
        "ECONOMIC_PERFORMANCE",
        "FISCAL_AND_STRUCTURAL",
        "OTHER_IRRELEVANT",
    ],
    part: Literal["QA", "IS"],
) -> str:
    return {
        "MONETARY_POLICY_AND_INFLATION": f"{part}_MP",
        "ECONOMIC_PERFORMANCE": f"{part}_EP",
        "FISCAL_AND_STRUCTURAL": f"{part}_FS",
        "OTHER_IRRELEVANT": f"{part}_OI",
    }[x]


def return_sentiment_agg_data(
    MODEL: Literal["finbert", "roberta"] = "finbert",
    just_answers: bool = True,
    with_label: bool = True,
) -> pd.DataFrame:
    ending = ""
    if with_label:
        ending = "_labeled"
    agg_qa: pd.DataFrame = pd.read_csv(
        f"{STATEMENTS_DIR}/sentiment/{MODEL}/agg_qa{ending}.csv",
        dtype={col: t for col, t in QA_COLUMNS.items() if col != "label" or with_label},
        parse_dates=["date"],
    )
    agg_intro: pd.DataFrame = pd.read_csv(
        f"{STATEMENTS_DIR}/sentiment/{MODEL}/agg_intro{ending}.csv",
        dtype={
            col: t for col, t in INTRO_COLUMNS.items() if col != "label" or with_label
        },
        parse_dates=["date"],
    )
    if with_label:
        agg_intro["label"] = agg_intro["label"].apply(label_formatter, part="IS")
        agg_qa["label"] = agg_qa["label"].apply(label_formatter, part="QA")
    else:
        agg_intro["label"] = "IS"
        agg_qa["label"] = "QA"
    if just_answers:
        agg_qa = agg_qa[agg_qa["is_question"] == False]
        agg_qa.drop(columns=["is_question"], inplace=True)
    else:
        agg_intro["is_question"] == False

    df_agg: pd.DataFrame = pd.concat([agg_intro, agg_qa])
    # Preklopíme tabuľku (pivot)
    df_pivot: pd.DataFrame = df_agg.pivot(
        index="date", columns="label", values=["mean", "max", "min", "std"]
    )

    # Zlúčime viacúrovňové názvy stĺpcov do jedného (napr. 'mean_QA_MP')
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_pivot.fillna(df_pivot.mean(), inplace=True)
    return df_pivot


def return_sentiment_chunk_data(
    MODEL: Literal["finbert", "roberta"] = "finbert",
    just_answers: bool = True,
    with_label: bool = True,
) -> pd.DataFrame:
    chunk_qa = pd.read_csv(
        f"{STATEMENTS_DIR}/sentiment/{MODEL}/chunk_qa_labeled.psv",
        sep="|",
        parse_dates=["date"],
    )
    chunk_qa["part"] = "QA"
    if just_answers:
        chunk_qa = chunk_qa[chunk_qa["is_question"] == False]
        chunk_qa.drop(columns=["is_question"], inplace=True)
    chunk_intro = pd.read_csv(
        f"{STATEMENTS_DIR}/sentiment/{MODEL}/chunk_intro_labeled.psv",
        sep="|",
        parse_dates=["date"],
    )
    chunk_intro["part"] = "IS"
    return pd.concat([chunk_intro, chunk_qa]).sort_values(["date","part", "chunk_id"]).reset_index().drop(columns="index")


if __name__ == "__main__":
    print(return_sentiment_chunk_data(with_label=False))
