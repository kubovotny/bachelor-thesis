import pandas as pd
from typing import Literal, Dict, List
from ..data.connection import return_sentiment, CHUNK_LIMITS

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
QA_OPTIONS = Literal["both_divided", "both_together", "just_answers", "just_questions"]


def label_formatter(row: pd.Series, question_labelled: bool = False) -> str:
    label_formatted = row["sentiment_model"]
    if "part" in row.index:
        label_formatted += f"_{row['part']}"
        if row["part"] == "QA" and question_labelled:
            label_formatted += f"_{'question' if row['is_question'] else 'answer'}"
    elif question_labelled:
        label_formatted += f"_{'question' if row['is_question'] else 'answer'}"
    if "topic" in row.index:
        label_formatted += (
            "_"
            + {
                "MONETARY_POLICY_AND_INFLATION": "MP",
                "ECONOMIC_PERFORMANCE": "EP",
                "FISCAL_AND_STRUCTURAL": "FS",
                "OTHER_IRRELEVANT": "OI",
            }[row["topic"]]
        )
    return label_formatted


def return_sentiment_agg(
    with_label: bool = True,
    word_limit: CHUNK_LIMITS = 200,
    IS_QA_division: bool = True,
    qa_division: bool = True,
):
    data = return_sentiment_chunk_data(word_limit, with_label)
    grouping_columns: List["str"] = ["date"]
    if IS_QA_division:
        grouping_columns.append("part")
    if with_label:
        grouping_columns.append("topic")
    if qa_division:
        grouping_columns.append("is_question")
    grouping_columns += ["sentiment_model"]
    data_agg = data.groupby(grouping_columns).agg(
        {"score": ["min", "mean", "max", "std"]}
    )
    data_agg.columns = data_agg.columns.droplevel(0)
    data_agg = data_agg.reset_index()
    data_agg.columns.name = None
    return data_agg


def return_sentiment_agg_pivot(
    word_limit: CHUNK_LIMITS = 200,
    IS_QA_division: bool = True,
    qa_options: QA_OPTIONS = "just_answers",
    with_label: bool = True,
) -> pd.DataFrame:
    agg_data = return_sentiment_agg(
        with_label, word_limit, IS_QA_division, qa_options != "both_together"
    )

    agg_data["label"] = agg_data.apply(
        (lambda row: label_formatter(row, qa_options == "both_divided")),
        axis=1,
    )
    if qa_options.startswith("just"):
        agg_data = agg_data[agg_data["is_question"] == (qa_options == "just_questions")]
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
    limit_version: CHUNK_LIMITS = 200,
    with_label: bool = True,
) -> pd.DataFrame:
    return return_sentiment(limit_version, with_label)


if __name__ == "__main__":
    print(
        return_sentiment_agg_pivot(
            word_limit=50,
            with_label=False,
            qa_options="both_together",
            IS_QA_division=False,
        )
    )
