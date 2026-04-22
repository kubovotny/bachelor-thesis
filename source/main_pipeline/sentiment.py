import pandas as pd
from chunker import STATEMENTS_DIR
from typing import Literal

def label_formatter(x,part:Literal["QA","IS"]):
    return {
            "MONETARY_POLICY_AND_INFLATION": f"{part}_MP",
            "ECONOMIC_PERFORMANCE": f"{part}_EP",
            "FISCAL_AND_STRUCTURAL": f"{part}_FS",
            "OTHER_IRRELEVANT": f"{part}_OI",
        }[x]

def return_sentiment_data(
    MODEL="finbert",
    just_answers=True,
):
    agg_qa = pd.read_csv(f"{STATEMENTS_DIR}/sentiment/{MODEL}/agg_qa_labeled.csv")
    agg_intro = pd.read_csv(f"{STATEMENTS_DIR}/sentiment/{MODEL}/agg_intro_labeled.csv")
    agg_intro["label"] = agg_intro["label"].apply(label_formatter,part="IS")
    agg_qa["label"] = agg_qa["label"].apply(label_formatter,part="QA")

    if just_answers:
        agg_qa = agg_qa[agg_qa["is_question"] == False]
        agg_qa.drop(columns=["is_question"], inplace=True)
    else:
        agg_intro["is_question"] == False

    df_agg = pd.concat([agg_intro, agg_qa])
    # Preklopíme tabuľku (pivot)
    df_pivot = df_agg.pivot(
        index="date", columns="label", values=["mean", "max", "min", "std"]
    )

    # Zlúčime viacúrovňové názvy stĺpcov do jedného (napr. 'mean_MONETARY_POLICY_AND_INFLATION')
    df_pivot.columns = [f"{col[0]}_{col[1]}" for col in df_pivot.columns]
    df_pivot = df_pivot.reset_index()

    df_pivot.fillna(df_pivot.mean(), inplace=True)

    df_pivot["diff_sentiment"] = df_pivot["mean_IS_MP"] - df_pivot["mean_QA_MP"]
    df_pivot["date"] = pd.to_datetime(df_pivot["date"])
    return df_pivot


if __name__ == "__main__":
    print(return_sentiment_data())
