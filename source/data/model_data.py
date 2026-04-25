import pandas as pd
from .sentiment import return_sentiment_agg_data
from .market import return_market_data
from typing import Literal


def return_data(
    sentiment_model: Literal["finbert", "roberta"] = "finbert",
    with_label: bool = False,
    market_data: Literal[
        "Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"
    ] = "shocks_ecb_mpd_me_d.csv",
) -> pd.DataFrame:
    return pd.merge(
        return_market_data(market_data),
        return_sentiment_agg_data(sentiment_model, with_label=with_label),
        on="date",
    )

if __name__ == "__main__":
    print(return_data())