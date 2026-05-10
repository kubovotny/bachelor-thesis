import pandas as pd
from .sentiment import return_sentiment_agg_pivot
from .market import return_market_data
from typing import Literal, List

LIST_OF_MARKET_DATA = Literal[
    "shocks_ecb_mpd_me_d.csv", "Dataset_EA-MPD.xlsx", "ECB Money Market.xlsx", "all"
]


def return_data(
    with_label: bool = False,
    market_data: (
        List[LIST_OF_MARKET_DATA] | LIST_OF_MARKET_DATA
    ) = "shocks_ecb_mpd_me_d.csv",
    word_limit: Literal[50, 100, 150, 200, 250, 300, 350] = 200,
) -> pd.DataFrame:
    return pd.merge(
        return_market_data(market_data),
        return_sentiment_agg_pivot(with_label=with_label, word_limit=word_limit),
        on="date",
    )


if __name__ == "__main__":
    print(return_data(word_limit=50))
    print(return_data(word_limit=350))
