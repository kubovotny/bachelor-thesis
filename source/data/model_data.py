import pandas as pd
from .sentiment import return_sentiment_agg_pivot, QA_OPTIONS, CHUNK_LIMIT_TYPE
from .market import return_market_data, LIST_OF_MARKET_DATA
from typing import List, Literal


def return_data(
    market_data: (
        List[LIST_OF_MARKET_DATA] | LIST_OF_MARKET_DATA
    ) = "shocks_ecb_mpd_me_d.csv",
    word_limit: CHUNK_LIMIT_TYPE = 200,
    IS_QA_division: bool = True,
    qa_options: QA_OPTIONS = "just_answers",
    with_label: bool = False,
    topic_model: Literal["moritz", "facebook"] = "facebook",
    drop_irrelevant: bool = False,
) -> pd.DataFrame:
    return pd.merge(
        return_market_data(market_data),
        return_sentiment_agg_pivot(
            word_limit=word_limit,
            IS_QA_division=IS_QA_division,
            qa_options=qa_options,
            with_label=with_label,
            topic_model=topic_model,
            drop_irrelevant=drop_irrelevant,
        ),
        on="date",
    )


if __name__ == "__main__":
    print(return_data(word_limit=50))
    print(return_data(word_limit=350))
