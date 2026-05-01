from .classifier import label_paragraph, label_choose
from ..data.connection import insert_topics, return_chunks, return_limits
from typing import Literal
import pandas as pd
from .. import FILENAME


def label_maker(
    limit: Literal[50, 100, 150, 200, 250, 300, 350] | None = None,
) -> None | pd.DataFrame:
    if limit is not None:
        data = return_chunks(limit)
        data["result"] = label_paragraph(list(data["chunk"]))
        data[["topic", "prob"]] = (
            data["result"]
            .apply(lambda x: label_choose(**x))
            .str.split(",", expand=True)
        )
        insert_topics(df=data)
        return data
    limits = return_limits
    for limit in limits:
        label_maker(limit)


if __name__ == "__main__":
    label_maker()
