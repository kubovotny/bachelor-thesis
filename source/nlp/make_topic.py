from .classifier import label_paragraph, label_choose
from ..data.connection import insert_topics, return_chunks, return_limits, CHUNK_LIMITS
from ..data import partition_indices
from typing import Literal
import pandas as pd
import time


def label_maker(
    limit: CHUNK_LIMITS | None = None,
) -> None | pd.DataFrame:
    if limit is not None:
        data = return_chunks(limit)
        start_label = time.time()
        data["result"] = label_paragraph(data["chunk"])
        data[["label_rowid", "prob"]] = (
            data["result"]
            .apply(lambda x: label_choose(**x))
            .str.split(",", expand=True)
        )
        start_store = time.time()
        insert_topics(df=data)
        print(
            f"LIMIT {limit} - LABEL TIME: {start_store - start_label}, STORE TIME: {time.time() - start_store}"
        )
        return data
    limits = return_limits()
    limits.sort(reverse=True)
    for limit in limits:
        label_maker(limit)


if __name__ == "__main__":
    label_maker()
