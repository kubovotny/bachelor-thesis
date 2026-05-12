from .classifier import label_paragraph, label_choose
from ..data.connection import (
    insert_topics,
    return_chunks,
    return_limits,
    CHUNK_LIMIT_TYPE,
)
from ..data import partition_indices
from typing import Literal
import pandas as pd
import time


def label_maker(
    limit: CHUNK_LIMIT_TYPE | None = None,
    sample_size: int | None = None,
    save_to_db: bool = True,
) -> None | pd.DataFrame:
    if limit is not None:
        data = return_chunks(limit)
        if sample_size is not None:
            data = data.sample(sample_size)
        print(data)
        start_label = time.time()
        data["result"] = pd.Series(
            label_paragraph(data["chunk"].astype(str).to_list())
        ).to_list()
        print(label_paragraph(data["chunk"].astype(str).to_list()))
        data[["label_rowid", "prob"]] = (
            data["result"]
            .apply(lambda x: label_choose(**x))
            .str.split(",", expand=True)
        )
        start_store = time.time()
        if save_to_db:
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
    label_maker(200, sample_size=1, save_to_db=False)
