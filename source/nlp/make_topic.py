from .classifier import label_paragraph, label_choose
from ..data.connection import insert_topics, return_chunks, return_limits
from ..data import partition_indices
from typing import Literal
import pandas as pd
import time


def label_maker(
    limit: Literal[50, 100, 150, 200, 250, 300, 350] | None = None,
) -> None | pd.DataFrame:
    if limit is not None:
        start = time.time()
        full_data = return_chunks(limit)
        lr = partition_indices(full_data, 1000)
        for l, r in lr:
            start_label = time.time()
            data = full_data.loc[l:r,:]
            data["result"] = label_paragraph(list(data["chunk"]))
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
            full_data.loc[l:r,["lable_rowid","prob"]] = data[["label_rowid","prob"]]
        print(f"LIMIT {limit} - TIME: {time.time() - start}")
        return data
    limits = return_limits().sort(reverse=True)
    for limit in limits:
        label_maker(limit)


if __name__ == "__main__":
    label_maker(200)
