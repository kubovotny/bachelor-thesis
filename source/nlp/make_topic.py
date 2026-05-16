from .classifier import label_paragraph, label_choose_multi
from ..data.queries import insert_topics, return_chunks, clear_topics_for_limit
from ..data.schema import CHUNK_LIMIT_TYPE
import pandas as pd
import time


def label_maker(
    limit: CHUNK_LIMIT_TYPE | None = None,
    sample_size: int | None = None,
    save_to_db: bool = True,
    threshold=0.45,
) -> None | pd.DataFrame:
    data = return_chunks(limit)
    if sample_size:
        data = data.sample(sample_size, random_state=42)
    start_label = time.time()
    data["result"] = label_paragraph(data["chunk"].astype(str).to_list())
    data["pairs"] = data["result"].apply(label_choose_multi, threshold=threshold)
    long_df = data.explode("pairs").dropna(subset=["pairs"])
    long_df[["label_rowid", "prob"]] = pd.DataFrame(
        long_df["pairs"].tolist(), index=long_df.index
    )
    start_store = time.time()
    if save_to_db:
        clear_topics_for_limit(limit)
        insert_topics(df=long_df)
    print(
        f"LIMIT {limit} - LABEL TIME: {start_store - start_label}, STORE TIME: {time.time() - start_store}"
    )
    return long_df


if __name__ == "__main__":
    for limit in 350,200,50,1:
        label_maker(limit, save_to_db=True)
