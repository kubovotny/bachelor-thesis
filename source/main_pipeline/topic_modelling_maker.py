from classifier import label_paragraph, label_choose
from chunker import chunk_press, chunk_qa, STATEMENTS_DIR
from typing import Literal, List
import pandas as pd

FILENAME = "scraped_v2"


def label_maker(part: Literal["intro", "qa"] = "intro") -> None:
    columns_to_drop: List[str] = ["result"]
    data: pd.DataFrame
    if part == "intro":
        data = chunk_press(FILENAME)
    else:
        data = chunk_qa(FILENAME)
        columns_to_drop.append("qa_len")
    data["result"] = label_paragraph(list(data["chunk"]))
    data["label"] = data["result"].apply(lambda x: label_choose(**x))
    data["prob"] = data["result"].apply(lambda x: x["scores"][0])
    data = data.drop(columns=columns_to_drop)
    data.to_csv(f"{STATEMENTS_DIR}/labeled_{part}.psv", sep="|")


if __name__ == "__main__":
    for part in ["intro", "qa"]:
        label_maker(part)
