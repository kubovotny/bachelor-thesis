import pandas as pd
from Statement import STATEMENTS_DIR


def qa_length(qa: str) -> int:
    return len(qa) if type(qa) is str else 0


def qa_dates_missing(data: pd.DataFrame) -> set[str]:
    data["qa_len"] = data["qa"].apply(qa_length)
    return set(data.query("qa_len < 10")["date"])


if __name__ == "__main__":
    correct_data: pd.DataFrame = pd.read_csv(f"{STATEMENTS_DIR}/scraped.csv", sep="|")
    correct_missing: set[str] = qa_dates_missing(correct_data)

    data_to_check: pd.DataFrame = pd.read_csv(
        f"{STATEMENTS_DIR}/scraped_v2.csv", sep="|"
    )

    missing_qa: set[str] = qa_dates_missing(data_to_check)

    print(sorted(missing_qa))
