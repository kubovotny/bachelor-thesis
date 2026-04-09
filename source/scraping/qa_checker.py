import pandas as pd
from Statement import STATEMENTS_DIR


def qa_length(qa):
    return len(qa) if type(qa) is str else 0


def qa_dates_missing(data):
    data["qa_len"] = data["qa"].apply(qa_length)
    return set(data.query("qa_len < 10")["date"])


if __name__ == "__main__":
    correct_data = pd.read_csv(f"{STATEMENTS_DIR}/scraped.csv", sep="|")
    correct_missing = qa_dates_missing(correct_data)

    data_to_check = pd.read_csv(f"{STATEMENTS_DIR}/scraped_v2.csv", sep="|")

    missing_qa = qa_dates_missing(data_to_check)

    print(missing_qa - correct_missing)
