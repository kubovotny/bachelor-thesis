import pandas as pd
from typing import List, Dict, Literal
from datetime import datetime
from .. import DATABASE_DIR

SHEETNAME = "Press Conference Window"

countries = ["DE", "ES", "IT", "FR"]
years = [2, 10]
XLSX_COLUMNS: List[str] = [
    "date",
    "OIS_1M",
    "OIS_3M",
    "OIS_6M",
    "OIS_1Y",
    *[f"{c}{y}Y" for y in years for c in countries],
    "STOXX50",
]

XLSX_COLUMNS_TYPES: Dict[str, str] = {col: "float64" for col in XLSX_COLUMNS[1:]}
CSV_COLUMNS = ["pc1", "STOXX50", "MP_pm", "CBI_pm", "MP_median", "CBI_median"]
CSV_COLUMNS_TYPES: Dict[str, str] = {col: "float64" for col in CSV_COLUMNS}


def clean_date(date: str) -> datetime:
    if type(date) is not str:
        return date
    if "/" in date:
        date = "-".join(date.split("/")[::-1])
    return pd.to_datetime(date)


def return_market_data(
    which: Literal[
        "Dataset_EA-MPD.xlsx", "shocks_ecb_mpd_me_d.csv"
    ] = "shocks_ecb_mpd_me_d.csv",
) -> pd.DataFrame:
    ending = which.split(".")[-1]
    if ending == "xlsx":
        market_data: pd.DataFrame = pd.read_excel(
            f"{DATABASE_DIR}/{which}",
            sheet_name=SHEETNAME,
            usecols=XLSX_COLUMNS,
            dtype=XLSX_COLUMNS_TYPES,
        )
    else:
        market_data: pd.DataFrame = pd.read_csv(
            f"{DATABASE_DIR}/{which}", dtype=CSV_COLUMNS_TYPES
        )

    market_data["date"] = market_data.date.apply(clean_date)
    market_data = market_data.set_index("date")
    market_data = market_data.dropna(how="all").reset_index()
    return market_data


if __name__ == "__main__":
    print(return_market_data())
1
