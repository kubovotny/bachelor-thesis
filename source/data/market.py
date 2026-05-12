import pandas as pd
from typing import List, Dict, Literal, Hashable
from datetime import datetime
from .. import DATA_DIR
from pandas._typing import DtypeArg

MPD_SHEETNAME = "Press Conference Window"
ECB_SHEETNAME = "MM"

countries = ["DE", "ES", "IT", "FR"]
years = [2, 10]
MPD_COLUMNS: List[str] = [
    "date",
    "OIS_1M",
    "OIS_3M",
    "OIS_6M",
    "OIS_1Y",
    *[f"{c}{y}Y" for y in years for c in countries],
    "STOXX50",
]
ECB_COLUMNS: List[str] = [
    "date",
    "MRO",
    "MRO announced",
    "MRO effective",
    "DF announced",
    "DF effective",
    "Wu-Xia shadow rate",
]

MPD_COLUMNS_TYPES: Dict[str, str] = {col: "float64" for col in MPD_COLUMNS[1:]}
ECB_COLUMN_TYPES: Dict[str, str] = {col: "float64" for col in ECB_COLUMNS[1:]}

CSV_COLUMNS = ["pc1", "STOXX50", "MP_pm", "CBI_pm", "MP_median", "CBI_median"]
CSV_COLUMNS_TYPES: Dict[Hashable, DtypeArg] = {col: "float64" for col in CSV_COLUMNS}

LIST_OF_MARKET_DATA = Literal[
    "shocks_ecb_mpd_me_d.csv", "Dataset_EA-MPD.xlsx", "ECB Money Market.xlsx", "all"
]

def clean_date(date: str | datetime) -> datetime:
    if not isinstance(date,str):
        return date
    if "/" in date:
        date = "-".join(date.split("/")[::-1])
    return pd.to_datetime(date)


def return_market_data(
    which: List[LIST_OF_MARKET_DATA] | LIST_OF_MARKET_DATA = "shocks_ecb_mpd_me_d.csv",
) -> pd.DataFrame:
    if which == "all":
        which = [
            "shocks_ecb_mpd_me_d.csv",
            "Dataset_EA-MPD.xlsx",
            "ECB Money Market.xlsx",
        ]
    if isinstance(which, list):
        market_data = return_market_data(which[0])
        for name in which[1:]:
            market_data = pd.merge(market_data, return_market_data(name), on="date")
        return market_data
    if "Dataset_EA-MPD.xlsx" == which:
        market_data: pd.DataFrame = pd.read_excel(
            f"{DATA_DIR}/Dataset_EA-MPD.xlsx",
            sheet_name=MPD_SHEETNAME,
            usecols=MPD_COLUMNS,
            dtype=MPD_COLUMNS_TYPES,
        )
    elif "ECB Money Market.xlsx" == which:
        market_data: pd.DataFrame = pd.read_excel(
            f"{DATA_DIR}/ECB Money Market.xlsx",
            sheet_name=ECB_SHEETNAME,
            usecols=ECB_COLUMNS,
            dtype=ECB_COLUMN_TYPES,
        )
    else:
        market_data: pd.DataFrame = pd.read_csv(
            f"{DATA_DIR}/shocks_ecb_mpd_me_d.csv", dtype=CSV_COLUMNS_TYPES
        )
    market_data["date"] = market_data.date.apply(clean_date)
    market_data = market_data.set_index("date")
    market_data = market_data.dropna(how="all").reset_index()
    return market_data


if __name__ == "__main__":
    print(return_market_data())
