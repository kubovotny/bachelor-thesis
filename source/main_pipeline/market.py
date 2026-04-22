from chunker import STATEMENTS_DIR
import pandas as pd

FILENAME = "Dataset_EA-MPD.xlsx"
SHEETNAME = "Press Conference Window"

countries = ["DE", "ES", "IT", "FR"]
years = [2, 10]
columns = [
    "date",
    "OIS_1M",
    "OIS_3M",
    "OIS_1Y",
    *[f"{c}{y}Y" for y in years for c in countries],
    "STOXX50",
]

MARKET_DIR = "/".join(STATEMENTS_DIR.split("/")[:-1]) + "/market"


def clean_date(date):
    if type(date) is not str:
        return date
    if "/" in date:
        date = "-".join(date.split("/")[::-1])
    return pd.to_datetime(date)


def return_market_data():
    market_data = pd.read_excel(
        f"{MARKET_DIR}/{FILENAME}", sheet_name=SHEETNAME, usecols=columns
    )
    market_data["date"] = market_data.date.apply(clean_date)
    market_data = market_data.dropna(how="all")

    return market_data
