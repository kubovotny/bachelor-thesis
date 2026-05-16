import requests
from typing import Dict, List
from bs4 import BeautifulSoup
from statement import scrape_statement
import pandas as pd
from .. import DATA_DIR



def get_page(url: str) -> bytes | None:
    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error getting page {url}: {e}")
        return None
    

def get_list_of_statements_for_year(year: int) -> Dict[str, str]:
    url: str = (
        "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html".format(
            year=year
        )
    )
    print(f"Scraping year: {year}")
    page_content: bytes | None = get_page(url)

    statements: Dict[str, str] = {}

    if page_content:
        soup: BeautifulSoup = BeautifulSoup(page_content, "html.parser")
        dts: List[BeautifulSoup] = [
            dt for dt in soup.select("dt[isodate]") if not dt.find_parent("dl")
        ]

        for dt in reversed(dts):
            isodate: str = dt.get("isodate")
            if not isodate:
                continue

            dd: BeautifulSoup = dt.find_next_sibling("dd")
            if not dd:
                continue

            title_div: BeautifulSoup = dd.find("div", class_="title")
            if not title_div:
                continue

            link: BeautifulSoup = title_div.find("a")
            if not link or not link.get("href"):
                continue

            article_url: str = "https://www.ecb.europa.eu" + link.get("href")
            statements[isodate] = article_url
    return statements


if __name__ == "__main__":
    # scrape_statement("https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/2003/html/is030306.en.html","2003-03-02")
    data = []
    for i in range(1998, 2026):
        statements = get_list_of_statements_for_year(i)
        for date, url in statements.items():
            elements = scrape_statement(url, date)
            data.append(elements)
    df = pd.DataFrame(data)
    df.index.name = "statement_id"
    df.to_csv(f"{DATA_DIR}/scraped_v2.csv", index=True, sep="|", encoding="utf-8")