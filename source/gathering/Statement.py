from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from Scraper import get_page
import pandas as pd
import re


def split_statement_to_intro_and_qa(all_elements: List[BeautifulSoup]) -> Tuple[List[BeautifulSoup], List[BeautifulSoup]]:
    separator_index = -1
    qa_pattern = re.compile(r"(I am[\w\s,]{10,30}questions\.)|(We[\w\s,]*questions[\w\s]*\.)|(\s?Click here [\w\s,]*answers\.)")
    # We need to search through the elements to find which one contains the pattern
    for i, element in enumerate(all_elements):
        if re.search(qa_pattern, element.get_text(separator="  ", strip=True)) and i > 2:
            separator_index = i
            break
    separator_index2 = -1
    qa_pattern2 = re.compile(r"Transcript of the questions.*ECB")
    for i, element in enumerate(all_elements):
        if re.search(qa_pattern2, element.get_text(separator="  ", strip=True)):
            separator_index2 = i - 1
            break
    if separator_index2 != -1:
        separator_index = separator_index2
    press_elements = []
    qa_elements = []
    # SPLIT
    if separator_index != -1:
        # PRESS STATEMENT
        press_elements = all_elements[:separator_index+1]
        # Q&A
        qa_elements = all_elements[separator_index+1:]
    else:
        press_elements = all_elements
    return press_elements, qa_elements

def scrape_statement(url: str, date: str)-> Dict[str, str]:
    page_content: bytes | None = get_page(url)

    if page_content:
        soup: BeautifulSoup = BeautifulSoup(page_content, "html.parser")
        main_content: List[BeautifulSoup] = soup.find("main")
        if not main_content:
            print(f"Warning: No main content found for {url}")
            return
        all_elements: List[BeautifulSoup] = []
        for child in main_content.find_all(recursive=False):
            if child.name not in ["div", "p", "ul", "ol"]:
                continue
            classes = child.get("class", [])
            if any(
                c
                in [
                    "title",
                    "address-box",
                    "-top-arrow",
                    "related-topics",
                    "upper-connetion",
                    "lower-connection",
                    "see-also-boxes",
                ]
                for c in classes
            ):
                continue
            for grandchild in child.children:
                if hasattr(grandchild, "name") and grandchild.name is not None:
                    if (
                        grandchild.name in ["div", "p", "ul", "ol"]
                        or "h" in grandchild.name
                    ):
                        all_elements.append(grandchild)
    intro, qa = split_statement_to_intro_and_qa(all_elements)
    press_text = "\t".join([e.get_text(separator="  ", strip=True) for e in intro])
    qa_text = "\t".join([e.get_text(separator="  ", strip=True) for e in qa])
    return {
        "date": date,
        "url": url,
        "press": press_text.replace('"', "''"),
        "qa": qa_text.replace('"', "'"),
    }

def get_list_of_statements_for_year(year: int)-> Dict[str, str]:
    url: str = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html".format(year=year)
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
    data = []
    for i in range(2013, 2014):
        statements = get_list_of_statements_for_year(i)
        for date, url in statements.items():
            elements = scrape_statement(url, date)
            data.append(elements)
    pd.DataFrame(data).to_csv("statements2013.csv", index=False, sep="|", encoding="utf-8")
