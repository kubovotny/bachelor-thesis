from typing import List, Tuple, Dict
from bs4 import BeautifulSoup
from .Scraper import get_page
import pandas as pd
import re
from datetime import datetime
from .. import DATA_DIR


def split_statement_to_intro_and_qa(
    all_elements: List[BeautifulSoup],
) -> Tuple[List[BeautifulSoup], List[BeautifulSoup]]:

    # We need to search through the elements to find which one contains the pattern

    separator_index = -1
    qa_pattern2 = re.compile(r"My (first )?question .*")
    for i, element in enumerate(all_elements):
        if re.search(qa_pattern2, element.get_text(separator="  ", strip=True)):
            separator_index = i - 1
            break

    separator_index2 = -1
    qa_pattern = re.compile(
        r"(I am[\w\s,]{10,30}questions\.)|(We[\w\s,]*questions[\w\s]*\.)|(\s?Click here [\w\s,]*answers\.)"
    )
    for i, element in enumerate(all_elements):
        if (
            re.search(qa_pattern, element.get_text(separator="  ", strip=True))
            and i > 2
        ):
            separator_index2 = i
            break

    if separator_index2 != -1:
        separator_index = separator_index2

    separator_index3 = -1
    qa_pattern2 = re.compile(r"Transcript of the questions.*ECB")
    for i, element in enumerate(all_elements):
        if re.search(qa_pattern2, element.get_text(separator="  ", strip=True)):
            separator_index3 = i - 1
            break
    if separator_index3 != -1:
        separator_index = separator_index3
    intro_elements = []
    qa_elements = []
    # SPLIT
    if separator_index != -1:
        # INTRODUCTORY STATEMENT
        intro_elements = all_elements[: separator_index + 1]
        # Q&A
        qa_elements = all_elements[separator_index + 1 :]
    else:
        intro_elements = all_elements
    return intro_elements, qa_elements


def recursive_italic_bold_parser(tag: BeautifulSoup) -> str:
    mini_text: str = tag.get_text()
    if len(mini_text) <= 1:
        return ""
    if tag.name is None:
        return mini_text
    if tag.name in ["b", "strong", "em"]:
        return f"[{mini_text}]"
    tags: List[str] = []
    for tg in tag:
        tags.append(recursive_italic_bold_parser(tg))
    return "".join(tags)


def qa_proccessor(qa: List[BeautifulSoup]) -> str:
    qa_paragraphs: List[str] = []
    for paragraph in qa:
        if len(paragraph.get_text().replace(" ", "")) < 5:
            continue
        qa_pattern2: re.Pattern[str] = re.compile(
            r"Transcript of the questions.*President(of the ECB)?", re.I
        )
        if re.search(qa_pattern2, paragraph.get_text(separator="  ", strip=True)):
            continue
        qa_paragraphs.append(recursive_italic_bold_parser(paragraph))
        # print(recursive_bold_parser(paragraph))
        # print()
    return "\t".join(qa_paragraphs)


def scrape_statement(url: str, date: str) -> Dict[str, str]:
    # print(date)
    page_content: bytes | None = get_page(url)

    if page_content:
        soup: BeautifulSoup = BeautifulSoup(page_content, "html.parser")
        main_content: BeautifulSoup = soup.find("main")
        if not main_content:
            print(f"Warning: No main content found for {url}")
            return
        all_elements: List[BeautifulSoup] = []
        for child in main_content.find_all(recursive=False):
            if child.name not in ["div", "p", "ul", "ol"]:
                continue
            classes: List[str] = child.get("class", [])
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
                    "notes",
                    "footnotes",
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
    intro_text: str = "\t".join([e.get_text(separator="  ", strip=True) for e in intro])

    qa_text: str = qa_proccessor(qa)
    return {
        "date": date,
        "url": url,
        "intro": intro_text.replace('"', "''"),
        "qa": qa_text.replace('"', "'"),
    }


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
