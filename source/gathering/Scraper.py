import requests
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd
from typing import List
import re


class ECBScraper:
    def __init__(self):
        self.base_url = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html"
        self.data = []

    def get_page(self, url: str) -> bytes | None:
        try:
            response: requests.Response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error getting page {url}: {e}")
            return None

    def get_sections(
        self, article_soup: BeautifulSoup, only_one=False
    ) -> List[BeautifulSoup] | BeautifulSoup | None:
        main: BeautifulSoup = article_soup.find("main")
        if main is None:
            return None
        if only_one:
            return main.find("div", class_="section")
        sections: List[BeautifulSoup] | None = main.select("div.section")
        return sections

    def scrape_statement(self, url: str):
        article_page_content: bytes | None = self.get_page(url)

        if article_page_content:
            article_soup: BeautifulSoup = BeautifulSoup(
                article_page_content, "html.parser"
            )
            section: BeautifulSoup | None = self.get_sections(article_soup, only_one=True)

            if section:
                press_text: str = ""
                qa_text: str = ""

                qa_separator: BeautifulSoup = section.find(
                    "p",
                    string=re.compile(
                        r"We\sare\s[\w\s,]* questions\.(?=Click here [\w\s,]\.)?"
                    ),
                )

                if qa_separator:
                    press_elements: List[str] = []
                    for sibling in qa_separator.previous_siblings:
                        press_elements.insert(0, str(sibling))
                    press_html: str = "".join(press_elements)
                    press_text: str = BeautifulSoup(press_html, "html.parser").get_text(
                        separator="\n", strip=True
                    )

                    qa_elements: List[str] = []
                    for sibling in qa_separator.next_siblings:
                        qa_elements.append(str(sibling))
                    qa_html: str = "".join(qa_elements).strip()
                    qa_text: str = BeautifulSoup(qa_html, "html.parser").get_text(
                        separator="\n", strip=True
                    )
                else:
                    # If separator is not found, all content is considered press
                    press_text: str = section.get_text(separator="\n", strip=True)
                self.data.append(
                    {
                        "date": isodate,
                        "url": article_url,
                        "press": press_text,
                        "qa": qa_text,
                    }
                )

    def scrape_year(self, year: int):
        url: str = self.base_url.format(year=year)
        print(f"Scraping year: {year}")
        page_content: bytes | None = self.get_page(url)

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
                self.scrape_statement(article_url)

    def scrape_all_years(self):
        for year in range(1998, datetime.now().year + 1):
            self.scrape_year(year)

    def save_to_csv(self, filename: str):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False, sep="|")


if __name__ == "__main__":
    scraper = ECBScraper()
    scraper.scrape_year(1998)
    scraper.save_to_csv("data_raw2.csv")
