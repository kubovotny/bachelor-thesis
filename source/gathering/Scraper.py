import requests
from datetime import datetime
from bs4 import BeautifulSoup
import pandas as pd

class ECBScraper:
    def __init__(self):
        self.base_url = "https://www.ecb.europa.eu/press/press_conference/monetary-policy-statement/{year}/html/index_include.en.html"
        self.data = []

    def get_page(self, url):
        try:
            response = requests.get(url)
            response.raise_for_status()
            return response.content
        except requests.exceptions.RequestException as e:
            print(f"Error getting page {url}: {e}")
            return None

    def scrape_year(self, year):
        url = self.base_url.format(year=year)
        print(f"Scraping year: {year}")
        page_content = self.get_page(url)

        if page_content:
            soup = BeautifulSoup(page_content, 'html.parser')
            dts = [dt for dt in soup.select('dt[isodate]') if not dt.find_parent('dl')]


            for dt in dts:
                isodate = dt.get('isodate')
                if not isodate:
                    continue

                dd = dt.find_next_sibling('dd')
                if not dd:
                    continue

                title_div = dd.find('div', class_='title')
                if not title_div:
                    continue
                
                link = title_div.find('a')
                if not link or not link.get('href'):
                    continue

                article_url = "https://www.ecb.europa.eu" + link.get('href')
                article_page_content = self.get_page(article_url)

                if article_page_content:
                    article_soup = BeautifulSoup(article_page_content, 'html.parser')
                    main = article_soup.find('main')
                    section = main.find('div', class_='section') if main else None

                    if section:
                        press_text = ''
                        qa_text = ''
                        
                        qa_separator = section.find('p', id='qa')
                        
                        if qa_separator:
                            press_elements = []
                            for sibling in qa_separator.previous_siblings:
                                press_elements.insert(0, str(sibling))
                            press_html = ''.join(press_elements)
                            press_text = BeautifulSoup(press_html, 'html.parser').get_text(separator='\n', strip=True)

                            qa_elements = []
                            for sibling in qa_separator.next_siblings:
                                qa_elements.append(str(sibling))
                            qa_html = ''.join(qa_elements)
                            qa_text = BeautifulSoup(qa_html, 'html.parser').get_text(separator='\n', strip=True)
                        else:
                            # If separator is not found, all content is considered press
                            press_text = section.get_text(separator='\n', strip=True)
                        
                        self.data.append({
                            'date': isodate,
                            'url': article_url,
                            'press': press_text,
                            'qa': qa_text
                        })
                    
    def scrape_all_years(self):
        for year in range(1998, datetime.now().year + 1):
            self.scrape_year(year)
    
    def save_to_csv(self, filename):
        df = pd.DataFrame(self.data)
        df.to_csv(filename, index=False)


if __name__ == '__main__':
    scraper = ECBScraper()
    scraper.scrape_year(2025)
    scraper.save_to_csv('data.csv')