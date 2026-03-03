import requests

def get_page(url: str) -> bytes | None:
    try:
        response: requests.Response = requests.get(url)
        response.raise_for_status()
        return response.content
    except requests.exceptions.RequestException as e:
        print(f"Error getting page {url}: {e}")
        return None
    