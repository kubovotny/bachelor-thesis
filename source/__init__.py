__version__ = "1.0.0"
from dotenv import load_dotenv
import os

load_dotenv(interpolate=True)
DATA_DIR = os.getenv("DATA_DIR")
PASSWORD = os.getenv("PASSWORD")
STATEMENTS_DIR = DATA_DIR + "/statements"
MARKET_DIR = DATA_DIR + "/market"
GROQ_KEY = os.environ["GROQ_KEY"]
FILENAME = "scraped_v2"
OUTPUT = os.getenv("IMAGES_OUTPUT_DIR")
if __name__ == "__main__":
    globals_list = list(globals().items())
    for a,b in globals_list:
        print(f"{a}: {b}")