__version__ = "1.0.0"
from dotenv import load_dotenv
import os

load_dotenv()
STATEMENTS_DIR = os.getenv("STATEMENTS_DIR")
MARKET_DIR = os.getenv("MARKET_DIR")
GROQ_KEY = os.environ["GROQ_KEY"]
FILENAME = "scraped_v2"