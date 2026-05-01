__version__ = "1.0.0"
from dotenv import load_dotenv
import os

load_dotenv(interpolate=True)
DATA_DIR = os.getenv("DATA_DIR")
PASSWORD = os.getenv("PASSWORD")
DATABASE_DIR = DATA_DIR + "/database"
GROQ_KEY = os.environ["GROQ_KEY"]
FILENAME = "scraped_v2"
OUTPUT = os.getenv("IMAGES_OUTPUT_DIR")
if __name__ == "__main__":
    globals_list = list(globals().items())
    for a,b in globals_list:
        print(f"{a}: {b}")