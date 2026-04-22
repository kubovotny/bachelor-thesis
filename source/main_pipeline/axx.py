from chunker import STATEMENTS_DIR
from classifier import get_sentiment,calculate_sentiment
import pandas as pd

# data = pd.read_csv(f"{STATEMENTS_DIR}/labeled_chunks_2022u.csv")[["chunk","topic","topic2"]].to_csv(f"{STATEMENTS_DIR}/labels_to_check.csv")

print(calculate_sentiment(get_sentiment("Rising interest rates in the US does not have a direct impact on our decisions.","roberta")))