from classifier import get_sentiment
import pandas as pd

STATEMENTS_DIR = "~/Documents/School/bachelor-thesis/source/statements"
POSTFIX = "_" "2022u"

label_chunks = pd.read_csv(f"{STATEMENTS_DIR}/labeled_chunks{POSTFIX}.csv")

sample = label_chunks.iloc[:,:10]
print(sample)