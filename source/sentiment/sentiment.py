import pandas as pd

import nltk

from nltk.sentiment.vader import SentimentIntensityAnalyzer

from nltk.corpus import stopwords

from nltk.tokenize import word_tokenize

from nltk.stem import WordNetLemmatizer

# nltk.download('all')

df = pd.read_csv('../gathering/data_raw1.csv', sep='|')

print(df.tail())

if __name__ == '__main__':...