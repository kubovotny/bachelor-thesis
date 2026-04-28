from ..data.sentiment import return_sentiment_chunk_data

data = return_sentiment_chunk_data(["finbert","roberta"],with_label=False)
print(data)