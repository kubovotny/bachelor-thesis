from ..data.sentiment import return_sentiment_chunk_data

# CORRECT — return_sentiment_chunk_data first arg is limit_version (int)
data = return_sentiment_chunk_data(word_limit=150, with_label=False)
# Both models are already in the data — they're in the sentiment_model column
pivot = data.pivot_table(index=["date","chunk"], columns="sentiment_model", values="score")
