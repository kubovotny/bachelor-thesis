from transformers import pipeline, TextClassificationPipeline
from typing import List, Dict, Any

classifier: TextClassificationPipeline | None = None


def get_sentiment(text: List[str] | str) -> List[Dict[str, Any]]:
    global classifier
    if classifier is None:
        classifier = pipeline("text-classification", model="ProsusAI/finbert")

    return classifier(text)
