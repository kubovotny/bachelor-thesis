from transformers import pipeline, TextClassificationPipeline
from typing import List, Dict, Any
from huggingface_hub import login
import os
import re

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)


classifier: TextClassificationPipeline | None = None

{
    "centralbankroberta": "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
    "finbert": "ProsusAI/finbert",
}


def get_sentiment(
    text: List[str | float] | str, model: str = "ProsusAI/finbert"
) -> List[Dict[str, Any]]:
    global classifier
    if classifier is None:
        classifier = pipeline("text-classification", model=model, token=HF_TOKEN)
    return classifier(text, top_k=3)


def calculate_sentiment(list_of_sentiments: List[Dict[str, str | float]]) -> float:
    score = 0
    for sentiment in list_of_sentiments:
        if any(sentiment["label"].lower() == x for x in ["positive", "hawkish"]):
            score += sentiment["score"]
        elif any(sentiment["label"].lower() == x for x in ["negative", "dovish"]):
            score -= sentiment["score"]

    return score


topic_classifier: TextClassificationPipeline | None = None


# TODO get_topic - decide whether to use facebook-bart-mnli or CBRoBERTa agent classifier and program it
def get_topic(text: List[str] | str):
    model = "Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier"
    global topic_classifier
    if topic_classifier is None:
        topic_classifier = pipeline("text-classification", model=model, token=HF_TOKEN)
    result = topic_classifier(text)
    print(result)
    return result
