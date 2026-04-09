from transformers import pipeline, TextClassificationPipeline
from typing import List, Dict, Any
from huggingface_hub import login
import os
import re

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)


classifier: TextClassificationPipeline | None = None


def get_sentiment(text: List[str|float] | str) -> List[Dict[str, Any]]:
    global classifier
    if classifier is None:
        classifier = pipeline(
            "text-classification", model="ProsusAI/finbert", token=HF_TOKEN
        )

    return classifier(text, top_k=3)


def calculate_sentiment(list_of_sentiments: List[Dict[str, str | float]]) -> float:
    score = 0
    for sentiment in list_of_sentiments:
        if sentiment["label"] == "positive":
            score += sentiment["score"]
        elif sentiment["label"] == "negative":
            score -= sentiment["score"]

    return score


def qa_splitter(paragraph: str) -> Dict[str, bool | str]:
    text: str = paragraph.replace("[", "").replace("]", "").strip()
    verdict = False
    if re.search(re.compile("^(Question|Q:)"), text) or (
        paragraph.count("[") == 1
        and paragraph.startswith("[")
        and paragraph.endswith("]")
    ):
        verdict = True
    return {"is_question": verdict, "text": clean_qa_from_flags(text)}


def clean_qa_from_flags(qa: str) -> str:
    qa_cleaned = re.sub(re.compile("(^[A-Z ]{1,15}:)", re.IGNORECASE),"",qa).strip()
    return qa_cleaned
