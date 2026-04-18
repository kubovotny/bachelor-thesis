from transformers import pipeline, TextClassificationPipeline
from typing import List, Dict, Any
from huggingface_hub import login
import os
import re

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)


classifier: TextClassificationPipeline | None = None

models = {
    "roberta": "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
    "finbert": "ProsusAI/finbert",
}


def get_sentiment(
    text: List[str | float] | str, model: str = "finbert"
) -> List[Dict[str, Any]]:
    global classifier
    if classifier is None:
        if model in models:
            model = models[model]
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


ZERO_SHOT_LABELS = {
    "ECONOMIC_ANALYSIS": "economic activity, GDP output growth, and employment developments",
    "INFLATION": "consumer price inflation, price developments, and wage pressures",
    "RISK_ASSESSMENT": "upside and downside risks to the economic outlook",
    "FINANCIAL_CONDITIONS": "financial market conditions, bond yields, and bank lending rates",
    "MONETARY_ANALYSIS": "monetary analysis, money supply growth, and credit dynamics",
    "FORWARD_GUIDANCE": "monetary policy stance, future interest rate guidance, and policy conclusions",
    "FISCAL_POLICY": "government fiscal policy, public debt, and national budgets",
    "STRUCTURAL_REFORM": "structural reforms, productivity, and labor market policies",
}

ZERO_SHOT_DESC2LABEL = {b: a for a, b in ZERO_SHOT_LABELS.items()}

topic_classifier: TextClassificationPipeline | None = None


def label_paragraph(text: str) -> str:
    global topic_classifier
    if topic_classifier is None:
        topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            token=HF_TOKEN,
            device="cuda",
        )
    result = topic_classifier(
        text, candidate_labels=list(ZERO_SHOT_DESC2LABEL.keys()), top_k=3
    )
    return result


def wise_label_choose(
    sequence: str = None, labels: list[str] = [], scores: list[float] = []
) -> str:
    if labels[0] == "RISK_ASSESSMENT":
        if scores[0] * 0.9 < scores[1]:
            return ZERO_SHOT_DESC2LABEL[labels[1]]
    return ZERO_SHOT_DESC2LABEL[labels[0]]
