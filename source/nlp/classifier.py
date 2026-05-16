from transformers import (
    pipeline,
    TextClassificationPipeline,
    ZeroShotClassificationPipeline,
)
from typing import Dict, Literal
from huggingface_hub import login
import os
from ..data.queries import return_topic_labels
import torch
import pandas as pd
from .. import DATA_DIR
import pandas as pd
import time

SENTIMENT_MODEL_NAME = Literal["finbert", "roberta"]
SENTIMENT_MODELS: Dict[SENTIMENT_MODEL_NAME, str] = {
    "roberta": "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
    "finbert": "ProsusAI/finbert",
}
BATCH_SIZE = 256

_classifier_cache: dict[str, TextClassificationPipeline] = {}
try:
    HF_TOKEN = os.environ["HF_TOKEN"]
    login(token=HF_TOKEN)
except:
    HF_TOKEN = None


def get_sentiment(text, model: SENTIMENT_MODEL_NAME = "finbert"):
    if model not in _classifier_cache:
        _classifier_cache[model] = pipeline(
            "text-classification",
            model=SENTIMENT_MODELS[model],
            token=HF_TOKEN,
            device=0 if torch.cuda.is_available() else -1,
            top_k=None,  # full distribution, no truncation
        )
    return _classifier_cache[model](text, batch_size=BATCH_SIZE)


# classifier.py lines 47–53 — drop the hawkish/dovish branches; they never trigger now
def calculate_sentiment(list_of_sentiments, apply_divisor=False) -> float:
    score, divisor = 0.0, 0.0
    for sentiment in list_of_sentiments:
        lbl = str(sentiment["label"]).lower()
        if lbl == "positive":
            score += float(sentiment["score"])
            divisor += float(sentiment["score"])
        elif lbl == "negative":
            score -= float(sentiment["score"])
            divisor += float(sentiment["score"])
    if divisor < 1e-2:
        return 0.0
    return score / (divisor if apply_divisor else 1.0)


ZERO_SHOT_LABELS: Dict[
    Literal[
        "MONETARY_POLICY_AND_INFLATION",
        "ECONOMIC_PERFORMANCE",
        "FISCAL_AND_STRUCTURAL",
        "OTHER_IRRELEVANT",
    ],
    tuple[str, int],
] = return_topic_labels(0)


# {
#     "ECONOMIC_ANALYSIS": "economic activity, GDP output growth, and employment developments",
#     "INFLATION": "consumer price inflation, price developments, and wage pressures",
#     "RISK_ASSESSMENT": "upside and downside risks to the economic outlook",
#     "FINANCIAL_CONDITIONS": "financial market conditions, bond yields, and bank lending rates",
#     "MONETARY_ANALYSIS": "monetary analysis, money supply growth, and credit dynamics",
#     "FORWARD_GUIDANCE": "monetary policy stance, future interest rate guidance, and policy conclusions",
#     "FISCAL_POLICY": "government fiscal policy, public debt, and national budgets",
#     "STRUCTURAL_REFORM": "structural reforms, productivity, and labor market policies",
#     "OTHER_IRRELEVANT": "general greetings, non-economic questions, unrelated remarks, climate change, or personal comments",
# }
def inverse_dict(original: Dict) -> Dict:
    return {d: (n, r) for n, (d, r) in original.items()}


ZERO_SHOT_DESC2LABEL: Dict[
    str,
    tuple[
        Literal[
            "MONETARY_POLICY_AND_INFLATION",
            "ECONOMIC_PERFORMANCE",
            "FISCAL_AND_STRUCTURAL",
            "OTHER_IRRELEVANT",
        ],
        int,
    ],
] = inverse_dict(ZERO_SHOT_LABELS)
_topic_classifier_cache: Dict[str, ZeroShotClassificationPipeline] = {}
TOPIC_MODEL_NAME = Literal["facebook", "moritz"]
TOPIC_MODELS: Dict[TOPIC_MODEL_NAME, str] = {
    "facebook": "facebook/bart-large-mnli",
    "moritz": "MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
}


def label_paragraph(
    text, multi_label: bool = True, model: TOPIC_MODEL_NAME = "facebook"
):
    if isinstance(text, str):
        text = [text]
    if model not in _topic_classifier_cache:
        _topic_classifier_cache[model] = pipeline(
            "zero-shot-classification",
            model=SENTIMENT_MODELS[model],  # KEEP — theory.tex still valid
            token=HF_TOKEN,
            device=0 if torch.cuda.is_available() else -1,
            torch_dtype=torch.float16,
        )
    results = []
    for out in _topic_classifier_cache[model](
        text if isinstance(text, list) else text.to_list(),
        candidate_labels=list(ZERO_SHOT_DESC2LABEL.keys()),
        batch_size=128,
        multi_label=multi_label,  # ← key change
        hypothesis_template="This passage discusses {}.",  # ← domain-tuned template
    ):
        results.append(out)
    return results


def label_choose_multi(out: dict, threshold: float = 0.45):
    """Return list of (label_rowid, prob) tuples above threshold; fall back to top-1 if none."""
    pairs = [
        (ZERO_SHOT_DESC2LABEL[lbl][0], s)
        for lbl, s in zip(out["labels"], out["scores"])
        if s >= threshold
    ]
    if not pairs:
        pairs = [(ZERO_SHOT_DESC2LABEL[out["labels"][0]][0], out["scores"][0])]
    return pairs


def label_choose(
    sequence: str | None = None, labels: list[str] = [], scores: list[float] = []
) -> str:
    return f"{ZERO_SHOT_DESC2LABEL[labels[0]][1]},{scores[0]}"


if __name__ == "__main__":
    start = time.time()
    data = pd.read_csv(f"{DATA_DIR}/scraped_v2.psv").sample(500, random_state=42)
    data["result"] = pd.Series(label_paragraph(data["chunk"].to_list())).to_list()
    data["topic"] = data["result"].apply(lambda x: ZERO_SHOT_DESC2LABEL[x["labels"][0]])
    data["prob"] = data["result"].apply(lambda x: x["scores"][0])
    end = time.time()
    for _, row in data.iterrows():
        print(row["topic"], row["prob"], sep=",")
    # print(*data["result"],sep="\n\n")
    print(end - start, len(data))
