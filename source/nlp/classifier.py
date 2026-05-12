from transformers import (
    pipeline,
    TextClassificationPipeline,
    ZeroShotClassificationPipeline,
)
from typing import List, Dict, Any, Literal
from huggingface_hub import login
import os
from ..data.connection import return_topic_labels
import torch
import pandas as pd

HF_TOKEN = os.environ["HF_TOKEN"]
login(token=HF_TOKEN)


classifier: TextClassificationPipeline | None = None

MODELS: Dict[Literal["finbert", "roberta"], str] = {
    "roberta": "Moritz-Pfeifer/CentralBankRoBERTa-sentiment-classifier",
    "finbert": "ProsusAI/finbert",
}


def get_sentiment(
    text: List[str] | str, model: Literal["finbert", "roberta"] = "finbert"
) -> List[Dict[str, str | float]] | List[List[Dict[str, str | float]]]:
    global classifier
    if model in MODELS:
        model_name = MODELS[model]
    else:
        raise NameError(f"{model} model is not in menu.")
    classifier = pipeline(
        "text-classification",
        model=model_name,
        token=HF_TOKEN,
        device=0,
    )
    return classifier(text, top_k=3, batch_size=256)


def calculate_sentiment(
    list_of_sentiments: List[Dict[str, str | float]], apply_divisor=False
) -> float:
    score: float = 0
    divisor: float = 0
    for sentiment in list_of_sentiments:
        if any(str(sentiment["label"]).lower() == x for x in ["positive", "hawkish"]):
            score += float(sentiment["score"])
            divisor += float(sentiment["score"])
        elif any(str(sentiment["label"]).lower() == x for x in ["negative", "dovish"]):
            score -= float(sentiment["score"])
            divisor += float(sentiment["score"])
    if divisor < 1e-2:
        return 0
    return score / (divisor if apply_divisor else 1)


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
topic_classifier: ZeroShotClassificationPipeline | None = None


def label_paragraph(text: str | List[str]) -> List[Dict[str, List[str | float] | str]]:
    global topic_classifier
    # Zabezpečíme, aby text bol list, kvôli tqdm a batchingu
    if isinstance(text, str):
        text = [text]

    if topic_classifier is None:
        topic_classifier = pipeline(
            "zero-shot-classification",
            model="facebook/bart-large-mnli",
            token=HF_TOKEN,
            device=0,
            torch_dtype=torch.float16,
        )

    results = []
    # Znížený batch_size na 32 pre stabilitu na 12GB VRAM
    for out in topic_classifier(
        text.to_list() if not isinstance(text, list) else text,
        candidate_labels=list(ZERO_SHOT_DESC2LABEL.keys()),
        batch_size=128,
    ):
        results.append(out)

    return results


def label_choose(
    sequence: str | None = None, labels: list[str] = [], scores: list[float] = []
) -> str:
    return f"{ZERO_SHOT_DESC2LABEL[labels[0]][1]},{scores[0]}"


if __name__ == "__main__":
    from .. import DATA_DIR
    import pandas as pd
    import time

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
