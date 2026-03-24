import re
from transformers import pipeline
from nltk.tokenize import sent_tokenize
from huggingface_hub import login
import os

HF_TOKEN = "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"
os.environ["HF_TOKEN"] = HF_TOKEN
login(token=HF_TOKEN)


def clean_paragraph(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # strip leading/trailing whitespace
    text = text.strip()

    # collapse multiple spaces/newlines into one space
    text = re.sub(r"\s+", " ", text)

    # remove very short or empty paragraphs (less than 60 chars = noise)
    if len(text) < 100:
        return ""

    return text


def process_long_paragraph(long_text, max_words=200):
    sentences = sent_tokenize(long_text)
    safe_chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Check how long our current chunk would be if we add this sentence
        if len(current_chunk.split()) + len(sentence.split()) <= max_words:
            current_chunk += sentence + " "
        else:
            # The chunk is full! Save it, and start a new one
            if current_chunk:
                safe_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Don't forget the last piece!
    if current_chunk:
        safe_chunks.append(current_chunk.strip())

    return safe_chunks


classifier = None

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


def label_paragraph(text: str) -> str:
    global classifier
    if classifier is None:
        classifier = pipeline(
            "zero-shot-classification", model="facebook/bart-large-mnli", token=HF_TOKEN,
            device="cuda"
        )
    result = classifier(text, candidate_labels=list(ZERO_SHOT_DESC2LABEL.keys()))
    return result


def wise_label_choose(
    sequence: str = None, labels: list[str] = [], scores: list[float] = []
) -> str:
    if labels[0] == "RISK_ASSESSMENT":
        if scores[0] * 0.9 < scores[1]:
            return ZERO_SHOT_DESC2LABEL[labels[1]]
    return ZERO_SHOT_DESC2LABEL[labels[0]]


# Load the AgentClassifier model
agent_classifier = pipeline(
    "text-classification", model="Moritz-Pfeifer/CentralBankRoBERTa-agent-classifier"
)

# Perform agent classification
agent_result = agent_classifier(
    "At its meeting today, the Governing Council reviewed, as usual, the main monetary, financial and other economic indicators in line with its monetary policy strategy. Following this discussion, it saw no need to change the interest rates on the ECB's monetary policy instruments. The interest rate on the main refinancing operations thus remains 2.5%. In addition, the interest rate on the marginal lending facility continues to be 3.5% and the interest rate on the deposit facility is maintained at 1.5%. Let me give you some details about our current assessment of the monetary policy stance, and thereby provide explanations for the decisions taken today."
)
print("Agent Classification:", agent_result[0]["label"])
