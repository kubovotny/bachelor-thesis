# verify_table.py — five minutes
from ..nlp.classifier import get_sentiment, calculate_sentiment
rows = [
    ("Inflation is far too high and is projected to remain above our target for too long.", "IS"),
    ("Economic activity has stagnated in recent quarters.", "QA"),
    ("We will continue to follow a data-dependent approach.", "IS"),
]
for txt, sec in rows:
    fb = calculate_sentiment(get_sentiment(txt, "finbert")[0])
    rb = calculate_sentiment(get_sentiment(txt, "roberta")[0])
    print(f"{sec} | {txt[:55]!r:60} | FB={fb:+.2f} | RB={rb:+.2f}")