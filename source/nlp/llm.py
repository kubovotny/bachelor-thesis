import json
import time
import pandas as pd
from groq import Groq
from ..data.connection import return_chunks
from .. import DATABASE_DIR, GROQ_KEY, FILENAME

# Inicializácia
client = Groq(api_key=GROQ_KEY)

# VYLEPŠENÝ PROMPT PRE BATCH (Viacero textov naraz)
BATCH_SYSTEM_PROMPT = """
You are an elite macroeconomist and ECB policy analyst. I will provide you with a LIST of 10 text excerpts from ECB press conferences.
To ensure accuracy, proceed strictly in two logical steps and respond EXCLUSIVELY in JSON format.
Be sensitive to subtle nuances. If the text implies a direction even slightly, reflect it in the score. Don't defaults to 0.0 unless the text is purely administrative.

STEP 1: TOPIC CLASSIFICATION
Determine the primary topic of the text. You MUST choose exactly one from the following list: ["Monetary Policy", "Economic Performance", "Financial Stability", "Other Irrelevant"].

STEP 2: SENTIMENT ANALYSIS (Interest Rate Outlook)
Evaluate what signal this text sends to financial markets regarding future interest rates. Assign a score from -1.0 (Strongly Dovish / Rate Cuts / Easing) to 1.0 (Strongly Hawkish / Rate Hikes / Tightening). A purely neutral or factual text is 0.0.


JSON OUTPUT STRUCTURE (Strictly adhere to these keys):
{
  "results": [
    { "topic_reasoning": "Brief explanation of your topic choice (max 1 sentence).",
      "topic": "Selected topic",
      "sentiment_reasoning": "Brief justification for your sentiment score (max 1 sentence).",
      "sentiment": 0.75
    }, ... (total 10 items)
  ]
}
"""


def evaluate_batch(texts):
    # Prevedieme zoznam textov na jeden očíslovaný reťazec
    formatted_input = "\n\n".join([f"TEXT {i+1}: {t}" for i, t in enumerate(texts)])

    try:
        response = client.chat.completions.create(
            model="llama-3.3-70b-versatile",
            response_format={"type": "json_object"},
            messages=[
                {"role": "system", "content": BATCH_SYSTEM_PROMPT},
                {"role": "user", "content": formatted_input},
            ],
            temperature=0.0,
        )
        return json.loads(response.choices[0].message.content)["results"]
    except Exception as e:
        print(f"Batch Error: {e}")
        # V prípade chyby vrátime zoznam error objektov, aby sme zachovali dĺžku
        return [{"topic": "Error", "sentiment": 0.0, "reasoning": str(e)}] * len(texts)


all_results = []
batch_size = 10
df = return_chunks()

for i in range(0, len(df), batch_size):
    batch_texts = df["chunk"].iloc[i : i + batch_size].tolist()

    print(f"Spracovávam riadky {i} až {i + len(batch_texts)}...")

    batch_res = evaluate_batch(batch_texts)
    all_results.extend(batch_res)

    # Každých 100 riadkov priebežne ukladaj (DÔLEŽITÉ pri takomto objeme!)
    if i % 100 == 0:...# TODO saving to dabase

    # Krátka pauza, aby Groq "nelapal po dychu"
    time.sleep(1.2)

# Pridanie výsledkov do pôvodného DF
results_df = pd.DataFrame(all_results)
df = pd.concat([df.reset_index(drop=True), results_df], axis=1)
# TODO save to dabase