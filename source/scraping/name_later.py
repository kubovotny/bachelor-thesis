import pandas as pd
from datetime import datetime

# udržuj to čo najjednoduchšie !!!!
# economic outlook vs monetary...
# sentiment nebude dobrý, lebo zmeny medzi konferenciami

STATEMENTS_DIR = "~/Documents/School/bachelor-thesis/source/statements"
POSTFIX = "_" "2022u"

df = pd.read_csv("source/statements/scraped.csv", sep="|", encoding="utf-8")
df["date"] = pd.to_datetime(df["date"], errors="coerce")
print(df.count())

after2022 = df[df["date"] > "2021-12-31"]

df.query("date > '2022-01-01'").to_csv(f"{STATEMENTS_DIR}/sample{POSTFIX}.csv",sep="|")