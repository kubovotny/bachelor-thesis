import pandas as pd
from ..data.sentiment import return_sentiment_agg_pivot

df = return_sentiment_agg_pivot()
df2 = return_sentiment_agg_pivot(IS_QA_division=True, with_label=False)
# Uistíme sa, že dáta sú chronologicky sorted
df = df.sort_values("date").reset_index(drop=True)
df2 = df2.sort_values("date").reset_index(drop=True)

topics = ["MP", "EP", "FS", "OI"]

print("\n" + "=" * 25 + " ROZBOR PODĽA JEDNOTLIVÝCH MÍTINGOV " + "=" * 25)

# Cyklus cez každý jeden riadok (míting) v DataFrame
for idx, row_data in df.iterrows():
    # Sformátujeme dátum na pekný text
    date_str = pd.to_datetime(row_data["date"]).strftime("%Y-%m-%d")

    print(f"\nCONFERENCE: {date_str}")
    print("-" * 55)

    date_summary = []
    for section in ["IS", "QA"]:
        row = {
            "Sekcia": (
                "Introductory Statement (IS)"
                if section == "IS"
                else "Questions & Answers (QA)"
            )
        }
        section_total = 0

        for topic in topics:
            col_name = f"finbert_{section}_{topic}_count"
            # Vytiahneme hodnotu z aktuálneho riadku, ak je NaN alebo chýba, dáme 0
            val = row_data[col_name]
            val = int(val) if pd.notna(val) and col_name in df.columns else 0

            row[topic] = val
            section_total += val

        row["COUNT_SUM"] = section_total
        row["COUNT_ACTUAL"] = df2.loc[idx].get(f"finbert_{section}_count", 0)
        date_summary.append(row)

    # Vytvorenie DataFrame pre tento konkrétny dátum
    date_df = pd.DataFrame(date_summary)

    # Pridáme riadok s celkovým súčtom pre tento dátum
    grand_total = {"Sekcia": "CELKOVO"}
    for topic in topics:
        grand_total[topic] = date_df[topic].sum()
        grand_total[topic] = date_df[topic].sum()
    grand_total["COUNT_SUM"] = date_df["COUNT_SUM"].sum()
    grand_total["COUNT_ACTUAL"] = date_df["COUNT_ACTUAL"].sum()

    date_df = pd.concat([date_df, pd.DataFrame([grand_total])], ignore_index=True)

    # Vypíšeme mini-tabuľku mítingu
    print(
        date_df.to_string(
            index=False,
            formatters={
                c: lambda x: f"{x:,}" for c in topics + ["COUNT_SUM", "COUNT_ACTUAL"]
            },
        )
    )
    print("-" * 55)