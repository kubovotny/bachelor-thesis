import sqlite3
import pandas as pd
from source.data.connection import connect_to_db

conn, _ = connect_to_db()

# All meeting dates in the database
db_dates = pd.read_sql(
    "SELECT DISTINCT DATE(date) as date FROM statements ORDER BY date",
    conn
)

# Dates that actually have sentiment computed at chunk_limit=150
sent_dates = pd.read_sql("""
    SELECT DISTINCT DATE(st.date) as date
    FROM sentiments se
    JOIN chunks ch ON ch.rowid = se.chunk_rowid
    JOIN statements st ON st.rowid = ch.statement_id
    WHERE ch.chunk_limit = 200
    ORDER BY date
""", conn)

conn.close()

# Find meetings in DB but missing from sentiment
missing = set(db_dates["date"]) - set(sent_dates["date"])
print(f"Meetings in DB:        {len(db_dates)}")
print(f"Meetings with sentiment: {len(sent_dates)}")
print(f"\nMissing dates:")
for d in sorted(missing):
    print(f"  {d}")