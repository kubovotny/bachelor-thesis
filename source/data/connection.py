import sqlite3
import pandas as pd
from .. import DATA_DIR, PASSWORD
from typing import Literal

DATABASE = f"{DATA_DIR}/statements.db"
conn = sqlite3.connect(DATABASE)
cur = conn.cursor()


def drop_and_make_tables():
    TABLE_SCHEMA = {
        "statements": """
              date            TEXT,
              url             TEXT""",
        "chunks": """
              statement_id    INT,
              part            INT,
              chunk_id        INT,
              is_question     BOOLEAN,
              chunk_limit     INT,
              chunk           TEXT,
              PRIMARY KEY (statement_id, part, chunk_id, chunk_limit),
              FOREIGN KEY (statement_id) REFERENCES statements(rowid)""",
        "sentiment_models": """
              name            TEXT""",
        "sentiments": """
              chunk_rowid     INT,
              model_id        INT,
              score           REAL,
              PRIMARY KEY (chunk_rowid, model_id),
              FOREIGN KEY (chunk_rowid) REFERENCES chunks(rowid),
              FOREIGN KEY (model_id) REFERENCES sentiment_models(rowid)""",
        "topic_labels": """
              name          TEXT,
              description   TEXT,
              version       INT,
              PRIMARY KEY(name, version)""",
        "topics": """
              chunk_rowid           INT,
              label_rowid           INT,
              prob                  REAL,
              PRIMARY KEY (chunk_rowid, label_rowid),
              FOREIGN KEY (chunk_rowid) REFERENCES chunks(rowid)
              FOREIGN KEY (label_rowid) REFERENCES topic_labels(rowid)""",
    }
    for table in reversed(TABLE_SCHEMA.keys()):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    for name, schema in TABLE_SCHEMA.items():
        cur.execute(f"CREATE TABLE {name}({schema})")

    cur.execute('INSERT INTO sentiment_models VALUES("finbert"),("roberta")')
    conn.commit()


def return_limits():
    sql = "SELECT DISTINCT chunk_limit FROM chunks;"
    return [l[0] for l in cur.execute(sql).fetchall()]


def return_topic_labels(version=0):
    sql = "SELECT rowid, name, description FROM topic_labels WHERE version=?;"
    return {n: (d, r) for r, n, d in cur.execute(sql, (version,)).fetchall()}


def return_chunks(limit_version:Literal[50, 100, 150, 200, 250, 300, 350]=200):
    sql = "SELECT rowid as chunk_rowid, chunk FROM chunks WHERE chunk_limit=?;"
    return pd.read_sql(sql, conn, params=(limit_version,))


def return_sentiment(limit_version:Literal[50, 100, 150, 200, 250, 300, 350]=200, with_topic: bool = False):
    sql = f"""SELECT DATE(st.date) date, CASE ch.part WHEN 0 THEN "IS" ELSE "QA" END part, ch.is_question=1 is_question, ch.chunk, se.score, sm.name sentiment_model\
{", tl.name topic, t.prob topic_prob" if with_topic else ""} FROM sentiments se
JOIN chunks ch ON ch.rowid = se.chunk_rowid
JOIN statements st ON st.rowid = ch.statement_id
JOIN sentiment_models sm ON sm.rowid = se.model_id
{("JOIN topics t ON t.chunk_rowid = se.chunk_rowid\n" + 
 "JOIN topic_labels tl ON tl.rowid = t.label_rowid\n" )if with_topic else ""}\
WHERE ch.chunk_limit = ?
ORDER BY st.date, ch.part, ch.chunk_id;
"""
    return pd.read_sql(sql, conn, parse_dates="date", params=(limit_version,))

def concat_intro_qa(
    df_intro: pd.DataFrame | None = None, df_qa: pd.DataFrame | None = None
) -> pd.DataFrame:
    if df_intro is not None:
        df_intro["part"] = 0
        df_intro["is_question"] = False
    else:
        df_intro = pd.DataFrame()
    if df_qa is not None:
        df_qa["part"] = 1
    else:
        df_qa = pd.DataFrame()
    return pd.concat([df_intro, df_qa])


def insert_statements(df: pd.DataFrame):
    df[["date", "url"]].sort_values("date").to_sql(
        "statements", conn, if_exists="delete_rows", index=False
    )
    conn.commit()


def insert_chunks(
    df_intro: pd.DataFrame | None = None, df_qa: pd.DataFrame | None = None
):
    df = concat_intro_qa(df_intro, df_qa)
    df[
        ["statement_id", "part", "chunk_id", "is_question", "chunk", "chunk_limit"]
    ].sort_values(["statement_id", "part", "chunk_id"]).to_sql(
        "chunks", conn, if_exists="append", index=False
    )

    conn.commit()


def insert_sentiments(
    model: Literal["finbert", "roberta"] = None,
    df_intro: pd.DataFrame | None = None,
    df_qa: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
):
    assert model is not None or df is not None, "Model or df must be defined"

    if df is not None:
        df[["chunk_rowid", "model_id", "score"]].to_sql(
            "sentiments", conn, if_exists="append", index=False
        )
        return
    part = (
        1
        if df_intro is None and df_qa is not None
        else 0 if df_intro is not None and df_qa is None else 2
    )
    model_id = cur.execute(
        "SELECT rowid FROM sentiment_models sm WHERE sm.name = ?", (model,)
    ).fetchone()[0]
    cur.execute(
        f"DELETE FROM sentiments WHERE model_id = ? and part = ?;", (model_id, part)
    )
    df = concat_intro_qa(df_intro, df_qa)[
        ["statement_id", "part", "chunk_id", "is_question", "score"]
    ]

    for _, (statement_id, part, chunk_id, is_question, score) in df.iterrows():
        cur.execute(
            """INSERT INTO sentiments 
                    VALUES(
                    (SELECT rowid FROM chunks WHERE statement_id = ? AND part = ? AND chunk_id = ? AND is_question = ?),
                    ?,?
                    )""",
            (statement_id, part, chunk_id, is_question, model_id, score),
        )

    conn.commit()


def insert_topic_labels():
    topics = {
        "MONETARY_POLICY_AND_INFLATION": "inflation, price stability, interest rate decisions, monetary policy stance, financing conditions, bank lending, and market interest rates",
        "ECONOMIC_PERFORMANCE": "economic growth, GDP outlook, unemployment, labor market developments, macroeconomic risks, demand, consumption, and investment",
        "FISCAL_AND_STRUCTURAL": "government budgets, national debt, public spending, and structural reforms",
        "OTHER_IRRELEVANT": "general greetings, purely political questions, unrelated remarks, climate change, or personal comments (excluding macroeconomic impacts)",
    }
    sql = "INSERT INTO topic_labels VALUES(0, ?, ?);"
    for item in topics.items():
        cur.execute(sql, item)
    conn.commit()


def insert_topics(
    df_intro: pd.DataFrame | None = None,
    df_qa: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
):
    if df is not None:
        df[["chunk_rowid", "label_rowid", "prob"]].to_sql(
            "topics", conn, index=False, if_exists="append"
        )
        return
    df = concat_intro_qa(df_intro, df_qa)[
        ["statement_id", "part", "chunk_id", "is_question", "label", "prob"]
    ]
    for _, (statement_id, part, chunk_id, is_question, label, prob) in df.iterrows():
        cur.execute(
            """INSERT INTO topics 
                    VALUES(
                    (SELECT rowid FROM chunks WHERE statement_id = ? AND part = ? AND chunk_id = ? AND is_question = ?),
                    (SELECT rowid FROM topic_labels WHERE name  = ?),?
                    )""",
            (statement_id, part, chunk_id, is_question, label, prob),
        )

    conn.commit()


if __name__ == "__main__":
    if input() == PASSWORD:
        drop_and_make_tables()
        data = pd.read_csv(f"{DATA_DIR}/statements/scraped_v2.psv", sep="|")
        insert_statements(data)
        insert_topic_labels()
    # intro = pd.read_csv(f"{DATA_DIR}/statements/intro.psv", sep="|")
    # qa = pd.read_csv(f"{DATA_DIR}/statements/qa.psv", sep="|")
    # insert_chunks(intro, qa)
    # intro = pd.read_csv(
    #     f"{DATA_DIR}/statements/sentiment/finbert/chunk_intro_labeled.psv", sep="|"
    # )
    # qa = pd.read_csv(
    #     f"{DATA_DIR}/statements/sentiment/finbert/chunk_qa_labeled.psv", sep="|"
    # )
    # insert_sentiments("finbert", intro, qa)

    # intro = pd.read_csv(f"{DATA_DIR}/statements/labeled_intro.psv", sep="|")
    # qa = pd.read_csv(f"{DATA_DIR}/statements/labeled_qa.psv", sep="|")
    # insert_topic(intro, qa)
    # print(return_sentiment(True))
"""SELECT DATE(st.date) date, CASE WHENch.part=0,"IS","QA"), ch.is_question, ch.chunk, se.score, sm.name sentiment_model, tl.name topic, t.prob topic_prob  FROM sentiments se
JOIN chunks ch ON ch.rowid = se.chunk_rowid
JOIN statements st ON st.rowid = ch.statement_id
JOIN sentiment_models sm ON sm.rowid = se.model_id
JOIN topics t ON t.chunk_rowid = se.chunk_rowid
JOIN topic_labels tl ON tl.rowid = t.label_rowid
WHERE ch.chunk_limit = 200;"""