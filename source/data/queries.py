from .connection import connect_to_db
import pandas as pd
from .. import DATA_DIR, PASSWORD
from typing import Literal, List

DATABASE = f"{DATA_DIR}/statements.db"
CHUNK_LIMIT_TYPE = Literal[1, 50, 100, 150, 200, 250, 300, 350]
CHUNK_LIMITS: List[CHUNK_LIMIT_TYPE] = [1, 50, 100, 150, 200, 250, 300, 350]


def return_limits():
    _, cur = connect_to_db()
    sql = "SELECT DISTINCT chunk_limit FROM chunks;"
    return [l[0] for l in cur.execute(sql).fetchall()]


def return_statements():
    conn, cur = connect_to_db()
    sql = "SELECT rowid statement_id, * FROM statements;"
    return pd.read_sql(sql, conn)


def return_topic_labels(version=0):
    conn, cur = connect_to_db()
    sql = "SELECT rowid, name, description FROM topic_labels WHERE version=?;"
    return {n: (d, r) for r, n, d in cur.execute(sql, (version,)).fetchall()}


def return_chunks(
    limit_version: CHUNK_LIMIT_TYPE | None = 200,
):
    conn, cur = connect_to_db()
    if limit_version is None:
        return pd.read_sql("SELECT rowid as chunk_rowid, * FROM chunks;", conn)
    else:
        return pd.read_sql(
            "SELECT rowid as chunk_rowid, chunk FROM chunks WHERE chunk_limit=?;",
            conn,
            params=(limit_version,),
        )


def return_sentiment(
    limit_version: CHUNK_LIMIT_TYPE | None = 200,
    with_label: bool = False,
):
    conn, cur = connect_to_db()
    sql = f"""SELECT DATE(st.date) date, ch.rowid, CASE ch.part WHEN 0 THEN "IS" ELSE "QA" END part, ch.is_question=1 is_question, ch.chunk, se.score, sm.name sentiment_model\
{", tl.name topic, t.prob topic_prob" if with_label else ""} FROM sentiments se
JOIN chunks ch ON ch.rowid = se.chunk_rowid
JOIN statements st ON st.rowid = ch.statement_id
JOIN sentiment_models sm ON sm.rowid = se.model_id
{("JOIN topics t ON t.chunk_rowid = se.chunk_rowid\n" + 
 "JOIN topic_labels tl ON tl.rowid = t.label_rowid\n" )if with_label else ""}\
{"WHERE ch.chunk_limit = ?" if limit_version is not None else ""}
ORDER BY st.date, ch.part, ch.chunk_id;
"""
    if limit_version is not None:
        return pd.read_sql(sql, conn, parse_dates=["date"], params=(limit_version,))
    else:
        return pd.read_sql(sql, conn, parse_dates=["date"])


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
    conn, cur = connect_to_db()
    df[["date", "url"]].sort_values("date").to_sql(
        "statements", conn, if_exists="delete_rows", index=False
    )
    conn.commit()


def insert_chunks(
    df_intro: pd.DataFrame | None = None, df_qa: pd.DataFrame | None = None
):
    conn, cur = connect_to_db()
    df = concat_intro_qa(df_intro, df_qa)
    df[
        ["statement_id", "part", "chunk_id", "is_question", "chunk", "chunk_limit"]
    ].sort_values(["statement_id", "part", "chunk_id"]).to_sql(
        "chunks", conn, if_exists="append", index=False
    )

    conn.commit()


def insert_sentiments(
    model: Literal["finbert", "roberta"] | None = None,
    df_intro: pd.DataFrame | None = None,
    df_qa: pd.DataFrame | None = None,
    df: pd.DataFrame | None = None,
):
    conn, cur = connect_to_db()
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
    conn, cur = connect_to_db()
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
    conn, cur = connect_to_db()
    if df is not None:
        df[["chunk_rowid", "label_rowid", "prob"]].to_sql(
            "topics", conn, index=False, if_exists="append"
        )
        return
    df = concat_intro_qa(df_intro, df_qa)[
        ["statement_id", "part", "chunk_id", "is_question", "label", "prob"]
    ]
    for _, (statement_id, part, chunk_id, is_question, label, prob) in df.iterrows():
        print(
            """INSERT INTO topics 
                    VALUES(
                    (SELECT rowid FROM chunks WHERE statement_id = ? AND part = ? AND chunk_id = ? AND is_question = ?),
                    (SELECT rowid FROM topic_labels WHERE name  = ?),?
                    )""",
            (statement_id, part, chunk_id, is_question, label, prob),
        )

    conn.commit()


def clear_topics_for_limit(chunk_limit: int):
    conn, cur = connect_to_db()
    cur.execute(
        """
        DELETE FROM topics
        WHERE chunk_rowid IN (SELECT rowid FROM chunks WHERE chunk_limit = ?)
    """,
        (chunk_limit,),
    )
    conn.commit()


if __name__ == "__main__":
    if input() == PASSWORD:
        data = pd.read_csv(f"{DATA_DIR}/statements/scraped_v2.psv", sep="|")
        insert_statements(data)
        insert_topic_labels()
