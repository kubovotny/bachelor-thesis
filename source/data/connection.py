import sqlite3
import pandas as pd
from .. import DATA_DIR
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
              chunk           TEXT,
              PRIMARY KEY (statement_id, part, chunk_id),
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
        "topics": """
              chunk_rowid     INT,
              label           TEXT,
              prob            REAL,
              PRIMARY KEY (chunk_rowid),
              FOREIGN KEY (chunk_rowid) REFERENCES chunks(rowid)""",
    }
    for table in reversed(TABLE_SCHEMA.keys()):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    for name, schema in TABLE_SCHEMA.items():
        cur.execute(f"CREATE TABLE {name}({schema})")

    cur.execute('INSERT INTO sentiment_models VALUES("finbert"),("roberta")')
    conn.commit()


def return_sentiment(with_label: bool = False):
    sql = f"""SELECT DATE(st.date) date, ch.chunk, se.score, sm.name sentiment_model{", t.label topic, t.prob topic_prob" if with_label else ""}  FROM sentiments se
JOIN chunks ch ON ch.rowid = se.chunk_rowid
JOIN statements st ON st.rowid = ch.statement_id
JOIN sentiment_models sm ON sm.rowid = se.model_id
{"JOIN topics t ON t.chunk_rowid = se.chunk_rowid" if with_label else ""};
"""
    return pd.read_sql(sql, conn, parse_dates="date")


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
    df[["statement_id", "part", "chunk_id", "is_question", "chunk"]].sort_values(
        ["statement_id", "part", "chunk_id"]
    ).to_sql("chunks", conn, if_exists="append", index=False)

    conn.commit()


def insert_sentiments(
    model: Literal["finbert", "roberta"],
    df_intro: pd.DataFrame | None = None,
    df_qa: pd.DataFrame | None = None,
):
    model_id = cur.execute(
        "SELECT rowid FROM sentiment_models sm WHERE sm.name = ?", (model,)
    ).fetchone()[0]
    cur.execute(f"DELETE FROM sentiments WHERE model_id = ?;", (model_id,))
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


def insert_topic(
    df_intro: pd.DataFrame | None = None, df_qa: pd.DataFrame | None = None
):
    df = concat_intro_qa(df_intro, df_qa)[
        ["statement_id", "part", "chunk_id", "is_question", "label", "prob"]
    ]
    for _, (statement_id, part, chunk_id, is_question, label, prob) in df.iterrows():
        cur.execute(
            """INSERT INTO topics 
                    VALUES(
                    (SELECT rowid FROM chunks WHERE statement_id = ? AND part = ? AND chunk_id = ? AND is_question = ?),
                    ?,?
                    )""",
            (statement_id, part, chunk_id, is_question, label, prob),
        )

    conn.commit()


if __name__ == "__main__":
    drop_and_make_tables()
    data = pd.read_csv(f"{DATA_DIR}/statements/scraped_v2.psv", sep="|")
    insert_statements(data)
    intro = pd.read_csv(f"{DATA_DIR}/statements/intro.psv", sep="|")
    qa = pd.read_csv(f"{DATA_DIR}/statements/qa.psv", sep="|")
    insert_chunks(intro, qa)
    intro = pd.read_csv(
        f"{DATA_DIR}/statements/sentiment/finbert/chunk_intro_labeled.psv", sep="|"
    )
    qa = pd.read_csv(
        f"{DATA_DIR}/statements/sentiment/finbert/chunk_qa_labeled.psv", sep="|"
    )
    insert_sentiments("finbert", intro, qa)

    intro = pd.read_csv(f"{DATA_DIR}/statements/labeled_intro.psv", sep="|")
    qa = pd.read_csv(f"{DATA_DIR}/statements/labeled_qa.psv", sep="|")
    insert_topic(intro, qa)
    print(return_sentiment(True))
