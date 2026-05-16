import sqlite3
from .. import PASSWORD
from .schema import DATABASE, TABLE_SCHEME

def connect_to_db():
    conn = sqlite3.connect(DATABASE)
    cur = conn.cursor()
    return conn, cur


def drop_and_make_tables():
    for table in reversed(TABLE_SCHEME.keys()):
        cur.execute(f"DROP TABLE IF EXISTS {table}")
    for name, schema in TABLE_SCHEME.items():
        cur.execute(f"CREATE TABLE {name}({schema})")

    cur.execute('INSERT INTO sentiment_models VALUES("finbert"),("roberta");')
    cur.execute('INSERT INTO topic_models VALUES("facebook"),("moritz");')
    conn.commit()



if __name__ == "__main__":
    conn, cur = connect_to_db()
    # return_database_scheme()
    if input() == PASSWORD:
        drop_and_make_tables()
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
