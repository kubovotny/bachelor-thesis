from .. import DATA_DIR
from typing import Literal, List

DATABASE = f"{DATA_DIR}/statements.db"
CHUNK_LIMIT_TYPE = Literal[1, 50, 200, 350]
CHUNK_LIMITS: List[CHUNK_LIMIT_TYPE] = [1, 50, 200, 350]


TABLE_SCHEME = {
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
              model_id              INT,
              PRIMARY KEY (chunk_rowid, model_id, label_rowid),
              FOREIGN KEY (chunk_rowid) REFERENCES chunks(rowid)
              FOREIGN KEY (label_rowid) REFERENCES topic_labels(rowid),
              FOREIGN KEY (model_id) REFERENCES topic_models(rowid)""",
    "topic_models": """
              name            TEXT""",
}


def return_database_scheme():
    for name, schema in TABLE_SCHEME.items():
        print(f"CREATE TABLE {name}({schema});")

if __name__ == "__main__":
    return_database_scheme()