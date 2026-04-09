from nltk.tokenize import sent_tokenize
import re
import pandas as pd


STATEMENTS_DIR = "~/Documents/School/bachelor-thesis/source/statements"


def clean_paragraph(text: str) -> str:
    if not isinstance(text, str):
        return ""

    # strip leading/trailing whitespace
    text = text.strip()

    # collapse multiple spaces/newlines into one space
    text = re.sub(r"\s+", " ", text)

    # remove very short or empty paragraphs (less than 60 chars = noise)
    if len(text) < 100:
        return ""

    return text


def process_long_paragraph(long_text, max_words=200):
    if type(long_text) is not str:
        return pd.NA
    sentences = sent_tokenize(long_text)
    safe_chunks = []
    current_chunk = ""
    for sentence in sentences:
        # Check how long our current chunk would be if we add this sentence
        if len(current_chunk.split()) + len(sentence.split()) <= max_words:
            current_chunk += sentence + " "
        else:
            # The chunk is full! Save it, and start a new one
            if current_chunk:
                safe_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "

    # Don't forget the last piece!
    if current_chunk:
        safe_chunks.append(current_chunk.strip())

    return safe_chunks

def make_percentile(data:pd.DataFrame)->pd.DataFrame:
    data["chunk_id"] = data.groupby("statement_id").cumcount()
    data["chunk_percentile"] = (
        data["chunk_id"]
        / data.groupby("statement_id")["chunk_id"].max()
    )
    return data

def chunk_it(filename: str) -> pd.DataFrame:
    data = pd.read_csv(f"{STATEMENTS_DIR}/{filename}.csv", sep="|")
    data.drop(columns=["Unnamed: 0", "url"], inplace=True)
    data.index.name = "statement_id"

    data["paragraph"] = data["press"].str.split("\t")
    data["date"] = pd.to_datetime(data["date"])

    df = data.explode("paragraph").drop(columns=["qa", "press"])
    df["paragraph"] = df["paragraph"].apply(clean_paragraph)
    df = df.query("paragraph != ''")

    df = data.explode("paragraph").drop(columns=["qa", "press"])
    df["paragraph"] = df["paragraph"].apply(clean_paragraph)
    df = df.query("paragraph != ''")

    df["chunk"] = df["paragraph"].apply(process_long_paragraph)

    df_w_chunked = df.explode("chunk").drop(columns=["paragraph"])
    df_w_chunked = make_percentile(df_w_chunked)
    return df_w_chunked
