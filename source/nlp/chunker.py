from nltk.tokenize import sent_tokenize
import re
import pandas as pd
from typing import Dict, List
from .. import DATA_DIR
from ..data.connection import insert_chunks, CHUNK_LIMIT_TYPE, CHUNK_LIMITS

DATABASE_SAVING = False


def load_statement_file(filename: str) -> pd.DataFrame:
    data = pd.read_csv(f"{DATA_DIR}/{filename}.psv", sep="|", index_col="statement_id")
    data.drop(columns=["url"], inplace=True)
    data["date"] = pd.to_datetime(data["date"])
    return data


def clean_paragraph(text: str, limit: int = 100) -> str:
    if not isinstance(text, str):
        return ""

    # strip leading/trailing whitespace
    text = text.strip()

    # collapse multiple spaces/newlines into one space
    text = re.sub(r"\s+", " ", text)

    # remove very short or empty paragraphs (less than 60 chars = noise)
    if len(text) < limit:
        return ""

    return text


def process_long_paragraph(
    long_text: str, max_words: int = 200, min_tail_words: int = 30
) -> List[str] | pd.api.typing.NAType:
    if type(long_text) is not str:
        return pd.NA
    sentences: List[str] = sent_tokenize(long_text)
    safe_chunks: List[str] = []
    current_chunk: str = ""
    for sentence in sentences:
        if len(current_chunk.split()) + len(sentence.split()) <= max_words:
            current_chunk += sentence + " "
        else:
            if current_chunk:
                safe_chunks.append(current_chunk.strip())
            current_chunk = sentence + " "
    if current_chunk:
        safe_chunks.append(current_chunk.strip())

    # Ak máme viac ako 1 blok, skontrolujeme ten úplne posledný
    if len(safe_chunks) > 1:
        last_chunk = safe_chunks[-1]

        # Ak je posledný blok príliš krátky (pod min_tail_words)
        if len(last_chunk.split()) < min_tail_words:
            # Prilepíme ho k predposlednému bloku
            safe_chunks[-2] = safe_chunks[-2] + " " + last_chunk
            # A túto samostatnú sirotu vymažeme zo zoznamu
            safe_chunks.pop()

    return safe_chunks


def make_percentile(data: pd.DataFrame) -> pd.DataFrame:
    data["chunk_id"] = data.groupby("statement_id").cumcount()
    data["chunk_percentile"] = (
        data["chunk_id"] / data.groupby("statement_id")["chunk_id"].max()
    )
    return data


def paragraphs_intro(filename: str) -> pd.DataFrame:
    data: pd.DataFrame = load_statement_file(filename)

    data["text"] = data["intro"].str.split("\t")

    df = data.explode("text").drop(columns=["qa", "intro"])
    return df


def chunk_intro(filename: str, limit: CHUNK_LIMIT_TYPE | None = None) -> pd.DataFrame:
    df = paragraphs_intro(filename)
    df["text"] = df["text"].apply(clean_paragraph, limit=100 if limit != 1 else 40)
    df = df.query("text != ''")
    if limit is None:
        limit_range: List[CHUNK_LIMIT_TYPE] = CHUNK_LIMITS
    else:
        limit_range = [limit]
    for limit in limit_range:
        df["chunk"] = df["text"].apply(
            process_long_paragraph,
            max_words=limit,
            min_tail_words=30 if limit != 1 else 0,
        )
        df_w_chunked = df.explode("chunk").drop(columns=["text"])
        df_w_chunked = make_percentile(df_w_chunked)
        df_w_chunked["chunk_limit"] = limit
        if DATABASE_SAVING:
            insert_chunks(df_w_chunked.reset_index())
    else:
        df_w_chunked = pd.DataFrame()
    return df_w_chunked


def clean_qa(qa: str) -> str:
    qa_cleaned = re.sub(re.compile(r"(^[A-Z ]{1,15}:)", re.IGNORECASE), "", qa)
    qa_cleaned = re.sub(re.compile(r"…|\.{3}"), "", qa_cleaned).strip()
    return qa_cleaned


def qa_splitter(paragraph: str) -> Dict[str, bool | str]:
    text: str = paragraph.replace("[", "").replace("]", "").strip()
    verdict = False
    if re.search(re.compile("^(Question|Q:)"), text) or (
        paragraph.count("[") == 1
        and paragraph.startswith("[")
        and paragraph.endswith("]")
    ):
        verdict = True
    return {"is_question": verdict, "text": clean_qa(text)}


def qa_multiple_proccesser(
    paragraphs: List[str],
) -> List[Dict[str, bool | str]] | float:
    if type(paragraphs) is float:
        return paragraphs
    qa_proccessed: List[Dict[str, bool | str]] = []
    for qa_paragraph in paragraphs:
        qa_proccessed.append(qa_splitter(qa_paragraph))
    return qa_proccessed


def paragraphs_qa(filename: str) -> pd.DataFrame:
    data = load_statement_file(filename)

    data["qa_paragraphs"] = data["qa"].str.split("\t")

    data["QA_processed"] = data["qa_paragraphs"].apply(qa_multiple_proccesser)
    data = data.drop(columns=["intro", "qa", "qa_paragraphs"]).dropna()

    if ...:
        # THIS was manually checked, if it divide correctly - 100% correct
        # check_edge = 280
        # for i,row in data.iterrows():
        #     if i < check_edge:
        #         continue
        #     print(row["date"])
        #     for QA_p in row["QA_processed"]:
        #         print(10 * "\t" if not QA_p["is_question"] else "", QA_p["text"][:50])
        #     if i >= check_edge + 10:
        #         break
        ...

    df_with_qa = data.explode("QA_processed")
    df_with_qa["is_question"] = df_with_qa["QA_processed"].apply(
        lambda x: x["is_question"]
    )
    df_with_qa["text"] = df_with_qa["QA_processed"].apply(lambda x: x["text"])
    return df_with_qa.drop(columns="QA_processed")


def chunk_qa(filename: str, limit: CHUNK_LIMIT_TYPE | None = None) -> pd.DataFrame:
    df_with_qa = paragraphs_qa(filename)
    df_with_qa["text"] = df_with_qa["text"].apply(
        clean_paragraph, limit=80 if limit != 1 else 40
    )
    df_with_qa = df_with_qa.query("text != ''")
    # df_with_qa["len"] = df_with_qa.text.str.split(" ").str.len()
    limit_range: List[CHUNK_LIMIT_TYPE]
    if limit is None:
        limit_range = CHUNK_LIMITS
    else:
        limit_range = [limit]
    # print(df_with_qa.query("len > 200"))
    for limit in limit_range:
        df_with_qa["qa_len"] = df_with_qa["text"].str.split().str.len()
        df_with_qa["chunk"] = df_with_qa["text"].apply(
            process_long_paragraph,
            max_words=limit,
            min_tail_words=30 if limit != 1 else 0,
        )
        df_qa_chunked = make_percentile(
            df_with_qa.explode("chunk").drop(columns=["text"])
        )
        df_qa_chunked["chunk_limit"] = limit
        if DATABASE_SAVING:
            insert_chunks(df_qa=df_qa_chunked.reset_index())
    else:
        df_qa_chunked = pd.DataFrame()
    return df_qa_chunked


def q_to_a_merged(filename: str) -> pd.DataFrame:
    data = load_statement_file(filename)
    dt: pd.Series
    dt = data["qa"].str.split("\t").apply(qa_multiple_proccesser)

    paired_list: List[List[str]] = []
    row: List[Dict[str, bool | str]] | float
    isq: bool
    text: str
    q: bool
    for row in dt:
        q = True
        pairs: List[str] = []
        pair: str = ""
        z: List[str] = []
        if isinstance(row, (float, int)):
            list_row = []
        else:
            list_row = row
        for dct in list_row:
            isq, text = bool(dct["is_question"]), str(dct["text"])
            if isq:
                if not q:
                    pair += "\t".join(z)  # answer adding
                    z = []
                    pairs.append(pair)
                    q = True
                z.append(text)
            else:
                if q:
                    pair = "\t".join(z) + "|"  # question adding
                    z = []
                    q = False
                z.append(text)
        pair += "\t".join(z)
        pairs.append(pair)
        paired_list.append(pairs)
    data["q_to_a"] = paired_list
    data = data.explode("q_to_a")
    data[["question", "answer"]] = (
        data["q_to_a"].str.split("|", expand=True).apply(lambda x: x.str.strip())
    )
    data = data.drop(columns=["intro", "qa", "q_to_a"])
    return data


def check_qa():
    data = chunk_qa("scraped_v2").drop(columns="intro")
    data["chunk_word_count"] = data["chunk"].str.split().str.len()
    data["chunk_len"] = data["chunk"].str.len()
    print(
        data.groupby(["chunk_len", "is_question"])["chunk"]
        .count()
        .groupby("is_question")
        .cumsum()
    )
    df = data.query("chunk_len <= 85")[["date", "chunk_len", "chunk"]]
    # df.to_csv(
    #     f"{STATEMENTS_DIR}/qa_short_answers.csv", sep="|"
    # )
    for i, row in df.sort_values("chunk_len").iterrows():
        print(row["date"].date(), row["chunk_len"], row["chunk"])


if __name__ == "__main__":
    DATABASE_SAVING = False
    print(chunk_intro("scraped_v2"))
    print(chunk_qa("scraped_v2"))
    # q_to_a_merged("scraped_v2").to_csv(f"{DATA_DIR}/qa_paired.csv", sep="|")
