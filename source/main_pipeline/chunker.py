from nltk.tokenize import sent_tokenize
import re
import pandas as pd
from typing import Dict, List, Any


STATEMENTS_DIR = "~/Documents/School/bachelor-thesis/source/statements"
FILE_SAVING = True


def load_statement_file(filename):
    data = pd.read_csv(f"{STATEMENTS_DIR}/{filename}.csv", sep="|")
    data.drop(columns=["Unnamed: 0", "url"], inplace=True)
    data.index.name = "statement_id"
    return data


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


def process_long_paragraph(long_text, max_words=200, min_tail_words=30):
    if type(long_text) is not str:
        return pd.NA
    sentences = sent_tokenize(long_text)
    safe_chunks = []
    current_chunk = ""
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


def chunk_press(filename: str) -> pd.DataFrame:
    data = load_statement_file(filename)

    data["paragraph"] = data["press"].str.split("\t")
    data["date"] = pd.to_datetime(data["date"])


    df = data.explode("paragraph").drop(columns=["qa", "press"])
    df["paragraph"] = df["paragraph"].apply(clean_paragraph)
    df = df.query("paragraph != ''")

    df["chunk"] = df["paragraph"].apply(process_long_paragraph)

    df_w_chunked = df.explode("chunk").drop(columns=["paragraph"])
    df_w_chunked = make_percentile(df_w_chunked)
    if FILE_SAVING:
        df_w_chunked.to_csv(f"{STATEMENTS_DIR}/intro.psv", sep="|")
    return df_w_chunked


def clean_qa_from_flags(qa: str) -> str:
    qa_cleaned = re.sub(re.compile("(^[A-Z ]{1,15}:)", re.IGNORECASE), "", qa).strip()
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
    return {"is_question": verdict, "text": clean_qa_from_flags(text)}


def qa_multiple_proccesser(paragraphs: List):
    if type(paragraphs) is float:
        return paragraphs
    qa_proccessed = []
    for qa_paragraph in paragraphs:
        qa_proccessed.append(qa_splitter(qa_paragraph))
    return qa_proccessed


def chunk_qa(filename: str) -> pd.DataFrame:
    data = load_statement_file(filename).sample(random_state=42)
    data["qa_paragraphs"] = data["qa"].str.split("\t")
    data["qa_paragraphs"] = data["qa_paragraphs"].apply(clean_paragraph)
    data = data.query("qa_paragraphs != ''")
    data["QA_processed"] = data["qa_paragraphs"].apply(qa_multiple_proccesser)
    data = data.drop(columns=["qa", "qa_paragraphs"]).dropna()
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
    df_with_qa = df_with_qa.drop(columns="QA_processed")
    # df_with_qa["len"] = df_with_qa.text.str.split(" ").str.len()

    # print(df_with_qa.query("len > 200"))
    df_with_qa["qa_len"] = df_with_qa["text"].str.split().str.len()
    df_with_qa["chunk"] = df_with_qa["text"].apply(process_long_paragraph)
    df_qa_chunked = make_percentile(df_with_qa.explode("chunk").drop(columns=["text"]))
    if FILE_SAVING:
        df_qa_chunked.to_csv(f"{STATEMENTS_DIR}/qa.psv", sep="|")
    return df_qa_chunked

def q_to_a_merged(filename: str) -> pd.DataFrame:
    data = load_statement_file(filename)
    dt = data["qa"].str.split("\t").apply(qa_multiple_proccesser)
    
    paired_list = []
    for row in dt:
        q = True
        pairs:list[str] = []
        pair = ""
        z:list[str] = []
        if type(row) is float:
            row = []
        for dct in row:
            isq, text=  dct.values()
            if isq:
                if not q:
                    pair += "\t".join(z) # answer adding
                    z = []
                    pairs.append(pair)
                    q = True
                z.append(text)
            else:
                if q:
                    pair = "\t".join(z) + "|" # question adding
                    z = []
                    q = False
                z.append(text)
        pair += "\t".join(z)
        pairs.append(pair)
        paired_list.append(pairs)
    data["q_to_a"] = paired_list
    data = data.explode("q_to_a")
    data[["question","answer"]] = data["q_to_a"].str.split("|",expand=True).apply(lambda x: x.str.strip())
    data = data.drop(columns=["press", "qa","q_to_a"])
    return data


if __name__ == "__main__":
    q_to_a_merged("scraped_v2").to_csv(f"{STATEMENTS_DIR}/qa_paired.csv", sep="|")