from pathlib import Path

CACHE_DIR = Path("artifacts")
RAW_DATA_PATH = CACHE_DIR / "raw_dataset.parquet"
CHUNKS_DF_PATH = CACHE_DIR / "chunks_df.parquet"
EMBEDDINGS_PATH = CACHE_DIR / "embeddings.npy"
CACHE_DIR.mkdir(parents=True, exist_ok=True)

FORCE_RECOMPUTE = False
FORCE_RECOMPUTE_DATA = FORCE_RECOMPUTE
FORCE_RECOMPUTE_CHUNKS = FORCE_RECOMPUTE
FORCE_RECOMPUTE_EMBEDDINGS = FORCE_RECOMPUTE

# ---

from llama_cpp import Llama
import torch

print(f"CUDA Available: {torch.cuda.is_available()}")

# ---

import os
import time
import re
import json
import torch
import numpy as np
import pandas as pd
from pathlib import Path
from getpass import getpass
import gc

# LLM & Chunking
from llama_cpp import Llama
from langchain_text_splitters import RecursiveCharacterTextSplitter
from sentence_transformers import SentenceTransformer
from transformers import AutoTokenizer

# Vector Search
import faiss

# Evaluation (Gemini API)
import google.generativeai as genai

def get_secret(name, prompt):
    try:
        from google.colab import userdata
        value = userdata.get(name)
    except Exception:
        value = None
        try:
            from dotenv import load_dotenv
            load_dotenv()
        except ImportError:
            pass
    return value or os.getenv(name) or getpass(prompt).strip()

# Device Configuration
device = "cuda" if torch.cuda.is_available() else "cpu"
print(f"Project initialized on device: {device}")

# ---

from datasets import load_dataset
import pandas as pd

if RAW_DATA_PATH.exists() and not FORCE_RECOMPUTE_DATA:
    df = pd.read_parquet(RAW_DATA_PATH)
else:
    dataset = load_dataset(
        "KillerShoaib/Jeffrey-Epstein-Emails-From-Epstein-Files"
    )
    df = dataset["train"].to_pandas()
    df.to_parquet(RAW_DATA_PATH, index=False)


df.head()

# ---

df.shape
df.info()
df.isna().sum()

# ---

print(f"Number of unique senders: {df['from_email'].nunique()}")
print(f"Number of unique recipients: {df['to'].nunique()}")

# ---

df["body"].str.len().describe()

# ---

df["subject"].value_counts().head(10)

# ---

df["body_len"] = df["body"].str.len()

short_bodies = df[df["body_len"] < 20]
df.sort_values("body_len")[["doc_id", "subject", "body", "body_len"]].head(20)

# ---

import re

def normalize_text(text):
    text = str(text)
    text = re.sub(r"\s+", " ", text)
    return text.strip()

for col in ["subject", "preview", "from_name", "from_email", "to", "date", "body"]:
    df[col] = df[col].apply(normalize_text)

# ---

import re

def count_recipients_by_angle_brackets(to_field):
    if not isinstance(to_field, str) or to_field.strip() == "":
        return 0

    return max(1, len(re.findall(r"<[^<>]+>", to_field)))

df["num_recipients"] = df["to"].apply(count_recipients_by_angle_brackets)
df[["to", "num_recipients"]].sort_values("num_recipients", ascending=False).head(20)

# ---

import pandas as pd

df["parsed_date"] = pd.to_datetime(df["date"], errors="coerce")

print("Unparseable dates:", df["parsed_date"].isna().sum())
print("Date range:", df["parsed_date"].min(), "to", df["parsed_date"].max())

# ---

dedup_key = ["subject", "from_email", "to", "num_recipients", "date", "body"]

print("Full-row duplicates:", df.duplicated().sum())
print("Content-based duplicates:", df.duplicated(subset=dedup_key).sum())

# ---

duplicate_groups = (
    df[df.duplicated(subset=dedup_key, keep=False)]
    .groupby(dedup_key)
    .agg(
        n_rows=("doc_id", "size"),
        doc_ids=("doc_id", lambda x: list(x)),
        n_unique_doc_ids=("doc_id", "nunique"),
        n_unique_previews=("preview", "nunique"),
        n_unique_from_names=("from_name", "nunique")
    )
    .reset_index()
    .sort_values("n_rows", ascending=False)
)

duplicate_groups.head(10)

# ---

df_grouped = (
    df.groupby(dedup_key, dropna=False)
    .agg(
        doc_ids=("doc_id", lambda x: sorted(set(x))),
        n_original_rows=("doc_id", "size"),
        preview=("preview", "first"),
        from_name=("from_name", "first"),
        parsed_date=("parsed_date", "first")
    )
    .reset_index()
)

print("Original rows:", len(df))
print("Rows after content-based grouping:", len(df_grouped))
print("Rows collapsed:", len(df) - len(df_grouped))

# ---

def build_header(row):
    # Show up to 5 document IDs to avoid long headers
    doc_ids = ", ".join(row["doc_ids"][:5])

    if len(row["doc_ids"]) > 5:
        doc_ids += f", ... ({len(row['doc_ids'])} total)"

    # Truncate recipient list if it's too long
    recipients = str(row["to"])
    if len(recipients) > 200:
        recipients = recipients[:200].rstrip() + " ... (truncated)"

    # Truncate subject if it's too long
    subject = str(row["subject"])
    if len(subject) > 200:
        subject = subject[:200].rstrip() + " ... (truncated)"

    return f"""Subject: {subject}
From: {row['from_name']} <{row['from_email']}>
To: {recipients} of {row['num_recipients']} total recipients
Date: {row['date']}
Source document IDs: {doc_ids}
"""

# ---

tokenizer = AutoTokenizer.from_pretrained("BAAI/bge-base-en-v1.5")

splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
    tokenizer=tokenizer,
    chunk_size=512,
    chunk_overlap=64,
    separators=["\n\n", "\n", ". ", " ", ""]
)

# ---

if CHUNKS_DF_PATH.exists() and not FORCE_RECOMPUTE_CHUNKS:
    chunks_df = pd.read_parquet(CHUNKS_DF_PATH)
else:
    chunk_records = []

    for row_idx, row in df_grouped.iterrows():
        header = build_header(row)

        # Calculate tokens in header to adjust body chunk size
        # We use a safety margin of 15 tokens for special tokens and chunk info text
        header_tokens = len(tokenizer.encode(header, add_special_tokens=False))
        max_body_tokens = 512 - header_tokens - 15

        # Ensure we don't have a negative or tiny chunk size
        chunk_size = max(128, max_body_tokens)

        local_splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=chunk_size,
            chunk_overlap=int(chunk_size * 0.15),
            separators=["\n\n", "\n", ". ", " ", ""]
        )

        body_chunks = local_splitter.split_text(row["body"])

        for chunk_idx, body_chunk in enumerate(body_chunks):
            chunk_text = f"""{header}

Body chunk {chunk_idx + 1} of {len(body_chunks)}:
{body_chunk}"""

            chunk_records.append({
                "chunk_id": f"{row_idx}_{chunk_idx}",
                "email_id": row_idx,
                "chunk_index": chunk_idx,
                "num_chunks": len(body_chunks),

                # Metadata
                "doc_ids": row["doc_ids"],
                "n_original_rows": row["n_original_rows"],
                "subject": row["subject"],
                "from_name": row["from_name"],
                "from_email": row["from_email"],
                "to": row["to"],
                "date": row["date"],

                # Text used for retrieval and prompt
                "text_for_embedding": chunk_text,
                "text_for_prompt": chunk_text,
            })

    chunks_df = pd.DataFrame(chunk_records)
    chunks_df.to_parquet(CHUNKS_DF_PATH, index=False)

chunks_df.head()

# ---

print(f"Description of number of chunks per email:\n{chunks_df['num_chunks'].describe()}")

# ---

chunks_df[chunks_df["text_for_embedding"].str.len() < 5]

# ---

chunks_len = chunks_df["text_for_embedding"].apply(
    lambda x: len(tokenizer.encode(x, add_special_tokens=True))
)
longest = chunks_len.max()
longest_chunk = chunks_df.iloc[chunks_len.idxmax()]


print(f"Longest chunk with header: {longest} tokens")
print(f"Text of longest chunk:\n{longest_chunk['text_for_prompt']}")

# ---

def compute_embeddings(chunks_df, embedding_model, batch_size=64):

    texts = chunks_df["text_for_embedding"].tolist()

    embeddings = embedding_model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True
    )

    return embeddings.astype("float32")

# ---

def load_or_compute_embeddings(
    chunks_df,
    embedding_model,
    embeddings_path=EMBEDDINGS_PATH,
    force_recompute=False
):
    """
    Load embeddings from disk if available and compatible.
    Otherwise compute and save them.
    """
    if embeddings_path.exists() and not force_recompute:
        print(f"Loading embeddings from {embeddings_path}...")
        embeddings = np.load(embeddings_path)

        if embeddings.shape[0] == len(chunks_df):
            return embeddings.astype("float32")

        print(
            "Saved embeddings are incompatible with chunks_df "
            f"({embeddings.shape[0]} embeddings vs {len(chunks_df)} chunks). "
            "Recomputing..."
        )

    else:
        print("No saved embeddings found. Computing embeddings...")

    embeddings = compute_embeddings(chunks_df, embedding_model)

    np.save(embeddings_path, embeddings)
    print(f"Saved embeddings to {embeddings_path}")

    return embeddings

# ---

from sentence_transformers import SentenceTransformer

embedding_model = SentenceTransformer(
    "BAAI/bge-base-en-v1.5",
    device=device
)

embeddings = load_or_compute_embeddings(
    chunks_df=chunks_df,
    embedding_model=embedding_model,
    force_recompute=FORCE_RECOMPUTE_EMBEDDINGS
)

embeddings.shape

# ---

import faiss

embedding_matrix = embeddings.astype("float32")

index = faiss.IndexFlatIP(embedding_matrix.shape[1])
index.add(embedding_matrix)

# ---

def retrieve(query, k=5):
    query_embedding = embedding_model.encode(
        [query],
        normalize_embeddings=True
    ).astype("float32")

    scores, indices = index.search(query_embedding, k)

    results = chunks_df.iloc[indices[0]].copy()
    results["score"] = scores[0]

    return results

# ---

retrieve("What emails mention Bill Clinton private orgy?", k=5)

# ---

def build_prompt(query, k=5):
    """
    Retrieves relevant documents and crafts a structured prompt for the Gemma LLM.
    """
    retrieved_df = retrieve(query, k=k)

    # extract and format the context from retrieved documents
    context_parts = []
    for idx, (_, row) in enumerate(retrieved_df.iterrows()):
        context_parts.append(f"[Segment {idx+1}]\n{row['text_for_prompt']}")

    context_str = "\n\n".join(context_parts)

    # gemma promt template
    # Template: <start_of_turn>user\n{content}<end_of_turn>\n<start_of_turn>model\n
    system_instruction = (
        "You are an investigative assistant analyzing a dataset of emails from the Epstein files. "
        "Your task is to answer the user's question accurately using ONLY the provided email segments. "
        "For each piece of information you provide, you MUST cite the 'Source document IDs' found in the segment headers. "
        "If the provided context does not contain the answer, explicitly state that the information is not available in the retrieved data."
        "Answer only relevant questions based on the provided context. Do not speculate or provide information that is not supported by the retrieved email segments."
    )

    user_content = f"""{system_instruction}

    RELEVANT EMAIL SEGMENTS:
    {context_str}

    USER QUESTION: {query}"""

    prompt = f"<start_of_turn>user\n{user_content}<end_of_turn>\n<start_of_turn>model\n"

    return prompt

# Example test
example_prompt = build_prompt("What was discussed regarding Saudi money?")
print(f"Generated prompt length: {len(example_prompt)} characters")
print("--- Prompt Preview ---")
print(example_prompt[:1000] + "...")

# ---

import gc

# --- Memory Management ---
def clear_memory():
    gc.collect()
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
        torch.cuda.ipc_collect()

clear_memory()

# --- Authentication Setup ---
HF_TOKEN = get_secret(
    "HF_TOKEN",
    "Enter Hugging Face Token (Required for gated Gemma access): "
)

# --- Model Configuration ---
model_id = "igorls/gemma-4-E4B-it-heretic-GGUF"
filename = "gemma-4-E4B-it-heretic-Q4_K_M.gguf"

print(f"Loading {model_id} via llama-cpp...")

llm = Llama.from_pretrained(
    repo_id=model_id,
    filename=filename,
    n_ctx=8192,
    n_gpu_layers=-1,
    token=HF_TOKEN
)

# ---

def answer_question(query, k=5, max_new_tokens=1024):
    # 1. Build the contextual prompt
    prompt = build_prompt(query, k=k)
    
    # 2. Generate the response using llama-cpp-python
    # GGUF models via llama-cpp handle their own formatting internally
    # based on the prompt template provided in build_prompt.
    response = llm(
        prompt,
        max_tokens=max_new_tokens,
        echo=False,        # Don't include the prompt in the output
        temperature=0.1,
        top_p=0.95,
        stop=["<end_of_turn>", "<|endoftext|>"]  # Ensure it stops correctly
    )
    
    # 3. Extract the text
    answer = response["choices"][0]["text"].strip()
    
    return answer

# Example of a real RAG query
query = "What are the main discussions regarding the Saudi contract and who was involved?"
print(f"QUERY: {query}\n")
response = answer_question(query)
print(f"ANSWER:\n{response}")

# ---

# Paths for caching evaluation results
TEST_SET_PATH = CACHE_DIR / "micro_test_set.json"
EVAL_RESULTS_PATH = CACHE_DIR / "evaluation_results.json"

def setup_gemini():
    key = get_secret(
        "GOOGLE_API_KEY",
        "Enter Google API Key (or press Enter to load pre-computed results from disk): "
    )
    if not key:
        print("No API key provided. System will attempt to load results from artifacts/ folder.")
        return None

    genai.configure(api_key=key)
    return genai.GenerativeModel('gemini-2.5-flash')

gemini_model = setup_gemini()

# ---

def generate_ground_truth(df, n=5):
    # Check for cache first
    if TEST_SET_PATH.exists():
        print(f"Loading test set from {TEST_SET_PATH}...")
        with open(TEST_SET_PATH, 'r') as f:
            return json.load(f)

    if not gemini_model:
        print("No API key and no cached test set found.")
        return []

    print("Generating new test set via Gemini API...")
    sample_df = df[df["body"].str.len() > 200].sample(n)
    test_set = []

    for _, row in sample_df.iterrows():
        context = build_header(row) + "\n\nBody: " + row["body"]
        prompt = f"Based on the following email passage, generate one complex question and its correct answer. Return JSON with keys 'question' and 'answer'.\n\nEMAIL PASSAGE:\n{context}"

        try:
            response = gemini_model.generate_content(prompt)
            json_str = response.text.strip().replace('```json', '').replace('```', '')
            qa = json.loads(json_str)
            qa["source_doc_ids"] = row["doc_ids"]
            test_set.append(qa)
            time.sleep(13) # sleep to avoid hitting the api rate limiter (for free plans you can only ask a small amout of api request every 5 minutes)
        except Exception as e:
            print(f'Gemini API Error: {e}')
            continue

    # Save cache
    with open(TEST_SET_PATH, 'w') as f:
        json.dump(test_set, f)
    return test_set

micro_test_set = generate_ground_truth(df_grouped)
print(f"Test set ready with {len(micro_test_set)} Q/A pairs.")

# ---

def evaluate_system(test_set):
    # Check for cache first
    if EVAL_RESULTS_PATH.exists():
        print(f"Loading pre-computed evaluation results from {EVAL_RESULTS_PATH}...")
        with open(EVAL_RESULTS_PATH, 'r') as f:
            return json.load(f)

    if not gemini_model:
        print("No API key and no cached results found.")
        return []

    print("Running RAG evaluation and Expert-Judging via Gemini API...")
    results = []
    for item in test_set:
        # Get RAG answer
        rag_answer = answer_question(item["question"])

        judge_prompt = f"""You are an expert judge evaluating a RAG system.
Compare the 'System Answer' against the 'Gold Answer' for the given 'Question'.

Question: {item['question']}
Gold Answer: {item['answer']}
System Answer: {rag_answer}

Score the System Answer from 1 to 10 based on:
1. Accuracy: Does it match the facts in the Gold Answer?
2. Faithfulness: Does it avoid hallucinating information not in the Gold Answer?
3. Citations: Does it mention Source Document IDs if they were provided?

Return only a JSON object with keys 'score' (int) and 'reasoning' (string)."""

        try:
            eval_response = gemini_model.generate_content(judge_prompt)
            eval_json = json.loads(eval_response.text.strip().replace('```json', '').replace('```', ''))
            results.append({
                "question": item["question"], "gold": item["answer"],
                "rag": rag_answer, "score": eval_json["score"], "reasoning": eval_json["reasoning"]
            })
        except: continue

    # Save results
    with open(EVAL_RESULTS_PATH, 'w') as f:
        json.dump(results, f)
    return results

evaluation_results = evaluate_system(micro_test_set)
if evaluation_results:
    eval_df = pd.DataFrame(evaluation_results)
    print(f"Average System Score: {eval_df['score'].mean():.2f}/10")
    display(eval_df)
else:
    print("No evaluation results to display. Please provide an API key or ensure cache files exist in artifacts/")

# ---

#code for testing
