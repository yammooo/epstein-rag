"""Document ingestion and preparation for the RAG notebook."""

from __future__ import annotations

import hashlib
import json
import re
import unicodedata
from pathlib import Path

import pandas as pd


DOCUMENT_COLUMNS = [
    "source_ids",
    "source_type",
    "title",
    "text",
    "date",
    "sender",
    "recipients",
    "page",
    "n_original_rows",
]


def load_email_dataset(cache_path: Path, force: bool = False) -> pd.DataFrame:
    """Load the email dataset, using a local Parquet cache when available."""
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    from datasets import load_dataset

    dataset = load_dataset("KillerShoaib/Jeffrey-Epstein-Emails-From-Epstein-Files")
    emails = dataset["train"].to_pandas()
    emails.to_parquet(cache_path, index=False)
    return emails


def normalize_metadata(value: object) -> str:
    """Normalize one-line metadata without turning missing values into 'nan'."""
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def normalize_body(value: object) -> str:
    """Preserve paragraphs while removing OCR and email whitespace noise."""
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def normalize_email_fields(emails: pd.DataFrame) -> pd.DataFrame:
    """Normalize the email fields used for profiling, deduplication, and retrieval."""
    emails = emails.copy()
    for column in ["subject", "preview", "from_name", "from_email", "to", "date"]:
        emails[column] = emails[column].map(normalize_metadata)
    emails["body"] = emails["body"].map(normalize_body)
    return emails


def count_recipients(value: object) -> int:
    """Estimate recipient count from email address brackets in a To field."""
    if not isinstance(value, str) or not value.strip():
        return 0
    return max(1, len(re.findall(r"<[^<>]+>", value)))


def add_email_fields(emails: pd.DataFrame) -> pd.DataFrame:
    """Add parsed date and recipient-count fields used by email-only analysis."""
    emails = emails.copy()
    emails["num_recipients"] = emails["to"].map(count_recipients)
    emails["parsed_date"] = pd.to_datetime(emails["date"], errors="coerce")
    return emails


def email_dedup_key() -> list[str]:
    """Return the exact-content key used to collapse duplicate email records."""
    return ["subject", "from_email", "to", "num_recipients", "date", "body"]


def duplicate_email_groups(emails: pd.DataFrame) -> pd.DataFrame:
    """Summarize duplicate emails for display in the notebook."""
    key = email_dedup_key()
    return (
        emails[emails.duplicated(subset=key, keep=False)]
        .groupby(key, dropna=False)
        .agg(
            n_rows=("doc_id", "size"),
            doc_ids=("doc_id", lambda ids: list(ids)),
            n_unique_doc_ids=("doc_id", "nunique"),
            n_unique_previews=("preview", "nunique"),
            n_unique_from_names=("from_name", "nunique"),
        )
        .reset_index()
        .sort_values("n_rows", ascending=False)
    )


def deduplicate_emails(emails: pd.DataFrame) -> pd.DataFrame:
    """Collapse identical emails while retaining every original source ID."""
    key = email_dedup_key()
    grouped = (
        emails.groupby(key, dropna=False)
        .agg(
            source_ids=("doc_id", lambda ids: sorted({str(value) for value in ids})),
            n_original_rows=("doc_id", "size"),
            sender_name=("from_name", "first"),
            parsed_date=("parsed_date", "first"),
        )
        .reset_index()
    )
    return grouped.rename(
        columns={
            "subject": "title",
            "from_email": "sender_email",
            "to": "recipients",
            "body": "text",
        }
    )


def emails_to_documents(grouped_emails: pd.DataFrame) -> pd.DataFrame:
    """Convert deduplicated emails to the common document schema."""
    documents = pd.DataFrame(
        {
            "source_ids": grouped_emails["source_ids"],
            "source_type": "email",
            "title": grouped_emails["title"],
            "text": grouped_emails["text"],
            "date": grouped_emails["date"],
            "sender": grouped_emails.apply(
                lambda row: _format_sender(row["sender_name"], row["sender_email"]), axis=1
            ),
            "recipients": grouped_emails["recipients"],
            "page": pd.NA,
            "n_original_rows": grouped_emails["n_original_rows"],
        }
    )
    return documents[DOCUMENT_COLUMNS]


def load_or_ocr_pdfs(
    pdf_dir: Path,
    cache_path: Path,
    force: bool = False,
) -> pd.DataFrame:
    """OCR PDFs with PP-OCRv6 into one common-schema record per page."""
    pdf_paths = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    settings = {
        "ocr_version": "PP-OCRv6",
        "engine": "transformers",
        "use_doc_orientation_classify": False,
        "use_doc_unwarping": False,
        "use_textline_orientation": False,
    }
    fingerprint = _files_fingerprint(pdf_paths, settings)
    cached = _load_frame_if_current(cache_path, fingerprint, force)
    if cached is not None:
        return cached

    if not pdf_paths:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)

    import torch
    from paddleocr import PaddleOCR

    device = "gpu" if torch.cuda.is_available() else "cpu"
    ocr = PaddleOCR(**settings, device=device)

    records: list[dict[str, object]] = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name}...")
        for page_number, result in enumerate(ocr.predict_iter(str(pdf_path)), start=1):
            text = _ocr_result_text(result)
            if not text:
                print(f"Skipping empty OCR page {page_number} in {pdf_path.name}.")
                continue

            records.append(
                {
                    "source_ids": [f"pdf:{pdf_path.stem}:p{page_number}"],
                    "source_type": "pdf",
                    "title": pdf_path.stem,
                    "text": text,
                    "date": "",
                    "sender": "",
                    "recipients": "",
                    "page": page_number,
                    "n_original_rows": 1,
                }
            )

    documents = pd.DataFrame(records, columns=DOCUMENT_COLUMNS)
    _save_frame_with_fingerprint(documents, cache_path, fingerprint)
    return documents


def _ocr_result_text(result) -> str:
    """Extract normalized text lines from one PaddleOCR page result."""
    lines = result.json["res"]["rec_texts"]
    return normalize_body("\n".join(line.strip() for line in lines if line and line.strip()))


def combine_documents(*frames: pd.DataFrame) -> pd.DataFrame:
    """Combine source-specific documents and validate the common RAG input."""
    documents = pd.concat(frames, ignore_index=True)
    documents = documents[DOCUMENT_COLUMNS]
    if documents["text"].str.strip().eq("").any():
        raise ValueError("Documents with empty text cannot be indexed.")
    if not documents["source_ids"].map(lambda source_ids: len(source_ids) > 0).all():
        raise ValueError("Every document needs at least one source ID.")
    return documents


def build_header(document: pd.Series) -> str:
    """Create a retrieval header that preserves source-specific provenance."""
    source_ids = ", ".join(map(str, document["source_ids"][:5]))
    if len(document["source_ids"]) > 5:
        source_ids += f", ... ({len(document['source_ids'])} total)"

    title = _truncate(document["title"], 200)
    if document["source_type"] == "pdf":
        return f"""Type: PDF
Document: {title}
Page: {document['page']}
Source document IDs: {source_ids}"""

    return f"""Type: Email
Subject: {title}
From: {_truncate(document['sender'], 200)}
To: {_truncate(document['recipients'], 200)}
Date: {document['date']}
Source document IDs: {source_ids}"""


def load_or_build_chunks(
    documents: pd.DataFrame,
    tokenizer,
    cache_path: Path,
    force: bool = False,
    max_tokens: int = 512,
) -> pd.DataFrame:
    """Build token-aware chunks and invalidate the cache when documents change."""
    fingerprint = _frame_fingerprint(
        documents,
        {"max_tokens": max_tokens, "tokenizer": getattr(tokenizer, "name_or_path", "")},
    )
    cached = _load_frame_if_current(cache_path, fingerprint, force)
    if cached is not None:
        return cached

    from langchain_text_splitters import RecursiveCharacterTextSplitter

    records: list[dict[str, object]] = []
    for document_index, document in documents.iterrows():
        header = build_header(document)
        header_tokens = len(tokenizer.encode(header, add_special_tokens=False))
        max_body_tokens = max_tokens - header_tokens - 15
        if max_body_tokens < 1:
            raise ValueError(f"Header for {document['source_ids']} is too long to chunk.")

        splitter = RecursiveCharacterTextSplitter.from_huggingface_tokenizer(
            tokenizer=tokenizer,
            chunk_size=max_body_tokens,
            chunk_overlap=int(max_body_tokens * 0.15),
            separators=["\n\n", "\n", ". ", " ", ""],
        )
        body_chunks = splitter.split_text(document["text"])

        for chunk_index, body_chunk in enumerate(body_chunks):
            text = f"{header}\n\nBody chunk {chunk_index + 1} of {len(body_chunks)}:\n{body_chunk}"
            records.append(
                {
                    "chunk_id": f"{document_index}_{chunk_index}",
                    "document_index": document_index,
                    "chunk_index": chunk_index,
                    "num_chunks": len(body_chunks),
                    "source_ids": document["source_ids"],
                    "source_type": document["source_type"],
                    "title": document["title"],
                    "date": document["date"],
                    "page": document["page"],
                    "text": text,
                }
            )

    chunks = pd.DataFrame(records)
    _save_frame_with_fingerprint(chunks, cache_path, fingerprint)
    return chunks


def _format_sender(name: object, email: object) -> str:
    name = normalize_metadata(name)
    email = normalize_metadata(email)
    if name and email:
        return f"{name} <{email}>"
    return name or email

def _truncate(value: object, limit: int) -> str:
    value = normalize_metadata(value)
    return value if len(value) <= limit else value[:limit].rstrip() + " ... (truncated)"



def _files_fingerprint(paths: list[Path], settings: dict[str, object]) -> str:
    payload = {
        "files": [
            {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
        "settings": settings,
    }
    return _fingerprint(payload)


def _frame_fingerprint(frame: pd.DataFrame, settings: dict[str, object]) -> str:
    payload = {
        "records": json.loads(frame.to_json(orient="records", date_format="iso")),
        "settings": settings,
    }
    return _fingerprint(payload)


def _fingerprint(payload: dict[str, object]) -> str:
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _meta_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _load_frame_if_current(cache_path: Path, fingerprint: str, force: bool) -> pd.DataFrame | None:
    meta_path = _meta_path(cache_path)
    if force or not cache_path.exists() or not meta_path.exists():
        return None

    with meta_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("fingerprint") != fingerprint:
        return None
    return pd.read_parquet(cache_path)


def _save_frame_with_fingerprint(frame: pd.DataFrame, cache_path: Path, fingerprint: str) -> None:
    frame.to_parquet(cache_path, index=False)
    with _meta_path(cache_path).open("w", encoding="utf-8") as handle:
        json.dump({"fingerprint": fingerprint}, handle)
