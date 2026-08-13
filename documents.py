"""Document ingestion, cleaning, metadata normalization, deduplication, OCR, and chunking for RAG."""

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
    """Load the Epstein email dataset from Hugging Face or read from local Parquet cache.

    Args:
        cache_path: File system path pointing to the Parquet cache location.
        force: If True, bypass existing cache and re-download from Hugging Face.

    Returns:
        pd.DataFrame containing raw email dataset records.
    """
    if cache_path.exists() and not force:
        return pd.read_parquet(cache_path)

    from datasets import load_dataset

    dataset = load_dataset("KillerShoaib/Jeffrey-Epstein-Emails-From-Epstein-Files")
    emails = dataset["train"].to_pandas()
    emails.to_parquet(cache_path, index=False)
    return emails


def normalize_metadata(value: object) -> str:
    """Normalize a single-line metadata field using NFKC normalization and whitespace collapsing.

    Args:
        value: Input metadata value (string, float, NaN, or non-string object).

    Returns:
        Cleaned metadata string, or an empty string if input is missing/NaN.
    """
    if pd.isna(value):
        return ""
    return re.sub(r"\s+", " ", unicodedata.normalize("NFKC", str(value))).strip()


def normalize_body(value: object) -> str:
    """Clean and standardize document body text while preserving paragraph structure.

    Performs Unicode NFKC normalization, standardizes CRLF line endings to LF,
    collapses inline spaces/tabs, and caps consecutive blank lines to at most two newlines.

    Args:
        value: Input text body value (string, NaN, or non-string object).

    Returns:
        Cleaned text string with standardized paragraph breaks.
    """
    if pd.isna(value):
        return ""

    text = unicodedata.normalize("NFKC", str(value))
    text = text.replace("\r\n", "\n").replace("\r", "\n")
    text = re.sub(r"[ \t]+", " ", text)
    text = re.sub(r"\n{3,}", "\n\n", text)
    text = text.strip()
    return text


def normalize_email_fields(emails: pd.DataFrame) -> pd.DataFrame:
    """Normalize metadata and body text fields across an entire email DataFrame.

    Args:
        emails: Input DataFrame containing raw email columns ('subject', 'preview',
            'from_name', 'from_email', 'to', 'date', 'body').

    Returns:
        pd.DataFrame: A copy of the input DataFrame with normalized text fields.
    """
    emails = emails.copy()
    for column in ["subject", "preview", "from_name", "from_email", "to", "date"]:
        emails[column] = emails[column].map(normalize_metadata)
    emails["body"] = emails["body"].map(normalize_body)
    return emails


def count_recipients(value: object) -> int:
    """Estimate the number of email recipients by counting angle-bracketed email patterns.

    Args:
        value: Raw text from an email 'To' header field.

    Returns:
        Estimated count of recipient addresses (0 if empty/invalid, minimum 1 if text is present).
    """
    if not isinstance(value, str) or not value.strip():
        return 0
    return max(1, len(re.findall(r"<[^<>]+>", value)))


def add_email_fields(emails: pd.DataFrame) -> pd.DataFrame:
    """Enrich an email DataFrame with computed recipient count and parsed datetime columns.

    Args:
        emails: DataFrame containing normalized email fields ('to', 'date').

    Returns:
        pd.DataFrame: Copy of input DataFrame enriched with 'num_recipients' (int)
            and 'parsed_date' (datetime64[ns]) columns.
    """
    emails = emails.copy()
    emails["num_recipients"] = emails["to"].map(count_recipients)
    emails["parsed_date"] = pd.to_datetime(emails["date"], errors="coerce")
    return emails


def email_dedup_key() -> list[str]:
    """Return column names defining exact content equality for email deduplication.

    Returns:
        List of column names used as composite unique key:
        ['subject', 'from_email', 'to', 'num_recipients', 'date', 'body'].
    """
    return ["subject", "from_email", "to", "num_recipients", "date", "body"]


def duplicate_email_groups(emails: pd.DataFrame) -> pd.DataFrame:
    """Summarize duplicate email groups for exploratory analysis and profiling.

    Args:
        emails: DataFrame with prepared email fields and original 'doc_id' column.

    Returns:
        pd.DataFrame: Aggregated DataFrame listing duplicate groups with row counts,
            associated document ID lists, and uniqueness metrics, sorted descending by size.
    """
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
    """Collapse identical emails while preserving all original document source IDs.

    Groups records matching the exact deduplication key and consolidates their
    doc_ids into a sorted list of unique source IDs.

    Args:
        emails: DataFrame containing prepared email records with 'doc_id'.

    Returns:
        pd.DataFrame: Grouped DataFrame with consolidated 'source_ids' lists,
            original row counts, and standardized column names ('title', 'sender_email',
            'recipients', 'text').
    """
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
    """Map deduplicated email records into the unified RAG document schema.

    Args:
        grouped_emails: Deduplicated emails DataFrame from deduplicate_emails().

    Returns:
        pd.DataFrame: DataFrame strictly conforming to DOCUMENT_COLUMNS schema with
            source_type='email'.
    """
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
    dpi: int = 300,
) -> pd.DataFrame:
    """Extract page text from PDF files via Tesseract OCR, cached with file fingerprinting.

    Renders each PDF page to an image, executes PyTesseract optical character recognition,
    normalizes the extracted body text, and produces document records for each valid page.
    Automatically invalidates cache if input PDFs or rendering settings change.

    Args:
        pdf_dir: Directory containing input '.pdf' files.
        cache_path: Path to Parquet cache file.
        force: If True, bypass cache and re-run OCR execution.
        dpi: Image rendering resolution for pdf2image conversion (default 300).

    Returns:
        pd.DataFrame: DataFrame conforming to DOCUMENT_COLUMNS with source_type='pdf'.
    """
    pdf_paths = sorted(pdf_dir.glob("*.pdf")) if pdf_dir.exists() else []
    fingerprint = _files_fingerprint(pdf_paths, {"dpi": dpi})
    cached = _load_frame_if_current(cache_path, fingerprint, force)
    if cached is not None:
        return cached

    if not pdf_paths:
        return pd.DataFrame(columns=DOCUMENT_COLUMNS)

    import cv2
    import numpy as np
    import pytesseract
    from pdf2image import convert_from_path

    records: list[dict[str, object]] = []
    for pdf_path in pdf_paths:
        print(f"Processing {pdf_path.name}...")
        pages = convert_from_path(pdf_path, dpi=dpi)
        for page_number, page in enumerate(pages, start=1):
            image = np.array(page)[:, :, ::-1].copy()
            text = normalize_body(pytesseract.image_to_string(image, config="--psm 3"))
            if not text:
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


def combine_documents(*frames: pd.DataFrame) -> pd.DataFrame:
    """Concatenate document DataFrames from multiple sources and validate data completeness.

    Args:
        *frames: Arbitrary number of document DataFrames matching DOCUMENT_COLUMNS.

    Returns:
        pd.DataFrame: Combined unified document DataFrame.

    Raises:
        ValueError: If any document contains empty text or lacks source IDs.
    """
    documents = pd.concat(frames, ignore_index=True)
    documents = documents[DOCUMENT_COLUMNS]
    if documents["text"].str.strip().eq("").any():
        raise ValueError("Documents with empty text cannot be indexed.")
    if not documents["source_ids"].map(lambda source_ids: len(source_ids) > 0).all():
        raise ValueError("Every document needs at least one source ID.")
    return documents


def build_header(document: pd.Series) -> str:
    """Construct a structured context header preserving document provenance.

    Args:
        document: A single row (pd.Series) from a document DataFrame.

    Returns:
        Formatted multi-line text header string containing metadata (Type, Subject/Title,
        Sender, Recipients, Page, Source document IDs).
    """
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
    """Split documents into token-budget-aware chunks with prepended provenance headers.

    Calculates header token count, allocates remaining token budget for document body,
    splits text recursively using Hugging Face tokenizer token counts, prepends metadata header
    to each chunk, and caches output with SHA-256 fingerprint validation.

    Args:
        documents: Input document DataFrame adhering to DOCUMENT_COLUMNS schema.
        tokenizer: Pre-trained Hugging Face tokenizer instance.
        cache_path: File system path to Parquet chunk cache.
        force: If True, re-split documents ignoring existing cache.
        max_tokens: Total token limit allowed per chunk including header (default 512).

    Returns:
        pd.DataFrame: DataFrame containing chunk records with chunk_id, document index,
            chunk index, headers, and full prepended text.

    Raises:
        ValueError: If a document header exceeds max_tokens allocation.
    """
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
    """Format sender display name and email address into a standard 'Name <email>' string.

    Args:
        name: Sender display name.
        email: Sender email address.

    Returns:
        Formatted email sender string, or single non-empty string fallback.
    """
    name = normalize_metadata(name)
    email = normalize_metadata(email)
    if name and email:
        return f"{name} <{email}>"
    return name or email


def _truncate(value: object, limit: int) -> str:
    """Truncate text to specified character limit, appending truncation notice if shortened.

    Args:
        value: Raw text object to truncate.
        limit: Maximum allowable character length.

    Returns:
        Truncated text string.
    """
    value = normalize_metadata(value)
    return value if len(value) <= limit else value[:limit].rstrip() + " ... (truncated)"


def _files_fingerprint(paths: list[Path], settings: dict[str, object]) -> str:
    """Compute deterministic SHA-256 fingerprint for a set of files and configuration settings.

    Args:
        paths: List of file system paths.
        settings: Key-value dictionary of processing settings.

    Returns:
        64-character SHA-256 hex digest string.
    """
    payload = {
        "files": [
            {"name": path.name, "size": path.stat().st_size, "mtime_ns": path.stat().st_mtime_ns}
            for path in paths
        ],
        "settings": settings,
    }
    return _fingerprint(payload)


def _frame_fingerprint(frame: pd.DataFrame, settings: dict[str, object]) -> str:
    """Compute deterministic SHA-256 fingerprint for a DataFrame and configuration settings.

    Args:
        frame: Input pandas DataFrame.
        settings: Key-value dictionary of processing parameters.

    Returns:
        64-character SHA-256 hex digest string.
    """
    payload = {
        "records": json.loads(frame.to_json(orient="records", date_format="iso")),
        "settings": settings,
    }
    return _fingerprint(payload)


def _fingerprint(payload: dict[str, object]) -> str:
    """Compute SHA-256 hex digest from a JSON-serializable dictionary.

    Args:
        payload: Dictionary payload to hash.

    Returns:
        64-character SHA-256 hex digest string.
    """
    encoded = json.dumps(payload, sort_keys=True, ensure_ascii=False).encode("utf-8")
    return hashlib.sha256(encoded).hexdigest()


def _meta_path(cache_path: Path) -> Path:
    """Derive the .meta.json sidecar file path corresponding to a cache file.

    Args:
        cache_path: Primary cache file path (e.g., data.parquet).

    Returns:
        Path object pointing to sidecar metadata JSON file (e.g., data.parquet.meta.json).
    """
    return cache_path.with_suffix(cache_path.suffix + ".meta.json")


def _load_frame_if_current(cache_path: Path, fingerprint: str, force: bool) -> pd.DataFrame | None:
    """Load cached DataFrame if cache exists, metadata sidecar exists, and fingerprint matches.

    Args:
        cache_path: Path to cached Parquet file.
        fingerprint: Target SHA-256 fingerprint string.
        force: If True, forces cache miss and returns None.

    Returns:
        Loaded DataFrame if valid and current; None otherwise.
    """
    meta_path = _meta_path(cache_path)
    if force or not cache_path.exists() or not meta_path.exists():
        return None

    with meta_path.open(encoding="utf-8") as handle:
        metadata = json.load(handle)
    if metadata.get("fingerprint") != fingerprint:
        return None
    return pd.read_parquet(cache_path)


def _save_frame_with_fingerprint(frame: pd.DataFrame, cache_path: Path, fingerprint: str) -> None:
    """Save DataFrame as Parquet file and write its SHA-256 fingerprint to metadata sidecar.

    Args:
        frame: DataFrame to serialize.
        cache_path: Parquet target file path.
        fingerprint: SHA-256 fingerprint string representing dataset and parameter state.
    """
    frame.to_parquet(cache_path, index=False)
    with _meta_path(cache_path).open("w", encoding="utf-8") as handle:
        json.dump({"fingerprint": fingerprint}, handle)

