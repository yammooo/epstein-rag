"""Provider-neutral helpers for RAG ground-truth generation, retrieval evaluation, and LLM answer judging."""

from __future__ import annotations

import json
import time
from pathlib import Path

import pandas as pd

from documents import build_header


def make_text_generator(provider: str, model: str, api_key: str | None):
    """Create a unified text generation closure for evaluation providers (Gemini or OpenRouter).

    Args:
        provider: Provider identifier ('gemini' or 'openrouter').
        model: Target model name/ID string.
        api_key: Authentication API key. Returns None if key is missing or empty.

    Returns:
        Callable[[str], str] | None: Function taking prompt string and returning text response,
            or None if no API key is provided.

    Raises:
        ValueError: If provider is not 'gemini' or 'openrouter'.
    """
    if not api_key:
        return None

    if provider == "gemini":
        from google import genai

        client = genai.Client(api_key=api_key)
        return lambda prompt: client.models.generate_content(model=model, contents=prompt).text

    if provider == "openrouter":
        import requests

        def generate(prompt: str) -> str:
            response = requests.post(
                "https://openrouter.ai/api/v1/chat/completions",
                headers={"Authorization": f"Bearer {api_key}"},
                json={
                    "model": model,
                    "messages": [{"role": "user", "content": prompt}],
                },
                timeout=60,
            )
            response.raise_for_status()
            return response.json()["choices"][0]["message"]["content"]

        return generate

    raise ValueError("Evaluation provider must be 'gemini' or 'openrouter'.")


def generate_ground_truth(
    documents: pd.DataFrame,
    generate_text,
    cache_path: Path,
    force: bool = False,
    n: int = 15,
    request_delay: int = 13,
    random_state: int = 42,
) -> list[dict]:
    """Generate or load a reproducible benchmark test set of source-grounded question/answer pairs.

    Samples candidate documents longer than 1000 characters, prompts the generation model to
    create multi-fact questions and ground-truth gold answers, attaches document source IDs,
    and caches the test set as a JSON file.

    Args:
        documents: Master document DataFrame complying with DOCUMENT_COLUMNS.
        generate_text: Text generation closure returned by make_text_generator().
        cache_path: File system path for JSON test set cache.
        force: If True, bypass cache and regenerate test set.
        n: Number of document evaluation cases to generate (default 15).
        request_delay: Delay in seconds between API calls for rate limiting (default 13).
        random_state: Random seed for reproducible document sampling (default 42).

    Returns:
        list[dict]: List of test item dicts containing 'question', 'answer', and 'source_ids'.
    """
    if cache_path.exists() and not force:
        with cache_path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        if _has_source_ids(cached):
            return cached
        print("Cached test set has no source IDs; generating a replacement.")
    if generate_text is None:
        return []

    candidates = documents[documents["text"].str.len() > 1000]
    sample = candidates.sample(n=min(n, len(candidates)), random_state=random_state)
    test_set = []

    for _, document in sample.iterrows():
        prompt = f"""Create exactly one source-grounded RAG evaluation case from the
document below.

The question must require combining 2–3 concrete facts from this document, such
as people, dates, locations, amounts, actions, or named documents.

Question requirements:
- One direct question, maximum 45 words.
- Include at least two specific, meaningful details from the document.
- Do not use vague wording such as "main topic", "what was discussed", or
  "what happened in the email".
- Do not mention source IDs or say "in this email/document".
- It must be answerable using only this document.

Answer requirements:
- Give the direct answer in 1–3 sentences, maximum 100 words.
- Include all facts needed to answer the question.
- Do not add facts not supported by the document.

Return only valid JSON in this exact format:
{{
  "question": "...",
  "answer": "...",
}}

SOURCE DOCUMENT:
{build_header(document)}

BODY:
{document["text"]}"""
        try:
            item = _parse_json_response(generate_text(prompt))
            item["source_ids"] = document["source_ids"]
            test_set.append(item)
            time.sleep(request_delay)
        except Exception as error:
            print(f"Test-set generation error: {error}")

    if test_set:
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(test_set, handle, ensure_ascii=False, indent=2)
    return test_set


def evaluate_retrieval(
    test_set: list[dict],
    retrieve_question,
    cutoffs: tuple[int, ...] = (5, 8, 10),
) -> pd.DataFrame:
    """Evaluate retrieval performance against a benchmark test set across cutoffs and rank metrics.

    Args:
        test_set: Benchmark test set containing 'question' and expected 'source_ids'.
        retrieve_question: Retrieval function taking (query, max_k) and returning retrieved chunks DataFrame.
        cutoffs: Depth cutoff thresholds for Hit@K calculation (default (5, 8, 10)).

    Returns:
        pd.DataFrame: Evaluation matrix listing question, expected source IDs, rank of first hit
            (first_hit_rank), and boolean hit flags at specified cutoffs (hit_at_5, hit_at_8, hit_at_10).
    """
    rows = []
    max_k = max(cutoffs)

    for item in test_set:
        expected_ids = set(map(str, item["source_ids"]))
        results = retrieve_question(item["question"], max_k)
        retrieved_ids = [set(map(str, source_ids)) for source_ids in results["source_ids"]]
        first_hit_rank = next(
            (
                rank
                for rank, source_ids in enumerate(retrieved_ids, start=1)
                if expected_ids & source_ids
            ),
            None,
        )

        row = {
            "question": item["question"],
            "expected_source_ids": item["source_ids"],
            "first_hit_rank": first_hit_rank,
        }
        row.update(
            {
                f"hit_at_{k}": first_hit_rank is not None and first_hit_rank <= k
                for k in cutoffs
            }
        )
        rows.append(row)

    return pd.DataFrame(rows)


def evaluate_answers(
    test_set: list[dict],
    answer_question,
    generate_text,
    cache_path: Path,
    force: bool = False,
    request_delay: int = 13,
) -> list[dict]:
    """Evaluate generated RAG system answers against gold answers using an LLM-as-a-Judge.

    Args:
        test_set: Benchmark test set containing questions and gold answers.
        answer_question: RAG generation function taking query string and returning answer.
        generate_text: LLM generation closure for judge model.
        cache_path: File system path for JSON evaluation cache.
        force: If True, bypass cache and re-evaluate answers.
        request_delay: Delay in seconds between API requests (default 13).

    Returns:
        list[dict]: Evaluation entries containing 'question', 'gold', 'source_ids', 'rag',
            numerical 'score' (1-10), and text 'reasoning'.
    """
    if cache_path.exists() and not force:
        with cache_path.open(encoding="utf-8") as handle:
            cached = json.load(handle)
        if _matches_test_set(cached, test_set):
            return cached
        print("Cached evaluation does not match the current test set; evaluating again.")
    if generate_text is None:
        return []

    results = []
    for item in test_set:
        rag_answer = answer_question(item["question"])
        judge_prompt = f"""You are an expert judge evaluating a RAG system.
Compare the System Answer against the Gold Answer for the Question.

Question: {item['question']}
Gold Answer: {item['answer']}
System Answer: {rag_answer}
Expected Source Document IDs: {item.get('source_ids', [])}

Score from 1 to 10 based on accuracy, faithfulness, and citations.
Return only JSON with keys 'score' (int) and 'reasoning' (string)."""
        try:
            judgment = _parse_json_response(generate_text(judge_prompt))
            results.append(
                {
                    "question": item["question"],
                    "gold": item["answer"],
                    "source_ids": item.get("source_ids", []),
                    "rag": rag_answer,
                    "score": judgment["score"],
                    "reasoning": judgment["reasoning"],
                }
            )
            time.sleep(request_delay)
        except Exception as error:
            print(f"Evaluation error: {error}")

    if results:
        with cache_path.open("w", encoding="utf-8") as handle:
            json.dump(results, handle, ensure_ascii=False, indent=2)
    return results


def _has_source_ids(test_set: object) -> bool:
    """Validate that a cached test set is a non-empty list of items with valid source_ids.

    Args:
        test_set: Object loaded from JSON cache.

    Returns:
        bool: True if test set structure is valid and contains source IDs; False otherwise.
    """
    return (
        isinstance(test_set, list)
        and bool(test_set)
        and all(isinstance(item.get("source_ids"), list) and item["source_ids"] for item in test_set)
    )


def _matches_test_set(results: object, test_set: list[dict]) -> bool:
    """Check if cached evaluation results match the question/answer/source_ids of the active test set.

    Args:
        results: Object loaded from cached evaluation JSON file.
        test_set: Active test set list.

    Returns:
        bool: True if cached evaluation matches current test set 1-to-1; False otherwise.
    """
    if not isinstance(results, list):
        return False
    cached_cases = [(item.get("question"), item.get("gold"), item.get("source_ids")) for item in results]
    current_cases = [(item.get("question"), item.get("answer"), item.get("source_ids")) for item in test_set]
    return cached_cases == current_cases


def _parse_json_response(text: str) -> dict:
    """Strip markdown code fences and parse JSON string into a Python dictionary.

    Args:
        text: Raw text output string from LLM response.

    Returns:
        dict: Parsed dictionary payload.
    """
    return json.loads(text.strip().removeprefix("```json").removesuffix("```").strip())

