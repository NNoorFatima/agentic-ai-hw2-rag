"""
Load CRAG dataset from dataset/crag_task_1_and_2_dev_v4.jsonl.
Do not change the file path or this module's location in the project.
"""

import json
from pathlib import Path
from typing import Generator, Optional


# Default path relative to project root (when running from project root)
DEFAULT_DATASET_PATH = "dataset/crag_task_1_and_2_dev_v4.jsonl"


def load_examples(
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Load CRAG JSONL and yield one example per row.

    Args:
        path: Path to the .jsonl file. Defaults to dataset/crag_task_1_and_2_dev_v4.jsonl
              (relative to current working directory).
        limit: If set, yield at most this many examples (useful for testing).

    Yields:
        Dict for each example with at least:
          - query: str
          - answer: str
          - alt_ans: list[str]
          - search_results: list[dict] with 'page_snippet', 'page_name', 'page_url', etc.
        Also includes interaction_id, domain, question_type for debugging.
    """
    file_path = Path(path or DEFAULT_DATASET_PATH)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    if not file_path.exists():
        raise FileNotFoundError(f"Dataset not found: {file_path}")

    count = 0
    with open(file_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                item = json.loads(line)
            except json.JSONDecodeError as e:
                raise ValueError(f"Invalid JSON at line {count + 1}: {e}") from e

            # Yield the fields needed for RAG + optional metadata
            # Use .get() for search_results so missing or malformed rows don't crash the loader
            search_results = item.get("search_results")
            if not isinstance(search_results, list):
                search_results = []

            yield {
                "interaction_id": item.get("interaction_id"),
                "query": item.get("query", ""),
                "answer": item.get("answer", ""),
                "alt_ans": item.get("alt_ans") or [],
                "search_results": search_results,
                "domain": item.get("domain"),
                "question_type": item.get("question_type"),
            }
            count += 1
            if limit is not None and count >= limit:
                return


def get_passages_for_retrieval(example: dict, use_snippet: bool = True) -> list[str]:
    """
    Get a list of text passages from one example for embedding/retrieval.
    Each passage is either page_snippet (short) or extracted text from page_result (HTML).

    Args:
        example: One item yielded by load_examples().
        use_snippet: If True, use page_snippet; if False, use page_result (raw HTML - caller may strip tags).

    Returns:
        List of 5 strings (one per search result).
    """
    passages = []
    for sr in example["search_results"]:
        if use_snippet:
            passages.append(sr.get("page_snippet") or "")
        else:
            passages.append(sr.get("page_result") or "")
    return passages


if __name__ == "__main__":
    # Quick check: load first 3 examples and print summary
    print("Loading first 3 examples from CRAG dataset...\n")
    for i, ex in enumerate(load_examples(limit=3)):
        print(f"--- Example {i + 1} ---")
        print(f"  interaction_id: {ex['interaction_id']}")
        print(f"  query: {ex['query'][:80]}...")
        print(f"  answer: {ex['answer'][:60]}...")
        print(f"  alt_ans count: {len(ex['alt_ans'])}")
        print(f"  search_results count: {len(ex['search_results'])}")
        if ex["search_results"]:
            first = ex["search_results"][0]
            print(f"  first result keys: {list(first.keys())}")
            snip = (first.get("page_snippet") or "")[:100]
            print(f"  first snippet: {snip}...")
        print()
    print("Data loader check done.")
