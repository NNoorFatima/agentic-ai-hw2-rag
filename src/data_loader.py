"""
data_loader.py
--------------
Load CRAG dataset. Supports both the original load_examples() API
and the load_dataset() wrapper used by corpus.py / run_evaluation.py.
"""

import json
from pathlib import Path
from typing import Generator, Iterator, Optional, List, Tuple, Any

DEFAULT_DATASET_PATH = "dataset/crag_task_1_and_2_dev_v4.jsonl"


# ── Original API (from uploaded data_loader.py) ───────────────────────────────

def load_examples(
    path: Optional[str] = None,
    limit: Optional[int] = None,
) -> Generator[dict, None, None]:
    """
    Load CRAG JSONL and yield one dict per row.
    Keys: interaction_id, query, answer, alt_ans, search_results, domain, question_type
    """
    file_path = Path(path or DEFAULT_DATASET_PATH)
    if not file_path.is_absolute():
        file_path = Path.cwd() / file_path

    if not file_path.exists():
        # Also try one level up (src/ subdirectory case)
        alt = Path.cwd().parent / file_path
        if alt.exists():
            file_path = alt
        else:
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

            search_results = item.get("search_results")
            if not isinstance(search_results, list):
                search_results = []

            yield {
                "interaction_id": item.get("interaction_id"),
                "query":          item.get("query", ""),
                "answer":         item.get("answer", ""),
                "alt_ans":        item.get("alt_ans") or [],
                "search_results": search_results,
                "domain":         item.get("domain"),
                "question_type":  item.get("question_type"),
            }
            count += 1
            if limit is not None and count >= limit:
                return


def get_passages_for_retrieval(example: dict, use_snippet: bool = True) -> List[str]:
    """Return list of text passages from one example's search_results."""
    passages = []
    for sr in example["search_results"]:
        key = "page_snippet" if use_snippet else "page_result"
        passages.append(sr.get(key) or "")
    return passages


# ── Compatibility wrapper used by corpus.py / run_evaluation.py ───────────────

def load_dataset(
    dataset_path: str,
    max_examples: Optional[int] = None,
) -> Iterator[Tuple[str, str, Any, List[dict]]]:
    """
    Wrapper around load_examples() that yields
    (query, answer, alt_ans, search_results) tuples.
    Used by corpus.py and run_evaluation.py.
    """
    for ex in load_examples(path=dataset_path, limit=max_examples):
        yield (
            ex["query"],
            ex["answer"],
            ex["alt_ans"],
            ex["search_results"],
        )


def load_all(dataset_path: str, max_examples: Optional[int] = None) -> List[dict]:
    """Load entire dataset into a list of dicts."""
    return list(load_examples(path=dataset_path, limit=max_examples))


if __name__ == "__main__":
    print("Loading first 3 examples from CRAG dataset...\n")
    for i, ex in enumerate(load_examples(limit=3)):
        print(f"--- Example {i + 1} ---")
        print(f"  interaction_id: {ex['interaction_id']}")
        print(f"  query:          {ex['query'][:80]}...")
        print(f"  answer:         {ex['answer'][:60]}...")
        print(f"  alt_ans count:  {len(ex['alt_ans'])}")
        print(f"  search_results: {len(ex['search_results'])}")
        if ex["search_results"]:
            first = ex["search_results"][0]
            snip  = (first.get("page_snippet") or "")[:100]
            print(f"  first snippet:  {snip}...")
        print()
    print("Data loader check done.")