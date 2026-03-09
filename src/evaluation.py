"""
evaluation.py
-------------
Evaluation for RAG pipelines. Compares predicted answers to gold answers
with aggressive normalisation to handle LLM citation noise.
"""

import re
import string
from typing import List, Optional, Union


# ── Normalisation ─────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    """
    Strip all noise added by LLM pipelines and normalise for comparison.
    Removes: citations [1], References:, Source:, I don't know qualifiers,
             articles, punctuation, extra whitespace.
    """
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()

    # Remove everything after "references:" or "note:"
    text = re.sub(r'references?:.*',          '', text, flags=re.IGNORECASE | re.DOTALL)
    text = re.sub(r'note:.*',                 '', text, flags=re.IGNORECASE | re.DOTALL)

    # Remove source/citation markers
    text = re.sub(r'\[source:[^\]]*\]',       '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(source[^\)]*\)',        '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[\d,\s]+\]',            '', text)   # [1] [2,3]
    text = re.sub(r'\bsource:\s*https?://\S+','', text, flags=re.IGNORECASE)

    # Remove "I don't know" qualifiers so they don't accidentally match
    text = re.sub(r"i don'?t know\.?",        '', text, flags=re.IGNORECASE)

    # Remove articles
    text = re.sub(r'\b(a|an|the)\b',          ' ', text)

    # Remove punctuation
    text = text.translate(str.maketrans('', '', string.punctuation))

    # Collapse whitespace
    return re.sub(r'\s+', ' ', text).strip()


def _tokens(text: str) -> set:
    return set(normalize(text).split())


def _token_f1(pred: str, gold: str) -> float:
    p = _tokens(pred)
    g = _tokens(gold)
    if not p or not g:
        return 0.0
    common = p & g
    if not common:
        return 0.0
    precision = len(common) / len(p)
    recall    = len(common) / len(g)
    return 2 * precision * recall / (precision + recall)


# ── Correctness check ─────────────────────────────────────────────────────────

def is_correct(
    pred: str,
    answer: str,
    alt_ans: Union[str, List[str], None] = None,
    f1_threshold: float = 0.5,
) -> bool:
    """
    Returns True if prediction matches any gold answer using:
    1. Exact match after normalisation
    2. Gold contained inside prediction
    3. Prediction contained inside gold
    4. Token F1 >= f1_threshold

    Note: all comparisons are done AFTER aggressive normalisation,
    so "[Christopher Nolan [4]]" and "Christopher Nolan" will match.
    """
    pred_norm = normalize(pred)
    if not pred_norm:
        return False

    # Build candidate list
    candidates = []
    if answer:
        candidates.append(answer)
    if isinstance(alt_ans, str) and alt_ans:
        candidates.append(alt_ans)
    elif isinstance(alt_ans, list):
        candidates.extend([a for a in alt_ans if a])

    for gold in candidates:
        gold_norm = normalize(gold)
        if not gold_norm:
            continue

        if pred_norm == gold_norm:                return True
        if gold_norm in pred_norm:                return True
        if pred_norm in gold_norm:                return True
        if _token_f1(pred, gold) >= f1_threshold: return True

    return False


# ── Compute accuracy ──────────────────────────────────────────────────────────

def compute_accuracy(results: List[dict], f1_threshold: float = 0.5) -> dict:
    """
    Parameters
    ----------
    results : list of dicts with keys: predicted_answer, answer, alt_ans

    Returns
    -------
    dict with accuracy, correct, total, per_result
    """
    correct    = 0
    per_result = []

    for r in results:
        pred = r.get("predicted_answer", "")
        gold = r.get("answer", "")
        alt  = r.get("alt_ans", None)
        ok   = r.get("correct", None)   # use pre-computed if available

        if ok is None:
            ok = is_correct(pred, gold, alt, f1_threshold)

        per_result.append(ok)
        if ok:
            correct += 1

    total    = len(results)
    accuracy = correct / total if total > 0 else 0.0

    return {"accuracy": accuracy, "correct": correct, "total": total, "per_result": per_result}


# ── Print summary ─────────────────────────────────────────────────────────────

def print_summary(pipeline_results: dict):
    print('\n' + '=' * 55)
    print(f"{'Pipeline':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print('=' * 55)
    for name, res in sorted(pipeline_results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"{name:<20} {res['accuracy']:>9.1%} {res['correct']:>10} {res['total']:>10}")
    print('=' * 55 + '\n')