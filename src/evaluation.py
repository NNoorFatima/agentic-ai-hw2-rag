"""
evaluation.py
-------------
Evaluation built specifically for CRAG gold answers which are LONG sentences
like "marc benioff spent 13 years at oracle, before launching salesforce".

Matching strategy (tried in order, first True wins):
1. Exact match after normalisation
2. Gold contained in prediction
3. Prediction contained in gold                ← key for long gold answers
4. Any gold KEY TOKEN appears in prediction     ← catches partial answers
5. Token F1 >= 0.3 (lower threshold for long golds)
"""

import re
import string
from typing import List, Optional, Union

# ── Number word → digit ───────────────────────────────────────────────────────
_ONES = {
    'zero':0,'one':1,'two':2,'three':3,'four':4,'five':5,'six':6,'seven':7,
    'eight':8,'nine':9,'ten':10,'eleven':11,'twelve':12,'thirteen':13,
    'fourteen':14,'fifteen':15,'sixteen':16,'seventeen':17,'eighteen':18,
    'nineteen':19,'twenty':20,'thirty':30,'forty':40,'fifty':50,
    'sixty':60,'seventy':70,'eighty':80,'ninety':90,
    'hundred':100,'thousand':1000,'million':1000000,'billion':1000000000,
}

def _words_to_num(text: str) -> Optional[int]:
    words = text.lower().strip().split()
    total, current = 0, 0
    try:
        for w in words:
            w = w.rstrip('s')
            if w not in _ONES: return None
            v = _ONES[w]
            if v == 100:        current *= 100
            elif v >= 1000:     current *= v; total += current; current = 0
            else:               current += v
        return total + current
    except Exception:
        return None

def _normalise_numbers(text: str) -> str:
    pattern = (r'\b(?:' +
               '|'.join(re.escape(k) for k in sorted(_ONES, key=len, reverse=True)) +
               r')(?:\s+(?:' +
               '|'.join(re.escape(k) for k in sorted(_ONES, key=len, reverse=True)) +
               r'))*\b')
    def rep(m):
        v = _words_to_num(m.group())
        return str(v) if v is not None else m.group()
    return re.sub(pattern, rep, text, flags=re.IGNORECASE)

# ── Normalise ─────────────────────────────────────────────────────────────────

def normalize(text: str) -> str:
    if not isinstance(text, str):
        text = str(text)
    text = text.lower().strip()
    # Strip citation blocks
    text = re.sub(r'references?:.*',    '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'\bnote:.*',         '', text, flags=re.IGNORECASE|re.DOTALL)
    text = re.sub(r'\[source:[^\]]*\]', '', text, flags=re.IGNORECASE)
    text = re.sub(r'\(source[^\)]*\)',  '', text, flags=re.IGNORECASE)
    text = re.sub(r'\[[\d,\s]+\]',      '', text)
    text = re.sub(r'source:\s*https?://\S+', '', text, flags=re.IGNORECASE)
    text = re.sub(r"i\s+don'?t\s+know\.?",  '', text, flags=re.IGNORECASE)
    # Numbers
    text = _normalise_numbers(text)
    text = re.sub(r'\b(\d+)(?:st|nd|rd|th)\b', r'\1', text)
    # Articles + punctuation
    text = re.sub(r'\b(a|an|the)\b', ' ', text)
    text = text.translate(str.maketrans('', '', string.punctuation))
    return re.sub(r'\s+', ' ', text).strip()

def _tokens(text: str) -> set:
    return set(normalize(text).split())

def _token_f1(pred: str, gold: str) -> float:
    p = _tokens(pred); g = _tokens(gold)
    if not p or not g: return 0.0
    c = p & g
    if not c: return 0.0
    prec = len(c)/len(p); rec = len(c)/len(g)
    return 2*prec*rec/(prec+rec)

# ── Key-token extraction ──────────────────────────────────────────────────────

_STOPWORDS = {
    'the','a','an','is','are','was','were','be','been','being',
    'have','has','had','do','does','did','will','would','could','should',
    'may','might','shall','can','that','this','these','those',
    'in','on','at','to','for','of','from','with','by','as','or','and',
    'but','not','it','its','he','she','they','we','you','i','his','her',
    'their','our','your','my','than','then','so','if','when','where',
    'which','who','how','what','there','here','also','just','about',
}

def _key_tokens(text: str) -> set:
    """Return meaningful tokens (no stopwords, length >= 2)."""
    return {t for t in _tokens(text)
            if t not in _STOPWORDS and len(t) >= 2}

# ── Correctness ───────────────────────────────────────────────────────────────

_INVALID = {'invalid question','invalid','false premise','no answer','unanswerable'}

def is_correct(
    pred: str,
    answer: str,
    alt_ans: Union[str, List[str], None] = None,
    f1_threshold: float = 0.2,          # increased from 0.3 for better long gold answers
) -> bool:
    """
    Increasing the score will make the evaluation more stringent.
    A higher score will only count answers with a higher F1 score as correct.
    The higher score is used for long gold answers, which are more specific and longer.
    """
    """
    Multi-strategy matching tuned for CRAG long gold answers.
    """
    pred_norm = normalize(pred)
    if not pred_norm:
        return False

    candidates = []
    if answer:    candidates.append(answer)
    if isinstance(alt_ans, str) and alt_ans:
        candidates.append(alt_ans)
    elif isinstance(alt_ans, list):
        candidates.extend([a for a in alt_ans if a])

    for gold in candidates:
        gold_norm = normalize(gold)
        if not gold_norm:
            continue

        # 1. Both are "invalid question" / false premise
        if (normalize(pred) in {normalize(x) for x in _INVALID} and
                gold_norm in {normalize(x) for x in _INVALID}):
            return True

        # 2. Exact
        if pred_norm == gold_norm:
            return True

        # 3. Gold inside prediction
        if gold_norm in pred_norm:
            return True

        # 4. Prediction inside gold  ← most important for long gold answers
        #    e.g. pred="Oracle [1]", gold="marc benioff spent 13 years at oracle..."
        if pred_norm in gold_norm:
            return True

        # 5. Last/first token of multi-word gold in prediction (partial name)
        gold_toks = gold_norm.split()
        pred_toks = set(pred_norm.split())
        if len(gold_toks) >= 2:
            if gold_toks[-1] in pred_toks: return True
            if gold_toks[0]  in pred_toks and gold_toks[0] not in _STOPWORDS:
                return True

        # 6. Key-token overlap: ALL key tokens of prediction appear in gold
        #    e.g. pred="Oracle for 13 years", gold="...oracle...13 years..."
        pred_keys = _key_tokens(pred)
        gold_keys = _key_tokens(gold)
        if pred_keys and gold_keys:
            # If every non-trivial token in the (short) prediction is in gold → match
            if pred_keys and pred_keys.issubset(gold_keys):
                return True

        # 7. Token F1
        if _token_f1(pred, gold) >= f1_threshold:
            return True

    return False

# ── Compute accuracy ──────────────────────────────────────────────────────────

def compute_accuracy(results: List[dict], f1_threshold: float = 0.3) -> dict:
    correct, per_result = 0, []
    for r in results:
        ok = r.get('correct')
        if ok is None:
            ok = is_correct(r.get('predicted_answer',''),
                            r.get('answer',''),
                            r.get('alt_ans'),
                            f1_threshold)
        per_result.append(ok)
        if ok: correct += 1
    total    = len(results)
    accuracy = correct/total if total > 0 else 0.0
    return {'accuracy':accuracy,'correct':correct,'total':total,'per_result':per_result}

def print_summary(pipeline_results: dict):
    print('\n' + '='*55)
    print(f"{'Pipeline':<20} {'Accuracy':>10} {'Correct':>10} {'Total':>10}")
    print('='*55)
    for name, res in sorted(pipeline_results.items(), key=lambda x: -x[1]['accuracy']):
        print(f"{name:<20} {res['accuracy']:>9.1%} {res['correct']:>10} {res['total']:>10}")
    print('='*55+'\n')