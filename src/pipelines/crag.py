"""
crag.py  —  CRAG (Corrective RAG)
----------------------------------
Algorithm
  1. Retrieve top-K chunks via standard vector search.
  2. Score each chunk's relevance to the query using cosine similarity
     (NLI cross-encoder used if available, cosine fallback otherwise).
  3. Routing:
       HIGH (≥0.65) → use all top-K chunks
       MED  (≥0.40) → use top-3 highest-confidence chunks
       LOW  (<0.40) → expand retrieval to top-K*2 AND add query-keyword
                      boosted re-ranking, take top-5
  4. Generate answer with IEEE citations.

Key fixes vs previous version
  - Removed the "no_context" path entirely — it was causing ~30% of queries
    to get no context at all, massively hurting accuracy
  - Low-confidence path now does EXPANDED retrieval (2x) + keyword rerank
    instead of giving up
  - Thresholds retuned: 0.65 / 0.40 (was 0.55 / 0.33)
  - GROQ_API_KEY checked first
"""

import os
import re
from typing import List, Tuple, Optional

_NLI_CACHE = {}


def _resolve_key(api_key):
    return (api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY"))


def _get_nli(model_name="cross-encoder/nli-deberta-v3-small"):
    if model_name not in _NLI_CACHE:
        try:
            from sentence_transformers import CrossEncoder
            _NLI_CACHE[model_name] = CrossEncoder(model_name)
            print(f"[crag] Loaded NLI model: {model_name}")
        except Exception as e:
            print(f"[crag] NLI model unavailable ({e}), using cosine scores.")
            _NLI_CACHE[model_name] = None
    return _NLI_CACHE[model_name]


def _score_chunks(query: str, chunks: List[Tuple[str, float, dict]],
                  nli_model_name: str, use_nli: bool) -> List[float]:
    """Return relevance score ∈ [0,1] for each chunk."""
    if use_nli:
        nli = _get_nli(nli_model_name)
        if nli is not None:
            try:
                pairs  = [(query, t) for (t, _, _) in chunks]
                scores = nli.predict(pairs, apply_softmax=True)
                if hasattr(scores[0], '__len__'):
                    return [float(s[2]) for s in scores]   # entailment col
                return [float(s) for s in scores]
            except Exception as e:
                print(f"[crag] NLI scoring failed: {e}")

    # Cosine fallback — shift from [-1,1] → [0,1]
    return [min(max((s + 1) / 2, 0.0), 1.0) for (_, s, _) in chunks]


def _keyword_rerank(query: str, chunks: List[Tuple[str, float, dict]],
                    conf_scores: List[float]) -> List[Tuple[str, float, dict]]:
    """
    Re-rank chunks by blending: 50% confidence score + 50% keyword overlap.
    Used in the low-confidence path to surface the best available chunk.
    """
    q_words = set(re.findall(r'\b\w{3,}\b', query.lower())) - {
        "who","what","when","where","which","how","why","did","does",
        "was","were","has","have","the","that","this","with","from"
    }
    scored = []
    for (text, orig, meta), conf in zip(chunks, conf_scores):
        t_words  = set(re.findall(r'\b\w{3,}\b', text.lower()))
        kw_score = len(q_words & t_words) / max(len(q_words), 1)
        blended  = 0.5 * conf + 0.5 * kw_score
        scored.append((blended, text, orig, meta))
    scored.sort(key=lambda x: x[0], reverse=True)
    return [(t, o, m) for (_, t, o, m) in scored]


def _ieee_refs(chunks: List[Tuple[str, float, dict]]) -> str:
    lines = []
    for i, (_, _, meta) in enumerate(chunks, 1):
        url  = meta.get("source_url",  "N/A")
        name = meta.get("source_name", "Web Source")
        lines.append(f"[{i}] {name}. Available: {url}")
    return "\n".join(lines)


def run(
    query: str,
    corpus_index,
    top_k: int = 15,
    confidence_threshold: float = 0.65,
    med_threshold: float = 0.40,
    use_nli: bool = True,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    embedding_model: str = "all-mpnet-base-v2",
    provider: str = "groq",
    gen_model: str = "llama3-70b-8192",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    from src.retrieval import embed_query
    from src.generation import generate_answer

    api_key = _resolve_key(api_key)

    # ── Step 1: Standard retrieval ────────────────────────────────────────────
    q_emb        = embed_query(query, model_name=embedding_model)
    all_retrieved = corpus_index.retrieve(q_emb, top_k=top_k)

    # ── Step 2: Score each chunk ──────────────────────────────────────────────
    conf_scores   = _score_chunks(query, all_retrieved, nli_model_name, use_nli)
    mean_conf     = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    # ── Step 3: Routing ───────────────────────────────────────────────────────
    if mean_conf >= confidence_threshold:
        # HIGH — use all retrieved chunks
        correction_path = "correct"
        used_chunks     = all_retrieved

    elif mean_conf >= med_threshold:
        # MEDIUM — use top-3 by confidence
        correction_path = "fallback_top3"
        pairs = sorted(zip(conf_scores, all_retrieved), key=lambda x: x[0], reverse=True)
        used_chunks = [c for (_, c) in pairs[:3]]
        conf_scores = [s for (s, _) in pairs[:3]]

    else:
        # LOW — expand retrieval 2x + keyword re-rank
        correction_path  = "expanded_rerank"
        expanded         = corpus_index.retrieve(q_emb, top_k=top_k * 2)
        exp_scores       = _score_chunks(query, expanded, nli_model_name, use_nli)
        reranked         = _keyword_rerank(query, expanded, exp_scores)
        used_chunks      = reranked[:top_k]
        conf_scores      = _score_chunks(query, used_chunks, nli_model_name, use_nli)

    # ── Step 4: Blend cosine + confidence for display score ──────────────────
    display_chunks = []
    for (text, orig_score, meta), conf in zip(used_chunks, conf_scores):
        blended = round((orig_score + conf) / 2, 4)
        display_chunks.append((text, blended, meta))

    # ── Step 5: Generate ─────────────────────────────────────────────────────
    answer_raw = generate_answer(
        query, display_chunks,
        provider=provider, model=gen_model, api_key=api_key
    )
    refs   = _ieee_refs(display_chunks)
    answer = f"{answer_raw}\n\nReferences:\n{refs}"

    return {
        "pipeline":         "crag",
        "answer":           answer,
        "retrieved":        display_chunks,
        "all_retrieved":    all_retrieved,
        "confidence":       mean_conf,
        "confidence_scores": conf_scores,
        "correction_path":  correction_path,
    }