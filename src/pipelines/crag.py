# """
# CRAG (Corrective RAG): assess retrieval confidence; use or correct retrieval based on it, then generate.
# Do not remove or rename this file.
# """

# # TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# # Retrieve from global index -> assess confidence (e.g. NLI, consistency, or LLM judge) ->
# # if high: use retrieved chunks for generation; if low: skip retrieval or use fallback -> generate answer.
"""
crag.py
-------
CRAG (Corrective RAG) Pipeline

Algorithm
---------
1. Embed query → retrieve top-k chunks from global index as usual.
2. Assess retrieval confidence:
   - Use a cross-encoder NLI model to judge relevance of each chunk to the query.
   - Average the confidence scores.
   - If confidence >= threshold → use retrieved chunks for generation (CORRECT path).
   - If confidence < threshold → fall back:
       (a) Reduce to only the best single chunk, OR
       (b) Generate directly from the query without context (INCORRECT path).
3. Generate final answer, citing sources (IEEE-style) at the end.

The intuition: when the index is noisy, retrieved chunks may be off-topic.
Corrective RAG detects this and avoids letting bad context mislead the LLM.
"""

import os
from typing import List, Tuple, Optional


_NLI_MODEL_CACHE = {}


def _get_nli_model(model_name: str = "cross-encoder/nli-deberta-v3-small"):
    """Load (and cache) a cross-encoder NLI model for relevance scoring."""
    if model_name not in _NLI_MODEL_CACHE:
        try:
            from sentence_transformers import CrossEncoder
            _NLI_MODEL_CACHE[model_name] = CrossEncoder(model_name)
        except Exception as exc:
            print(f"[crag] Could not load NLI model '{model_name}': {exc}. Using heuristic scorer.")
            _NLI_MODEL_CACHE[model_name] = None
    return _NLI_MODEL_CACHE[model_name]


def _score_relevance_nli(
    query: str,
    chunks: List[Tuple[str, float, dict]],
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
) -> List[float]:
    """
    Score relevance of each chunk to the query using a cross-encoder NLI model.
    Returns a list of relevance scores in [0, 1].
    Entailment label is treated as "relevant".
    """
    nli_model = _get_nli_model(nli_model_name)

    if nli_model is None:
        # Fallback: use the cosine similarity scores already in retrieved
        return [min(max(score, 0.0), 1.0) for (_, score, _) in chunks]

    try:
        pairs = [(query, text) for (text, _, _) in chunks]
        scores = nli_model.predict(pairs, apply_softmax=True)
        # scores shape: (N, 3) for [contradiction, neutral, entailment]
        # Use entailment probability as relevance
        if hasattr(scores, '__len__') and len(scores) > 0 and hasattr(scores[0], '__len__'):
            return [float(s[2]) for s in scores]  # entailment column
        else:
            return [float(s) for s in scores]
    except Exception as exc:
        print(f"[crag] NLI scoring failed: {exc}. Using cosine scores.")
        return [min(max(score, 0.0), 1.0) for (_, score, _) in chunks]


def _score_relevance_cosine(chunks: List[Tuple[str, float, dict]]) -> List[float]:
    """Use already-computed cosine similarity scores (normalised to [0,1])."""
    scores = [score for (_, score, _) in chunks]
    if not scores:
        return []
    # Cosine scores from FAISS IP (after L2-norm) are in [-1, 1]; shift to [0, 1]
    return [min(max((s + 1) / 2, 0.0), 1.0) for s in scores]


def _format_ieee_citations(retrieved: List[Tuple[str, float, dict]]) -> str:
    """Build IEEE-style reference list from retrieved chunks."""
    refs = []
    for i, (text, score, meta) in enumerate(retrieved, 1):
        url = meta.get("source_url", "N/A")
        name = meta.get("source_name", "Web Source")
        refs.append(f"[{i}] {name}. Available: {url}")
    return "\n".join(refs)


def run(
    query: str,
    corpus_index,
    top_k: int = 5,
    confidence_threshold: float = 0.55,
    use_nli: bool = True,
    nli_model_name: str = "cross-encoder/nli-deberta-v3-small",
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = "openai",
    gen_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    CRAG pipeline.

    Returns
    -------
    dict with keys:
        answer           : str (with IEEE citations at end)
        retrieved        : list[(text, score, meta)]  — chunks used for generation
        all_retrieved    : list[(text, score, meta)]  — all retrieved before filtering
        pipeline         : str
        confidence       : float  — mean relevance confidence
        confidence_scores: list[float]
        correction_path  : str  — "correct" | "fallback_single" | "no_context"
    """
    from src.retrieval import embed_query
    from src.generation import generate_answer

    # Step 1: Standard retrieval
    q_emb = embed_query(query, model_name=embedding_model)
    all_retrieved = corpus_index.retrieve(q_emb, top_k=top_k)

    # Step 2: Assess retrieval confidence
    if use_nli:
        conf_scores = _score_relevance_nli(query, all_retrieved, nli_model_name)
    else:
        conf_scores = _score_relevance_cosine(all_retrieved)

    mean_confidence = sum(conf_scores) / len(conf_scores) if conf_scores else 0.0

    # Step 3: Decide correction path
    if mean_confidence >= confidence_threshold:
        # HIGH confidence: use all retrieved chunks
        correction_path = "correct"
        used_chunks = all_retrieved

    elif mean_confidence >= confidence_threshold * 0.6:
        # MEDIUM confidence: use only top-1 chunk (most relevant)
        correction_path = "fallback_single"
        # Re-rank by NLI confidence
        ranked_by_conf = sorted(
            zip(conf_scores, all_retrieved),
            key=lambda x: x[0],
            reverse=True,
        )
        used_chunks = [ranked_by_conf[0][1]] if ranked_by_conf else all_retrieved[:1]

    else:
        # LOW confidence: generate without retrieved context
        correction_path = "no_context"
        used_chunks = []

    # Step 4: Generate answer
    if used_chunks:
        # Attach NLI-based confidence as score override for display
        used_with_conf = []
        for (text, orig_score, meta), conf in zip(
            used_chunks,
            conf_scores[:len(used_chunks)] if correction_path == "correct" else [conf_scores[0]],
        ):
            display_score = (orig_score + conf) / 2  # blend cosine + NLI
            used_with_conf.append((text, display_score, meta))
        answer_raw = generate_answer(
            query, used_with_conf, provider=provider, model=gen_model, api_key=api_key
        )
        # Append IEEE-style references
        refs = _format_ieee_citations(used_with_conf)
        answer = f"{answer_raw}\n\nReferences:\n{refs}"
    else:
        # Generate purely from LLM knowledge (no context injection)
        answer_raw = generate_answer(
            query, [], provider=provider, model=gen_model, api_key=api_key
        )
        answer = f"{answer_raw}\n\n(Note: Retrieved context was deemed insufficiently relevant; answer generated from model knowledge only.)"

    return {
        "pipeline": "crag",
        "answer": answer,
        "retrieved": used_chunks,
        "all_retrieved": all_retrieved,
        "confidence": mean_confidence,
        "confidence_scores": conf_scores,
        "correction_path": correction_path,
    }