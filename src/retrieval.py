"""
retrieval.py
------------
Embed a query and retrieve top-k chunks from the global CorpusIndex.

Important: we prefix queries with "Q: " to match the query-enriched
format used when building the index (embed_text = "Q: {query}\nA: {snippet}").
This asymmetric encoding is standard practice and significantly improves
retrieval quality on short factual queries.
"""

from typing import List, Tuple, Optional
import numpy as np

_MODEL_CACHE = {}


def _get_model(name: str):
    if name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[name] = SentenceTransformer(name)
    return _MODEL_CACHE[name]


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a single query string.
    Prefixes with 'Q: ' to match the query-enriched index format.
    """
    model = _get_model(model_name)
    # Match the format used at index build time
    text  = f"Q: {query}"
    emb   = model.encode([text], convert_to_numpy=True)[0]
    return emb.astype("float32")


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """Embed a list of strings. Returns (N, dim) float32 array."""
    model = _get_model(model_name)
    return model.encode(texts, convert_to_numpy=True,
                        show_progress_bar=False).astype("float32")


def retrieve(
    query: str,
    corpus_index,
    top_k: int = 10,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Tuple[str, float, dict]]:
    """
    Standard single-query retrieval.
    Returns list of (chunk_text, cosine_score, metadata).
    """
    q_emb = embed_query(query, model_name)
    return corpus_index.retrieve(q_emb, top_k=top_k)