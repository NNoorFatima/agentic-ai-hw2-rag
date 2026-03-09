"""
retrieval.py
------------
Base retrieval: embed a query and retrieve top-k chunks from the global CorpusIndex.

Public API
----------
embed_query(query, model_name) -> np.ndarray
retrieve(query, corpus_index, top_k, model_name) -> list[(text, score, metadata)]
"""

from typing import List, Tuple, Optional
import numpy as np


_MODEL_CACHE: dict = {}


def _get_model(model_name: str):
    """Cache sentence-transformer models to avoid reloading."""
    if model_name not in _MODEL_CACHE:
        from sentence_transformers import SentenceTransformer
        _MODEL_CACHE[model_name] = SentenceTransformer(model_name)
    return _MODEL_CACHE[model_name]


def embed_query(query: str, model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a single query string into a dense vector.

    Parameters
    ----------
    query      : the natural-language question
    model_name : sentence-transformers model identifier

    Returns
    -------
    1-D numpy float32 array
    """
    model = _get_model(model_name)
    emb = model.encode([query], convert_to_numpy=True)[0]
    return emb.astype("float32")


def embed_texts(texts: List[str], model_name: str = "all-MiniLM-L6-v2") -> np.ndarray:
    """
    Embed a list of strings. Returns (N, dim) float32 array.
    """
    model = _get_model(model_name)
    return model.encode(texts, convert_to_numpy=True, show_progress_bar=False).astype("float32")


def retrieve(
    query: str,
    corpus_index,
    top_k: int = 5,
    model_name: str = "all-MiniLM-L6-v2",
) -> List[Tuple[str, float, dict]]:
    """
    Standard single-query retrieval from the global index.

    Parameters
    ----------
    query        : natural-language question
    corpus_index : CorpusIndex from corpus.py
    top_k        : number of chunks to return
    model_name   : embedding model name

    Returns
    -------
    list of (chunk_text, similarity_score, metadata_dict)
    """
    q_emb = embed_query(query, model_name)
    return corpus_index.retrieve(q_emb, top_k=top_k)


if __name__ == "__main__":
    # Quick smoke-test (requires a built index)
    import sys
    sys.path.insert(0, ".")
    from src.corpus import load_index

    idx = load_index("data")
    results = retrieve("Who directed Inception?", idx, top_k=3)
    for text, score, meta in results:
        print(f"Score={score:.4f} | URL={meta.get('source_url','')}")
        print(f"  {text[:120]}\n")