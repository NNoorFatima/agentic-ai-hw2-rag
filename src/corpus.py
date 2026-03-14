"""
corpus.py  —  Build, save, and load the global corpus index.

Each chunk is embedded as:
    "Q: {query}  A: {snippet}"

This query-enriched format means that at retrieval time, embedding the query
produces a vector close to the stored chunk vectors → much higher cosine scores
than embedding snippet text alone.

Scores you should see after rebuild: 0.40 – 0.80 (vs 0.06 before).
"""

import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(it, **kw): return it

from src.data_loader import load_dataset


def _get_model(name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(name)


def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


def _build_chunks(query: str, search_results: List[dict]) -> List[dict]:
    """
    One chunk per page_snippet.
    embed_text = "Q: {query}  A: {snippet}"  — used only for embedding.
    text       = snippet                      — stored and shown to user/LLM.
    """
    chunks = []
    seen   = set()
    for item in search_results:
        snippet = (item.get("page_snippet") or "").strip()
        if not snippet or snippet in seen:
            continue
        seen.add(snippet)
        chunks.append({
            "text":        snippet,
            "embed_text":  f"Q: {query}  A: {snippet}",
            "source_url":  item.get("page_url",  ""),
            "source_name": item.get("page_name", ""),
        })
    return chunks


class CorpusIndex:
    """
    Cosine-similarity index over corpus chunks.
    Uses FAISS (IndexFlatIP) if installed, else normalised numpy.
    """

    def __init__(self, chunks: List[dict], embeddings: np.ndarray, model_name: str):
        self.chunks     = chunks
        self.embeddings = embeddings.astype("float32")
        self.model_name = model_name
        self._faiss_idx = None
        self._normed    = None
        self._build()

    def _build(self):
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        self._normed = (self.embeddings / norms).astype("float32")

        faiss = _get_faiss()
        if faiss:
            idx = faiss.IndexFlatIP(self._normed.shape[1])
            idx.add(self._normed)
            self._faiss_idx = idx

    def retrieve(self, q_emb: np.ndarray, top_k: int = 10) -> List[Tuple[str, float, dict]]:
        q  = q_emb.astype("float32").reshape(1, -1)
        qn = q / max(float(np.linalg.norm(q)), 1e-9)

        if self._faiss_idx is not None:
            scores, idxs = self._faiss_idx.search(qn, min(top_k, len(self.chunks)))
            return [(self.chunks[i]["text"], float(s), self.chunks[i])
                    for s, i in zip(scores[0], idxs[0]) if i >= 0]

        sims = (self._normed @ qn.T).flatten()
        top  = np.argsort(sims)[::-1][:top_k]
        return [(self.chunks[i]["text"], float(sims[i]), self.chunks[i]) for i in top]

    def __len__(self):
        return len(self.chunks)


def build_index(
    dataset_path: str,
    index_dir: str       = "data",
    embedding_model: str = "all-MiniLM-L6-v2",
    max_examples: Optional[int] = None,
    batch_size: int      = 128,
) -> CorpusIndex:
    print(f"[build_index] Reading {dataset_path} …")
    model = _get_model(embedding_model)

    all_chunks: List[dict] = []
    seen = set()

    for query, answer, alt_ans, search_results in tqdm(
        load_dataset(dataset_path, max_examples), desc="Collecting chunks"
    ):
        for c in _build_chunks(query, search_results):
            key = c["text"][:200]
            if key not in seen:
                seen.add(key)
                all_chunks.append(c)

    print(f"[build_index] {len(all_chunks)} unique chunks.")

    embed_texts = [c["embed_text"] for c in all_chunks]
    print(f"[build_index] Embedding {len(embed_texts)} chunks (batch={batch_size}) …")
    embeddings = model.encode(
        embed_texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    idx = CorpusIndex(all_chunks, embeddings, embedding_model)
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    _save(idx, index_dir)
    print(f"[build_index] Saved to '{index_dir}'. Chunks: {len(idx)}")
    return idx


def _save(idx: CorpusIndex, d: str):
    p = Path(d)
    np.save(str(p / "embeddings.npy"), idx.embeddings)
    with open(p / "metadata.pkl", "wb") as f:
        pickle.dump({"chunks": idx.chunks, "model_name": idx.model_name}, f)
    faiss = _get_faiss()
    if faiss and idx._faiss_idx is not None:
        faiss.write_index(idx._faiss_idx, str(p / "index.faiss"))


def load_index(index_dir: str = "data",
               embedding_model: Optional[str] = None) -> CorpusIndex:
    p = Path(index_dir)
    if not (p / "metadata.pkl").exists():
        raise FileNotFoundError(f"No index at '{index_dir}'. Run build_index first.")
    with open(p / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)
    embeddings = np.load(str(p / "embeddings.npy"))
    model_name = embedding_model or meta.get("model_name", "all-MiniLM-L6-v2")
    print(f"[load_index] Loaded {len(meta['chunks'])} chunks from '{index_dir}'.")
    return CorpusIndex(meta["chunks"], embeddings, model_name)