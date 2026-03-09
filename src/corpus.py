"""
corpus.py
---------
Build the global corpus from the CRAG dataset.
Embeds all chunks with sentence-transformers and builds/saves/loads a FAISS index.

Public API
----------
build_index(dataset_path, index_dir, embedding_model, max_examples)
    -> CorpusIndex

load_index(index_dir, embedding_model)
    -> CorpusIndex

class CorpusIndex:
    retrieve(query_embedding, top_k) -> list[(chunk_text, score, metadata)]
"""

import os
import pickle
import re
from pathlib import Path
from typing import List, Tuple, Optional, Dict, Any

import numpy as np

try:
    from tqdm import tqdm
except ImportError:
    def tqdm(iterable, **kwargs):
        return iterable

from src.data_loader import load_dataset

# Lazy imports for heavy deps
def _get_sentence_transformer(model_name: str):
    from sentence_transformers import SentenceTransformer
    return SentenceTransformer(model_name)


def _get_faiss():
    try:
        import faiss
        return faiss
    except ImportError:
        return None


# ──────────────────────────────────────────────────────────────────────────────
# Text extraction helpers
# ──────────────────────────────────────────────────────────────────────────────

def _extract_text_from_html(html: str, max_chars: int = 2000) -> str:
    """Strip HTML tags and return plain text (truncated)."""
    try:
        from bs4 import BeautifulSoup
        soup = BeautifulSoup(html, "lxml")
        for tag in soup(["script", "style", "nav", "footer", "header"]):
            tag.decompose()
        text = soup.get_text(separator=" ", strip=True)
    except Exception:
        # Fallback: naive tag strip
        text = re.sub(r"<[^>]+>", " ", html)
        text = re.sub(r"\s+", " ", text).strip()
    return text[:max_chars]


def _build_chunks_from_row(search_results: List[dict], use_full_html: bool = False) -> List[dict]:
    """
    Extract text chunks from a single dataset row's search_results.

    Returns a list of chunk dicts:
        {text, source_url, source_name}
    """
    chunks = []
    for item in search_results:
        snippet = (item.get("page_snippet") or "").strip()
        url = item.get("page_url", "")
        name = item.get("page_name", "")

        if snippet:
            chunks.append({"text": snippet, "source_url": url, "source_name": name})

        if use_full_html:
            html = item.get("page_result", "") or ""
            if html:
                full_text = _extract_text_from_html(html, max_chars=1500)
                # Avoid duplicating identical snippet content
                if full_text and full_text[:100] != snippet[:100]:
                    chunks.append(
                        {"text": full_text, "source_url": url, "source_name": name}
                    )
    return chunks


# ──────────────────────────────────────────────────────────────────────────────
# CorpusIndex class
# ──────────────────────────────────────────────────────────────────────────────

class CorpusIndex:
    """
    Wraps a FAISS (or numpy fallback) vector index over the global corpus.

    Attributes
    ----------
    chunks    : list[dict]   – {'text', 'source_url', 'source_name'}
    embeddings: np.ndarray   – shape (N, dim)
    model_name: str
    """

    def __init__(self, chunks: List[dict], embeddings: np.ndarray, model_name: str):
        self.chunks = chunks
        self.embeddings = embeddings.astype("float32")
        self.model_name = model_name
        self._faiss_index = None
        self._build_faiss()

    def _build_faiss(self):
        faiss = _get_faiss()
        if faiss is None:
            print("[CorpusIndex] FAISS not available – using numpy cosine search.")
            return
        dim = self.embeddings.shape[1]
        index = faiss.IndexFlatIP(dim)  # Inner-product (cosine after normalisation)
        norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
        norms = np.where(norms == 0, 1e-9, norms)
        normed = self.embeddings / norms
        index.add(normed)
        self._faiss_index = index
        self._normed_embeddings = normed

    def retrieve(
        self, query_embedding: np.ndarray, top_k: int = 5
    ) -> List[Tuple[str, float, dict]]:
        """
        Parameters
        ----------
        query_embedding : 1-D numpy array (will be normalised internally)
        top_k           : number of results

        Returns
        -------
        list of (chunk_text, score, metadata_dict)
        where score ∈ [0, 1] (cosine similarity)
        """
        q = query_embedding.astype("float32").reshape(1, -1)
        norm = np.linalg.norm(q)
        if norm > 0:
            q = q / norm

        if self._faiss_index is not None:
            scores, indices = self._faiss_index.search(q, min(top_k, len(self.chunks)))
            results = []
            for score, idx in zip(scores[0], indices[0]):
                if idx < 0:
                    continue
                chunk = self.chunks[idx]
                results.append((chunk["text"], float(score), chunk))
            return results
        else:
            # Numpy fallback
            norms = np.linalg.norm(self.embeddings, axis=1, keepdims=True)
            norms = np.where(norms == 0, 1e-9, norms)
            normed = self.embeddings / norms
            sims = (normed @ q.T).flatten()
            top_indices = np.argsort(sims)[::-1][:top_k]
            return [
                (self.chunks[i]["text"], float(sims[i]), self.chunks[i])
                for i in top_indices
            ]

    def __len__(self):
        return len(self.chunks)


# ──────────────────────────────────────────────────────────────────────────────
# build_index / save / load
# ──────────────────────────────────────────────────────────────────────────────

def build_index(
    dataset_path: str,
    index_dir: str = "data",
    embedding_model: str = "all-MiniLM-L6-v2",
    max_examples: Optional[int] = None,
    use_full_html: bool = False,
    batch_size: int = 64,
) -> CorpusIndex:
    """
    Build the global corpus and embedding index from scratch.

    Steps
    -----
    1. Iterate all rows via data_loader.
    2. Extract text chunks from search_results.
    3. Deduplicate chunks.
    4. Embed all chunks with the chosen sentence-transformer.
    5. Build CorpusIndex (FAISS or numpy).
    6. Save to index_dir.

    Returns CorpusIndex.
    """
    print(f"[build_index] Loading dataset from '{dataset_path}' …")
    model = _get_sentence_transformer(embedding_model)

    all_chunks: List[dict] = []
    seen_texts = set()

    for _query, _answer, _alt_ans, search_results in tqdm(
        load_dataset(dataset_path, max_examples), desc="Collecting chunks"
    ):
        for chunk in _build_chunks_from_row(search_results, use_full_html):
            t = chunk["text"].strip()
            if t and t not in seen_texts:
                seen_texts.add(t)
                all_chunks.append(chunk)

    print(f"[build_index] Total unique chunks: {len(all_chunks)}")

    texts = [c["text"] for c in all_chunks]
    print(f"[build_index] Embedding {len(texts)} chunks …")
    embeddings = model.encode(
        texts,
        batch_size=batch_size,
        show_progress_bar=True,
        convert_to_numpy=True,
    )

    corpus_index = CorpusIndex(all_chunks, embeddings, embedding_model)

    # Save
    Path(index_dir).mkdir(parents=True, exist_ok=True)
    _save_index(corpus_index, index_dir)
    print(f"[build_index] Index saved to '{index_dir}'.")
    return corpus_index


def _save_index(corpus_index: CorpusIndex, index_dir: str):
    faiss = _get_faiss()
    path = Path(index_dir)

    # Save FAISS index
    if faiss is not None and corpus_index._faiss_index is not None:
        faiss.write_index(corpus_index._faiss_index, str(path / "index.faiss"))

    # Save embeddings + metadata
    np.save(str(path / "embeddings.npy"), corpus_index.embeddings)
    with open(path / "metadata.pkl", "wb") as f:
        pickle.dump(
            {"chunks": corpus_index.chunks, "model_name": corpus_index.model_name},
            f,
        )


def load_index(
    index_dir: str = "data",
    embedding_model: Optional[str] = None,
) -> CorpusIndex:
    """
    Load a previously saved CorpusIndex from index_dir.
    """
    path = Path(index_dir)
    if not (path / "metadata.pkl").exists():
        raise FileNotFoundError(
            f"No saved index found at '{index_dir}'. Run build_index first."
        )

    with open(path / "metadata.pkl", "rb") as f:
        meta = pickle.load(f)

    chunks: List[dict] = meta["chunks"]
    model_name: str = embedding_model or meta.get("model_name", "all-MiniLM-L6-v2")
    embeddings: np.ndarray = np.load(str(path / "embeddings.npy"))

    corpus_index = CorpusIndex(chunks, embeddings, model_name)

    # Optionally reload FAISS index from disk (already rebuilt in __init__)
    print(f"[load_index] Loaded {len(chunks)} chunks from '{index_dir}'.")
    return corpus_index


if __name__ == "__main__":
    import sys

    dataset = sys.argv[1] if len(sys.argv) > 1 else "dataset/crag_task_1_and_2_dev_v4.jsonl"
    idx = build_index(dataset, index_dir="data", max_examples=50)
    print(f"Index size: {len(idx)}")
    # Quick sanity check
    model = _get_sentence_transformer("all-MiniLM-L6-v2")
    q_emb = model.encode(["Who directed Inception?"], convert_to_numpy=True)[0]
    results = idx.retrieve(q_emb, top_k=3)
    for text, score, meta in results:
        print(f"  Score={score:.4f} | {text[:80]}")