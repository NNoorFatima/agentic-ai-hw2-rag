"""
graph_rag.py  —  Graph-Augmented RAG
--------------------------------------
Algorithm
  1. Retrieve seed nodes (top seed_k by vector similarity).
  2. Build / reuse a chunk-similarity graph:
       edge(i,j) exists if cosine(emb_i, emb_j) >= threshold.
  3. BFS-expand from seeds up to max_hops.
  4. Score ALL candidates (seeds + neighbours) against the query.
  5. Take top_k highest-scoring candidates → LLM generation.

Key fixes vs previous version
  - seed_k is now 2×top_k so the graph has a much wider starting set
  - similarity_threshold lowered to 0.60 (was 0.72) — denser graph,
    more useful expansions on a 9k-chunk corpus
  - max_hops=2 (was 1) for deeper traversal
  - index lookup uses a pre-built text→index dict (O(1)) instead of
    list.index() which was O(N) and caused missed seeds
  - GROQ_API_KEY checked first
"""

import os
from typing import List, Tuple, Optional, Dict, Set
import numpy as np


def _resolve_key(api_key):
    return (api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY"))


# ── Graph construction ────────────────────────────────────────────────────────

def _build_graph(corpus_index, threshold: float) -> Dict[int, List[int]]:
    emb   = corpus_index.embeddings.astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = emb / norms

    n   = len(normed)
    adj: Dict[int, List[int]] = {i: [] for i in range(n)}

    batch = 512
    for start in range(0, n, batch):
        end  = min(start + batch, n)
        sims = normed[start:end] @ normed.T          # (B, N)
        for li, gi in enumerate(range(start, end)):
            nbrs = np.where(sims[li] >= threshold)[0].tolist()
            adj[gi] = [j for j in nbrs if j != gi]

    return adj


def _get_graph(corpus_index, threshold: float) -> Dict[int, List[int]]:
    key = f"_graph_{threshold}"
    if not hasattr(corpus_index, key):
        n = len(corpus_index.chunks)
        print(f"[graph_rag] Building similarity graph ({n} nodes, threshold={threshold}) …")
        setattr(corpus_index, key, _build_graph(corpus_index, threshold))
        print(f"[graph_rag] Graph ready.")
    return getattr(corpus_index, key)


def _get_text_index(corpus_index) -> Dict[str, int]:
    """Pre-built text→chunk_index map for O(1) seed lookup."""
    if not hasattr(corpus_index, "_text_index"):
        corpus_index._text_index = {c["text"]: i for i, c in enumerate(corpus_index.chunks)}
    return corpus_index._text_index


# ── BFS expansion ─────────────────────────────────────────────────────────────

def _bfs(seeds: List[int], graph: Dict[int, List[int]],
         max_hops: int, max_nodes: int) -> Set[int]:
    visited  = set(seeds)
    frontier = set(seeds)
    for _ in range(max_hops):
        nxt = set()
        for node in frontier:
            for nb in graph.get(node, []):
                if nb not in visited:
                    nxt.add(nb); visited.add(nb)
        frontier = nxt
        if len(visited) >= max_nodes:
            break
    return visited


# ── Pipeline run ──────────────────────────────────────────────────────────────

def run(
    query: str,
    corpus_index,
    top_k: int = 10,
    seed_k: int = None,            # defaults to top_k*2
    similarity_threshold: float = 0.60,
    max_hops: int = 2,
    max_candidates: int = 200,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = "groq",
    gen_model: str = "llama3-70b-8192",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    from src.retrieval import embed_query
    from src.generation import generate_answer

    api_key = _resolve_key(api_key)
    if seed_k is None:
        seed_k = top_k * 2

    # 1. Seed retrieval
    q_emb      = embed_query(query, model_name=embedding_model)
    seed_hits  = corpus_index.retrieve(q_emb, top_k=seed_k)

    # 2. Map seed texts → indices (O(1) lookup)
    text_idx    = _get_text_index(corpus_index)
    seed_indices = []
    for (text, _, _) in seed_hits:
        idx = text_idx.get(text)
        if idx is not None:
            seed_indices.append(idx)

    # 3. Graph BFS expansion
    graph      = _get_graph(corpus_index, threshold=similarity_threshold)
    candidates = _bfs(seed_indices, graph, max_hops=max_hops, max_nodes=max_candidates)
    if not candidates:
        candidates = set(seed_indices)

    # 4. Score candidates against query
    emb   = corpus_index.embeddings.astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = emb / norms

    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)
    cand_list = list(candidates)
    sims      = (normed[cand_list] @ q_norm).tolist()

    top_scored = sorted(zip(cand_list, sims), key=lambda x: x[1], reverse=True)[:top_k]

    # 5. Build retrieved list
    retrieved = []
    for idx, score in top_scored:
        chunk = corpus_index.chunks[idx]
        retrieved.append((chunk["text"], float(score), chunk))

    # 6. Generate
    answer = generate_answer(query, retrieved,
                              provider=provider, model=gen_model, api_key=api_key)
    return {
        "pipeline":       "graph_rag",
        "answer":         answer,
        "retrieved":      retrieved,
        "seed_indices":   seed_indices,
        "expanded_count": len(candidates),
    }