# """
# Graph RAG: graph-augmented retrieval over the corpus (e.g. entity/relation graph or similarity graph), then generate.
# Do not remove or rename this file.
# """

# # TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# # Build or use a graph over the corpus (entities, relations, or chunk similarity) ->
# # retrieve in a graph-aware way (e.g. entity linking, subgraph, spread from seeds) ->
# # convert selected graph neighborhood to text -> generate answer from that context.
"""
graph_rag.py
------------
Graph RAG Pipeline

Algorithm
---------
1. Build a chunk-similarity graph over the corpus (done once per corpus_index).
   - Nodes = corpus chunks
   - Edges = cosine similarity above a threshold between two chunks

2. For a query:
   a. Retrieve seed nodes (top-k by vector similarity, same as baseline).
   b. Expand: collect 1-hop or 2-hop neighbours of seed nodes in the graph.
   c. Score all candidates (seeds + neighbours) by their similarity to the query.
   d. Take top-k from this expanded candidate set.
   e. Pass to LLM for generation.

Why it helps on noisy corpora:
- If a directly-retrieved chunk is a near-duplicate of a more relevant chunk,
  graph traversal surfaces the better chunk.
- Multi-hop questions benefit from chunks that are conceptually linked even
  if not lexically similar to the query.

Note: Graph construction is lazy and cached on the corpus_index object
to avoid rebuilding on every call.
"""

import os
from typing import List, Tuple, Optional, Dict, Set
import numpy as np


# ──────────────────────────────────────────────────────────────────────────────
# Graph construction (lazy, cached per index)
# ──────────────────────────────────────────────────────────────────────────────

def _build_chunk_graph(
    corpus_index,
    similarity_threshold: float = 0.75,
) -> Dict[int, List[int]]:
    """
    Build an adjacency list where edge (i, j) exists iff
    cosine_similarity(embeddings[i], embeddings[j]) >= similarity_threshold.

    Uses batched matrix multiplication for efficiency.
    Returns dict: node_id -> list of neighbour node_ids
    """
    import networkx as nx

    emb = corpus_index.embeddings.astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = emb / norms

    n = len(normed)
    adjacency: Dict[int, List[int]] = {i: [] for i in range(n)}

    # Process in batches to avoid OOM on large corpora
    batch_size = 512
    for start in range(0, n, batch_size):
        end = min(start + batch_size, n)
        batch = normed[start:end]          # (B, dim)
        sims = batch @ normed.T            # (B, N)

        for local_i, global_i in enumerate(range(start, end)):
            row = sims[local_i]
            neighbours = np.where(row >= similarity_threshold)[0].tolist()
            # Exclude self
            neighbours = [j for j in neighbours if j != global_i]
            adjacency[global_i] = neighbours

    return adjacency


def _get_graph(corpus_index, similarity_threshold: float = 0.75) -> Dict[int, List[int]]:
    """Lazily build and cache the chunk graph on the corpus_index object."""
    cache_key = f"_graph_{similarity_threshold}"
    if not hasattr(corpus_index, cache_key):
        print(f"[graph_rag] Building chunk similarity graph (threshold={similarity_threshold}) …")
        graph = _build_chunk_graph(corpus_index, similarity_threshold)
        setattr(corpus_index, cache_key, graph)
        print(f"[graph_rag] Graph built: {len(graph)} nodes.")
    return getattr(corpus_index, cache_key)


# ──────────────────────────────────────────────────────────────────────────────
# Graph-aware retrieval
# ──────────────────────────────────────────────────────────────────────────────

def _expand_via_graph(
    seed_indices: List[int],
    graph: Dict[int, List[int]],
    max_hops: int = 2,
    max_candidates: int = 50,
) -> Set[int]:
    """BFS/DFS expansion from seed nodes up to max_hops."""
    visited: Set[int] = set(seed_indices)
    frontier: Set[int] = set(seed_indices)

    for _ in range(max_hops):
        next_frontier: Set[int] = set()
        for node in frontier:
            for neighbour in graph.get(node, []):
                if neighbour not in visited:
                    next_frontier.add(neighbour)
                    visited.add(neighbour)
        frontier = next_frontier
        if len(visited) >= max_candidates:
            break

    return visited


def run(
    query: str,
    corpus_index,
    top_k: int = 5,
    seed_k: int = 5,
    similarity_threshold: float = 0.72,
    max_hops: int = 1,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = "openai",
    gen_model: str = "gpt-4o-mini",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    Graph RAG pipeline.

    Returns
    -------
    dict with keys: answer, retrieved, pipeline, seed_indices, expanded_count
    """
    from src.retrieval import embed_query
    from src.generation import generate_answer

    # Step 1: Standard seed retrieval
    q_emb = embed_query(query, model_name=embedding_model)
    seed_results = corpus_index.retrieve(q_emb, top_k=seed_k)

    # Find indices of seed chunks in the corpus
    chunk_texts = [c["text"] for c in corpus_index.chunks]
    seed_indices = []
    for (text, score, meta) in seed_results:
        try:
            idx = chunk_texts.index(text)
            seed_indices.append(idx)
        except ValueError:
            pass  # chunk not found (shouldn't happen)

    # Step 2: Graph expansion
    graph = _get_graph(corpus_index, similarity_threshold=similarity_threshold)
    candidate_indices = _expand_via_graph(seed_indices, graph, max_hops=max_hops)

    # Step 3: Score all candidates against query embedding
    emb = corpus_index.embeddings.astype("float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1e-9, norms)
    normed = emb / norms

    q_norm = q_emb / (np.linalg.norm(q_emb) + 1e-9)

    candidate_list = list(candidate_indices)
    if not candidate_list:
        candidate_list = seed_indices

    candidate_embs = normed[candidate_list]          # (M, dim)
    scores = (candidate_embs @ q_norm).tolist()      # (M,)

    scored = sorted(
        zip(candidate_list, scores),
        key=lambda x: x[1],
        reverse=True,
    )[:top_k]

    # Step 4: Build retrieved list
    retrieved = []
    for idx, score in scored:
        chunk = corpus_index.chunks[idx]
        retrieved.append((chunk["text"], float(score), chunk))

    # Step 5: Generate answer
    answer = generate_answer(query, retrieved, provider=provider, model=gen_model, api_key=api_key)

    return {
        "pipeline": "graph_rag",
        "answer": answer,
        "retrieved": retrieved,
        "seed_indices": seed_indices,
        "expanded_count": len(candidate_indices),
    }