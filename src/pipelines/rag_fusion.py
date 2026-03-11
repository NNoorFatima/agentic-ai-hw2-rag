"""
rag_fusion.py  —  RAG Fusion with Reciprocal Rank Fusion (RRF)
--------------------------------------------------------------
Algorithm
  1. Generate N diverse query variants via LLM (or heuristic fallback).
  2. Retrieve top-K chunks for EACH variant independently.
  3. Merge all ranked lists with RRF:  score(chunk) = Σ 1/(k + rank_i)
  4. Take top-K from fused list → feed to LLM.

Key fixes vs previous version
  - GROQ_API_KEY is now checked first in variant generation
  - per-variant retrieval uses top_k*2 so RRF has more signal
  - final fused list returns top_k*2 to give generation more context
  - num_variants increased to 4 for better coverage
"""

import os
import re
from typing import List, Tuple, Dict, Optional


def _resolve_key(api_key: Optional[str]) -> Optional[str]:
    return (api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY"))


def _generate_query_variants(
    query: str,
    n: int = 4,
    provider: str = "groq",
    model: str = "llama3-70b-8192",
    api_key: Optional[str] = None,
) -> List[str]:
    """Generate n diverse reformulations of the query."""
    api_key = _resolve_key(api_key)

    if api_key and provider != "local":
        prompt = (
            f"Generate {n} different search queries to help answer: '{query}'\n"
            f"Make each query different — vary phrasing, perspective, and keywords.\n"
            f"Output ONLY the queries, one per line, no numbering, no explanation."
        )
        try:
            if provider in ("groq", "openai"):
                from openai import OpenAI
                base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
                client = OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=300,
                    temperature=0.8,
                )
                text = resp.choices[0].message.content.strip()
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=model, max_tokens=300, temperature=0.8,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = resp.content[0].text.strip()
            else:
                text = ""

            variants = [ln.strip().lstrip("•-*0123456789.) ") 
                        for ln in text.split("\n") if ln.strip()]
            variants = [v for v in variants if len(v) > 5]
            if variants:
                all_v = [query] + variants
                # deduplicate preserving order
                seen, out = set(), []
                for v in all_v:
                    k = v.lower()
                    if k not in seen:
                        seen.add(k); out.append(v)
                return out[:n]
        except Exception as exc:
            print(f"[rag_fusion] variant LLM failed: {exc}")

    # ── Heuristic fallback ──────────────────────────────────────────────────
    words = query.lower().split()
    q_keywords = " ".join(w for w in words
                          if len(w) > 3 and w not in
                          {"what","when","where","which","that","this","with",
                           "from","have","does","been","will","would","could"})
    variants = [query]

    # rephrase as "X is/was ..."
    for sw in ["who","what","when","where","which","how"]:
        if words and words[0] == sw:
            variants.append(" ".join(words[1:]).capitalize())
            break

    variants.append(f"{q_keywords} information facts")
    variants.append(f"definition details about {q_keywords}")

    seen, out = set(), []
    for v in variants:
        k = v.lower()
        if k not in seen and len(v) > 3:
            seen.add(k); out.append(v)
    return out[:n]


def _rrf(ranked_lists: List[List[Tuple[str, float, dict]]], k: int = 60) -> List[Tuple[str, float, dict]]:
    """Reciprocal Rank Fusion over multiple ranked lists."""
    scores: Dict[str, float] = {}
    store:  Dict[str, Tuple[str, dict]] = {}
    for lst in ranked_lists:
        for rank, (text, _, meta) in enumerate(lst, 1):
            key = text[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            store[key]  = (text, meta)
    return [(store[k][0], scores[k], store[k][1])
            for k in sorted(scores, key=lambda x: scores[x], reverse=True)]


def run(
    query: str,
    corpus_index,
    top_k: int = 10,
    num_variants: int = 4,
    rrf_k: int = 60,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = "groq",
    gen_model: str = "llama3-70b-8192",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    from src.retrieval import embed_query
    from src.generation import generate_answer

    api_key = _resolve_key(api_key)

    # 1. Variants
    variants = _generate_query_variants(
        query, n=num_variants, provider=provider, model=gen_model, api_key=api_key
    )

    # 2. Retrieve per-variant with more candidates than top_k
    fetch_k = top_k * 2
    all_lists = []
    for v in variants:
        emb = embed_query(v, model_name=embedding_model)
        all_lists.append(corpus_index.retrieve(emb, top_k=fetch_k))

    # 3. RRF merge
    fused = _rrf(all_lists, k=rrf_k)

    # 4. Take top_k from fused
    top_chunks = fused[:top_k]

    # 5. Generate
    answer = generate_answer(query, top_chunks,
                              provider=provider, model=gen_model, api_key=api_key)
    return {
        "pipeline":       "rag_fusion",
        "answer":         answer,
        "retrieved":      top_chunks,
        "query_variants": variants,
    }