"""
rag_fusion.py
-------------
RAG Fusion Pipeline

Algorithm
---------
1. Given a query, generate N query variants using an LLM (or heuristic).
2. For each variant, retrieve top-k chunks from the global index.
3. Merge all ranked lists using Reciprocal Rank Fusion (RRF).
4. Take top-k from the fused list.
5. Pass fused context to LLM for final answer generation.

RRF score formula: sum over queries of  1 / (k + rank)
where k is a constant (typically 60) and rank is 1-indexed position.
"""

from typing import List, Tuple, Dict, Optional
import os
import re


def _generate_query_variants(query: str, n: int = 3, provider: str = "groq",
                              model: str = "llama3-8b-8192", api_key: Optional[str] = None) -> List[str]:
    """
    Generate n rephrased/expanded versions of the query.
    Falls back to simple heuristic variants if LLM is unavailable.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    if api_key:
        prompt = (
            f"Generate {n} different search queries that could help answer: '{query}'\n"
            f"Return only the queries, one per line, no numbering."
        )
        try:
            if provider in ("openai", "groq"):
                from openai import OpenAI
                base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
                client = OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[{"role": "user", "content": prompt}],
                    max_tokens=200,
                    temperature=0.7,
                )
                text = resp.choices[0].message.content.strip()
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=model, max_tokens=200, temperature=0.7,
                    messages=[{"role": "user", "content": prompt}]
                )
                text = resp.content[0].text.strip()
            else:
                text = ""

            variants = [line.strip() for line in text.split("\n") if line.strip()]
            if variants:
                return [query] + variants[:n-1]
        except Exception as exc:
            print(f"[rag_fusion] LLM variant generation failed: {exc}. Using heuristics.")

    # Heuristic fallback: simple query reformulations
    words = query.lower().split()
    variants = [query]

    # Variant 1: "Tell me about X"
    variants.append(f"Tell me about {query}")

    # Variant 2: Strip question words and rephrase
    for stop in ["who", "what", "when", "where", "which", "how", "why"]:
        if words and words[0] == stop:
            rephrased = " ".join(words[1:]).capitalize()
            variants.append(rephrased)
            break
    else:
        variants.append(f"Information about {query}")

    # Variant 3: keyword extraction style
    keywords = [w for w in words if len(w) > 3 and w not in
                {"what", "when", "where", "which", "that", "this", "with", "from", "have", "does"}]
    variants.append(" ".join(keywords[:6]))

    return list(dict.fromkeys(variants))[:n]  # deduplicate, keep order


def _reciprocal_rank_fusion(
    ranked_lists: List[List[Tuple[str, float, dict]]],
    rrf_k: int = 60,
) -> List[Tuple[str, float, dict]]:
    """
    Merge multiple ranked lists using Reciprocal Rank Fusion.

    Parameters
    ----------
    ranked_lists : list of ranked result lists, each element is (text, score, meta)
    rrf_k        : RRF constant (default 60)

    Returns
    -------
    Merged list sorted by fused RRF score (descending)
    """
    # Use chunk text as unique key
    rrf_scores: Dict[str, float] = {}
    chunk_store: Dict[str, Tuple[str, dict]] = {}

    for ranked_list in ranked_lists:
        for rank, (text, score, meta) in enumerate(ranked_list, start=1):
            key = text[:200]  # deduplicate by first 200 chars
            rrf_scores[key] = rrf_scores.get(key, 0.0) + 1.0 / (rrf_k + rank)
            chunk_store[key] = (text, meta)

    sorted_keys = sorted(rrf_scores, key=lambda k: rrf_scores[k], reverse=True)
    return [(chunk_store[k][0], rrf_scores[k], chunk_store[k][1]) for k in sorted_keys]


def run(
    query: str,
    corpus_index,
    top_k: int = 5,
    num_variants: int = 3,
    rrf_k: int = 60,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = "groq",
    gen_model: str = "llama3-8b-8192",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    RAG Fusion pipeline.

    Returns
    -------
    dict with keys: answer, retrieved, pipeline, query_variants
    """
    from src.retrieval import embed_query
    from src.generation import generate_answer

    # Step 1: Generate query variants
    variants = _generate_query_variants(query, n=num_variants, provider=provider,
                                         model=gen_model, api_key=api_key)

    # Step 2: Retrieve for each variant
    all_ranked_lists = []
    for variant in variants:
        q_emb = embed_query(variant, model_name=embedding_model)
        results = corpus_index.retrieve(q_emb, top_k=top_k)
        all_ranked_lists.append(results)

    # Step 3: RRF fusion
    fused = _reciprocal_rank_fusion(all_ranked_lists, rrf_k=rrf_k)

    # Step 4: Top-k from fused
    top_chunks = fused[:top_k]

    # Step 5: Generate answer
    answer = generate_answer(
        query, top_chunks,
        provider=provider, model=gen_model, api_key=api_key
    )

    return {
        "pipeline": "rag_fusion",
        "answer": answer,
        "retrieved": top_chunks,
        "query_variants": variants,
    }