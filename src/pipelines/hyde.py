"""
hyde.py  —  HyDE (Hypothetical Document Embedding)
---------------------------------------------------
Algorithm
  1. LLM writes a 3-4 sentence hypothetical passage that WOULD answer the query.
  2. Embed the hypothetical passage (not the raw query).
  3. Retrieve top-K real chunks closest to that embedding.
  4. ALSO retrieve top-K using the raw query embedding (hybrid).
  5. Merge both ranked lists with RRF for best coverage.
  6. LLM generates the final answer from retrieved real chunks.

Key fixes vs previous version
  - GROQ_API_KEY is now checked first
  - Hybrid retrieval (HyDE + raw query) via RRF — handles cases where the
    hypothetical doc drifts away from the actual corpus vocabulary
  - fetch_k = top_k * 2 before merging
"""

import os
from typing import List, Tuple, Dict, Optional


def _resolve_key(api_key):
    return (api_key
            or os.environ.get("GROQ_API_KEY")
            or os.environ.get("OPENAI_API_KEY")
            or os.environ.get("ANTHROPIC_API_KEY"))


def _generate_hypothetical_doc(
    query: str,
    provider: str = "groq",
    model: str = "llama3-70b-8192",
    api_key: Optional[str] = None,
) -> str:
    api_key = _resolve_key(api_key)
    system = (
        "You are an expert fact writer. "
        "Write a short 3-4 sentence factual passage — like a Wikipedia paragraph — "
        "that directly and completely answers the question. "
        "Include the specific facts, names, dates, or numbers the answer requires. "
        "Do NOT say 'I' or 'the answer is'. Write as a reference article."
    )
    prompt = f"Question: {query}\n\nFactual passage:"

    if api_key and provider != "local":
        try:
            if provider in ("groq", "openai"):
                from openai import OpenAI
                base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
                client = OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user",   "content": prompt},
                    ],
                    max_tokens=250,
                    temperature=0.5,
                )
                doc = resp.choices[0].message.content.strip()
                if doc:
                    return doc
            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=model, max_tokens=250, temperature=0.5,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                doc = resp.content[0].text.strip()
                if doc:
                    return doc
        except Exception as exc:
            print(f"[hyde] hypothetical doc failed: {exc}")

    # Heuristic fallback — expand query into pseudo-document
    import re
    keywords = " ".join(
        w for w in re.findall(r'\b\w{4,}\b', query.lower())
        if w not in {"what","when","where","which","that","this","with","from",
                     "have","does","been","will","would","could","directed","released"}
    )
    return (
        f"{query.rstrip('?')}. "
        f"This topic involves {keywords}. "
        f"The key facts related to {keywords} include specific names, dates, and details "
        f"found in encyclopedic and news sources."
    )


def _rrf(lists, k=60):
    scores, store = {}, {}
    max_cosine = {}
    for lst in lists:
        for rank, (text, cos_score, meta) in enumerate(lst, 1):
            key = text[:200]
            scores[key] = scores.get(key, 0.0) + 1.0 / (k + rank)
            store[key]  = (text, meta)
            max_cosine[key] = max(max_cosine.get(key, 0.0), cos_score)
            
    # Sort by RRF score, but return the real cosine score
    fused = sorted(scores.keys(), key=lambda x: scores[x], reverse=True)
    return [(store[k][0], max_cosine[k], store[k][1]) for k in fused]

def run(
    query: str,
    corpus_index,
    top_k: int = 15,
    embedding_model: str = "all-mpnet-base-v2",
    provider: str = "groq",
    gen_model: str = "llama3-70b-8192",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    from src.retrieval import embed_query
    from src.generation import generate_answer

    api_key = _resolve_key(api_key)
    fetch_k = top_k * 2

    # 1. Generate hypothetical document
    hyp_doc = _generate_hypothetical_doc(
        query, provider=provider, model=gen_model, api_key=api_key
    )

    # 2. Embed hypothetical doc → retrieve
    hyp_emb  = embed_query(hyp_doc, model_name=embedding_model)
    hyp_hits = corpus_index.retrieve(hyp_emb, top_k=fetch_k)

    # 3. Also retrieve with raw query embedding (hybrid safety net)
    raw_emb  = embed_query(query, model_name=embedding_model)
    raw_hits = corpus_index.retrieve(raw_emb, top_k=fetch_k)

    # 4. Merge with RRF
    merged = _rrf([hyp_hits, raw_hits])
    retrieved = merged[:top_k]

    # 5. Generate answer from real retrieved chunks
    answer = generate_answer(query, retrieved,
                              provider=provider, model=gen_model, api_key=api_key)
    return {
        "pipeline":         "hyde",
        "answer":           answer,
        "retrieved":        retrieved,
        "hypothetical_doc": hyp_doc,
    }