"""
hyde.py
-------
HyDE (Hypothetical Document Embedding) Pipeline

Algorithm
---------
1. Given a query, use an LLM to generate a hypothetical document (1–2 paragraphs)
   that *might* contain the answer.
2. Embed the hypothetical document (NOT the raw query).
3. Retrieve top-k chunks from the global index by similarity to this embedding.
4. Pass the real retrieved chunks + query to the LLM for the final answer.

Why it works: a hypothetical document shares vocabulary and style with actual
answer-containing documents, so its embedding is closer to relevant chunks than
the raw question's embedding.
"""

import os
from typing import List, Tuple, Optional


def _generate_hypothetical_doc(
    query: str,
    provider: str = "groq",
    model: str = "llama3-8b-8192",
    api_key: Optional[str] = None,
) -> str:
    """
    Ask an LLM to write a short hypothetical document answering the query.
    Falls back to an extended query string if LLM is unavailable.
    """
    api_key = api_key or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")

    system = (
        "You are a knowledgeable assistant. "
        "Write a short 2–3 sentence passage that directly answers the following question. "
        "Write as if you are a web page or encyclopedia article. "
        "Be factual and use relevant terminology."
    )
    prompt = f"Question: {query}\n\nPassage:"

    if api_key:
        try:
            if provider in ("openai", "groq"):
                from openai import OpenAI
                base_url = "https://api.groq.com/openai/v1" if provider == "groq" else None
                client = OpenAI(api_key=api_key, base_url=base_url)
                resp = client.chat.completions.create(
                    model=model,
                    messages=[
                        {"role": "system", "content": system},
                        {"role": "user", "content": prompt},
                    ],
                    max_tokens=200,
                    temperature=0.7,
                )
                return resp.choices[0].message.content.strip()

            elif provider == "anthropic":
                import anthropic
                client = anthropic.Anthropic(api_key=api_key)
                resp = client.messages.create(
                    model=model,
                    max_tokens=200,
                    temperature=0.7,
                    system=system,
                    messages=[{"role": "user", "content": prompt}],
                )
                return resp.content[0].text.strip()

        except Exception as exc:
            print(f"[hyde] Hypothetical doc generation failed: {exc}. Using fallback.")

    # Heuristic fallback: expand the query into a pseudo-document
    return (
        f"The answer to '{query}' can be found in the following information. "
        f"{query.rstrip('?')} is a well-documented topic. "
        f"According to various sources, the relevant facts about {query} are as follows."
    )


def run(
    query: str,
    corpus_index,
    top_k: int = 5,
    embedding_model: str = "all-MiniLM-L6-v2",
    provider: str = "groq",
    gen_model: str = "llama3-8b-8192",
    api_key: Optional[str] = None,
    **kwargs,
) -> dict:
    """
    HyDE pipeline.

    Returns
    -------
    dict with keys: answer, retrieved, pipeline, hypothetical_doc
    """
    from src.retrieval import embed_query
    from src.generation import generate_answer

    # Step 1: Generate hypothetical document
    hyp_doc = _generate_hypothetical_doc(query, provider=provider, model=gen_model, api_key=api_key)

    # Step 2: Embed the hypothetical document
    hyp_emb = embed_query(hyp_doc, model_name=embedding_model)

    # Step 3: Retrieve using hypothetical doc embedding
    retrieved = corpus_index.retrieve(hyp_emb, top_k=top_k)

    # Step 4: Generate final answer from real retrieved chunks
    answer = generate_answer(query, retrieved, provider=provider, model=gen_model, api_key=api_key)

    return {
        "pipeline": "hyde",
        "answer": answer,
        "retrieved": retrieved,
        "hypothetical_doc": hyp_doc,
    }