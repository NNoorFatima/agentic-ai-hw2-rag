"""
CRAG (Corrective RAG): assess retrieval confidence; use or correct retrieval based on it, then generate.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# Retrieve from global index -> assess confidence (e.g. NLI, consistency, or LLM judge) ->
# if high: use retrieved chunks for generation; if low: skip retrieval or use fallback -> generate answer.
