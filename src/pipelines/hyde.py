"""
HyDE: generate hypothetical document, retrieve from global index by similarity to it, generate final answer.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (hypothetical_doc, retrieved_passages, answer).
# Generate hypothetical passage with LLM -> embed it -> index.retrieve(hypothetical_embedding, top_k) -> generate from retrieved chunks.
