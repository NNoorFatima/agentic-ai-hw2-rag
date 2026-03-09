"""
Graph RAG: graph-augmented retrieval over the corpus (e.g. entity/relation graph or similarity graph), then generate.
Do not remove or rename this file.
"""

# TODO: Implement run(query, index, embedder, top_k, generator) -> (retrieved_passages, answer).
# Build or use a graph over the corpus (entities, relations, or chunk similarity) ->
# retrieve in a graph-aware way (e.g. entity linking, subgraph, spread from seeds) ->
# convert selected graph neighborhood to text -> generate answer from that context.
