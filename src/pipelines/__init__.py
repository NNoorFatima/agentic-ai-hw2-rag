# Pipelines: crag, graph_rag, rag_fusion, hyde.
# Do not remove or rename this package or the pipeline modules.
"""
pipelines package
-----------------
Exposes the four RAG pipeline run functions.

Each pipeline function signature:
    run(query, corpus_index, **kwargs) -> dict

Returns dict with keys:
    answer      : str
    retrieved   : list[(text, score, metadata)]
    pipeline    : str (pipeline name)
"""

try:
    from src.pipelines.rag_fusion import run as run_rag_fusion
except ImportError:
    from .rag_fusion import run as run_rag_fusion

try:
    from src.pipelines.hyde import run as run_hyde
except ImportError:
    from .hyde import run as run_hyde

try:
    from src.pipelines.crag import run as run_crag
except ImportError:
    from .crag import run as run_crag

try:
    from src.pipelines.graph_rag import run as run_graph_rag
except ImportError:
    from .graph_rag import run as run_graph_rag

PIPELINES = {
    "rag_fusion": run_rag_fusion,
    "hyde": run_hyde,
    "crag": run_crag,
    "graph_rag": run_graph_rag,
}

__all__ = ["run_rag_fusion", "run_hyde", "run_crag", "run_graph_rag", "PIPELINES"]