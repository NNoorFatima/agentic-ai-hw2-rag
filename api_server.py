"""
api_server.py
-------------
Flask backend that loads the corpus index and exposes endpoints for the frontend.

Endpoints
---------
GET  /api/health               – health check
POST /api/query                – run a pipeline on a query
GET  /api/samples              – return sample queries from the dataset
"""

import json
import os
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from flask import Flask, request, jsonify
from flask_cors import CORS

app = Flask(__name__)
CORS(app)

# Global state
_corpus_index = None
_sample_queries = []


def _get_index():
    global _corpus_index
    if _corpus_index is None:
        from src.corpus import load_index
        index_dir = os.environ.get("INDEX_DIR", "data")
        embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")
        print(f"[api_server] Loading index from '{index_dir}' …")
        _corpus_index = load_index(index_dir=index_dir, embedding_model=embedding_model)
        print(f"[api_server] Index loaded: {len(_corpus_index)} chunks")
    return _corpus_index


def _load_samples():
    global _sample_queries
    if not _sample_queries:
        dataset = os.environ.get("DATASET_PATH", "dataset/crag_task_1_and_2_dev_v4.jsonl")
        try:
            from src.data_loader import load_dataset
            samples = []
            for q, a, aa, _ in load_dataset(dataset, max_examples=20):
                samples.append({"query": q, "answer": a})
            _sample_queries = samples
        except Exception as exc:
            print(f"[api_server] Could not load samples: {exc}")
            _sample_queries = [
                {"query": "Who directed Inception?", "answer": "Christopher Nolan"},
                {"query": "Which athlete has won more Grand Slams, Federer or Nadal?", "answer": "Nadal"},
            ]
    return _sample_queries


@app.route("/api/health", methods=["GET"])
def health():
    return jsonify({"status": "ok"})


@app.route("/api/samples", methods=["GET"])
def samples():
    return jsonify({"samples": _load_samples()})


@app.route("/api/query", methods=["POST"])
def query_endpoint():
    data = request.get_json(force=True)
    query = data.get("query", "").strip()
    pipeline_name = data.get("pipeline", "rag_fusion")
    top_k = int(data.get("top_k", 5))

    if not query:
        return jsonify({"error": "query is required"}), 400

    if pipeline_name not in ["rag_fusion", "hyde", "crag", "graph_rag"]:
        return jsonify({"error": f"Unknown pipeline: {pipeline_name}"}), 400

    provider = os.environ.get("LLM_PROVIDER", "groq")
    model = os.environ.get("LLM_MODEL", "llama3-70b-8192")
    api_key = os.environ.get("GROQ_API_KEY") or os.environ.get("OPENAI_API_KEY") or os.environ.get("ANTHROPIC_API_KEY")
    embedding_model = os.environ.get("EMBEDDING_MODEL", "all-MiniLM-L6-v2")

    try:
        corpus_index = _get_index()
        from src.pipelines import PIPELINES
        pipeline_fn = PIPELINES[pipeline_name]

        result = pipeline_fn(
            query=query,
            corpus_index=corpus_index,
            top_k=top_k,
            embedding_model=embedding_model,
            provider=provider,
            gen_model=model,
            api_key=api_key,
        )

        # Serialise retrieved chunks (remove numpy types)
        retrieved_serialised = []
        for (text, score, meta) in result.get("retrieved", []):
            retrieved_serialised.append({
                "text": text,
                "score": round(float(score), 4),
                "source_url": meta.get("source_url", ""),
                "source_name": meta.get("source_name", ""),
            })

        response = {
            "pipeline": pipeline_name,
            "query": query,
            "answer": result.get("answer", ""),
            "retrieved": retrieved_serialised,
        }

        # Pipeline-specific extra fields
        if pipeline_name == "rag_fusion":
            response["query_variants"] = result.get("query_variants", [])
        elif pipeline_name == "hyde":
            response["hypothetical_doc"] = result.get("hypothetical_doc", "")
        elif pipeline_name == "crag":
            response["confidence"] = round(float(result.get("confidence", 0)), 4)
            response["confidence_scores"] = [round(float(s), 4) for s in result.get("confidence_scores", [])]
            response["correction_path"] = result.get("correction_path", "")
        elif pipeline_name == "graph_rag":
            response["expanded_count"] = result.get("expanded_count", 0)

        return jsonify(response)

    except Exception as exc:
        import traceback
        traceback.print_exc()
        return jsonify({"error": str(exc)}), 500


if __name__ == "__main__":
    port = int(os.environ.get("PORT", 5000))
    # Pre-load index at startup
    try:
        _get_index()
    except Exception as e:
        print(f"[api_server] Warning: Could not pre-load index: {e}")
    app.run(host="0.0.0.0", port=port, debug=False)