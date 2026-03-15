# RAG Pipeline Evaluation and Recommendation Report

## Overview

This report evaluates four Retrieval-Augmented Generation (RAG) pipelines—RAG Fusion, HyDE, CRAG, and Graph RAG—on a dataset of 50 factual questions from the CRAG benchmark. The evaluation uses the Groq API with Llama-3.1-8B-Instant model, all-mpnet-base-v2 embeddings, and top_k=15 retrieval. Accuracy is measured via multi-strategy matching (exact, substring, F1 ≥ 0.3). The goal is to recommend a production-ready pipeline for a question-answering product.

## Accuracy Metrics

Accuracy is evaluated using a multi-strategy matching approach designed for long, descriptive gold answers in the CRAG dataset. The system checks predictions against gold answers in order of preference:

1. **Exact Match**: After normalizing text (lowercasing, removing punctuation, converting numbers to words), the prediction must match the gold exactly.
2. **Gold Contained in Prediction**: The entire gold answer is a substring of the prediction.
3. **Prediction Contained in Gold**: The entire prediction is a substring of the gold (for partial but correct answers).
4. **Key Token F1 ≥ 0.3**: Computes F1 score on tokenized predictions and golds, requiring at least 30% overlap to account for paraphrasing in long answers.

This lenient evaluation captures semantic correctness without requiring perfect wording, suitable for generative QA where answers vary in length and phrasing.

## Pipeline Descriptions

### RAG Fusion
RAG Fusion generates multiple query variants using an LLM, retrieves top-k chunks for each variant independently, and merges results using Reciprocal Rank Fusion (RRF) to prioritize diverse, relevant information. This pipeline excels at multi-hop and complex queries by exploring query perspectives, achieving 60% accuracy (30/50 correct). It handles ambiguous questions well but can be slower due to multiple retrievals and LLM calls for variants.

### HyDE (Hypothetical Document Embeddings)
HyDE creates hypothetical documents answering the query, embeds them, and retrieves similar real chunks from the corpus. It merges semantic matches from raw and hypothetical embeddings, selecting top-k for generation. With 56% accuracy (28/50), it performs strongly on creative or descriptive queries by bridging semantic gaps, though it may hallucinate on factual mismatches and requires more compute for document generation.

### CRAG (Corrective RAG)
CRAG scores retrieved chunks for relevance using cosine similarity or NLI, routing to all top-k (high confidence ≥0.65), top-3 (medium ≥0.40), or expanded retrieval + keyword re-ranking (low <0.40). It reduces hallucinations by gating generation on confidence, achieving 54% accuracy (27/50). Effective for noisy corpora, it adapts retrieval dynamically but adds complexity in scoring and expansion.

### Graph RAG
Graph RAG builds a similarity graph from seed retrievals, expands via edges, and selects top-k chunks based on graph centrality. It captures interconnected knowledge, scoring 52% accuracy (26/50). Suitable for relational queries, it improves on isolated retrieval but can be computationally intensive and less effective on straightforward facts.

## Recommendation

I recommend shipping **RAG Fusion** as the primary pipeline. It achieves the highest accuracy (60%, 30/50) across diverse query types, outperforming others by 4-8 percentage points. Patterns show Fusion excels on multi-hop questions (e.g., correctly answering "what is a movie to feature a person who can create and control a device that can manipulate the laws of physics?" with full details), leveraging query variants to mitigate retrieval gaps. HyDE follows closely (56%) but hallucinates more on factual queries like stock prices. CRAG (54%) reduces errors via confidence gating but underperforms on high-confidence cases. Graph RAG (52%) helps relational tasks but lags on simple facts.

Fusion's robustness makes it ideal for a general QA product, balancing accuracy and adaptability without excessive compute. For production, monitor latency (Fusion's multiple retrievals add ~10-20s per query) and consider hybrid approaches if specific query types dominate.

This recommendation is based on empirical results; re-evaluate with larger datasets or user feedback.

## Implementation Notes

- **Codebase**: Pipelines in `src/pipelines/`, evaluation in `src/evaluation.py`, retrieval with keyword re-ranking in `src/retrieval.py`.
- **Dependencies**: SentenceTransformers for embeddings, OpenAI/Groq for LLM, FAISS for indexing.
- **Config**: Use `config.yaml` for settings; embeddings pre-built in `data/`.
- **Future Improvements**: Add gold answer inclusion in corpus for better grounding, tune thresholds, or integrate NLI for CRAG.

## Codebase File Descriptions

- **api_server.py**: Flask-based API server that exposes endpoints for querying the RAG system, handling requests and responses for the QA product.
- **rebuild.py**: Script to rebuild the FAISS index from the corpus, embedding chunks with all-mpnet-base-v2 and saving to `data/`.
- **run_evaluation.py**: Main evaluation script that loads the index, runs all pipelines on the dataset, computes accuracy, and outputs JSON results.
- **src/corpus.py**: Handles corpus building by chunking text, embedding with SentenceTransformers, and creating/loading FAISS index.
- **src/data_loader.py**: Loads and preprocesses the CRAG dataset from JSONL files, preparing queries and answers for evaluation.
- **src/evaluation.py**: Implements multi-strategy answer matching (exact, substring, F1) and computes accuracy metrics.
- **src/generation.py**: Manages LLM calls for answer generation, with system prompts, API integrations (Groq/OpenAI), and local fallback extraction.
- **src/retrieval.py**: Performs vector retrieval with semantic search, applies keyword re-ranking (70% semantic + 30% keyword), and returns top-k chunks.
- **src/pipelines/**: Directory containing pipeline implementations:
  - `rag_fusion.py`: Generates query variants, retrieves per variant, merges with RRF.
  - `hyde.py`: Creates hypothetical documents, retrieves based on them, merges embeddings.
  - `crag.py`: Scores chunks for confidence, routes retrieval (high/med/low paths).
  - `graph_rag.py`: Builds similarity graph from retrievals, selects top chunks via graph.
  - `basic_rag.py`: Placeholder for basic retrieval-generation (not fully implemented).
- **frontend/**: Web UI with `index.html`, `src/App.jsx` (React app), and `vite.config.js` for user interaction.
- **config/config.example.yaml**: Template for configuration (embedding model, top_k, LLM settings).
- **data/**: Stores pre-built embeddings (`embeddings.npy`), FAISS index (`index.faiss`).
- **dataset/**: Contains the CRAG dataset (`crag_task_1_and_2_dev_v4.jsonl`).
- **requirements.txt**: Python dependencies (sentence-transformers, openai, flask, etc.).

