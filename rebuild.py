"""
rebuild_and_run.py
------------------
ONE SCRIPT that:
1. Deletes the old index
2. Rebuilds with query-enriched embeddings
3. Verifies scores are now correct (should be 0.3-0.7, not 0.06)
4. Runs evaluation on 50 examples

Run:
    set GROQ_API_KEY=gsk_your_key
    python rebuild_and_run.py
"""

import sys
import os
import shutil
from pathlib import Path
from src.corpus import build_index, load_index

# from transformers.convert_slow_tokenizers_checkpoints_to_fast import args

sys.path.insert(0, str(Path(__file__).parent))

# ── Step 1: Delete old index ──────────────────────────────────────────────────
# INDEX_DIR   = "data"
DATASET     = "dataset/crag_task_1_and_2_dev_v4.jsonl"

# print("=" * 60)
# print("STEP 1: Deleting old index")
# print("=" * 60)
# if Path(INDEX_DIR).exists():
#     shutil.rmtree(INDEX_DIR)
#     print(f"  Deleted '{INDEX_DIR}/' folder")
# else:
#     print(f"  '{INDEX_DIR}/' does not exist, skipping")

# ── Step 2: Rebuild index ─────────────────────────────────────────────────────
# print()
# print("=" * 60)
# print("STEP 2: Building new query-enriched index")
# print("  Each chunk embedded as: 'Q: {query}  A: {snippet}'")
# print("  Expected time: 10-15 minutes")
# print("=" * 60)

# from src.corpus import build_index
# corpus_index = build_index(
#     dataset_path=DATASET,
#     index_dir=INDEX_DIR,
#     embedding_model="all-MiniLM-L6-v2",
#     batch_size=128,
# )
# print(f"\n  Index built: {len(corpus_index)} chunks")
index_dir     = "data"
metadata_path = Path(index_dir) / "metadata.pkl"
corpus_index = load_index(index_dir=index_dir, embedding_model="all-MiniLM-L6-v2")
# ── Step 3: Verify scores ─────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 3: Verifying retrieval scores")
print("=" * 60)
from src.retrieval import embed_query

test_queries = [
    "how many 3-point attempts did steve nash average per game",
    "where did the ceo of salesforce previously work",
    "which movie won the oscar best visual effects in 2021",
]

all_ok = True
for q in test_queries:
    emb     = embed_query(q)
    results = corpus_index.retrieve(emb, top_k=3)
    scores  = [round(s, 3) for (_, s, _) in results]
    top_txt = results[0][0][:80] if results else "NONE"
    status  = "OK" if scores[0] > 0.15 else "LOW"
    if scores[0] <= 0.15:
        all_ok = False
    print(f"  [{status}] scores={scores}")
    print(f"        Q: {q[:60]}")
    print(f"        A: {top_txt}")
    print()

if all_ok:
    print("  Scores look good (>0.15). Proceeding to evaluation.")
else:
    print("  WARNING: Scores still low. Something is wrong with the index.")
    print("  Check that corpus.py has 'embed_text' in chunks.")
    sys.exit(1)

# ── Step 4: Run evaluation ────────────────────────────────────────────────────
print()
print("=" * 60)
print("STEP 4: Running evaluation (50 examples, all 4 pipelines)")
print("=" * 60)

import time, json
from src.data_loader import load_dataset
from src.pipelines import PIPELINES
from src.evaluation import compute_accuracy, print_summary, is_correct
provider = "groq"
model    = "llama-3.1-8b-instant"
# Get the API key from environment variables
# First, check if GROQ_API_KEY is set
# If not, fall back to OPENAI_API_KEY
# If not, fall back to ANTHROPIC_API_KEY
api_key = (os.getenv("GROQ_API_KEY") or
              os.getenv("OPENAI_API_KEY") or
              os.getenv("ANTHROPIC_API_KEY"))

if not api_key:
    print("WARNING: No API key found. Using local extractive fallback.")
    provider = "local"
    model    = None
else:
    print(f"  Provider : {provider}")
    print(f"  Model    : {model}")
    print(f"  API key  : {api_key[:8]}...")

dev_examples = []
for q, a, alt, _ in load_dataset(DATASET, max_examples=20):
    dev_examples.append({"query": q, "answer": a, "alt_ans": alt})
print(f"  Examples : {len(dev_examples)}\n")

all_results = {}
for pipeline_name, pipeline_fn in PIPELINES.items():
    print(f"--- {pipeline_name.upper()} ---")
    run_results = []
    correct_so_far = 0
    for i, ex in enumerate(dev_examples):
        try:
            result  = pipeline_fn(
                query=ex["query"], corpus_index=corpus_index,
                top_k=10, embedding_model="all-MiniLM-L6-v2",
                provider=provider, gen_model=model, api_key=api_key,
            )
            pred      = result.get("answer", "")
            retrieved = result.get("retrieved", [])
            scores    = [round(s,3) for (_,s,_) in retrieved[:3]]
            ok        = is_correct(pred, ex["answer"], ex["alt_ans"])
            if ok: correct_so_far += 1
            run_results.append({
                "query": ex["query"], "answer": ex["answer"],
                "alt_ans": ex["alt_ans"], "predicted_answer": pred,
                "retrieval_scores": scores, "correct": ok,
            })
            flag = "✓" if ok else "✗"
            acc  = correct_so_far / (i+1)
            print(f"  [{i+1:2d}/50] {flag} acc={acc:.0%} scores={scores} | {ex['query'][:50]}")
        except Exception as exc:
            import traceback; traceback.print_exc()
            run_results.append({
                "query": ex["query"], "answer": ex["answer"],
                "alt_ans": ex["alt_ans"], "predicted_answer": "",
                "retrieval_scores": [], "correct": False, "error": str(exc),
            })
    all_results[pipeline_name] = run_results
    print(run_results, "\n") 
    final = sum(r["correct"] for r in run_results) / len(run_results)
    print(f"  → {pipeline_name}: {final:.1%}\n")

summary = {n: compute_accuracy(r) for n, r in all_results.items()}
print_summary(summary)

with open("evaluation_results.json","w") as f:
    json.dump({
        "accuracy_summary": {k:{"accuracy":v["accuracy"],"correct":v["correct"],"total":v["total"]}
                             for k,v in summary.items()},
        "per_pipeline_results": all_results,
    }, f, indent=2)
print("Results saved to evaluation_results.json")
best = max(summary, key=lambda k: summary[k]["accuracy"])
print(f"\n Best pipeline: {best.upper()} ({summary[best]['accuracy']:.1%})\n")