"""
run_evaluation.py
-----------------
Build (or load) the global index once, then run all four pipelines
on the dev set and report accuracy per pipeline.

Usage
-----
python run_evaluation.py [--dataset DATASET_PATH] [--index-dir DATA_DIR]
                         [--max-examples N] [--provider groq|openai|anthropic|local]
                         [--model MODEL_NAME] [--api-key KEY] [--rebuild]
                         [--output OUTPUT_FILE] [--top-k N]

Examples
--------
# With Groq (free) - RECOMMENDED
python run_evaluation.py --provider groq --model llama3-70b-8192 --max-examples 100

# With env var set
set GROQ_API_KEY=gsk_xxxx
python run_evaluation.py --provider groq --max-examples 100

# Extractive fallback only (no LLM)
python run_evaluation.py --provider local --max-examples 100
"""

import argparse
import json
import os
import sys
import time
from pathlib import Path

sys.path.insert(0, str(Path(__file__).parent))

from src.data_loader import load_dataset
from src.corpus import build_index, load_index
from src.pipelines import PIPELINES
from src.evaluation import compute_accuracy, print_summary, is_correct


def parse_args():
    parser = argparse.ArgumentParser(description="RAG Pipeline Evaluation")
    parser.add_argument("--dataset", default="dataset/crag_task_1_and_2_dev_v4.jsonl")
    parser.add_argument("--index-dir", default="data")
    parser.add_argument("--max-examples", type=int, default=100)
    parser.add_argument(
        "--provider", default="groq",
        choices=["groq", "openai", "anthropic", "local"],
        help="LLM provider. Use 'groq' for free Llama3 (get key at console.groq.com)"
    )
    parser.add_argument("--model", default=None,
                        help="Model name. Defaults: groq=llama3-70b-8192, openai=gpt-4o-mini")
    parser.add_argument("--api-key", default=None, help="API key (or set env var)")
    parser.add_argument("--rebuild", action="store_true", help="Force rebuild index")
    parser.add_argument("--output", default="evaluation_results.json")
    parser.add_argument("--top-k", type=int, default=10,
                        help="Chunks to retrieve per query (higher = better recall)")
    parser.add_argument("--embedding-model", default="all-MiniLM-L6-v2")
    parser.add_argument("--pipelines", nargs="+",
                        choices=["rag_fusion", "hyde", "crag", "graph_rag"],
                        default=["rag_fusion", "hyde", "crag", "graph_rag"])
    return parser.parse_args()


# Default models per provider
PROVIDER_DEFAULTS = {
    "groq":      "llama3-70b-8192",
    "openai":    "gpt-4o-mini",
    "anthropic": "claude-3-haiku-20240307",
    "local":     None,
}

# Env vars to check per provider
PROVIDER_ENV_KEYS = {
    "groq":      ["GROQ_API_KEY"],
    "openai":    ["OPENAI_API_KEY"],
    "anthropic": ["ANTHROPIC_API_KEY"],
    "local":     [],
}


def resolve_api_key(provider: str, cli_key: str) -> str:
    if cli_key:
        return cli_key
    for env_var in PROVIDER_ENV_KEYS.get(provider, []):
        val = os.environ.get(env_var)
        if val:
            return val
    # Also check all common ones as fallback
    for env_var in ["GROQ_API_KEY", "OPENAI_API_KEY", "ANTHROPIC_API_KEY"]:
        val = os.environ.get(env_var)
        if val:
            return val
    return None


def main():
    args = parse_args()

    provider = args.provider
    model    = args.model or PROVIDER_DEFAULTS.get(provider, "llama3-70b-8192")
    api_key  = resolve_api_key(provider, args.api_key)

    if provider != "local" and not api_key:
        print(f"\n⚠  WARNING: No API key found for provider '{provider}'.")
        print(f"   Set env var: {PROVIDER_ENV_KEYS.get(provider, ['GROQ_API_KEY'])[0]}=your_key")
        print(f"   Or pass: --api-key YOUR_KEY")
        print(f"   Falling back to LOCAL extractive mode (low accuracy expected).\n")
        provider = "local"
        model = None
    else:
        print(f"\n✓ Provider : {provider}")
        print(f"✓ Model    : {model}")
        print(f"✓ API key  : {'set (' + api_key[:8] + '...)' if api_key else 'NOT SET'}\n")

    # ── Step 1: Build or load index ──────────────────────────────────────────
    index_dir     = args.index_dir
    metadata_path = Path(index_dir) / "metadata.pkl"

    if args.rebuild or not metadata_path.exists():
        print("[run_evaluation] Building corpus index …")
        corpus_index = build_index(
            dataset_path=args.dataset,
            index_dir=index_dir,
            embedding_model=args.embedding_model,
        )
    else:
        print("[run_evaluation] Loading existing corpus index …")
        corpus_index = load_index(index_dir=index_dir, embedding_model=args.embedding_model)

    print(f"[run_evaluation] Corpus size: {len(corpus_index)} chunks\n")

    # ── Step 2: Load dev examples ─────────────────────────────────────────────
    print(f"[run_evaluation] Loading up to {args.max_examples} dev examples …")
    dev_examples = []
    for query, answer, alt_ans, _ in load_dataset(args.dataset, max_examples=args.max_examples):
        dev_examples.append({"query": query, "answer": answer, "alt_ans": alt_ans})
    print(f"[run_evaluation] Loaded {len(dev_examples)} examples.\n")
    print(f"[run_evaluation] Running pipelines: {args.pipelines}")
    print(f"[run_evaluation] top_k={args.top_k}\n")

    # ── Step 3: Run each pipeline ─────────────────────────────────────────────
    all_pipeline_results = {}

    for pipeline_name in args.pipelines:
        pipeline_fn = PIPELINES[pipeline_name]
        print(f"{'─'*60}")
        print(f"  Pipeline: {pipeline_name.upper()}")
        print(f"{'─'*60}")

        run_results = []
        correct_so_far = 0

        for i, ex in enumerate(dev_examples):
            query = ex["query"]
            try:
                t0 = time.time()
                result = pipeline_fn(
                    query=query,
                    corpus_index=corpus_index,
                    top_k=args.top_k,
                    embedding_model=args.embedding_model,
                    provider=provider,
                    gen_model=model,
                    api_key=api_key,
                )
                elapsed = time.time() - t0
                pred    = result.get("answer", "")
                retrieved = result.get("retrieved", [])
                scores_str = ", ".join(f"{s:.3f}" for (_, s, _) in retrieved[:3])

                ok = is_correct(pred, ex["answer"], ex["alt_ans"])
                if ok:
                    correct_so_far += 1

                run_results.append({
                    "query":            query,
                    "answer":           ex["answer"],
                    "alt_ans":          ex["alt_ans"],
                    "predicted_answer": pred,
                    "retrieval_scores": [round(s, 4) for (_, s, _) in retrieved],
                    "correct":          ok,
                    "elapsed_s":        round(elapsed, 2),
                })

                flag = "✓" if ok else "✗"
                running_acc = correct_so_far / (i + 1)
                print(f"  [{i+1:3d}/{len(dev_examples)}] {flag} acc={running_acc:.0%} "
                      f"| scores=[{scores_str}] | {query[:55]}")

            except Exception as exc:
                import traceback; traceback.print_exc()
                run_results.append({
                    "query":            query,
                    "answer":           ex["answer"],
                    "alt_ans":          ex["alt_ans"],
                    "predicted_answer": "",
                    "retrieval_scores": [],
                    "correct":          False,
                    "error":            str(exc),
                })

        all_pipeline_results[pipeline_name] = run_results
        final_acc = sum(r["correct"] for r in run_results) / len(run_results)
        print(f"\n  → {pipeline_name} final accuracy: {final_acc:.1%}\n")

    # ── Step 4: Accuracy summary ──────────────────────────────────────────────
    accuracy_summary = {}
    for name, results in all_pipeline_results.items():
        acc = compute_accuracy(results)
        accuracy_summary[name] = acc

    print_summary(accuracy_summary)

    # ── Step 5: Save results ──────────────────────────────────────────────────
    output = {
        "config": {
            "dataset":         args.dataset,
            "max_examples":    args.max_examples,
            "provider":        provider,
            "model":           model,
            "top_k":           args.top_k,
            "embedding_model": args.embedding_model,
        },
        "accuracy_summary": {
            k: {"accuracy": v["accuracy"], "correct": v["correct"], "total": v["total"]}
            for k, v in accuracy_summary.items()
        },
        "per_pipeline_results": all_pipeline_results,
    }

    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2, ensure_ascii=False)

    print(f"[run_evaluation] Results saved to '{args.output}'")

    if accuracy_summary:
        best = max(accuracy_summary, key=lambda k: accuracy_summary[k]["accuracy"])
        print(f"\n🏆  Best pipeline : {best.upper()} "
              f"({accuracy_summary[best]['accuracy']:.1%} accuracy)\n")


if __name__ == "__main__":
    main()