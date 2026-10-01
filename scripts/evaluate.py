"""
scripts/evaluate.py

Reproduce evaluation on Test/Dev Dataset (Sections 4.2 & 4.3):
- MRR@10
- Recall@50, Recall@200, Recall@1000
- Query Latency / Throughput
"""

import argparse
import time
import torch
from tqdm import tqdm

from scripts import config
from scripts.data import load_queries, load_qrels
from scripts.query_index import ColBERTSearcher

def calculate_mrr_at_k(qrels, results, k=10):
    mrr = 0.0
    evaluated = 0
    for qid, rels in qrels.items():
        if qid not in results:
            continue
        evaluated += 1
        for rank, did in enumerate(results[qid][:k], 1):
            if did in rels and rels[did] > 0:
                mrr += 1.0 / rank
                break
    return mrr / evaluated if evaluated > 0 else 0.0

def calculate_recall_at_k(qrels, results, k=50):
    recall = 0.0
    evaluated = 0
    for qid, rels in qrels.items():
        if qid not in results:
            continue
        evaluated += 1
        top_k = set(results[qid][:k])
        relevant_found = sum(1 for did in rels if did in top_k and rels[did] > 0)
        total_relevant = sum(1 for did in rels if rels[did] > 0)
        if total_relevant > 0:
            recall += relevant_found / total_relevant
    return recall / evaluated if evaluated > 0 else 0.0

def evaluate(data_dir=config.DATA_DIR, index_dir=config.INDEX_DIR, mode="e2e", limit=None):
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[evaluate] Hardware device: {device}")
    queries = load_queries(f"{data_dir}/queries.dev.tsv")
    qrels = load_qrels(f"{data_dir}/qrels.dev.tsv")

    if limit:
        target_qids = list(queries.keys())[:limit]
        queries = {q: queries[q] for q in target_qids}
        print(f"[evaluate] Subsetting to {limit} queries for quick validation.")

    searcher = ColBERTSearcher(index_dir=index_dir, device=device)

    results = {}
    print(f"[evaluate] Running retrieval (mode={mode})...")
    start_time = time.time()

    for qid, query_text in tqdm(queries.items(), desc="Evaluating"):
        retrieved_ids = searcher.search(query_text, k=1000, mode=mode)
        results[qid] = retrieved_ids

    elapsed = max(1e-6, time.time() - start_time)

    # Compute key metrics from Section 4.2 & 4.3
    mrr_10 = calculate_mrr_at_k(qrels, results, k=10)
    recall_50 = calculate_recall_at_k(qrels, results, k=50)
    recall_200 = calculate_recall_at_k(qrels, results, k=200)
    recall_1000 = calculate_recall_at_k(qrels, results, k=1000)

    print("\n" + "=" * 45)
    print("        COLBERT EVALUATION SUMMARY")
    print("=" * 45)
    print(f"Queries Evaluated : {len(queries):,}")
    print(f"Total Time        : {elapsed:.2f} s")
    print(f"Throughput        : {len(queries) / elapsed:.2f} queries/sec")
    print(f"Avg Latency       : {(elapsed / len(queries)) * 1000:.1f} ms/query")
    print("-" * 45)
    print(f"MRR@10            : {mrr_10:.4f}")
    print(f"Recall@50         : {recall_50:.4f}")
    print(f"Recall@200        : {recall_200:.4f}")
    print(f"Recall@1000       : {recall_1000:.4f}")
    print("=" * 45)

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Evaluate ColBERT on Test/Dev Set")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of queries")
    parser.add_argument("--mode", type=str, choices=["e2e", "re-rank"], default="e2e", help="Search mode")
    args = parser.parse_args()

    evaluate(mode=args.mode, limit=args.limit)
