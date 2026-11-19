import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['OPENAI_API_KEY'] = 'dummy'

import numpy as np
import torch
from tqdm import tqdm

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint

from pooled_score.src.data import DATASETS, ensure_bm25_index, read_qrels, read_queries
from pooled_score.src.pooling import CONFIGS, DIM, build_pooled_indexes, load_pooled_index


BM25_TOP_K = 1000
FULL_TOP_K = 1000
INDEX_ROOT = "/ml_data/aryan/pooled_score/indexes"


def padded_documents(vectors, offsets, doc_indices, device):
    lengths = np.asarray(
        [int(offsets[i + 1] - offsets[i]) for i in doc_indices], dtype=np.int64
    )
    max_pools = int(lengths.max())
    padded = np.zeros((len(doc_indices), max_pools, DIM), dtype=np.float16)
    for row, doc_index in enumerate(doc_indices):
        start = int(offsets[doc_index])
        end = int(offsets[doc_index + 1])
        padded[row, : end - start] = vectors[start:end]
    return (
        torch.from_numpy(padded).to(device),
        torch.from_numpy(lengths).to(device),
    )


def score_documents(query_embeddings, documents, lengths):
    query_embeddings = query_embeddings.to(documents.device).float()
    documents = documents.float()
    query_count, query_length, _ = query_embeddings.shape
    doc_count, max_pools, _ = documents.shape
    similarities = torch.matmul(
        query_embeddings.reshape(query_count * query_length, DIM),
        documents.reshape(doc_count * max_pools, DIM).T,
    ).reshape(query_count, query_length, doc_count, max_pools)
    invalid = torch.arange(max_pools, device=documents.device)[None, :] >= lengths[:, None]
    similarities.masked_fill_(invalid[None, None, :, :], float("-inf"))
    return similarities.max(dim=-1).values.sum(dim=1)


def retrieve_full(index, query_embeddings, top_k, doc_batch_size, query_batch_size, device):
    doc_ids, offsets, vectors = index
    num_queries = len(query_embeddings)
    num_docs = len(doc_ids)
    top_k = min(top_k, num_docs)
    best_scores = torch.full((num_queries, top_k), float("-inf"), device=device)
    best_indices = torch.full((num_queries, top_k), -1, dtype=torch.long, device=device)

    with torch.inference_mode():
        for doc_start in tqdm(range(0, num_docs, doc_batch_size), desc="Full retrieval"):
            doc_end = min(doc_start + doc_batch_size, num_docs)
            doc_indices = np.arange(doc_start, doc_end, dtype=np.int64)
            documents, lengths = padded_documents(vectors, offsets, doc_indices, device)
            batch_indices = torch.arange(doc_start, doc_end, device=device)

            for query_start in range(0, num_queries, query_batch_size):
                query_end = min(query_start + query_batch_size, num_queries)
                scores = score_documents(
                    query_embeddings[query_start:query_end], documents, lengths
                )
                merged_scores = torch.cat(
                    (best_scores[query_start:query_end], scores), dim=1
                )
                expanded_indices = batch_indices[None, :].expand(query_end - query_start, -1)
                merged_indices = torch.cat(
                    (best_indices[query_start:query_end], expanded_indices), dim=1
                )
                keep = min(top_k, doc_end)
                top_values, top_positions = merged_scores.topk(keep, dim=1)
                best_scores[query_start:query_end, :keep] = top_values
                best_indices[query_start:query_end, :keep] = torch.gather(
                    merged_indices,
                    1,
                    top_positions,
                )

    return best_indices.cpu().numpy()


def accumulate_metrics(totals, qid, ranked, qrels, dataset):
    relevant = qrels.get(qid, set())
    if not relevant:
        return

    totals["queries"] += 1
    for rank, doc_id in enumerate(ranked[:10], start=1):
        if doc_id in relevant:
            totals["MRR@10"] += 1.0 / rank
            break

    if dataset == "msmarco":
        hits = sum(doc_id in relevant for doc_id in ranked[:50])
        totals["Recall@50"] += hits / len(relevant)
    else:
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / rank
        totals["MAP"] += precision_sum / len(relevant)


def finish_metrics(totals, dataset):
    count = max(1, totals["queries"])
    metric = "Recall@50" if dataset == "msmarco" else "MAP"
    return {
        "MRR@10": totals["MRR@10"] / count,
        metric: totals[metric] / count,
        "evaluated_queries": totals["queries"],
    }


def new_totals():
    return {"queries": 0, "MRR@10": 0.0, "Recall@50": 0.0, "MAP": 0.0}


def run_dataset(name, settings, checkpoint, args, device):
    from pyserini.search.lucene import LuceneSearcher

    data_dir = settings["data_dir"]
    split = settings["split"]
    collection_path = os.path.join(data_dir, "sampled_collection.tsv")
    query_path = os.path.join(data_dir, f"sampled_queries.{split}.tsv")
    qrels_path = os.path.join(data_dir, f"sampled_qrels.{split}.tsv")
    bm25_index_dir = os.path.join(data_dir, "index", "bm25_index")
    pooled_index_dir = os.path.join(INDEX_ROOT, name)

    for path in (collection_path, query_path, qrels_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    queries = read_queries(query_path)
    qrels = read_qrels(qrels_path)
    queries = {qid: text for qid, text in queries.items() if qid in qrels and qrels[qid]}
    if not queries:
        raise ValueError(f"No judged queries found for {name}.")

    ensure_bm25_index(collection_path, bm25_index_dir)
    searcher = LuceneSearcher(bm25_index_dir)
    print(f"{name}: {len(queries):,} queries; retrieving BM25 top-{BM25_TOP_K}")

    build_pooled_indexes(
        checkpoint,
        collection_path,
        pooled_index_dir,
        args.model,
        args.doc_maxlen,
        args.batch_size,
    )
    with open(os.path.join(pooled_index_dir, "document_ids.json"), encoding="utf-8") as f:
        doc_ids = json.load(f)
    doc_id_to_index = {doc_id: i for i, doc_id in enumerate(doc_ids)}

    query_ids = list(queries)
    with torch.inference_mode():
        query_embeddings = checkpoint.queryFromText(
            [queries[qid] for qid in query_ids],
            bsize=args.batch_size,
            to_cpu=True,
        ).detach().cpu().half()

    bm25_totals = new_totals()
    bm25_candidates = {}
    for qid, query in tqdm(queries.items(), desc=f"{name} BM25 retrieval"):
        hits = searcher.search(query, k=BM25_TOP_K)
        run = [hit.docid for hit in hits if hit.docid in doc_id_to_index]
        bm25_candidates[qid] = np.asarray(
            [doc_id_to_index[doc_id] for doc_id in run], dtype=np.int64
        )
        accumulate_metrics(bm25_totals, qid, run, qrels, name)

    results = {
        "split": split,
        "documents": len(doc_ids),
        "queries": len(query_ids),
        "bm25_depth": BM25_TOP_K,
        "full_depth": FULL_TOP_K,
        "BM25": finish_metrics(bm25_totals, name),
        "configurations": {},
    }

    for config in CONFIGS:
        print(f"{name}: {config['name']}")
        index = load_pooled_index(pooled_index_dir, config)
        offsets = index[1]
        total_vectors = int(offsets[-1])
        metrics = {"full_corpus": new_totals(), "bm25_rerank": new_totals()}

        full_indices = retrieve_full(
            index,
            query_embeddings,
            FULL_TOP_K,
            args.doc_batch_size,
            args.query_batch_size,
            device,
        )
        for row, qid in enumerate(query_ids):
            ranked = [doc_ids[i] for i in full_indices[row]]
            accumulate_metrics(metrics["full_corpus"], qid, ranked, qrels, name)

        with torch.inference_mode():
            for row, qid in enumerate(tqdm(query_ids, desc=f"{name} BM25 rerank")):
                candidate_indices = bm25_candidates[qid]
                docs, lengths = padded_documents(
                    index[2], offsets, candidate_indices, device
                )
                scores = score_documents(
                    query_embeddings[row : row + 1], docs, lengths
                )[0]
                order = torch.argsort(scores, descending=True).cpu().numpy()
                ranked = [doc_ids[candidate_indices[i]] for i in order]
                accumulate_metrics(metrics["bm25_rerank"], qid, ranked, qrels, name)

        results["configurations"][config["name"]] = {
            "pooling": config,
            "total_pool_vectors": total_vectors,
            "avg_pools_per_document": total_vectors / len(doc_ids),
            "storage_mb": round(total_vectors * DIM * 2 / (1024 * 1024), 2),
            "full_corpus": finish_metrics(metrics["full_corpus"], name),
            "bm25_rerank": finish_metrics(metrics["bm25_rerank"], name),
        }
        del index, full_indices
        torch.cuda.empty_cache()

    del searcher, query_embeddings
    return results


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--model", default="jinaai/jina-colbert-v1-en")
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--doc-batch-size", type=int, default=512)
    parser.add_argument("--query-batch-size", type=int, default=32)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--doc-maxlen", type=int, default=220)
    parser.add_argument("--threads", type=int, default=8)
    args = parser.parse_args()
    if min(args.batch_size, args.doc_batch_size, args.query_batch_size) < 1:
        raise ValueError("Batch sizes must be positive.")

    torch.set_num_threads(max(1, args.threads))
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CUDA_VISIBLE_DEVICES is set to 2.")
    torch.cuda.set_device(0)
    device = torch.device("cuda:0")

    config = ColBERTConfig(
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=DIM,
        gpus=1,
    )
    checkpoint = Checkpoint(args.model, colbert_config=config)

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    all_results = {}
    for name, settings in datasets.items():
        all_results[name] = run_dataset(name, settings, checkpoint, args, device)
        os.makedirs("results", exist_ok=True)
        with open("results/pool_late_interaction_metrics.json", "w", encoding="utf-8") as f:
            json.dump(all_results, f, indent=2)
    print("Saved metrics to results/pool_late_interaction_metrics.json")


if __name__ == "__main__":
    main()
