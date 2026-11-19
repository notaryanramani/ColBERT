"""Evaluate the base and weighted models on MS MARCO and TREC CAR."""

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import faiss
import numpy as np
import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm
from transformers import AutoTokenizer

from weighted_score.src.base import ColBERT
from weighted_score.src.data import MAX_QUERY_LENGTH, MODEL_NAME
from weighted_score.src.weighted import WeightedColBERT


DATASETS = {
    "msmarco": {
        "data_dir": "/ml_data/aryan/msmarco_sample",
        "queries": ("sampled_queries.dev.tsv", "queries.dev.tsv"),
        "qrels": ("sampled_qrels.dev.tsv", "qrels.dev.tsv"),
    },
    "trec-car": {
        "data_dir": "/ml_data/aryan/trec-car_sample",
        "queries": ("sampled_queries.test.tsv", "queries.test.tsv"),
        "qrels": ("sampled_qrels.test.tsv", "qrels.test.tsv"),
    },
}


def read_queries(path):
    queries = {}
    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def read_qrels(path):
    qrels = {}
    with open(path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], set()).add(parts[2])
    return qrels


def find_data_file(data_dir, names):
    for name in names:
        path = os.path.join(data_dir, name)
        if os.path.isfile(path):
            return path
    raise FileNotFoundError(f"Could not find {names} in {data_dir}")


def find_index_dir(index_root, dataset, model_name):
    path = os.path.join(index_root, dataset, model_name)
    if os.path.isdir(path):
        return path

    # Support the earlier MS MARCO index layout, which had no dataset folder.
    if dataset == "msmarco":
        path = os.path.join(index_root, model_name)
        if os.path.isdir(path):
            return path

    raise FileNotFoundError(
        f"Index for {dataset}/{model_name} not found under {index_root}. "
        "Wait for indexing to finish or check the index output path."
    )


def mean_reciprocal_rank(qrels, results, k=10):
    scores = []
    for qid, relevant in qrels.items():
        ranked = results.get(qid, [])
        reciprocal_rank = 0.0
        for rank, doc_id in enumerate(ranked[:k], start=1):
            if doc_id in relevant:
                reciprocal_rank = 1.0 / rank
                break
        scores.append(reciprocal_rank)
    return float(np.mean(scores)) if scores else 0.0


def recall_at_k(qrels, results, k=50):
    scores = []
    for qid, relevant in qrels.items():
        if not relevant:
            continue
        found = set(results.get(qid, [])[:k]) & relevant
        scores.append(len(found) / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def mean_average_precision(qrels, results):
    scores = []
    for qid, relevant in qrels.items():
        if not relevant:
            continue
        ranked = results.get(qid, [])
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked, start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / rank
        scores.append(precision_sum / len(relevant))
    return float(np.mean(scores)) if scores else 0.0


def load_model(model_name, checkpoint, tokenizer, device):
    model_class = ColBERT if model_name == "base" else WeightedColBERT
    model = model_class(tokenizer=tokenizer).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()
    return model


def encode_query(model, tokenizer, query, device):
    tokens = tokenizer(
        f"[Q] {query}",
        truncation=True,
        max_length=MAX_QUERY_LENGTH,
        return_tensors="pt",
    )
    input_ids = tokens["input_ids"].to(device)
    attention_mask = tokens["attention_mask"].to(device)
    with torch.no_grad():
        query_vectors = model.encode(input_ids, attention_mask)
    return query_vectors, attention_mask


def get_candidates(faiss_index, offsets, doc_ids, query_vectors, top_k_tokens):
    query_vectors = query_vectors.squeeze(0).cpu().numpy().astype(np.float32)
    k = min(top_k_tokens, faiss_index.ntotal)
    _, token_ids = faiss_index.search(query_vectors, k)
    token_ids = token_ids[token_ids >= 0]

    # Each FAISS row is a token vector. Offsets identify its document bag.
    doc_indices = np.searchsorted(offsets[1:], token_ids, side="right")
    return sorted(set(int(i) for i in doc_indices if i < len(doc_ids)))


def rerank(model, query_vectors, query_mask, candidates, vectors, offsets,
           doc_ids, device, batch_size):
    ranked = []
    for start in range(0, len(candidates), batch_size):
        batch_indices = candidates[start:start + batch_size]
        bags = [
            torch.from_numpy(
                np.asarray(vectors[offsets[i]:offsets[i + 1]], dtype=np.float32)
            )
            for i in batch_indices
        ]
        lengths = torch.tensor([len(bag) for bag in bags], device=device)
        doc_vectors = pad_sequence(bags, batch_first=True).to(device)
        doc_mask = (
            torch.arange(doc_vectors.size(1), device=device)[None, :]
            < lengths[:, None]
        )

        with torch.no_grad():
            scores = model.score(
                query_vectors,
                doc_vectors,
                query_mask,
                doc_mask,
            )[0].cpu().tolist()
        ranked.extend(zip(batch_indices, scores))

    ranked.sort(key=lambda item: item[1], reverse=True)
    return [doc_ids[i] for i, _ in ranked]


def evaluate_one(dataset, model_name, args, tokenizer, device):
    paths = DATASETS[dataset]
    data_dir = getattr(args, f"{dataset.replace('-', '_')}_dir") or paths["data_dir"]
    query_path = find_data_file(data_dir, paths["queries"])
    qrels_path = find_data_file(data_dir, paths["qrels"])
    index_dir = find_index_dir(args.index_root, dataset, model_name)

    queries = read_queries(query_path)
    qrels = read_qrels(qrels_path)
    if args.limit is not None:
        queries = dict(list(queries.items())[:args.limit])
        qrels = {qid: rels for qid, rels in qrels.items() if qid in queries}

    doc_ids_path = os.path.join(index_dir, "document_ids.json")
    with open(doc_ids_path, "r", encoding="utf-8") as f_in:
        doc_ids = json.load(f_in)
    offsets = np.load(os.path.join(index_dir, "document_offsets.npy"))
    vectors = np.load(os.path.join(index_dir, "document_vectors.npy"), mmap_mode="r")

    cpu_index = faiss.read_index(os.path.join(index_dir, "faiss.index"))
    resources = None
    if args.faiss_device == "gpu":
        resources = faiss.StandardGpuResources()
        faiss_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    else:
        faiss_index = cpu_index

    checkpoint = args.base_checkpoint if model_name == "base" else args.weighted_checkpoint
    model = load_model(model_name, checkpoint, tokenizer, device)
    results = {}

    for qid, query in tqdm(queries.items(), desc=f"{dataset} {model_name}"):
        query_vectors, query_mask = encode_query(model, tokenizer, query, device)
        candidates = get_candidates(
            faiss_index,
            offsets,
            doc_ids,
            query_vectors,
            args.top_k_tokens,
        )
        results[qid] = rerank(
            model,
            query_vectors,
            query_mask,
            candidates,
            vectors,
            offsets,
            doc_ids,
            device,
            args.score_batch_size,
        )

    if dataset == "msmarco":
        metrics = {
            "MRR@10": mean_reciprocal_rank(qrels, results, k=10),
            "Recall@50": recall_at_k(qrels, results, k=50),
        }
    else:
        metrics = {
            "MAP": mean_average_precision(qrels, results),
            "MRR@10": mean_reciprocal_rank(qrels, results, k=10),
        }

    del model, faiss_index, cpu_index
    if resources is not None:
        del resources
        torch.cuda.empty_cache()
    return metrics


def main():
    parser = argparse.ArgumentParser(description="Evaluate ColBERT models on MS MARCO and TREC CAR")
    parser.add_argument("--index-root", default="/ml_data/aryan/colbert/indexes")
    parser.add_argument("--base-checkpoint", default="/ml_data/aryan/colbert/base_colbert.pt")
    parser.add_argument("--weighted-checkpoint", default="/ml_data/aryan/colbert/weighted_colbert.pt")
    parser.add_argument("--msmarco-dir", default=None)
    parser.add_argument("--trec-car-dir", default=None)
    parser.add_argument("--results", default="results/weighted_experiment_metrics.json")
    parser.add_argument("--top-k-tokens", type=int, default=1000)
    parser.add_argument("--score-batch-size", type=int, default=64)
    parser.add_argument("--faiss-device", choices=["gpu", "cpu"], default="gpu")
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()


    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})
    metrics = {}

    for dataset in DATASETS:
        metrics[dataset] = {}
        for model_name in ("weighted",):
            print(f"\nEvaluating {model_name} on {dataset}")
            metrics[dataset][model_name] = evaluate_one(
                dataset, model_name, args, tokenizer, device
            )
            print(metrics[dataset][model_name])

    os.makedirs(os.path.dirname(args.results) or ".", exist_ok=True)
    with open(args.results, "w", encoding="utf-8") as f_out:
        json.dump(metrics, f_out, indent=2)
    print(f"Saved metrics to {args.results}")


if __name__ == "__main__":
    main()
