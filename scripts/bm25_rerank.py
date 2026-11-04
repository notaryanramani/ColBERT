import argparse
import csv
import json
import os
import re
from collections import defaultdict

import torch
from rank_bm25 import BM25Okapi
from tqdm import tqdm

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


def tokenize(text):
    return re.findall(r"[a-z0-9]+", str(text).lower())


def load_scifact(data_dir, split):
    corpus = {}
    with open(os.path.join(data_dir, "corpus.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            doc_id = str(row["_id"])
            title = row.get("title", "")
            body = row.get("text", "")
            corpus[doc_id] = (title + " " + body).strip()

    queries = {}
    with open(os.path.join(data_dir, "queries.jsonl"), encoding="utf-8") as f:
        for line in f:
            row = json.loads(line)
            queries[str(row["_id"])] = str(row.get("text", ""))

    qrels = defaultdict(set)
    qrels_path = os.path.join(data_dir, "qrels", f"{split}.tsv")
    with open(qrels_path, encoding="utf-8") as f:
        reader = csv.DictReader(f, delimiter="\t")
        for row in reader:
            if int(row["score"]) > 0:
                qrels[str(row["query-id"])].add(str(row["corpus-id"]))

    return corpus, queries, dict(qrels)


def load_msmarco(data_dir, split):
    def read_two_column_tsv(path):
        result = {}
        with open(path, encoding="utf-8") as f:
            for line in f:
                parts = line.rstrip("\n").split("\t", 1)
                if len(parts) != 2:
                    continue
                if parts[0].lower() in {"qid", "query-id", "pid", "docid"}:
                    continue
                result[str(parts[0])] = parts[1]
        return result

    corpus = read_two_column_tsv(os.path.join(data_dir, "collection.tsv"))
    queries = read_two_column_tsv(os.path.join(data_dir, f"queries.{split}.tsv"))

    qrels = defaultdict(set)
    qrels_path = os.path.join(data_dir, f"qrels.{split}.tsv")
    with open(qrels_path, encoding="utf-8") as f:
        for line in f:
            parts = line.strip().split()
            if not parts or parts[0].lower() in {"qid", "query-id"}:
                continue
            # TREC format: qid 0 docid relevance
            # Three-column format: qid docid relevance
            if len(parts) >= 4:
                qid, doc_id, relevance = parts[0], parts[2], parts[3]
            elif len(parts) == 3:
                qid, doc_id, relevance = parts
            else:
                continue
            if int(relevance) > 0:
                qrels[str(qid)].add(str(doc_id))

    return corpus, queries, dict(qrels)


def retrieve_bm25(bm25, doc_ids, query, top_k):
    scores = bm25.get_scores(tokenize(query))
    ranked_indices = sorted(
        range(len(doc_ids)), key=lambda i: scores[i], reverse=True
    )[:top_k]
    return [(doc_ids[i], float(scores[i])) for i in ranked_indices]


def encode_candidate_documents(checkpoint, doc_ids, corpus, batch_size):
    embeddings = {}
    for start in tqdm(
        range(0, len(doc_ids), batch_size),
        desc="Encoding candidate documents",
    ):
        batch_ids = doc_ids[start : start + batch_size]
        batch_texts = [corpus[doc_id] for doc_id in batch_ids]

        with torch.inference_mode():
            result = checkpoint.docFromText(
                batch_texts,
                bsize=batch_size,
                keep_dims="flatten",
                to_cpu=True,
            )

        batch_embeddings = result[0]
        doc_lengths = result[1]

        offset = 0
        for doc_id, doc_len in zip(batch_ids, doc_lengths):
            doc_embedding = batch_embeddings[offset:offset + doc_len]
            embeddings[doc_id] = doc_embedding.detach().cpu().float().contiguous()
            offset += doc_len

    return embeddings


def maxsim_score(query_embedding, doc_embedding):
    # ColBERT MaxSim: for every query token, take its best document-token
    # similarity, then sum those maxima.
    similarities = torch.matmul(query_embedding, doc_embedding.T)
    return float(similarities.max(dim=1).values.sum().item())


def evaluate(runs, qrels):
    query_ids = [qid for qid in runs if qid in qrels and qrels[qid]]
    if not query_ids:
        raise ValueError("No query IDs overlap between the run and positive qrels.")

    mrr_total = 0.0
    recall_total = 0.0
    map_total = 0.0

    for qid in query_ids:
        relevant = qrels[qid]
        ranked = runs[qid]

        # MRR@10
        for rank, doc_id in enumerate(ranked[:10], start=1):
            if doc_id in relevant:
                mrr_total += 1.0 / rank
                break

        # Recall@50
        hits_50 = sum(doc_id in relevant for doc_id in ranked[:50])
        recall_total += hits_50 / len(relevant)

        # AP@100: denominator is the total number of relevant qrels
        hits = 0
        precision_sum = 0.0
        for rank, doc_id in enumerate(ranked[:100], start=1):
            if doc_id in relevant:
                hits += 1
                precision_sum += hits / rank
        map_total += precision_sum / len(relevant)

    count = len(query_ids)
    return {
        "MRR@10": mrr_total / count,
        "Recall@50": recall_total / count,
        "MAP@100": map_total / count,
        "evaluated_queries": count,
    }


def main():
    parser = argparse.ArgumentParser(
        description="BM25 retrieval followed by pretrained ColBERT MaxSim reranking."
    )
    parser.add_argument("--dataset", choices=["scifact", "msmarco"], default="scifact")
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=16)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--doc-maxlen", type=int, default=220)
    parser.add_argument("--threads", type=int, default=4)
    parser.add_argument("--model", default="jinaai/jina-colbert-v1-en")
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    if args.data_dir is None:
        args.data_dir = "data/scifact" if args.dataset == "scifact" else "data/msmarco"
    if args.split is None:
        args.split = "test" if args.dataset == "scifact" else "dev"
    if args.output is None:
        args.output = f"results/bm25_colbert_{args.dataset}_{args.split}.json"

    torch.set_num_threads(max(1, args.threads))

    print(f"Loading {args.dataset} ({args.split})...")
    if args.dataset == "scifact":
        corpus, queries, qrels = load_scifact(args.data_dir, args.split)
    else:
        corpus, queries, qrels = load_msmarco(args.data_dir, args.split)

    # Evaluate only queries that have positive relevance judgments.
    queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if not corpus or not queries or not qrels:
        raise ValueError("Dataset is empty or its paths/formats are incorrect.")

    doc_ids = list(corpus.keys())
    print(f"Documents: {len(doc_ids):,}")
    print(f"Queries with qrels: {len(queries):,}")

    print("Building BM25 index...")
    tokenized_corpus = [tokenize(corpus[doc_id]) for doc_id in tqdm(doc_ids, desc="Tokenizing")]
    bm25 = BM25Okapi(tokenized_corpus)

    print(f"Retrieving BM25 top-{args.top_k} for every query...")
    bm25_runs = {}
    bm25_scores = {}
    candidate_ids = set()

    for qid, query in tqdm(queries.items(), desc="BM25 retrieval"):
        results = retrieve_bm25(bm25, doc_ids, query, args.top_k)
        bm25_runs[qid] = [doc_id for doc_id, _ in results]
        bm25_scores[qid] = {doc_id: score for doc_id, score in results}
        candidate_ids.update(bm25_runs[qid])

    candidate_ids = list(candidate_ids)
    print(f"Unique candidate documents to encode: {len(candidate_ids):,}")

    print(f"Loading ColBERT checkpoint: {args.model}")
    colbert_config = ColBERTConfig(
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=128,
        gpus=0,
    )
    checkpoint = Checkpoint(args.model, colbert_config=colbert_config)

    doc_embeddings = encode_candidate_documents(
        checkpoint, candidate_ids, corpus, args.batch_size
    )

    print("Reranking BM25 candidates with ColBERT MaxSim...")
    colbert_runs = {}
    colbert_scores = {}

    for qid, query_text in tqdm(queries.items(), desc="ColBERT reranking"):
        with torch.inference_mode():
            query_embedding = checkpoint.queryFromText(
                [query_text], bsize=1, to_cpu=True
            )[0].detach().cpu().float()

        scores = {}
        for doc_id in bm25_runs[qid]:
            scores[doc_id] = maxsim_score(query_embedding, doc_embeddings[doc_id])

        # Python's sort is stable, so BM25 order breaks exact ColBERT-score ties.
        ranked = sorted(bm25_runs[qid], key=lambda doc_id: scores[doc_id], reverse=True)
        colbert_runs[qid] = ranked
        colbert_scores[qid] = scores

    print("\nEvaluation")
    print("-" * 54)
    print(f"{'Metric':<18}{'BM25':>16}{'BM25 + ColBERT':>20}")
    print("-" * 54)

    bm25_metrics = evaluate(bm25_runs, qrels)
    colbert_metrics = evaluate(colbert_runs, qrels)

    for metric in ["MRR@10", "Recall@50", "MAP@100"]:
        print(
            f"{metric:<18}{bm25_metrics[metric]:>16.4f}"
            f"{colbert_metrics[metric]:>20.4f}"
        )
    print("-" * 54)
    print(f"Evaluated queries: {bm25_metrics['evaluated_queries']}")

    output = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "candidate_depth": args.top_k,
        "metrics": {
            "BM25": bm25_metrics,
            "BM25+ColBERT": colbert_metrics,
        },
        "runs": {
            "BM25": bm25_runs,
            "BM25+ColBERT": colbert_runs,
        },
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output, f, indent=2)

    print(f"\nSaved metrics and rankings to: {args.output}")


if __name__ == "__main__":
    main()
