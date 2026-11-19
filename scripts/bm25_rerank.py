import argparse
import json
import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"
os.environ['OPENAI_API_KEY'] = 'dummy'

import torch
from torch.nn.utils.rnn import pad_sequence
from tqdm import tqdm

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint


DATASETS = {
    "msmarco": {
        "data_dir": "/ml_data/aryan/msmarco_sample",
        "split": "dev",
    },
    "trec-car": {
        "data_dir": "/ml_data/aryan/trec-car_sample",
        "split": "test",
    },
}
BM25_TOP_K = 125
RERANK_TOP_K = 125
INDEX_BATCH_SIZE = 1000
INDEX_THREADS = 4
BATCH_SIZE = 16


def read_tsv(path):
    rows = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                rows[parts[0]] = parts[1]
    return rows


def read_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], set()).add(parts[2])
    return qrels


def build_index(collection_path, index_dir):
    from pyserini.index.lucene import LuceneIndexer

    indexer = LuceneIndexer(index_dir, threads=INDEX_THREADS)
    batch = []
    with open(collection_path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue
            batch.append({"id": parts[0], "contents": parts[1]})
            if len(batch) >= INDEX_BATCH_SIZE:
                indexer.add_batch_dict(batch)
                batch = []
    if batch:
        indexer.add_batch_dict(batch)
    indexer.close()


def encode_documents(checkpoint, texts):
    with torch.inference_mode():
        result = checkpoint.docFromText(
            texts,
            bsize=BATCH_SIZE,
            keep_dims="flatten",
            to_cpu=False,
        )
    embeddings, lengths = result[0], result[1]
    output = []
    offset = 0
    for length in lengths:
        output.append(embeddings[offset:offset + length].detach().cpu().float().contiguous())
        offset += length
    return output


def maxsim_scores(query_embedding, doc_embeddings, batch_size=64):
    lengths = torch.tensor([len(embedding) for embedding in doc_embeddings])
    padded = pad_sequence(doc_embeddings, batch_first=True)
    query_embedding = query_embedding.to("cuda:0")
    scores = []
    for start in range(0, len(doc_embeddings), batch_size):
        docs = padded[start:start + batch_size].to("cuda:0")
        doc_lengths = lengths[start:start + batch_size].to("cuda:0")
        mask = torch.arange(docs.shape[1], device="cuda:0")[None, :] >= doc_lengths[:, None]
        similarities = torch.matmul(query_embedding, docs.transpose(1, 2))
        similarities.masked_fill_(mask[:, None, :], float("-inf"))
        scores.extend(similarities.max(dim=2).values.sum(dim=1).cpu().tolist())
    return scores


def evaluate(runs, qrels, dataset):
    query_ids = [qid for qid in runs if qid in qrels and qrels[qid]]
    if not query_ids:
        raise ValueError("No query IDs overlap between the run and positive qrels.")

    mrr_total = 0.0
    metric_total = 0.0
    for qid in query_ids:
        relevant = qrels[qid]
        ranked = runs[qid]
        for rank, doc_id in enumerate(ranked[:10], start=1):
            if doc_id in relevant:
                mrr_total += 1.0 / rank
                break
        if dataset == "msmarco":
            hits = sum(doc_id in relevant for doc_id in ranked[:50])
            metric_total += hits / len(relevant)
        else:
            hits = 0
            precision_sum = 0.0
            for rank, doc_id in enumerate(ranked, start=1):
                if doc_id in relevant:
                    hits += 1
                    precision_sum += hits / rank
            metric_total += precision_sum / len(relevant)

    count = len(query_ids)
    metric_name = "Recall@50" if dataset == "msmarco" else "MAP"
    return {
        "MRR@10": mrr_total / count,
        metric_name: metric_total / count,
        "evaluated_queries": count,
    }


def run_dataset(name, config, checkpoint, args):
    from pyserini.search.lucene import LuceneSearcher

    data_dir = config["data_dir"]
    split = config["split"]
    collection_path = os.path.join(data_dir, "sampled_collection.tsv")
    queries_path = os.path.join(data_dir, f"sampled_queries.{split}.tsv")
    qrels_path = os.path.join(data_dir, f"sampled_qrels.{split}.tsv")
    index_dir = os.path.join(data_dir, "index", "bm25_index")
    output_path = os.path.join("results", f"bm25_colbert_{name}_{split}.json")

    for path in (collection_path, queries_path, qrels_path):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    if not os.path.isfile(os.path.join(index_dir, "segments_1")):
        os.makedirs(os.path.dirname(index_dir), exist_ok=True)
        print(f"Building Pyserini index for {name}...")
        build_index(collection_path, index_dir)

    corpus = read_tsv(collection_path)
    queries = read_tsv(queries_path)
    qrels = read_qrels(qrels_path)
    queries = {qid: text for qid, text in queries.items() if qid in qrels}
    if not corpus or not queries or not qrels:
        raise ValueError(f"{name} data is empty or its paths/formats are incorrect.")

    print(f"{name}: {len(corpus):,} documents, {len(queries):,} queries")
    searcher = LuceneSearcher(index_dir)
    bm25_runs = {}
    reranked_runs = {}

    for qid, query in tqdm(queries.items(), desc=f"{name} BM25 and reranking"):
        hits = searcher.search(query, k=BM25_TOP_K)
        bm25_ids = [hit.docid for hit in hits]
        selected_ids = bm25_ids[:RERANK_TOP_K]
        selected_texts = [corpus[doc_id] for doc_id in selected_ids if doc_id in corpus]
        selected_ids = [doc_id for doc_id in selected_ids if doc_id in corpus]

        with torch.inference_mode():
            query_embedding = checkpoint.queryFromText(
                [query], bsize=1, to_cpu=True
            )[0].detach().cpu().float()
        doc_embeddings = encode_documents(checkpoint, selected_texts)
        scores = maxsim_scores(query_embedding, doc_embeddings)
        reranked_top = [
            selected_ids[i]
            for i in sorted(range(len(selected_ids)), key=scores.__getitem__, reverse=True)
        ]
        reranked_runs[qid] = reranked_top
        bm25_runs[qid] = bm25_ids

    metrics = {
        "BM25": evaluate(bm25_runs, qrels, name),
        "BM25+ColBERT": evaluate(reranked_runs, qrels, name),
    }
    metric_name = "Recall@50" if name == "msmarco" else "MAP"
    print(f"{name} MRR@10: {metrics['BM25']['MRR@10']:.4f} -> {metrics['BM25+ColBERT']['MRR@10']:.4f}")
    print(f"{name} {metric_name}: {metrics['BM25'][metric_name]:.4f} -> {metrics['BM25+ColBERT'][metric_name]:.4f}")

    output = {
        "dataset": name,
        "split": split,
        "model": args.model,
        "bm25_depth": BM25_TOP_K,
        "rerank_depth": RERANK_TOP_K,
        "metrics": metrics,
        "runs": {"BM25": bm25_runs, "BM25+ColBERT": reranked_runs},
    }
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(output, f)
    print(f"Saved results to {output_path}")


def main():
    global BATCH_SIZE
    parser = argparse.ArgumentParser()
    parser.add_argument("--dataset", choices=["all", *DATASETS], default="all")
    parser.add_argument("--batch-size", type=int, default=BATCH_SIZE)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--doc-maxlen", type=int, default=220)
    parser.add_argument("--model", default="jinaai/jina-colbert-v1-en")
    args = parser.parse_args()
    if args.batch_size < 1:
        raise ValueError("--batch-size must be positive.")
    BATCH_SIZE = args.batch_size

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA is required; CUDA_VISIBLE_DEVICES is set to 2.")
    torch.cuda.set_device(0)

    colbert_config = ColBERTConfig(
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        dim=128,
        gpus=1,
    )
    checkpoint = Checkpoint(args.model, colbert_config=colbert_config)

    datasets = DATASETS if args.dataset == "all" else {args.dataset: DATASETS[args.dataset]}
    for name, config in datasets.items():
        run_dataset(name, config, checkpoint, args)


if __name__ == "__main__":
    main()
