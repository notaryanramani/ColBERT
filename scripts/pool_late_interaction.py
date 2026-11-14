import argparse
import json
import math
import os
import time
import warnings
from collections import defaultdict

import numpy as np
import torch
import torch.nn.functional as F
from rank_bm25 import BM25Okapi
from sklearn.metrics import ndcg_score
from tqdm import tqdm

warnings.filterwarnings("ignore", category=FutureWarning)
warnings.filterwarnings("ignore", category=UserWarning)

from colbert.infra import ColBERTConfig
from colbert.modeling.checkpoint import Checkpoint
from colbert.modeling.colbert import ColBERT

from scripts.bm25_rerank import load_msmarco, retrieve_bm25, tokenize

# Searcher-only C++ extension is not needed for Checkpoint.docFromText / queryFromText
ColBERT.try_load_torch_extensions = classmethod(lambda cls, use_gpu: None)


def load_dataset_by_name(dataset_name: str, data_dir: str | None = None, split: str | None = None):
    """
    Loads msmarco or trec-car in a unified (corpus, queries, qrels) format.
    """
    if dataset_name == "msmarco":
        data_dir = data_dir or "data/msmarco"
        split = split or "dev"
        return load_msmarco(data_dir, split), data_dir, split
    elif dataset_name == "trec-car":
        data_dir = data_dir or "data/trec-car"
        split = split or "test"
        return load_msmarco(data_dir, split), data_dir, split
    else:
        raise ValueError(f"Unsupported dataset: {dataset_name}")



def pool_document_embeddings(
    doc_embedding: torch.Tensor,
    num_pools: int | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    pool_fn: str = "mean",
    normalize: bool = True,
) -> torch.Tensor:
    """
    Divide document token embeddings E_d of shape [|d|, dim] into K contiguous
    local pools [a_k, b_k] and aggregate each pool into a single representation P_k:

        P_k = Pool(E_{d_{a_k}}, ..., E_{d_{b_k}})

    Supports two natural ways of specifying the local segments [a_k, b_k]:
      - Fixed number of pools K (`num_pools=K`): partitions [0, |d|) into
        min(K, |d|) equal contiguous spans (exact `torch.tensor_split`).
      - Fixed local window size w (`window_size=w`, optional `stride=s`):
        divides [0, |d|) into contiguous local windows of w consecutive tokens
        (so K = ceil(|d| / s)), preserving local phrase context of length w.
    """
    n_tokens = doc_embedding.size(0)
    if n_tokens == 0:
        return doc_embedding

    if num_pools is not None:
        k = max(1, min(int(num_pools), n_tokens))
        if k >= n_tokens:
            return doc_embedding
        chunks = torch.tensor_split(doc_embedding, k, dim=0)
    elif window_size is not None:
        w = max(1, int(window_size))
        s = max(1, int(stride)) if stride is not None else w
        if w == 1 and s == 1:
            return doc_embedding
        chunks = [
            doc_embedding[start : start + w]
            for start in range(0, n_tokens, s)
            if start < n_tokens
        ]
    else:
        return doc_embedding

    if pool_fn == "mean":
        pooled = torch.stack([c.mean(dim=0) for c in chunks], dim=0)
    elif pool_fn == "max":
        pooled = torch.stack([c.max(dim=0).values for c in chunks], dim=0)
    else:
        raise ValueError(f"Unknown pool_fn: {pool_fn}")

    if normalize:
        pooled = F.normalize(pooled, p=2, dim=-1)

    return pooled


def pool_late_interaction_score(
    query_embedding: torch.Tensor,
    pooled_doc_embedding: torch.Tensor,
) -> float:
    """
    Compute late interaction between query tokens E_q [|q|, dim] and pooled
    document representations P [K, dim]:

        S(q, d) = sum_{i=1}^{|q|} max_{k=1,...,K} ( E_{q_i}^T P_k )
    """
    similarities = torch.matmul(query_embedding, pooled_doc_embedding.T)
    return float(similarities.max(dim=1).values.sum().item())


# ---------------------------------------------------------------------------
# Embedding Cache (encode once, evaluate all pooling configurations fast)
# ---------------------------------------------------------------------------

def get_or_compute_embeddings(
    cache_path: str,
    model_name: str,
    corpus: dict[str, str],
    queries: dict[str, str],
    query_maxlen: int = 32,
    doc_maxlen: int = 220,
    batch_size: int = 32,
):
    if os.path.exists(cache_path):
        print(f"Loading cached token embeddings from {cache_path} ...")
        data = torch.load(cache_path, map_location="cpu", weights_only=False)
        doc_ids = data["doc_ids"]
        flat_docs = data["flat_doc_embeddings"].float()
        doc_offsets = data["doc_offsets"]
        doc_embeddings = {
            doc_id: flat_docs[int(doc_offsets[i]) : int(doc_offsets[i + 1])]
            for i, doc_id in enumerate(doc_ids)
        }
        query_embeddings = {
            qid: emb.float() for qid, emb in data["query_embeddings"].items()
        }
        print(
            f"Loaded {len(doc_embeddings):,} docs ({flat_docs.size(0):,} total tokens, "
            f"avg {flat_docs.size(0) / len(doc_embeddings):.1f} tokens/doc) and "
            f"{len(query_embeddings):,} queries."
        )
        return doc_ids, doc_embeddings, query_embeddings

    print(f"Loading ColBERT checkpoint: {model_name}")
    colbert_config = ColBERTConfig(
        query_maxlen=query_maxlen,
        doc_maxlen=doc_maxlen,
        dim=128,
        gpus=0,
    )
    checkpoint = Checkpoint(model_name, colbert_config=colbert_config)

    doc_ids = list(corpus.keys())
    doc_tensors = []
    doc_lengths = []

    for start in tqdm(
        range(0, len(doc_ids), batch_size),
        desc="Encoding all corpus documents",
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
        batch_emb, lengths = result[0].detach().cpu().half(), result[1]
        doc_tensors.append(batch_emb)
        doc_lengths.extend(int(l) for l in lengths)

    flat_docs = torch.cat(doc_tensors, dim=0)
    doc_offsets = torch.tensor([0] + list(np.cumsum(doc_lengths)), dtype=torch.int64)

    query_embeddings_half = {}
    qids = list(queries.keys())
    qtexts = [queries[qid] for qid in qids]
    with torch.inference_mode():
        q_embs = checkpoint.queryFromText(qtexts, bsize=batch_size, to_cpu=True)
    for i, qid in enumerate(qids):
        query_embeddings_half[qid] = q_embs[i].detach().cpu().half()

    os.makedirs(os.path.dirname(cache_path) or ".", exist_ok=True)
    torch.save(
        {
            "doc_ids": doc_ids,
            "flat_doc_embeddings": flat_docs,
            "doc_offsets": doc_offsets,
            "query_embeddings": query_embeddings_half,
        },
        cache_path,
    )
    print(f"Saved token embeddings cache to {cache_path}")
    return get_or_compute_embeddings(
        cache_path, model_name, corpus, queries, query_maxlen, doc_maxlen, batch_size
    )


# Evaluation Metrics

def evaluate_all_metrics(runs: dict[str, list[str]], qrels: dict[str, set[str]]) -> dict:
    """
    Computes both:
      - finetuning/colbert.ipynb metrics: NDCG@10, MRR@10, Recall@100, MAP@10
      - scripts/bm25_rerank.py metrics:   MRR@10, Recall@50, MAP@100
    """
    query_ids = [qid for qid in runs if qid in qrels and qrels[qid]]
    ndcg10_list = []
    mrr10_list = []
    recall50_list = []
    recall100_list = []
    map10_list = []
    map100_list = []

    for qid in query_ids:
        relevant = qrels[qid]
        ranked = runs[qid]

        # NDCG@10 
        top10 = ranked[:10]
        if top10:
            y_true = np.array([[1 if doc in relevant else 0 for doc in top10]])
            y_score = np.array([[len(ranked) - i for i in range(len(top10))]])
            ndcg10_list.append(float(ndcg_score(y_true, y_score, k=10)))
        else:
            ndcg10_list.append(0.0)

        # MRR@10
        rr = 0.0
        for rank, doc_id in enumerate(top10, start=1):
            if doc_id in relevant:
                rr = 1.0 / rank
                break
        mrr10_list.append(rr)

        # Recall@50 and Recall@100
        hits_50 = sum(doc_id in relevant for doc_id in ranked[:50])
        hits_100 = sum(doc_id in relevant for doc_id in ranked[:100])
        recall50_list.append(hits_50 / len(relevant))
        recall100_list.append(hits_100 / len(relevant))

        # MAP@10 denominator min(|rel|, 10))
        hits = 0
        prec_sum10 = 0.0
        for rank, doc_id in enumerate(top10, start=1):
            if doc_id in relevant:
                hits += 1
                prec_sum10 += hits / rank
        map10_list.append(prec_sum10 / min(len(relevant), 10))

        # MAP@100 (denominator |rel|)
        hits = 0
        prec_sum100 = 0.0
        for rank, doc_id in enumerate(ranked[:100], start=1):
            if doc_id in relevant:
                hits += 1
                prec_sum100 += hits / rank
        map100_list.append(prec_sum100 / len(relevant))

    return {
        "NDCG@10": float(np.mean(ndcg10_list)),
        "MRR@10": float(np.mean(mrr10_list)),
        "Recall@50": float(np.mean(recall50_list)),
        "Recall@100": float(np.mean(recall100_list)),
        "MAP@10": float(np.mean(map10_list)),
        "MAP@100": float(np.mean(map100_list)),
        "evaluated_queries": len(query_ids),
    }


def precompute_pooled_corpus(
    doc_ids: list[str],
    doc_embeddings: dict[str, torch.Tensor],
    num_pools: int | None = None,
    window_size: int | None = None,
    stride: int | None = None,
    pool_fn: str = "mean",
    normalize: bool = True,
):
    """
    Pools every document in the corpus and packs them into a padded 3D tensor
    [num_docs, max_K, dim] + mask for fast vectorized MaxSim scoring across all
    5,183 documents, as well as a per-doc dict for candidate reranking.
    """
    pooled_dict = {}
    lengths = []
    for doc_id in doc_ids:
        p = pool_document_embeddings(
            doc_embeddings[doc_id],
            num_pools=num_pools,
            window_size=window_size,
            stride=stride,
            pool_fn=pool_fn,
            normalize=normalize,
        )
        pooled_dict[doc_id] = p
        lengths.append(p.size(0))

    total_vecs = int(sum(lengths))
    avg_vecs = total_vecs / len(doc_ids)
    storage_mb_fp16 = (total_vecs * 128 * 2) / (1024 * 1024)
    return pooled_dict, total_vecs, avg_vecs, storage_mb_fp16


def compute_all_scores_matrix(
    doc_ids: list[str],
    qids: list[str],
    query_embeddings: dict[str, torch.Tensor],
    pooled_docs: dict[str, torch.Tensor],
    doc_batch_size: int = 512,
    query_batch_size: int = 512,
) -> tuple[torch.Tensor, float]:
    """
    Computes the full [num_queries, num_docs] Pool-Based Late Interaction score
    matrix once using vectorized batched matrix multiplication. Both Full-Corpus
    Retrieval and BM25 Top-100 Reranking read directly from this matrix!
    """
    num_docs = len(doc_ids)
    dim = 128

    Q_tensor = torch.stack([query_embeddings[qid] for qid in qids], dim=0)  # [Q, 32, 128]
    num_q, q_len, _ = Q_tensor.shape
    all_scores = torch.empty((num_q, num_docs), dtype=torch.float32)

    t0 = time.perf_counter()
    for start in range(0, num_docs, doc_batch_size):
        end = min(start + doc_batch_size, num_docs)
        batch_ids = doc_ids[start:end]
        B = len(batch_ids)
        max_k = max(pooled_docs[d].size(0) for d in batch_ids)

        padded = torch.zeros((B, max_k, dim), dtype=torch.float32)
        valid_mask = torch.zeros((B, max_k), dtype=torch.bool)
        for i, d in enumerate(batch_ids):
            emb = pooled_docs[d]
            k_len = emb.size(0)
            padded[i, :k_len] = emb
            valid_mask[i, :k_len] = True

        padded_flat_T = padded.reshape(B * max_k, dim).T  # [128, B*max_k]
        mask_view = ~valid_mask.unsqueeze(0).unsqueeze(0)

        for q_start in range(0, num_q, query_batch_size):
            q_end = min(q_start + query_batch_size, num_q)
            q_chunk = Q_tensor[q_start:q_end]
            qb = q_end - q_start
            sim = torch.matmul(q_chunk.reshape(qb * q_len, dim), padded_flat_T).reshape(
                qb, q_len, B, max_k
            )
            sim.masked_fill_(mask_view, -1e9)
            all_scores[q_start:q_end, start:end] = sim.max(dim=-1).values.sum(dim=1)

    elapsed_ms = (time.perf_counter() - t0) * 1000.0
    return all_scores, elapsed_ms


def extract_runs_from_score_matrix(
    all_scores: torch.Tensor,
    doc_ids: list[str],
    doc_id_to_idx: dict[str, int],
    qids: list[str],
    bm25_runs: dict[str, list[str]],
    top_k: int = 100,
) -> tuple[dict[str, list[str]], dict[str, list[str]], float]:

    num_q, num_docs = all_scores.shape
    top_indices = torch.topk(all_scores, k=min(top_k, num_docs), dim=1).indices.numpy()
    full_runs = {
        qids[q_idx]: [doc_ids[i] for i in top_indices[q_idx]]
        for q_idx in range(num_q)
    }

    t0 = time.perf_counter()
    scores_np = all_scores.numpy()
    reranked_runs = {}
    for q_idx, qid in enumerate(qids):
        candidates = bm25_runs[qid]
        row = scores_np[q_idx]
        reranked_runs[qid] = sorted(
            candidates, key=lambda d: float(row[doc_id_to_idx[d]]), reverse=True
        )
    rerank_ms = (time.perf_counter() - t0) * 1000.0
    return full_runs, reranked_runs, rerank_ms



def find_qualitative_examples(
    queries: dict[str, str],
    corpus: dict[str, str],
    qrels: dict[str, set[str]],
    baseline_runs: dict[str, list[str]],
    pooled_runs: dict[str, list[str]],
    max_examples: int = 5,
) -> list[dict]:
    """
    Finds concrete SciFact queries where Pool-Based Late Interaction ranks a
    relevant ground-truth document higher than token-level MaxSim (e.g., where
    token-level MaxSim was distracted by isolated keyword overlap in a
    non-relevant document).
    """
    improvements = []
    for qid, relevant in qrels.items():
        if not relevant or qid not in baseline_runs or qid not in pooled_runs:
            continue
        base_list = baseline_runs[qid]
        pool_list = pooled_runs[qid]

        base_rank = next(
            (i + 1 for i, d in enumerate(base_list) if d in relevant), 999
        )
        pool_rank = next(
            (i + 1 for i, d in enumerate(pool_list) if d in relevant), 999
        )

        if pool_rank < base_rank and pool_rank <= 5:
            best_rel_doc = pool_list[pool_rank - 1]
            false_pos_doc = base_list[0] if base_list[0] not in relevant else None
            improvements.append(
                {
                    "qid": qid,
                    "query": queries[qid],
                    "baseline_rank": base_rank,
                    "pooled_rank": pool_rank,
                    "relevant_doc_id": best_rel_doc,
                    "relevant_doc_snippet": corpus[best_rel_doc][:260],
                    "baseline_top1_false_positive_id": false_pos_doc,
                    "baseline_top1_snippet": (
                        corpus[false_pos_doc][:260] if false_pos_doc else None
                    ),
                }
            )

    improvements.sort(key=lambda x: (x["pooled_rank"], -(x["baseline_rank"] - x["pooled_rank"])))
    return improvements[:max_examples]



def main():
    parser = argparse.ArgumentParser(
        description="Evaluate Pool-Based Late Interaction on MS MARCO or TREC-CAR."
    )
    parser.add_argument(
        "--dataset",
        choices=["msmarco", "trec-car"],
        default="msmarco",
    )
    parser.add_argument("--data-dir", default=None)
    parser.add_argument("--split", default=None)
    parser.add_argument("--top-k", type=int, default=100)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--query-maxlen", type=int, default=32)
    parser.add_argument("--doc-maxlen", type=int, default=220)
    parser.add_argument("--threads", type=int, default=8)
    parser.add_argument("--model", default="jinaai/jina-colbert-v1-en")
    parser.add_argument("--cache-path", default=None)
    parser.add_argument("--output", default=None)
    args = parser.parse_args()

    torch.set_num_threads(max(1, args.threads))

    (corpus, all_queries, qrels), data_dir, split = load_dataset_by_name(
        args.dataset, args.data_dir, args.split
    )
    args.data_dir = data_dir
    args.split = split
    if args.cache_path is None:
        args.cache_path = os.path.join(
            data_dir, f"{args.dataset}_colbert_token_embeddings.pt"
        )
    if args.output is None:
        args.output = f"results/pool_late_interaction_{args.dataset}_{split}.json"

    print(f"Loading {args.dataset} ({split}) from {data_dir} ...")
    # Only keep queries whose relevant docs exist in corpus
    corpus_ids_set = set(corpus.keys())
    qrels = {
        qid: {d for d in rels if d in corpus_ids_set}
        for qid, rels in qrels.items()
    }
    qrels = {qid: rels for qid, rels in qrels.items() if rels}
    queries = {qid: text for qid, text in all_queries.items() if qid in qrels}
    print(f"Corpus: {len(corpus):,} documents | Evaluated queries: {len(queries):,}")

    # 1. Encode all corpus documents & queries once (or load from cache)
    doc_ids, doc_embeddings, query_embeddings = get_or_compute_embeddings(
        cache_path=args.cache_path,
        model_name=args.model,
        corpus=corpus,
        queries=queries,
        query_maxlen=args.query_maxlen,
        doc_maxlen=args.doc_maxlen,
        batch_size=args.batch_size,
    )

    # 2. Load or compute BM25 top-100 runs (cached per dataset)
    bm25_cache_path = os.path.join(
        data_dir, f"{args.dataset}_{split}_bm25_top{args.top_k}.json"
    )
    if os.path.exists(bm25_cache_path):
        print(f"Loading cached BM25 top-{args.top_k} candidates from {bm25_cache_path} ...")
        with open(bm25_cache_path, encoding="utf-8") as f:
            loaded_json = json.load(f)
            bm25_runs = loaded_json["runs"]["BM25"] if "runs" in loaded_json else loaded_json
    else:
        print(f"Building BM25 index and retrieving top-{args.top_k} candidates...")
        tokenized_corpus = [tokenize(corpus[d]) for d in tqdm(doc_ids, desc="BM25 Tokenize")]
        bm25 = BM25Okapi(tokenized_corpus)
        bm25_runs = {}
        for qid, qtext in tqdm(queries.items(), desc="BM25 Retrieval"):
            bm25_runs[qid] = [d for d, _ in retrieve_bm25(bm25, doc_ids, qtext, args.top_k)]
        with open(bm25_cache_path, "w", encoding="utf-8") as f:
            json.dump(bm25_runs, f)
        print(f"Saved BM25 candidates cache to {bm25_cache_path}")

    bm25_metrics = evaluate_all_metrics(bm25_runs, qrels)

    # 3. Define Pooling Configurations to Evaluate
    #    - Baseline token-level MaxSim (|d| tokens, no pooling)
    #    - Fixed-K contiguous pools: K in {8, 16, 24, 32, 48, 64, 96} (mean & max pooling)
    #    - Local phrase window pools: window w in {2, 3, 4, 6, 8} tokens (non-overlapping & overlapping stride)
    configs = [
        {"name": "Token-Level ColBERT (Baseline, |d| tokens)", "family": "baseline", "num_pools": None, "window_size": None, "stride": None, "pool_fn": "mean"},
        # Fixed K pools (exact slide equation: divide doc into K pools P_1..P_K)
        {"name": "Pool-ColBERT (Fixed K=8, Mean)",   "family": "fixed_K_mean", "num_pools": 8,  "window_size": None, "stride": None, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Fixed K=16, Mean)",  "family": "fixed_K_mean", "num_pools": 16, "window_size": None, "stride": None, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Fixed K=24, Mean)",  "family": "fixed_K_mean", "num_pools": 24, "window_size": None, "stride": None, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Fixed K=32, Mean)",  "family": "fixed_K_mean", "num_pools": 32, "window_size": None, "stride": None, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Fixed K=48, Mean)",  "family": "fixed_K_mean", "num_pools": 48, "window_size": None, "stride": None, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Fixed K=64, Mean)",  "family": "fixed_K_mean", "num_pools": 64, "window_size": None, "stride": None, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Fixed K=96, Mean)",  "family": "fixed_K_mean", "num_pools": 96, "window_size": None, "stride": None, "pool_fn": "mean"},
        # Fixed K with Max Pooling comparison
        {"name": "Pool-ColBERT (Fixed K=32, Max)",   "family": "fixed_K_max",  "num_pools": 32, "window_size": None, "stride": None, "pool_fn": "max"},
        {"name": "Pool-ColBERT (Fixed K=64, Max)",   "family": "fixed_K_max",  "num_pools": 64, "window_size": None, "stride": None, "pool_fn": "max"},
        # Contiguous Local Window Pools (e.g. w consecutive tokens per pool, K = ceil(|d|/w))
        {"name": "Pool-ColBERT (Window w=2, Mean)",  "family": "window_mean",  "num_pools": None, "window_size": 2, "stride": 2, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Window w=3, Mean)",  "family": "window_mean",  "num_pools": None, "window_size": 3, "stride": 3, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Window w=4, Mean)",  "family": "window_mean",  "num_pools": None, "window_size": 4, "stride": 4, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Window w=6, Mean)",  "family": "window_mean",  "num_pools": None, "window_size": 6, "stride": 6, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Window w=8, Mean)",  "family": "window_mean",  "num_pools": None, "window_size": 8, "stride": 8, "pool_fn": "mean"},
        # Overlapping Local Context Windows (sliding n-gram pools preserving phrase boundaries)
        {"name": "Pool-ColBERT (Sliding w=2, s=1, Mean)", "family": "sliding_mean", "num_pools": None, "window_size": 2, "stride": 1, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Sliding w=3, s=1, Mean)", "family": "sliding_mean", "num_pools": None, "window_size": 3, "stride": 1, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Sliding w=3, s=2, Mean)", "family": "sliding_mean", "num_pools": None, "window_size": 3, "stride": 2, "pool_fn": "mean"},
        {"name": "Pool-ColBERT (Sliding w=4, s=2, Mean)", "family": "sliding_mean", "num_pools": None, "window_size": 4, "stride": 2, "pool_fn": "mean"},
    ]

    results_summary = []
    saved_runs = {"BM25": bm25_runs}
    baseline_tokens_total = None
    doc_id_to_idx = {d: i for i, d in enumerate(doc_ids)}
    qids = list(query_embeddings.keys())

    print("\nRunning Pool-Based Late Interaction experiments...")
    for cfg in tqdm(configs, desc="Evaluating pooling configs"):
        pooled_docs, total_vecs, avg_vecs, storage_mb = precompute_pooled_corpus(
            doc_ids=doc_ids,
            doc_embeddings=doc_embeddings,
            num_pools=cfg["num_pools"],
            window_size=cfg["window_size"],
            stride=cfg["stride"],
            pool_fn=cfg["pool_fn"],
            normalize=True,
        )
        if baseline_tokens_total is None:
            baseline_tokens_total = total_vecs

        compression_ratio = baseline_tokens_total / total_vecs
        reduction_pct = (1.0 - (total_vecs / baseline_tokens_total)) * 100.0

        # Compute [Q, D] score matrix once via batched matmul, then extract both
        # Full-Corpus top-k and BM25 top-k reranking in milliseconds.
        all_scores, full_ms = compute_all_scores_matrix(
            doc_ids=doc_ids,
            qids=qids,
            query_embeddings=query_embeddings,
            pooled_docs=pooled_docs,
        )
        full_runs, rerank_runs, rerank_ms = extract_runs_from_score_matrix(
            all_scores=all_scores,
            doc_ids=doc_ids,
            doc_id_to_idx=doc_id_to_idx,
            qids=qids,
            bm25_runs=bm25_runs,
            top_k=args.top_k,
        )
        rerank_metrics = evaluate_all_metrics(rerank_runs, qrels)
        full_metrics = evaluate_all_metrics(full_runs, qrels)

        entry = {
            **cfg,
            "total_vectors": total_vecs,
            "avg_vectors_per_doc": round(avg_vecs, 2),
            "storage_mb_fp16": round(storage_mb, 2),
            "compression_ratio": round(compression_ratio, 2),
            "vector_reduction_pct": round(reduction_pct, 1),
            "bm25_rerank_time_ms": round(rerank_ms, 1),
            "full_ranking_time_ms": round(full_ms, 1),
            "bm25_rerank_metrics": rerank_metrics,
            "full_corpus_metrics": full_metrics,
        }
        results_summary.append(entry)
        saved_runs[cfg["name"]] = {
            "bm25_rerank": rerank_runs,
            "full_corpus": full_runs,
        }

    # Print formatted tables
    ds_upper = args.dataset.upper()
    print("\n" + "=" * 115)
    print(f"SETUP 1: FULL-CORPUS RETRIEVAL OVER ALL {len(doc_ids):,} {ds_upper} DOCUMENTS ({len(queries):,} queries)")
    print("=" * 115)
    header = (
        f"{'Configuration':<40} | {'Avg K':>6} | {'Comp.':>6} | {'Size':>7} | "
        f"{'NDCG@10':>8} | {'MRR@10':>8} | {'Rec@100':>8} | {'MAP@10':>8} | {'Time(ms)':>8}"
    )
    print(header)
    print("-" * 115)
    for r in results_summary:
        m = r["full_corpus_metrics"]
        print(
            f"{r['name']:<40} | {r['avg_vectors_per_doc']:>6.1f} | "
            f"{r['compression_ratio']:>5.2f}x | {r['storage_mb_fp16']:>5.1f}MB | "
            f"{m['NDCG@10']:>8.4f} | {m['MRR@10']:>8.4f} | "
            f"{m['Recall@100']:>8.4f} | {m['MAP@10']:>8.4f} | {r['full_ranking_time_ms']:>8.1f}"
        )

    print("\n" + "=" * 115)
    print("SETUP 2: BM25 TOP-100 RERANKING (compare with scripts/bm25_rerank.py)")
    print("=" * 115)
    header2 = (
        f"{'Configuration':<40} | {'Avg K':>6} | {'Comp.':>6} | "
        f"{'MRR@10':>8} | {'Recall@50':>9} | {'MAP@100':>8} | {'NDCG@10':>8} | {'Time(ms)':>8}"
    )
    print(header2)
    print("-" * 115)
    print(
        f"{'BM25 Only (Lexical Baseline)':<40} | {'-':>6} | {'-':>6} | "
        f"{bm25_metrics['MRR@10']:>8.4f} | {bm25_metrics['Recall@50']:>9.4f} | "
        f"{bm25_metrics['MAP@100']:>8.4f} | {bm25_metrics['NDCG@10']:>8.4f} | {'-':>8}"
    )
    for r in results_summary:
        m = r["bm25_rerank_metrics"]
        print(
            f"{r['name']:<40} | {r['avg_vectors_per_doc']:>6.1f} | "
            f"{r['compression_ratio']:>5.2f}x | "
            f"{m['MRR@10']:>8.4f} | {m['Recall@50']:>9.4f} | "
            f"{m['MAP@100']:>8.4f} | {m['NDCG@10']:>8.4f} | {r['bm25_rerank_time_ms']:>8.1f}"
        )
    print("=" * 115)

    # Qualitative examples comparing baseline vs w=2 and K=64
    base_name = configs[0]["name"]
    qualitative = {}
    for target_name in [
        "Pool-ColBERT (Window w=2, Mean)",
        "Pool-ColBERT (Fixed K=64, Mean)",
        "Pool-ColBERT (Sliding w=3, s=1, Mean)",
    ]:
        qualitative[target_name] = find_qualitative_examples(
            queries=queries,
            corpus=corpus,
            qrels=qrels,
            baseline_runs=saved_runs[base_name]["full_corpus"],
            pooled_runs=saved_runs[target_name]["full_corpus"],
        )

    output_payload = {
        "dataset": args.dataset,
        "split": args.split,
        "model": args.model,
        "num_documents": len(doc_ids),
        "num_queries": len(queries),
        "bm25_only_metrics": bm25_metrics,
        "experiments": results_summary,
        "qualitative_examples": qualitative,
    }
    os.makedirs(os.path.dirname(args.output) or ".", exist_ok=True)
    with open(args.output, "w", encoding="utf-8") as f:
        json.dump(output_payload, f, indent=2)
    print(f"\nSaved full experiment results to {args.output}")


if __name__ == "__main__":
    main()

