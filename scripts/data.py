"""
Downloads the two datasets used in the ColBERT paper:
  1. MS MARCO Passage Ranking
  2. TREC CAR

Saving them in TSV format compatible with ColBERT.
Includes subsetting logic to ensure ground-truth passages are always kept 
when reducing the dataset size to fit on Colab/Kaggle quotas.

Usage:
  python -m src.data msmarco --limit 100000
  python -m src.data trec-car --limit 100000
  python -m src.data all --limit 100000
"""

import argparse
import os

import ir_datasets
from tqdm import tqdm

BUFFER_SIZE = 10_000

#  MS MARCO Passage Ranking
MSMARCO_DIR = os.path.join("data", "msmarco")
MSMARCO_COLLECTION = "msmarco-passage"
MSMARCO_SPLITS = {
    "dev":   "msmarco-passage/dev/small",
    "train": "msmarco-passage/train",
}


def download_msmarco(data_dir=MSMARCO_DIR, include_train=False, limit=None):
    """Download MS MARCO Passage Ranking dataset."""
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 60)
    print("  MS MARCO Passage Ranking")
    print("=" * 60)

    # 1. Save Queries and Qrels first
    _save_queries(MSMARCO_SPLITS["dev"], os.path.join(data_dir, "queries.dev.tsv"))
    dev_qrels_path = os.path.join(data_dir, "qrels.dev.tsv")
    _save_qrels(MSMARCO_SPLITS["dev"], dev_qrels_path)

    if include_train:
        _save_queries(MSMARCO_SPLITS["train"], os.path.join(data_dir, "queries.train.tsv"))
        _save_qrels(MSMARCO_SPLITS["train"], os.path.join(data_dir, "qrels.train.tsv"))

    # 2. Extract required PIDs so they aren't lost if we limit the collection
    required_pids = set()
    if limit is not None:
        required_pids = _extract_qrel_pids(dev_qrels_path)
        print(f"Subset requested: Ensuring {len(required_pids):,} relevant passages are kept.")

    # 3. Save Collection
    _save_docs(MSMARCO_COLLECTION, os.path.join(data_dir, "collection.tsv"), required_pids, limit)

    _print_summary(data_dir)



#  TREC CAR (Complex Answer Retrieval)
TRECCAR_DIR = os.path.join("data", "trec-car")
TRECCAR_COLLECTION = "car/v1.5"
TRECCAR_TEST = "car/v1.5/trec-y1/auto"


def download_trec_car(data_dir=TRECCAR_DIR, limit=None):
    """Download TREC CAR dataset (Year 1, automatic judgments)."""
    os.makedirs(data_dir, exist_ok=True)
    print("=" * 60)
    print("  TREC CAR (Complex Answer Retrieval)")
    print("=" * 60)

    # 1. Save Queries and Qrels first
    _save_queries(TRECCAR_TEST, os.path.join(data_dir, "queries.test.tsv"))
    test_qrels_path = os.path.join(data_dir, "qrels.test.tsv")
    _save_qrels(TRECCAR_TEST, test_qrels_path)

    # 2. Extract required PIDs
    required_pids = set()
    if limit is not None:
        required_pids = _extract_qrel_pids(test_qrels_path)
        print(f"Subset requested: Ensuring {len(required_pids):,} relevant passages are kept.")

    # 3. Save Collection
    _save_docs(TRECCAR_COLLECTION, os.path.join(data_dir, "collection.tsv"), required_pids, limit)

    _print_summary(data_dir)


#  Shared helpers
def _extract_qrel_pids(qrels_filepath):
    """Read a qrels file and return a set of all doc_ids present."""
    pids = set()
    if os.path.exists(qrels_filepath):
        with open(qrels_filepath, "r", encoding="utf-8") as f:
            for line in f:
                parts = line.strip().split("\t")
                if len(parts) >= 3:
                    pids.add(parts[2])  # qid \t 0 \t pid \t rel
    return pids


def _save_docs(dataset_id, filepath, required_pids=None, limit=None):
    """Save documents/passages. Optionally limit size while ensuring required_pids are kept."""
    dataset = ir_datasets.load(dataset_id)
    total = dataset.docs_count()
    
    if required_pids is None:
        required_pids = set()
    
    remaining_pids = set(required_pids)

    # If limit is smaller than required_pids, we must at least save the required ones.
    if limit is not None:
        target_size = max(limit, len(required_pids))
        print(f"\nSaving subset of passages (target: {target_size:,}) -> {filepath}")
    else:
        print(f"\nSaving all {total:,} passages -> {filepath}")

    saved_count = 0
    with open(filepath, "w", encoding="utf-8") as f:
        buffer = []
        
        for doc in tqdm(dataset.docs_iter(), total=total, desc="Passages"):
            is_required = doc.doc_id in remaining_pids
            
            if is_required:
                remaining_pids.remove(doc.doc_id)
            elif limit is not None and (saved_count + len(remaining_pids)) >= limit:
                if not remaining_pids:
                    break  # All required PIDs found and limit reached!
                continue
            
            clean_text = doc.text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            buffer.append(f"{doc.doc_id}\t{clean_text}\n")
            saved_count += 1
            
            if len(buffer) >= BUFFER_SIZE:
                f.writelines(buffer)
                buffer.clear()
                
        if buffer:
            f.writelines(buffer)
            
    print(f"Total passages saved: {saved_count:,}")


def _save_queries(dataset_id, filepath):
    """Save queries as qid<TAB>text TSV."""
    dataset = ir_datasets.load(dataset_id)
    total = dataset.queries_count()
    print(f"Saving {total:,} queries -> {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        for query in tqdm(dataset.queries_iter(), total=total, desc="Queries"):
            text = query.text.replace("\t", " ").replace("\n", " ").replace("\r", " ")
            f.write(f"{query.query_id}\t{text}\n")


def _save_qrels(dataset_id, filepath):
    """Save relevance judgments as qid<TAB>0<TAB>pid<TAB>rel (TREC format)."""
    dataset = ir_datasets.load(dataset_id)
    total = dataset.qrels_count()
    print(f"Saving {total:,} qrels -> {filepath}")

    with open(filepath, "w", encoding="utf-8") as f:
        buffer = []
        for qrel in tqdm(dataset.qrels_iter(), total=total, desc="Qrels"):
            buffer.append(f"{qrel.query_id}\t0\t{qrel.doc_id}\t{qrel.relevance}\n")
            if len(buffer) >= BUFFER_SIZE:
                f.writelines(buffer)
                buffer.clear()
        if buffer:
            f.writelines(buffer)


def _print_summary(data_dir):
    """Print saved file sizes."""
    print(f"\nFiles in {data_dir}/:")
    for f in sorted(os.listdir(data_dir)):
        size_mb = os.path.getsize(os.path.join(data_dir, f)) / (1024 * 1024)
        print(f"  {f:30s} {size_mb:>10.1f} MB")
    print()


# ======================================================================
#  Loading utilities (for later pipeline steps)
# ======================================================================

def load_collection(data_dir):
    """Load passages as {pid: text} dict from collection.tsv."""
    filepath = os.path.join(data_dir, "collection.tsv")
    collection = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in tqdm(f, desc="Loading collection"):
            pid, text = line.strip("\r\n").split("\t", maxsplit=1)
            collection[pid] = text
    print(f"Loaded {len(collection):,} passages")
    return collection


def load_queries(filepath):
    """Load queries as {qid: text} dict from a queries TSV."""
    queries = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            qid, text = line.strip("\r\n").split("\t", maxsplit=1)
            queries[qid] = text
    print(f"Loaded {len(queries):,} queries")
    return queries


def load_qrels(filepath):
    """Load relevance judgments as {qid: {pid: relevance}}."""
    qrels = {}
    with open(filepath, "r", encoding="utf-8") as f:
        for line in f:
            parts = line.strip("\r\n").split("\t")
            qid, pid, rel = parts[0], parts[2], int(parts[3])
            qrels.setdefault(qid, {})[pid] = rel
    print(f"Loaded qrels for {len(qrels):,} queries")
    return qrels


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Download ColBERT datasets")
    parser.add_argument("dataset", choices=["msmarco", "trec-car", "all"], default="msmarco", nargs="?")
    parser.add_argument("--limit", type=int, default=None, help="Limit number of passages to save (avoids Colab disk crash).")
    args = parser.parse_args()

    if args.dataset in ("msmarco", "all"):
        download_msmarco(limit=args.limit)
    if args.dataset in ("trec-car", "all"):
        download_trec_car(limit=args.limit)
