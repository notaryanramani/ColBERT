import argparse
import os


MSMARCO_DIR = "/ml_data/aryan/msmarco"
TRECCAR_DIR = "/ml_data/aryan/trec-car"

DATASETS = {
    "msmarco": {
        "collection": os.path.join(MSMARCO_DIR, "collection.tsv"),
        "qrels": os.path.join(MSMARCO_DIR, "qrels.dev.tsv"),
        "queries": os.path.join(MSMARCO_DIR, "queries.dev.tsv"),
        "split": "dev",
        "out_dir": "data/msmarco_sample",
    },
    "trec-car": {
        "collection": os.path.join(TRECCAR_DIR, "collection.tsv"),
        "qrels": os.path.join(TRECCAR_DIR, "qrels.test.tsv"),
        "queries": os.path.join(TRECCAR_DIR, "queries.test.tsv"),
        "split": "test",
        "out_dir": "data/trec-car_sample",
    },
}


def sample_dataset(collection_path, qrels_path, queries_path, output_dir, split="dev", limit=100000):
    """Subset a collection and keep the qrels and queries that still reference it."""
    os.makedirs(output_dir, exist_ok=True)
    
    out_collection = os.path.join(output_dir, "sampled_collection.tsv")
    out_qrels = os.path.join(output_dir, f"sampled_qrels.{split}.tsv")
    out_queries = os.path.join(output_dir, f"sampled_queries.{split}.tsv")

    # Prioritize documents with positive judgments.  Taking the first N rows of
    # MS MARCO's collection often yields no judged documents at all, which in
    # turn leaves no queries to retain.
    priority_pids = set()
    with open(qrels_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                priority_pids.add(parts[2])
                if limit is not None and len(priority_pids) >= limit:
                    break

    if limit is not None:
        print(
            f"Sampling up to {limit:,} documents from {collection_path} "
            f"({len(priority_pids):,} qrel-linked documents prioritized)..."
        )
    else:
        print(f"Sampling all documents from {collection_path}...")

    sampled_pids = set()
    remaining_priority_pids = set(priority_pids)
    
    # 1. Sample the documents
    with open(collection_path, "r", encoding="utf-8") as f_in, \
        open(out_collection, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            parts = line.strip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                pid, _ = parts
                is_priority = pid in remaining_priority_pids

                if is_priority:
                    remaining_priority_pids.remove(pid)
                elif limit is not None and (
                    len(sampled_pids) + len(remaining_priority_pids)
                ) >= limit:
                    continue

                sampled_pids.add(pid)
                f_out.write(line)

                if limit is not None and len(sampled_pids) >= limit and not remaining_priority_pids:
                    break

    print(f"Saved {len(sampled_pids):,} documents to {out_collection}")
    if remaining_priority_pids:
        print(
            f"Warning: {len(remaining_priority_pids):,} qrel-linked document IDs "
            "were absent from the collection."
        )

    # 2. Keep every judgment for a sampled document.  The query IDs collected
    # here drive the query subset in step 3, so each retained qrel has its
    # matching query text available in the sampled dataset.
    print(f"\nFiltering {qrels_path} to retain queries for sampled documents...")
    valid_qids = set()
    qrels_retained = 0
    
    with open(qrels_path, "r", encoding="utf-8") as f_in, \
         open(out_qrels, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4:
                qid, _, pid, _ = parts[:4]
                
                if pid in sampled_pids:
                    valid_qids.add(qid)
                    f_out.write(line)
                    qrels_retained += 1

    print(f"Retained {qrels_retained:,} qrels covering {len(valid_qids):,} unique queries.")
    print(f"Saved filtered qrels to {out_qrels}")

    # 3. Filter the Queries
    print(f"\nFiltering {queries_path}...")
    queries_retained = 0
    
    with open(queries_path, "r", encoding="utf-8") as f_in, \
         open(out_queries, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            parts = line.strip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                qid, _ = parts
                
                if qid in valid_qids:
                    f_out.write(line)
                    queries_retained += 1

    print(f"Retained {queries_retained:,} queries.")
    print(f"Saved filtered queries to {out_queries}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Downsample a retrieval dataset while preserving linked queries.")
    parser.add_argument("--dataset", choices=DATASETS, default="msmarco", help="Dataset whose mounted paths to use")
    parser.add_argument("--collection", help="Path to original collection TSV (overrides --dataset)")
    parser.add_argument("--qrels", help="Path to original qrels TSV (overrides --dataset)")
    parser.add_argument("--queries", help="Path to original queries TSV (overrides --dataset)")
    parser.add_argument("--out_dir", help="Output directory for the new sample (overrides --dataset)")
    parser.add_argument("--limit", type=int, default=100000, help="Number of documents to sample")
    
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    
    sample_dataset(
        collection_path=args.collection or dataset["collection"],
        qrels_path=args.qrels or dataset["qrels"],
        queries_path=args.queries or dataset["queries"],
        output_dir=args.out_dir or dataset["out_dir"],
        split=dataset["split"],
        limit=args.limit
    )
