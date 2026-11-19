import argparse
import os


MSMARCO_DIR = "/ml_data/aryan/msmarco"
TRECCAR_DIR = "/ml_data/aryan/trec-car"

DATASETS = {
    "msmarco": {
        "collection": os.path.join(MSMARCO_DIR, "collections.tsv"),
        "splits": {
            "train": {
                "qrels": os.path.join(MSMARCO_DIR, "qrels.train.tsv"),
                "queries": os.path.join(MSMARCO_DIR, "queries.train.tsv"),
            },
            "dev": {
                "qrels": os.path.join(MSMARCO_DIR, "qrels.dev.tsv"),
                "queries": os.path.join(MSMARCO_DIR, "queries.dev.tsv"),
            },
        },
        "default_split": "dev",
        "out_dir": "/ml_data/aryan/msmarco_sample",
    },
    "trec-car": {
        "collection": os.path.join(TRECCAR_DIR, "collections.tsv"),
        "splits": {
            "test": {
                "qrels": os.path.join(TRECCAR_DIR, "qrels.test.tsv"),
                "queries": os.path.join(TRECCAR_DIR, "queries.test.tsv"),
            },
        },
        "default_split": "test",
        "out_dir": "data/trec-car_sample",
    },
}


def sample_dataset(collection_path, splits, output_dir, limit=1000000):
    """Sample a collection and keep each split's qrels and queries in sync."""
    os.makedirs(output_dir, exist_ok=True)

    out_collection = os.path.join(output_dir, "sampled_collection.tsv")

    priority_pids = set()
    for split_paths in splits.values():
        with open(split_paths["qrels"], "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and int(parts[3]) > 0:
                    priority_pids.add(parts[2])

    if limit is not None:
        print(
            f"Sampling up to {limit:,} documents from {collection_path} "
            f"({len(priority_pids):,} qrel-linked documents across splits prioritized)..."
        )
    else:
        print(f"Sampling all documents from {collection_path}...")

    sampled_pids = set()
    remaining_priority_pids = set(priority_pids)

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

    for split, split_paths in splits.items():
        out_qrels = os.path.join(output_dir, f"sampled_qrels.{split}.tsv")
        out_queries = os.path.join(output_dir, f"sampled_queries.{split}.tsv")
        valid_qids = set()
        qrels_retained = 0

        print(f"\nFiltering {split_paths['qrels']} for sampled documents...")
        with open(split_paths["qrels"], "r", encoding="utf-8") as f_in, \
             open(out_qrels, "w", encoding="utf-8") as f_out:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and parts[2] in sampled_pids:
                    valid_qids.add(parts[0])
                    f_out.write(line)
                    qrels_retained += 1

        print(f"Retained {qrels_retained:,} qrels for {len(valid_qids):,} queries.")
        print(f"Saved filtered qrels to {out_qrels}")

        queries_retained = 0
        print(f"Filtering {split_paths['queries']}...")
        with open(split_paths["queries"], "r", encoding="utf-8") as f_in, \
             open(out_queries, "w", encoding="utf-8") as f_out:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t", maxsplit=1)
                if len(parts) == 2 and parts[0] in valid_qids:
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
    parser.add_argument("--limit", type=int, default=1000000, help="Number of documents to sample")
    
    args = parser.parse_args()
    dataset = DATASETS[args.dataset]
    splits = {key: value.copy() for key, value in dataset["splits"].items()}

    if args.qrels:
        splits[dataset["default_split"]]["qrels"] = args.qrels
    if args.queries:
        splits[dataset["default_split"]]["queries"] = args.queries

    sample_dataset(
        collection_path=args.collection or dataset["collection"],
        splits=splits,
        output_dir=args.out_dir or dataset["out_dir"],
        limit=args.limit,
    )
