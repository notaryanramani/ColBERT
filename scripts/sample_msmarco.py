import argparse
import os

def sample_dataset(collection_path, qrels_path, queries_path, output_dir, limit=100000):
    os.makedirs(output_dir, exist_ok=True)
    
    out_collection = os.path.join(output_dir, "sampled_collection.tsv")
    out_qrels = os.path.join(output_dir, "sampled_qrels.dev.tsv")
    out_queries = os.path.join(output_dir, "sampled_queries.dev.tsv")

    print(f"Sampling {limit:,} documents from {collection_path}...")
    sampled_pids = set()
    
    # 1. Sample the documents
    with open(collection_path, "r", encoding="utf-8") as f_in, \
         open(out_collection, "w", encoding="utf-8") as f_out:
        
        for i, line in enumerate(f_in):
            if i >= limit:
                break
            
            parts = line.strip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                pid, _ = parts
                sampled_pids.add(pid)
                f_out.write(line)

    print(f"Saved {len(sampled_pids):,} documents to {out_collection}")

    # 2. Filter the Qrels (only keep rows where pid is in sampled_pids)
    print(f"\nFiltering {qrels_path} to retain only reachable queries...")
    valid_qids = set()
    qrels_retained = 0
    
    with open(qrels_path, "r", encoding="utf-8") as f_in, \
         open(out_qrels, "w", encoding="utf-8") as f_out:
        
        for line in f_in:
            parts = line.strip("\r\n").split("\t")
            if len(parts) >= 4:
                qid, _, pid, rel = parts
                
                # If the relevant document was captured in our sample, keep the query!
                if pid in sampled_pids and int(rel) > 0:
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
    parser = argparse.ArgumentParser(description="Downsample MS MARCO by subsetting docs and filtering queries.")
    parser.add_argument("--collection", default="data/msmarco/collection.tsv", help="Path to original collection TSV")
    parser.add_argument("--qrels", default="data/msmarco/qrels.dev.tsv", help="Path to original dev qrels TSV")
    parser.add_argument("--queries", default="data/msmarco/queries.dev.tsv", help="Path to original dev queries TSV")
    parser.add_argument("--out_dir", default="data/msmarco_sample", help="Output directory for the new sample")
    parser.add_argument("--limit", type=int, default=100000, help="Number of documents to sample")
    
    args = parser.parse_args()
    
    sample_dataset(
        collection_path=args.collection,
        qrels_path=args.qrels,
        queries_path=args.queries,
        output_dir=args.out_dir,
        limit=args.limit
    )
