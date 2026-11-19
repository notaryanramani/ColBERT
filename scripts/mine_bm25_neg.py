import os
os.environ.setdefault("OPENAI_API_KEY", "dummy")
from tqdm import tqdm

DATA_DIR = "/ml_data/aryan/msmarco_sample"
SPLIT = "train"
COLLECTION_PATH = os.path.join(DATA_DIR, "sampled_collection.tsv")
QUERIES_PATH = os.path.join(DATA_DIR, f"sampled_queries.{SPLIT}.tsv")
QRELS_PATH = os.path.join(DATA_DIR, f"sampled_qrels.{SPLIT}.tsv")

os.makedirs(os.path.join(DATA_DIR, "index"), exist_ok=True)
INDEX_DIR = os.path.join(DATA_DIR, "index/bm25_index")
OUTPUT_PATH = os.path.join(DATA_DIR, "bm25_negatives.tsv")
NUM_NEGATIVES = 100
INDEX_BATCH_SIZE = 1000
INDEX_THREADS = 4


def load_positive_docs(qrels_path):
    positive_docs = {}

    with open(qrels_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) < 4 or int(parts[3]) <= 0:
                continue

            qid, docid = parts[0], parts[2]
            positive_docs.setdefault(qid, set()).add(docid)

    return positive_docs


def build_index(collection_path, index_dir, batch_size, threads):
    from pyserini.index.lucene import LuceneIndexer

    indexer = LuceneIndexer(index_dir, threads=threads)
    batch = []

    with open(collection_path, "r", encoding="utf-8") as f_in:
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            docid, text = parts
            batch.append({"id": docid, "contents": text})

            if len(batch) >= batch_size:
                indexer.add_batch_dict(batch)
                batch = []

    if batch:
        indexer.add_batch_dict(batch)

    indexer.close()


def mine_negatives(queries_path, qrels_path, index_dir, output_path, num_negatives):
    from pyserini.search.lucene import LuceneSearcher

    positive_docs = load_positive_docs(qrels_path)
    searcher = LuceneSearcher(index_dir)

    with open(queries_path, "r", encoding="utf-8") as f_in:
        total_queries = sum(1 for _ in f_in)

    with open(queries_path, "r", encoding="utf-8") as f_in, \
         open(output_path, "w", encoding="utf-8") as f_out:
        for line in tqdm(f_in, total=total_queries, desc="Mining BM25 negatives", unit="query"):
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            qid, query = parts
            positives = positive_docs.get(qid, set())
            k = num_negatives + len(positives) + 10 # same buffer
            hits = searcher.search(query, k=k)
            written = 0

            for hit in hits:
                if hit.docid in positives:
                    continue

                f_out.write(f"{qid}\t{hit.docid}\n")
                written += 1

                if written == num_negatives:
                    break


def main():
    for path in (COLLECTION_PATH, QUERIES_PATH, QRELS_PATH):
        if not os.path.isfile(path):
            raise FileNotFoundError(path)

    if not os.path.isdir(INDEX_DIR):
        build_index(
            COLLECTION_PATH,
            INDEX_DIR,
            batch_size=INDEX_BATCH_SIZE,
            threads=INDEX_THREADS,
        )

    mine_negatives(
        QUERIES_PATH,
        QRELS_PATH,
        INDEX_DIR,
        OUTPUT_PATH,
        num_negatives=NUM_NEGATIVES,
    )


if __name__ == "__main__":
    main()
