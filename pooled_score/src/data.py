import os


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


def read_queries(path):
    queries = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                queries[parts[0]] = parts[1]
    return queries


def read_qrels(path):
    qrels = {}
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t")
            if len(parts) >= 4 and int(parts[3]) > 0:
                qrels.setdefault(parts[0], set()).add(parts[2])
    return qrels


def iter_documents(path):
    with open(path, encoding="utf-8") as f:
        for line in f:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) == 2:
                yield parts[0], parts[1]


def ensure_bm25_index(collection_path, index_dir, batch_size=1000, threads=4):
    if os.path.isdir(index_dir) and any(
        name.startswith("segments_") for name in os.listdir(index_dir)
    ):
        return

    from pyserini.index.lucene import LuceneIndexer

    os.makedirs(index_dir, exist_ok=True)
    indexer = LuceneIndexer(index_dir, threads=threads)
    batch = []
    for doc_id, text in iter_documents(collection_path):
        batch.append({"id": doc_id, "contents": text})
        if len(batch) >= batch_size:
            indexer.add_batch_dict(batch)
            batch = []
    if batch:
        indexer.add_batch_dict(batch)
    indexer.close()


