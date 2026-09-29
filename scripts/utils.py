"""Indexing throughput helpers:
  1. length-based bucketing
  2. per-batch max length padding
  3. multi-core CPU tokenization (serial fallback for small inputs)
  4. (multi-GPU is handled in index_documents.py)
"""

from multiprocessing import Pool
from scripts import config

def length_bucket(doc_ids, collection, bucket_size=config.BUCKET_SIZE):
    """Yield length-sorted buckets of doc_ids."""
    for i in range(0, len(doc_ids), bucket_size):
        bucket = doc_ids[i:i + bucket_size]
        bucket.sort(key=lambda did: len(collection[did]))
        yield bucket


def _tok_worker(args):
    tokenizer, texts = args
    return tokenizer(texts, padding=False, truncation=True,
                    max_length=config.DOC_MAXLEN,
                    add_special_tokens=False)["input_ids"]


def tokenize_parallel(tokenizer, texts, workers=config.NUM_CPU_WORKERS):
    """Serialize small inputs; parallelize only if len(texts) >= 10_000."""
    if workers <= 1 or len(texts) < 10_000:
        return _tok_worker((tokenizer, texts))
    chunks = [texts[i::workers] for i in range(workers)]
    with Pool(workers) as p:
        results = p.map(_tok_worker, [(tokenizer, c) for c in chunks])
    merged = [None] * len(texts)
    for w, chunk in enumerate(results):
        for j, item in enumerate(chunk):
            merged[w + j * workers] = item
    return merged


def prepend_specials(tokenizer, rows, is_query=False):
    """Prepend [CLS] [Q]|[D] to each tokenized row."""
    cls_id = tokenizer.cls_token_id
    sp_id  = tokenizer.convert_tokens_to_ids(
        config.QUERY_TOKEN if is_query else config.DOC_TOKEN)
    return [[cls_id, sp_id] + r for r in rows]


def pad_batch(rows, pad_id):
    """Pad to max length within this batch."""
    import torch
    maxlen = max(len(r) for r in rows)
    ids  = torch.full((len(rows), maxlen), pad_id, dtype=torch.long)
    mask = torch.zeros((len(rows), maxlen), dtype=torch.long)
    for i, r in enumerate(rows):
        ids[i, :len(r)]  = torch.tensor(r, dtype=torch.long)
        mask[i, :len(r)] = 1
    return ids, mask

def punct_ids(tokenizer):
    return {tokenizer.convert_tokens_to_ids(t) for t in config.PUNCT_TOKENS}