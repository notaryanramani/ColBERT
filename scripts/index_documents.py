"""
Produces:
  indexes/doc_embeddings.fp16.npy    flat [total_tokens, 128]
  indexes/doc_offsets.npy            int64 [num_docs + 1]
  indexes/doc_ids.json               list[str] of length num_docs
"""

import os
import json
import time
import numpy as np
import torch
torch.set_num_threads(os.cpu_count())

from scripts.data import load_collection
from scripts import config
from scripts.colbert_model import load_model, build_tokenizer
from scripts.utils import (length_bucket, tokenize_parallel,
                           prepend_specials, pad_batch, punct_ids)

def _save_index(vecs, offsets, kept_doc_ids):
    os.makedirs(config.INDEX_DIR, exist_ok=True)
    out_dtype = np.float16 if config.STORE_DTYPE == "float16" else np.float32
    arr = np.vstack(vecs).astype(out_dtype)
    np.save(os.path.join(config.INDEX_DIR, "doc_embeddings.fp16.npy"), arr)
    np.save(os.path.join(config.INDEX_DIR, "doc_offsets.npy"),
            np.asarray(offsets, dtype=np.int64))
    with open(os.path.join(config.INDEX_DIR, "doc_ids.json"), "w") as f:
        json.dump(kept_doc_ids, f)
    return arr.shape


@torch.no_grad()
def _encode_ids(model, ids, mask, punct_tensor, device):
    ids, mask = ids.to(device), mask.to(device)
    emb = model.encode(ids, mask)
    keep = ~torch.isin(ids, punct_tensor.to(ids.device))
    keep[:, :2] = True
    out = []
    for b in range(ids.size(0)):
        out.append(emb[b][keep[b]].cpu())
    return out


def main():
    device = "cuda" if torch.cuda.is_available() else "cpu"
    print(f"[index] device={device}")
    print(f"[index] STORE_DTYPE={config.STORE_DTYPE}, EMB_DIM={config.EMB_DIM}")

    print("[1/4] Loading collection ...")
    collection = load_collection(config.DATA_DIR)
    doc_ids    = list(collection.keys())
    print(f"      {len(doc_ids):,} documents")

    print("[2/4] Building tokenizer + model ...")
    tokenizer = build_tokenizer()
    pset      = torch.tensor(sorted(punct_ids(tokenizer)), dtype=torch.long)
    pad_id    = tokenizer.pad_token_id
    model     = load_model(config.CKPT_PATH, device=device)

    if torch.cuda.device_count() > 1:
        print(f"      DataParallel across {torch.cuda.device_count()} GPUs")
        model = torch.nn.DataParallel(model)

    print("[3/4] Encoding documents (length-bucketed, parallel tokenization) ...")
    all_vecs      = []
    doc_offsets   = [0]
    kept_doc_ids  = []
    n_done        = 0
    t0            = time.time()
    last_print    = t0

    for bucket in length_bucket(doc_ids, collection):
        texts = [collection[d] for d in bucket]

        rows = tokenize_parallel(tokenizer, texts)
        rows = prepend_specials(tokenizer, rows, is_query=False)

        for i in range(0, len(rows), config.BATCH_SIZE):
            batch_rows = rows[i:i + config.BATCH_SIZE]
            ids, mask  = pad_batch(batch_rows, pad_id)
            embs       = _encode_ids(model, ids, mask, pset, device)

            batch_ids = bucket[i:i + config.BATCH_SIZE]
            for did, emb in zip(batch_ids, embs):
                all_vecs.append(emb.numpy())
                doc_offsets.append(doc_offsets[-1] + emb.shape[0])
                kept_doc_ids.append(did)

            n_done += len(batch_rows)
            now = time.time()
            if now - last_print > 15:
                rate = n_done / max(1e-6, now - t0) * 60
                print(f"      {n_done:>9,} docs | {rate:>8,.0f} docs/min",
                      flush=True)
                last_print = now

    print("[4/4] Saving ...")
    shape = _save_index(all_vecs, doc_offsets, kept_doc_ids)
    print(f"      embeddings: {shape}")
    print(f"      docs      : {len(kept_doc_ids):,}")
    print(f"      elapsed   : {(time.time()-t0)/60:.1f} min")


if __name__ == "__main__":
    main()
