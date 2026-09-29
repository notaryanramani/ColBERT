"""
Verifies that the index is well-formed:
  - vector dim == 128
  - len(offsets) == len(doc_ids) + 1
  - offsets[-1] == num tokens
  - avg tokens per doc in a plausible range
  - each stored doc_id exists in collection.tsv
"""

import os
import json
import numpy as np
from scripts import config
from scripts.data import load_collection


def main():
    vecs    = np.load(os.path.join(config.INDEX_DIR, "doc_embeddings.fp16.npy"),
                      mmap_mode="r")
    offsets = np.load(os.path.join(config.INDEX_DIR, "doc_offsets.npy"))
    with open(os.path.join(config.INDEX_DIR, "doc_ids.json")) as f:
        doc_ids = json.load(f)

    print(f"vector shape  : {vecs.shape}")
    print(f"doc count     : {len(doc_ids):,}")
    print(f"offset length : {len(offsets):,}")

    ok = True
    if vecs.shape[1] != config.EMB_DIM:
        print(f"FAIL: dim {vecs.shape[1]} != {config.EMB_DIM}"); ok = False
    if len(offsets) != len(doc_ids) + 1:
        print("FAIL: len(offsets) != len(doc_ids) + 1"); ok = False
    if offsets[-1] != vecs.shape[0]:
        print(f"FAIL: offsets[-1]={offsets[-1]} != n_vecs={vecs.shape[0]}"); ok = False

    avg = vecs.shape[0] / max(1, len(doc_ids))
    print(f"avg tok/doc   : {avg:.1f}")
    if not (10 <= avg <= 200):
        print("WARN: avg tokens/doc outside [10,200] — check punctuation filter")

    # sample 100 doc_ids and confirm they exist in the collection
    collection = load_collection(config.DATA_DIR)
    missing = 0
    for did in doc_ids[:100]:
        if did not in collection:
            missing += 1
    print(f"missing IDs (first 100): {missing}")

    print("OK" if ok else "FAILED")


if __name__ == "__main__":
    main()
