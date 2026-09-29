import os
import numpy as np
import faiss
from scripts import config


def main():
    emb_path = os.path.join(config.INDEX_DIR, "doc_embeddings.fp16.npy")
    print(f"[faiss] loading {emb_path}")
    vecs = np.load(emb_path).astype(np.float32)
    d    = vecs.shape[1]
    print(f"[faiss] {vecs.shape[0]:,} vectors of dim {d}")

    quantizer = faiss.IndexFlatL2(d)
    index     = faiss.IndexIVFPQ(quantizer, d,
                                 config.FAISS_NLIST,
                                 config.FAISS_M,
                                 config.FAISS_NBITS)
    print("[faiss] training ...")
    index.train(vecs)
    print("[faiss] adding ...")
    index.add(vecs)
    index.nprobe = config.FAISS_NPROBE

    out = os.path.join(config.INDEX_DIR, "colbert_ivfpq.faiss")
    faiss.write_index(index, out)
    print(f"[faiss] wrote {out}")


if __name__ == "__main__":
    main()
