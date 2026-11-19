"""Encode documents and build GPU FAISS indexes for both ColBERT models."""

import argparse
import json
import os
os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import faiss
import numpy as np
import torch
from tqdm import tqdm
from transformers import AutoTokenizer

from weighted_score.src.base import ColBERT
from weighted_score.src.data import COLLECTION_PATH, MAX_DOCUMENT_LENGTH, MODEL_NAME
from weighted_score.src.weighted import WeightedColBERT

TRECCAR_COLLECTION_PATH = "/ml_data/aryan/trec-car_sample/sampled_collection.tsv"


def read_documents(path, limit=None):
    with open(path, "r", encoding="utf-8") as f_in:
        count = 0
        for line in f_in:
            parts = line.rstrip("\r\n").split("\t", maxsplit=1)
            if len(parts) != 2:
                continue

            yield parts[0], parts[1]
            count += 1
            if limit is not None and count >= limit:
                break


def build_index(name, model_class, checkpoint, args, tokenizer, device):
    print(f"\nBuilding {name} index from {checkpoint}")
    model = model_class(tokenizer=tokenizer).to(device)
    state = torch.load(checkpoint, map_location=device)
    model.load_state_dict(state)
    model.eval()

    output_dir = os.path.join(args.output, args.dataset, name)
    os.makedirs(output_dir, exist_ok=True)
    all_vectors = []
    doc_ids = []
    offsets = [0]
    batch = []

    def encode_batch(rows):
        doc_batch_ids = [doc_id for doc_id, _ in rows]
        texts = [f"[D] {text}" for _, text in rows]
        tokens = tokenizer(
            texts,
            padding=True,
            truncation=True,
            max_length=MAX_DOCUMENT_LENGTH,
            return_tensors="pt",
        )
        input_ids = tokens["input_ids"].to(device)
        attention_mask = tokens["attention_mask"].to(device)

        with torch.no_grad():
            embeddings = model.encode(input_ids, attention_mask)

        for i, doc_id in enumerate(doc_batch_ids):
            # Drop padding only; each document keeps its own token-vector bag.
            bag = embeddings[i][attention_mask[i].bool()].cpu().numpy().astype(np.float32)
            all_vectors.append(bag)
            doc_ids.append(doc_id)
            offsets.append(offsets[-1] + len(bag))

    for row in tqdm(read_documents(args.collection, args.limit), desc=f"Encoding {name} documents"):
        batch.append(row)
        if len(batch) == args.batch_size:
            encode_batch(batch)
            batch = []

    if batch:
        encode_batch(batch)

    if not all_vectors:
        raise RuntimeError(f"No documents were read from {args.collection}")

    vectors = np.concatenate(all_vectors, axis=0)
    np.save(os.path.join(output_dir, "document_vectors.npy"), vectors.astype(np.float16))
    np.save(os.path.join(output_dir, "document_offsets.npy"), np.asarray(offsets, dtype=np.int64))
    with open(os.path.join(output_dir, "document_ids.json"), "w", encoding="utf-8") as f_out:
        json.dump(doc_ids, f_out)

    # FAISS row i matches vector i. Offsets map it to the document bag,
    # and document_ids stores the corresponding external document ID.
    cpu_index = faiss.IndexFlatIP(vectors.shape[1])
    resources = faiss.StandardGpuResources()
    gpu_index = faiss.index_cpu_to_gpu(resources, 0, cpu_index)
    gpu_index.add(vectors)
    faiss.write_index(
        faiss.index_gpu_to_cpu(gpu_index),
        os.path.join(output_dir, "faiss.index"),
    )

    print(f"Indexed {len(doc_ids):,} documents and {len(vectors):,} token vectors")
    print(f"Saved index files to {output_dir}")

    del model, state, vectors, gpu_index, cpu_index, resources, all_vectors
    torch.cuda.empty_cache()


def main():
    parser = argparse.ArgumentParser(description="Build ColBERT indexes for MS MARCO or TREC CAR")
    parser.add_argument("--dataset", choices=["msmarco", "trec-car"], default="msmarco")
    parser.add_argument("--collection", default=None)
    parser.add_argument("--base-checkpoint", default="/ml_data/aryan/colbert/base_colbert.pt")
    parser.add_argument("--weighted-checkpoint", default="/ml_data/aryan/colbert/weighted_colbert.pt")
    parser.add_argument("--output", default="/ml_data/aryan/colbert/indexes")
    parser.add_argument("--batch-size", type=int, default=128)
    parser.add_argument("--limit", type=int, default=None)
    args = parser.parse_args()

    if args.collection is None:
        if args.dataset == "trec-car":
            args.collection = TRECCAR_COLLECTION_PATH
        else:
            args.collection = COLLECTION_PATH

    device = torch.device("cuda:0")
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    tokenizer.add_special_tokens({"additional_special_tokens": ["[Q]", "[D]"]})

    # build_index("base", ColBERT, args.base_checkpoint, args, tokenizer, device)
    build_index("weighted", WeightedColBERT, args.weighted_checkpoint, args, tokenizer, device)


if __name__ == "__main__":
    main()
