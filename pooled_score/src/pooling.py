import json
import os

import numpy as np
import torch
import torch.nn.functional as F
from tqdm import tqdm

from .data import iter_documents


CONFIGS = [
    {"name": "fixed_k8", "kind": "fixed", "k": 8},
    {"name": "fixed_k16", "kind": "fixed", "k": 16},
    {"name": "window_w8_s2", "kind": "window", "window": 8, "stride": 2},
    {"name": "window_w16_s4", "kind": "window", "window": 16, "stride": 4},
]
DIM = 128


def pool_document(embedding, config):
    n_tokens = embedding.size(0)
    if n_tokens == 0:
        return embedding

    if config["kind"] == "fixed":
        count = min(config["k"], n_tokens)
        chunks = torch.tensor_split(embedding, count, dim=0)
    else:
        window = config["window"]
        stride = config["stride"]
        chunks = [
            embedding[start : start + window]
            for start in range(0, n_tokens, stride)
        ]

    pooled = torch.stack([chunk.mean(dim=0) for chunk in chunks])
    return F.normalize(pooled.float(), p=2, dim=-1)


def index_is_ready(index_dir, config_names, model_name, doc_maxlen):
    manifest_path = os.path.join(index_dir, "manifest.json")
    if not os.path.isfile(manifest_path):
        return False
    with open(manifest_path, encoding="utf-8") as f:
        manifest = json.load(f)
    if manifest.get("model") != model_name or manifest.get("doc_maxlen") != doc_maxlen:
        return False
    if not os.path.isfile(os.path.join(index_dir, "document_ids.json")):
        return False
    return all(
        os.path.isfile(os.path.join(index_dir, name, "vectors.f16"))
        and os.path.isfile(os.path.join(index_dir, name, "offsets.npy"))
        for name in config_names
    )


def build_pooled_indexes(
    checkpoint,
    collection_path,
    index_dir,
    model_name,
    doc_maxlen,
    batch_size,
):
    config_names = [config["name"] for config in CONFIGS]
    if index_is_ready(index_dir, config_names, model_name, doc_maxlen):
        return

    os.makedirs(index_dir, exist_ok=True)
    output_dirs = {}
    vector_files = {}
    offsets = {name: [0] for name in config_names}
    for config in CONFIGS:
        name = config["name"]
        output_dirs[name] = os.path.join(index_dir, name)
        os.makedirs(output_dirs[name], exist_ok=True)
        vector_files[name] = open(
            os.path.join(output_dirs[name], "vectors.f16.tmp"), "wb"
        )

    document_ids = []
    batch = []

    def encode_batch(rows):
        texts = [text for _, text in rows]
        with torch.inference_mode():
            encoded, lengths = checkpoint.docFromText(
                texts,
                bsize=batch_size,
                keep_dims="flatten",
                to_cpu=False,
            )[:2]

        offset = 0
        for (doc_id, _), length in zip(rows, lengths):
            document_ids.append(doc_id)
            doc_embedding = encoded[offset : offset + length]
            offset += length
            for config in CONFIGS:
                name = config["name"]
                pooled = pool_document(doc_embedding, config).detach().cpu().half().numpy()
                pooled.tofile(vector_files[name])
                offsets[name].append(offsets[name][-1] + len(pooled))

    try:
        for row in tqdm(iter_documents(collection_path), desc="Encoding and pooling documents"):
            batch.append(row)
            if len(batch) == batch_size:
                encode_batch(batch)
                batch = []
        if batch:
            encode_batch(batch)
    finally:
        for f in vector_files.values():
            f.close()

    ids_tmp = os.path.join(index_dir, "document_ids.json.tmp")
    with open(ids_tmp, "w", encoding="utf-8") as f:
        json.dump(document_ids, f)
    os.replace(ids_tmp, os.path.join(index_dir, "document_ids.json"))

    for name in config_names:
        output_dir = output_dirs[name]
        offsets_tmp = os.path.join(output_dir, "offsets.npy.tmp")
        with open(offsets_tmp, "wb") as f:
            np.save(f, np.asarray(offsets[name], dtype=np.int64))
        os.replace(offsets_tmp, os.path.join(output_dir, "offsets.npy"))
        os.replace(
            os.path.join(output_dir, "vectors.f16.tmp"),
            os.path.join(output_dir, "vectors.f16"),
        )

    manifest = {
        "model": model_name,
        "doc_maxlen": doc_maxlen,
        "documents": len(document_ids),
        "configs": CONFIGS,
    }
    manifest_tmp = os.path.join(index_dir, "manifest.json.tmp")
    with open(manifest_tmp, "w", encoding="utf-8") as f:
        json.dump(manifest, f, indent=2)
    os.replace(manifest_tmp, os.path.join(index_dir, "manifest.json"))


def load_pooled_index(index_dir, config):
    import json

    with open(os.path.join(index_dir, "document_ids.json"), encoding="utf-8") as f:
        doc_ids = json.load(f)
    config_dir = os.path.join(index_dir, config["name"])
    offsets = np.load(os.path.join(config_dir, "offsets.npy"), mmap_mode="r")
    vectors = np.memmap(
        os.path.join(config_dir, "vectors.f16"),
        mode="r",
        dtype=np.float16,
        shape=(int(offsets[-1]), DIM),
    )
    return doc_ids, offsets, vectors
