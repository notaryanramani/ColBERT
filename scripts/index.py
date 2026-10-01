import argparse
from pathlib import Path

import torch
from colbert import Indexer
from colbert.infra import ColBERTConfig, Run, RunConfig


def parse_args():
    parser = argparse.ArgumentParser(description="Build a CPU ColBERT index for MS MARCO")
    parser.add_argument("--collection", default="data/msmarco/collection.tsv")
    parser.add_argument("--checkpoint", default="jinaai/jina-colbert-v1-en")
    parser.add_argument("--root", default="experiments")
    parser.add_argument("--experiment", default="msmarco")
    parser.add_argument("--name", default="jina-colbert-v1-en.nbits=2")
    parser.add_argument("--doc-maxlen", type=int, default=160)
    parser.add_argument("--index-bsize", type=int, default=8)
    parser.add_argument("--nbits", type=int, default=2, choices=range(1, 9))
    parser.add_argument("--overwrite", action="store_true")
    return parser.parse_args()


def main():
    args = parse_args()
    collection = Path(args.collection)
    if not collection.is_file():
        raise FileNotFoundError(f"Collection not found: {collection}")

    if torch.cuda.is_available():
        print("CUDA is available; this script will still use one indexing process.")
    else:
        print("CUDA is not available; indexing on CPU. This can take a long time for full MS MARCO.")

    config = ColBERTConfig(
        root=args.root,
        doc_maxlen=args.doc_maxlen,
        query_maxlen=32,
        index_bsize=args.index_bsize,
        nbits=args.nbits,
        kmeans_niters=4,
        avoid_fork_if_possible=True,
    )

    with Run().context(RunConfig(nranks=1, gpus=0, experiment=args.experiment)):
        indexer = Indexer(checkpoint=args.checkpoint, config=config)
        path = indexer.index(
            name=args.name,
            collection=str(collection),
            overwrite="force_silent_overwrite" if args.overwrite else False,
        )

    print(f"Index written to {path}")


if __name__ == "__main__":
    main()
