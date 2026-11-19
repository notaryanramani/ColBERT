import os
import random

from torch.utils.data import Dataset, DataLoader
from transformers import AutoTokenizer


DATA_DIR = "/ml_data/aryan/msmarco_sample"
COLLECTION_PATH = os.path.join(DATA_DIR, "sampled_collection.tsv")
QUERIES_PATH = os.path.join(DATA_DIR, "sampled_queries.train.tsv")
QRELS_PATH = os.path.join(DATA_DIR, "sampled_qrels.train.tsv")
NEGATIVES_PATH = os.path.join(DATA_DIR, "bm25_negatives.tsv")
MODEL_NAME = "bert-base-uncased"
MAX_QUERY_LENGTH = 32
MAX_DOCUMENT_LENGTH = 128
NEGATIVE_POOL_SIZE = 5


class MSMARCOTrainDataset(Dataset):
    def __init__(self, tokenizer=None):
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(MODEL_NAME)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[Q]", "[D]"]}
        )
        self.documents = self._load_documents()
        self.queries = self._load_queries()
        self.positives = self._load_positives()
        self.negatives = self._load_negatives()
        self.examples = self._make_examples()

    def _load_documents(self):
        documents = {}

        with open(COLLECTION_PATH, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t", maxsplit=1)
                if len(parts) == 2:
                    documents[parts[0]] = parts[1]

        return documents

    def _load_queries(self):
        queries = {}

        with open(QUERIES_PATH, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t", maxsplit=1)
                if len(parts) == 2:
                    queries[parts[0]] = parts[1]

        return queries

    def _load_positives(self):
        positives = {}

        with open(QRELS_PATH, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) >= 4 and int(parts[3]) > 0:
                    positives.setdefault(parts[0], []).append(parts[2])

        return positives

    def _load_negatives(self):
        negatives = {}

        with open(NEGATIVES_PATH, "r", encoding="utf-8") as f_in:
            for line in f_in:
                parts = line.rstrip("\r\n").split("\t")
                if len(parts) != 2:
                    continue

                qid, docid = parts
                candidates = negatives.setdefault(qid, [])
                if len(candidates) < NEGATIVE_POOL_SIZE:
                    candidates.append(docid)

        return negatives

    def _make_examples(self):
        examples = []

        for qid, query in self.queries.items():
            positive_ids = [
                docid for docid in self.positives.get(qid, [])
                if docid in self.documents
            ]
            positive_set = set(positive_ids)
            negative_ids = [
                docid for docid in self.negatives.get(qid, [])
                if docid in self.documents and docid not in positive_set
            ]

            if positive_ids and negative_ids:
                examples.append((query, positive_ids, negative_ids))

        return examples

    def __len__(self):
        return len(self.examples)

    def __getitem__(self, index):
        query, positive_ids, negative_ids = self.examples[index]
        positive = self.documents[random.choice(positive_ids)]
        negative = self.documents[random.choice(negative_ids)]
        return query, positive, negative

    def collate(self, batch):
        queries, positives, negatives = zip(*batch)

        query_tokens = self.tokenizer(
            [f"[Q] {query}" for query in queries],
            padding=True,
            truncation=True,
            max_length=MAX_QUERY_LENGTH,
            return_tensors="pt",
        )
        document_tokens = self.tokenizer(
            [f"[D] {document}" for document in list(positives) + list(negatives)],
            padding=True,
            truncation=True,
            max_length=MAX_DOCUMENT_LENGTH,
            return_tensors="pt",
        )

        return {
            "query_input_ids": query_tokens["input_ids"],
            "query_attention_mask": query_tokens["attention_mask"],
            "document_input_ids": document_tokens["input_ids"],
            "document_attention_mask": document_tokens["attention_mask"],
        }


def get_dataloader(batch_size=64, tokenizer=None):
    dataset = MSMARCOTrainDataset(tokenizer=tokenizer)
    return DataLoader(
        dataset,
        batch_size=batch_size,
        shuffle=True,
        collate_fn=dataset.collate,
    )
