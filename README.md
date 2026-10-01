# ColBERT

This repository contains source code for CS6101 Course Project Autumn '26.
**Paper:** *ColBERT: Efficient and Effective Passage Search via Contextualized Late Interaction over BERT* (Khattab & Zaharia, SIGIR 2020)
## 1. Objective

Build a ColBERT-style vector index over the MS MARCO passage collection so that
each document is represented as a **bag of contextualized token embeddings**
(not a single vector), and so that late-interaction MaxSim scoring can be run
against the index at query time.
# Dataset Downloading and Loading Guide

Install the dependencies:

```bash
python -m venv venv
source venv/bin/active  # MacOS/Linux
venv\scripts\activate # Windows
pip install -r requirements.txt
```

Download the complete MS MARCO passage dataset and its `dev/small` queries and relevance judgments:

```bash
python scripts/data.py msmarco
```

Data folder structure:

```text
data/msmarco/
├── collection.tsv       
├── queries.dev.tsv      
└── qrels.dev.tsv        
```

`download_msmarco()` does not download the training queries or qrels by
default as we don't need training data for evaluation. To get those too, call it explicitly:

```python
from scripts.data import download_msmarco
download_msmarco(include_train=True)
```

For a small local smoke test, use `--limit`. The script still includes all passages needed by the dev qrels, so the resulting collection can be larger than the requested limit:

```bash
python scripts/data.py msmarco --limit 100000
```

## How to load the downloaded data

Documents are saved inside `data/msmarco/collection.tsv`. Each row has the document
identifier that must be retained in search results. Evaluate the retrieved
document identifiers against `data/msmarco/qrels.dev.tsv`, using the matching
queries in `data/msmarco/queries.dev.tsv`.

Use the predefined loaders for easy dataloading to index later:

```python
from scripts.data import load_collection, load_qrels, load_queries

data_dir = "data/msmarco"
collection = load_collection(data_dir) 
queries = load_queries(f"{data_dir}/queries.dev.tsv")  
qrels = load_qrels(f"{data_dir}/qrels.dev.tsv")  
```

The same layout is available for TREC CAR:

```bash
python scripts/data.py trec-car
```

Use `data/trec-car/collection.tsv`, `queries.test.tsv`, and `qrels.test.tsv`
when indexing and evaluating that dataset for TREC CAR.

# Task 2 — Indexing Documents into a Vector DB

Implements the paper's offline indexing (§3.4): the BERT document encoder
`f_D` is run once over the collection, producing one `m`-dim embedding per
token per document. Unlike a conventional vector DB that stores one vector
per document, ColBERT stores a *bag of embeddings* per document, so that
late-interaction MaxSim scoring (§3.3) can be applied at query time.

Install the dependencies by running below command
```bash
pip install numpy torch transformers faiss-cpu tqdm ir_datasets pandas safetensors
```
## Files

| File | Significance |
|---|---|
| `scripts/config.py` | All hyperparameters — encoder dims, batch sizes, storage dtype, FAISS params, paths. Single source of truth referenced by every other script. |
| `scripts/colbert_model.py` | ColBERT model definition: BERT encoders for query and document (§3.2), linear projection `768 → 128` with L2 normalization, and the `maxsim()` late-interaction operator (§3.3, Eq. 3). |
| `scripts/utils.py` | Indexing throughput helpers implementing the four §3.4 optimizations: `length_bucket()` (length-based bucketing), `pad_batch()` (per-batch max-length padding), `tokenize_parallel()` (multi-core tokenization with serial fallback for small slices), `prepend_specials()` (adds `[CLS] [D]` / `[CLS] [Q]`), `punct_ids()` (punctuation filter list). |
| `scripts/index_documents.py` | **Task 2 main script.** Loads `collection.tsv`, encodes every document through the BERT document encoder, applies the punctuation filter, saves the flat embedding array + offsets + doc IDs to `indexes/`. |
| `scripts/build_faiss_index.py` | Builds the FAISS `IndexIVFPQ` structure over the flat embeddings for end-to-end retrieval (§3.6). Uses `nlist=2000`, `m=16`, `nbits=8`, `nprobe=10` per the paper. |
| `scripts/verify_index.py` | Post-run validator. Confirms embedding dim, offsets length, offsets endpoint, average tokens per doc, and that sample doc IDs exist in the collection. |
| `scripts/query_index.py` | Retrieval smoke test. Supports `--mode re-rank` (§3.5, exhaustive MaxSim) and `--mode e2e` (§3.6, FAISS filter + MaxSim refine). |
| `Makefile` | One-shot pipeline runner: `make` executes the three steps above in sequence, stopping on the first failure.|
| `indexes/` | Output directory for all Task 2 artifacts. |

> **Note:** Replace `venv` with your own virtual environment name
> (e.g. `myenv`) in the activation commands above.

## Output

After a successful run, `indexes/` contains:

| File | Shape / Type | Description |
|---|---|---|
| `doc_embeddings.fp16.npy` | `[total_tokens, 128]` fp16 | Flat array of all L2-normalized token embeddings across the collection. |
| `doc_offsets.npy`         | `[num_docs + 1]` int64 | Doc `i` occupies `embeddings[offsets[i] : offsets[i+1]]`. |
| `doc_ids.json`            | `list[str]` | MS MARCO doc IDs in the same order as offsets. |
| `colbert_ivfpq.faiss`     | FAISS index | IVFPQ structure for approximate top-k search over all token embeddings (§3.6). |


The trailing `OK` line confirms all validation checks passed.

## How to run

With the virtualenv activated and from the repo root:

```bash
make

# Task 3
## Instructions to Run
### First Testing the Retrieval Pipeline

Run a single query to ensure that the FAISS index, embedding slices, and MaxSim operator work end-to-end without crashing:

```bash
python -m scripts.query_index --query "what is the capital of france" --mode e2e --k 10
```
### Fast Sanity Check (Subset of Queries)

Run evaluation on the first 50 queries to quickly check that `queries.dev.tsv` and `qrels.dev.tsv` parse correctly and that the MRR/Recall formulas compute without errors:

```bash
python -m scripts.evaluate --mode e2e --limit 50
```

### Full Evaluation & Benchmark Generation

Run the evaluation across all queries in the evaluation split and write the results to `evaluation.txt`.

```bash
python -m scripts.evaluate --mode e2e | tee evaluation.txt
```
