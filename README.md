# ColBERT

This repository contains source code for CS6101 Course Project Autumn '26.

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
