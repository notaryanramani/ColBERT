# Task 2 — Indexing Documents into a Vector DB
VENV    := myenv
PYTHON  := $(VENV)/bin/python

.PHONY: all index faiss verify clean-index

all: index faiss verify
        @echo ""
        @echo "=========================================="
        @echo " Task 2 complete. Index files in indexes/"
        @echo "=========================================="

index:
        @echo "[1/3] Indexing documents (§3.4) ..."
        $(PYTHON) -m scripts.index_documents

faiss:
        @echo "[2/3] Building FAISS IVFPQ index (§3.6) ..."
        $(PYTHON) -m scripts.build_faiss_index

verify:
        @echo "[3/3] Verifying index ..."
        $(PYTHON) -m scripts.verify_index

clean-index:
        rm -rf indexes/*
