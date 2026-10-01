"""
scripts/query_index.py

Implements ColBERT retrieval:
- e2e (§3.6): FAISS IVFPQ candidate retrieval + MaxSim refinement
- re-rank (§3.5): Direct late-interaction MaxSim scoring
"""

import os
import json
import argparse
import numpy as np
import torch
import faiss
from collections import defaultdict

from scripts import config
from scripts.colbert_model import load_model, build_tokenizer, ColBERT
from scripts.utils import prepend_specials, pad_batch

class ColBERTSearcher:
    def __init__(self, index_dir=config.INDEX_DIR, device=None):
        self.device = device or ("cuda" if torch.cuda.is_available() else "cpu")
        self.index_dir = index_dir

        print(f"[searcher] Loading metadata and index artifacts from {index_dir}...")
        with open(os.path.join(index_dir, "doc_ids.json"), "r") as f:
            self.doc_ids = json.load(f)
            
        self.doc_offsets = np.load(os.path.join(index_dir, "doc_offsets.npy"))
        
        emb_path = os.path.join(index_dir, "doc_embeddings.fp16.npy")
        self.doc_embeddings = np.load(emb_path, mmap_mode="r")
        faiss_path = os.path.join(index_dir, "colbert_ivfpq.faiss")
        if os.path.exists(faiss_path):
            self.faiss_index = faiss.read_index(faiss_path)
            self.faiss_index.nprobe = config.FAISS_NPROBE
        else:
            self.faiss_index = None
            print("[searcher] WARN: FAISS index not found. Only re-rank mode will work.")

        self.tokenizer = build_tokenizer()
        self.model = load_model(config.CKPT_PATH, device=self.device)
        self.model.eval()

    def encode_query(self, query_text):
        """Tokenize and encode a single query string to [1, Nq, emb_dim] tensor."""
        tokens = self.tokenizer(
            query_text,
            padding=False,
            truncation=True,
            max_length=config.QUERY_MAXLEN,
            add_special_tokens=False
        )["input_ids"]

        tokens = prepend_specials(self.tokenizer, [tokens], is_query=True)
        ids, mask = pad_batch(tokens, self.tokenizer.pad_token_id)
        
        ids, mask = ids.to(self.device), mask.to(self.device)
        with torch.no_grad():
            Q_emb = self.model.encode(ids, mask) # [1, Nq, 128]
        return Q_emb

    def _token_to_doc_idx(self, token_idx):
        """Binary search in doc_offsets to find the document index for a token index."""
        doc_idx = np.searchsorted(self.doc_offsets, token_idx, side="right") - 1
        return doc_idx

    def _retrieve_candidates_faiss(self, Q_emb, n_candidates=1000, tokens_per_query_token=200):
        """
        §3.6 Candidate Generation:
        Query FAISS with each query vector, map vector hits to doc IDs,
        and pick documents with the highest match counts / approximate scores.
        """
        # Q_emb: [1, Nq, 128]
        q_vecs = Q_emb.squeeze(0).cpu().numpy().astype(np.float32)
        
        # FAISS search for top-k nearest tokens for each query token
        distances, indices = self.faiss_index.search(q_vecs, tokens_per_query_token)
        candidate_scores = defaultdict(float)
        
        for q_i in range(indices.shape[0]):
            for rank_i, tok_idx in enumerate(indices[q_i]):
                if tok_idx < 0:
                    continue
                d_idx = self._token_to_doc_idx(tok_idx)
                if 0 <= d_idx < len(self.doc_ids):
                    # Higher similarity (smaller L2 distance) adds higher weight
                    candidate_scores[d_idx] += 1.0 / (1.0 + float(distances[q_i][rank_i]))
        
        # Sort candidate doc indices by score
        sorted_candidates = sorted(candidate_scores.keys(), key=lambda idx: candidate_scores[idx], reverse=True)
        return sorted_candidates[:n_candidates]

    def _score_candidates_maxsim(self, Q_emb, candidate_doc_indices):
        """
        §3.3 Late Interaction (MaxSim):
        Given candidate doc indices, fetch their document token embeddings from disk
        and compute exact MaxSim scoring.
        """
        if not candidate_doc_indices:
            return []

        scores = []
        for d_idx in candidate_doc_indices:
            start_off = self.doc_offsets[d_idx]
            end_off = self.doc_offsets[d_idx + 1]
            d_vecs = self.doc_embeddings[start_off:end_off].astype(np.float32)
            d_tensor = torch.from_numpy(d_vecs).unsqueeze(0).to(self.device) # [1, Nd, 128]

            with torch.no_grad():
                score = ColBERT.maxsim(Q_emb, d_tensor).item()
                
            scores.append((self.doc_ids[d_idx], score))

        scores.sort(key=lambda x: x[1], reverse=True)
        return scores

    def search(self, query_text, k=1000, mode="e2e", candidate_doc_ids=None):
        """
        Main retrieval interface.
        mode="e2e": FAISS filtering (§3.6) + MaxSim refinement.
        mode="re-rank": Exhaustive or provided candidate MaxSim scoring (§3.5).
        """
        Q_emb = self.encode_query(query_text)

        if mode == "e2e":
            if self.faiss_index is None:
                raise RuntimeError("FAISS index required for e2e mode.")
            # Retrieve pool of candidate documents via FAISS
            candidate_indices = self._retrieve_candidates_faiss(Q_emb, n_candidates=max(k, 1000))
        elif mode == "re-rank":
            if candidate_doc_ids is not None:
                id_to_idx = {did: idx for idx, did in enumerate(self.doc_ids)}
                candidate_indices = [id_to_idx[did] for did in candidate_doc_ids if did in id_to_idx]
            else:
                # Fallback to all documents if candidate list not supplied (used for small smoke tests)
                candidate_indices = list(range(len(self.doc_ids)))
        else:
            raise ValueError(f"Unknown mode: {mode}")

        ranked_docs = self._score_candidates_maxsim(Q_emb, candidate_indices)
        return [doc_id for doc_id, _ in ranked_docs[:k]]


def main():
    parser = argparse.ArgumentParser(description="ColBERT Retrieval Smoke Test")
    parser.add_argument("--query", type=str, default="what is the capital of france", help="Query string")
    parser.add_argument("--mode", type=str, choices=["e2e", "re-rank"], default="e2e", help="Search mode")
    parser.add_argument("--k", type=int, default=10, help="Top-k results")
    args = parser.parse_args()

    searcher = ColBERTSearcher()
    print(f"\n[query] '{args.query}' (mode={args.mode})")
    results = searcher.search(args.query, k=args.k, mode=args.mode)
    
    print("\nTop Results:")
    for rank, doc_id in enumerate(results, 1):
        print(f"  {rank:>2}. DocID: {doc_id}")

if __name__ == "__main__":
    main()
