import torch
import torch.nn as nn
import torch.nn.functional as F

from .base import ColBERT


class WeightedColBERT(ColBERT):
    def __init__(self, model_name="bert-base-uncased", embedding_dim=128, tokenizer=None):
        super().__init__(
            model_name=model_name,
            embedding_dim=embedding_dim,
            tokenizer=tokenizer,
        )
        self.query_token_scorer = nn.Linear(embedding_dim, 1)

    def score(self, q_emb, d_emb, q_mask, d_mask, q_hidden=None):
        sim = torch.einsum("bqh,cdh->bcqd", q_emb, d_emb)
        sim = sim.masked_fill(~d_mask.bool()[None, :, None, :], float("-inf"))
        token_scores = sim.max(dim=-1).values                    

        qm = q_mask.bool()
        feats = q_hidden if q_hidden is not None else q_emb
        logits = self.query_token_scorer(feats).squeeze(-1)     
        logits = logits.masked_fill(~qm, float("-inf"))
        n = qm.sum(-1, keepdim=True).clamp(min=1)
        weights = F.softmax(logits, dim=-1) * n                  
        weights = weights.masked_fill(~qm, 0.0)

        token_scores = token_scores.masked_fill(~qm[:, None, :], 0.0)
        return torch.einsum("bq,bcq->bc", weights, token_scores)