import torch
import torch.nn as nn
import torch.nn.functional as F
from transformers import AutoModel, AutoTokenizer


class ColBERT(nn.Module):
    def __init__(self, model_name="bert-base-uncased", embedding_dim=128, tokenizer=None):
        super().__init__()
        self.tokenizer = tokenizer or AutoTokenizer.from_pretrained(model_name)
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": ["[Q]", "[D]"]}
        )
        self.bert = AutoModel.from_pretrained(model_name)
        self.bert.resize_token_embeddings(len(self.tokenizer))
        self.projection = nn.Linear(self.bert.config.hidden_size, embedding_dim, bias=False)

    def encode(self, input_ids, attention_mask):
        output = self.bert(input_ids=input_ids, attention_mask=attention_mask)
        embeddings = self.projection(output.last_hidden_state)
        return F.normalize(embeddings, p=2, dim=-1)

    def score(self, query_embeddings, document_embeddings, query_mask, document_mask):
        similarities = torch.einsum("bqh,cdh->bcqd", query_embeddings, document_embeddings)
        document_mask = document_mask.bool()[None, :, None, :]
        similarities = similarities.masked_fill(~document_mask, float("-inf"))
        token_scores = similarities.max(dim=-1).values
        token_scores = token_scores * query_mask.bool()[:, None, :]
        return token_scores.sum(dim=-1)

    def forward(self, query_input_ids, query_attention_mask, document_input_ids, document_attention_mask):
        query_embeddings = self.encode(query_input_ids, query_attention_mask)
        document_embeddings = self.encode(document_input_ids, document_attention_mask)
        return self.score(
            query_embeddings,
            document_embeddings,
            query_attention_mask,
            document_attention_mask,
        )


