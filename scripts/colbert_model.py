import os
import torch
import torch.nn as nn
from transformers import BertModel, BertTokenizer

from scripts import config


class ColBERT(nn.Module):
    def __init__(self, bert_name=config.BERT_NAME, emb_dim=config.EMB_DIM):
        super().__init__()
        self.bert   = BertModel.from_pretrained(bert_name)
        hidden      = self.bert.config.hidden_size      # 768 for base
        self.linear = nn.Linear(hidden, emb_dim, bias=False)
        # room for [Q] and [D]
        self.bert.resize_token_embeddings(self.bert.config.vocab_size + 2)

    def encode(self, input_ids, attention_mask):
        """Returns L2-normalized token embeddings: [B, L, m]."""
        out = self.bert(input_ids=input_ids,
                        attention_mask=attention_mask).last_hidden_state
        emb = self.linear(out)
        emb = nn.functional.normalize(emb, p=2, dim=-1)
        return emb

    @staticmethod
    def maxsim(Eq, Ed):
        """Late interaction, Eq: [B,Nq,m], Ed: [B,Nd,m] -> [B]."""
        sim = torch.bmm(Eq, Ed.transpose(1, 2))         # [B, Nq, Nd]
        return sim.max(dim=2).values.sum(dim=1)


def build_tokenizer():
    tok = BertTokenizer.from_pretrained(config.BERT_NAME)
    tok.add_special_tokens(
        {"additional_special_tokens": [config.QUERY_TOKEN, config.DOC_TOKEN]})
    return tok


def load_model(ckpt_path=None, device="cuda"):
    model = ColBERT().to(device)
    if ckpt_path and os.path.isfile(ckpt_path):
        state = torch.load(ckpt_path, map_location=device)
        # tolerate DataParallel prefixes
        state = {k.replace("module.", ""): v for k, v in state.items()}
        model.load_state_dict(state)
        print(f"[load_model] loaded checkpoint: {ckpt_path}")
    else:
        print("[load_model] no checkpoint found — using untrained [Q]/[D]/linear")
    model.eval()
    return model
