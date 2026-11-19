import torch
import torch.nn as nn
import torch.nn.functional as F
from tqdm import tqdm

class ColBERTTrainer:
    def __init__(self, model, learning_rate=2e-5, device=None, max_grad_norm=1.0):
        self.device = torch.device(device or ("cuda" if torch.cuda.is_available() else "cpu"))
        self.model = model.to(self.device)
        self.optimizer = torch.optim.AdamW(self.model.parameters(), lr=learning_rate)
        self.max_grad_norm = max_grad_norm

    def train_step(self, batch):
        self.model.train()
        batch = {key: value.to(self.device) for key, value in batch.items()}

        with torch.autocast(
            device_type=self.device.type,
            dtype=torch.bfloat16,
            enabled=self.device.type == "cuda",
        ):
            scores = self.model(
                query_input_ids=batch["query_input_ids"],
                query_attention_mask=batch["query_attention_mask"],
                document_input_ids=batch["document_input_ids"],
                document_attention_mask=batch["document_attention_mask"],
            )
            targets = torch.arange(scores.size(0), device=self.device)
            loss = F.cross_entropy(scores.float(), targets)

        self.optimizer.zero_grad()
        loss.backward()
        nn.utils.clip_grad_norm_(self.model.parameters(), self.max_grad_norm)
        self.optimizer.step()

        return loss.detach().item()

    def train(self, dataloader, epochs=1):
        history = []

        for epoch in range(epochs):
            total_loss = 0.0
            batches = 0

            progress = tqdm(dataloader, desc=f"Epoch {epoch + 1}/{epochs}")

            for batch in progress:
                loss = self.train_step(batch)
                total_loss += loss
                batches += 1
                progress.set_postfix(loss=f"{loss:.4f}")

            history.append(total_loss / max(batches, 1))

        return history
