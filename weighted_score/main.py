import os

os.environ["CUDA_VISIBLE_DEVICES"] = "2"

import argparse

import torch

from weighted_score.src.base import ColBERT
from weighted_score.src.data import get_dataloader
from weighted_score.src.plot import plot_losses
from weighted_score.src.train import ColBERTTrainer
from weighted_score.src.weighted import WeightedColBERT


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--epochs", type=int, default=5)
    parser.add_argument("--batch-size", type=int, default=64)
    parser.add_argument("--model", choices=["base", "weighted", "both"], default="both")
    parser.add_argument("--init-from-base", default=None)
    args = parser.parse_args()

    if not torch.cuda.is_available():
        raise RuntimeError("CUDA device 2 is not available")

    torch.cuda.set_device(0)

    dataloader = get_dataloader(batch_size=args.batch_size)
    checkpoint_dir = "/ml_data/aryan/colbert"
    os.makedirs(checkpoint_dir, exist_ok=True)
    histories = {}

    model_classes = {
        "base": ColBERT,
        "weighted": WeightedColBERT,
    }
    model_names = ("base", "weighted") if args.model == "both" else (args.model,)

    for name in model_names:
        model_class = model_classes[name]
        model = model_class()

        if name == "weighted" and args.init_from_base:
            base_state = torch.load(args.init_from_base, map_location="cpu")
            model.load_state_dict(base_state, strict=False)
            print(f"Initialized weighted model from {args.init_from_base}")

        trainer = ColBERTTrainer(model, device="cuda:0")
        history = trainer.train(dataloader, epochs=args.epochs)
        histories[name] = history

        checkpoint_path = os.path.join(checkpoint_dir, f"{name}_colbert.pt")
        torch.save(trainer.model.state_dict(), checkpoint_path)

        print(f"{name} training loss: {history}")
        print(f"Saved checkpoint to {checkpoint_path}")

    plot_name = "loss_weighted.png" if args.model == "weighted" else "loss.png"
    plot_path = os.path.join(os.path.dirname(__file__), "src", plot_name)
    plot_losses(histories, plot_path)
    print(f"Saved loss plot to {plot_path}")


if __name__ == "__main__":
    main()
