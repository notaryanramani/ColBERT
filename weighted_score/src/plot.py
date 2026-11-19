import matplotlib.pyplot as plt


def plot_losses(histories, output_path):
    for name, history in histories.items():
        losses = history.tolist() if hasattr(history, "tolist") else list(history)
        plt.plot(range(1, len(losses) + 1), losses, marker="o", label=name)

    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training Loss")
    plt.grid(True)
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()
