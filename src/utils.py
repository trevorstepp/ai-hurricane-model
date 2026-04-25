import matplotlib.pyplot as plt

def plot_loss(train_loss: list[float], test_loss: list[float]) -> None:
    """
    """
    plt.plot(train_loss, label="Train")
    plt.plot(test_loss, label="Test")
    plt.legend()
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.title("Training versus Test Loss")
    plt.grid(axis="both", alpha=0.5)
    plt.tight_layout()
    plt.show()