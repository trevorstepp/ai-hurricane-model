import matplotlib.pyplot as plt
import pandas as pd
from pathlib import Path

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

def load_or_build(path: Path, build_fn) -> pd.DataFrame:
    """
    Load a DataFrame from a CSV file if it exists; otherwise, build it.

    Parameters
    ----
    path : Path
        Path to the CSV file.
    
    build_fn : Callable
        A function that returns a DataFrame. Called only if the CSV file does not exist.
    
    Returns
    ----
    DataFrame
        The loaded or newly built DataFrame.
    """
    if path.exists():
        return pd.read_csv(path, parse_dates=["datetime"])
    else:
        return build_fn()
    
def apply_movement(prev_lat: float, prev_lon: float, dlat: float, dlon: float) -> tuple[float, float]:
    """
    Convert predicted movement (dlat, dlon) into a new geographic position.

    Parameters
    ----
    prev_lat : float
        Previous latitude.
    prev_lon : float
        Previous longitude.
    dlat : float
        Predicted change in latitude.
    dlon : float
        Predicted change in longitude.

    Returns 
    ----
    tuple[float, float]
        New latitude and longitude after applying predicted movement.
    """
    return prev_lat + dlat, prev_lon + dlon