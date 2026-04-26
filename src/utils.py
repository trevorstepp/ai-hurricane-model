import matplotlib.pyplot as plt
import pandas as pd
import torch.nn as nn
import torch
from pathlib import Path
from sklearn.preprocessing import StandardScaler

from src.model import HurricaneLSTM

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

def save_model(state_dict: dict, scaler_X: StandardScaler, scaler_y: StandardScaler,
               model_params: dict, path: Path) -> None:
    """
    Save trained model and associated scalers.

    Parameters
    ----
    state_dict : dict
        State dictionary containing the trained model weights.
    scaler_X : StandardScaler
        Fitted scaler used to standardize input features.
    scaler_y : StandardScaler
        Fitted scaler used to standardize target values.
    model_params : dict
        Dictionary of model initialization parameters used to reconstruct the model
        architecture when loading in the model.
    path : Path
        File path to where the model will be saved.
    """
    torch.save({
        "model_state_dict": state_dict,
        "scaler_X": scaler_X,
        "scaler_y": scaler_y,
        "model_params": model_params
    }, path)

def load_model(model_class: type[nn.Module], path: Path) -> tuple[nn.Module, StandardScaler, StandardScaler]:
    """
    Retrieve the trained model and associated scalers.

    Parameters
    ----
    model_class : type[nn.Module]
        The model class used to reconstruct the model.
    path : Path
        File path to where the model is saved.
    
    Returns
    ----
    tuple[nn.Module, StandardScaler, StandardScaler]
        The reconstructed model and the scalers used to standardize input and target values.
    """
    load = torch.load(path, weights_only=False)

    params = load["model_params"]
    model = model_class(**params)
    model.load_state_dict(load["model_state_dict"])

    return model, load["scaler_X"], load["scaler_y"]