from src.data_processing.parse_hurdat2 import parse_hurdat2
from src.data_processing.movement_features import add_movement_features
from src.sequences import build_sequences
from src.split_and_scale import split_and_scale_data
from src.utils import plot_loss, load_or_build, save_model
from src.model import HurricaneLSTM

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path
from datetime import datetime
import numpy as np
import pandas as pd

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
MODELS_DIR = BASE_DIR / "models"

DATA_DIR.mkdir(parents=True, exist_ok=True)
MODELS_DIR.mkdir(parents=True, exist_ok=True)

def training_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                  epochs: int = 10, lr: float = 0.001) -> tuple[list[float], list[float], dict]:
    """
    Train the HurricaneLSTM model using training and test data loaders.

    Parameters
    ----

    Returns
    ----
    tuple[list[float], list[float], dict]
        Training and test loss per epoch and model state dictionary.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)

    best_loss = np.inf
    train_losses = []
    test_losses = []

    for epoch in range(epochs):
        model.train()
        train_loss = 0.0

        for X_batch, y_batch in train_loader:
            # reset gradients
            optimizer.zero_grad()

            outputs = model(X_batch)
            loss = criterion(outputs, y_batch)
            loss.backward()
            optimizer.step()

            train_loss += loss.item()
        
        train_loss /= len(train_loader)

        # evaluate model each epoch, stop when loss stops improving
        model.eval()
        test_loss = 0.0

        with torch.no_grad():
            for X_batch, y_batch in test_loader:
                preds = model(X_batch)
                loss = criterion(preds, y_batch)
                test_loss += loss.item()
        
        test_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}: Train = {train_loss:.4f}, Test = {test_loss:.4f}")

        # if we have a new best loss, keep that model stored
        best_state = None

        if test_loss < best_loss:
            best_loss = test_loss
            best_state = model.state_dict()

        train_losses.append(train_loss)
        test_losses.append(test_loss)
    
    return train_losses, test_losses, best_state

def main() -> None:
    """
    """
    hurdat2_path = DATA_DIR / "hurdat2.txt"
    parsed_hurdat2_csv = DATA_DIR / "parsed_hurdat2.csv"
    movement_features_csv = DATA_DIR / "movement_added.csv"

    hurdat2_df = load_or_build(
        parsed_hurdat2_csv,
        lambda: parse_hurdat2(hurdat2_path, parsed_hurdat2_csv)
    )

    movement_df = load_or_build(
        movement_features_csv,
        lambda: add_movement_features(hurdat2_df, movement_features_csv)
    )

    X, y = build_sequences(movement_df)  # X: (samples, time_steps, features)
                                         # y: (samples, 2)

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_and_scale_data(X, y)

    X_train_t = torch.tensor(X_train, dtype=torch.float32)
    y_train_t = torch.tensor(y_train, dtype=torch.float32)

    X_test_t = torch.tensor(X_test, dtype=torch.float32)
    y_test_t = torch.tensor(y_test, dtype=torch.float32)

    train_loader = DataLoader(
        TensorDataset(X_train_t, y_train_t),
        batch_size=64,
        shuffle=True
    )

    test_loader = DataLoader(
        TensorDataset(X_test_t, y_test_t),
        batch_size=64,
        shuffle=False
    )

    input_dim = X_train.shape[-1]
    hidden_dim = 64
    layer_dim = 2
    output_dim = y_train.shape[-1]
    model = HurricaneLSTM(
        input_dim=input_dim,
        hidden_dim=hidden_dim,
        layer_dim=layer_dim,
        output_dim=output_dim
    )

    train_losses, test_losses, model_dict = training_loop(model, train_loader, test_loader)

    timestamp = datetime.now().strftime("%Y%m%d_%H:%M:%S")
    save_path = MODELS_DIR / f"best_model_{timestamp}.pt"
    model_params = {
        "input_dim": input_dim,
        "hidden_dim": hidden_dim,
        "layer_dim": layer_dim,
        "output_dim": output_dim
    }
    save_model(model_dict, scaler_X, scaler_y, model_params, save_path)
    plot_loss(train_losses, test_losses)

if __name__ == "__main__":
    main()