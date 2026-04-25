from data_processing import parse_hurdat2, add_movement_features
from sequences import build_sequences
from split_and_scale import split_and_scale_data
from model import HurricaneLSTM

import torch
import torch.nn as nn
from torch.utils.data import TensorDataset, DataLoader
from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def training_loop(model: nn.Module, train_loader: DataLoader, test_loader: DataLoader,
                  epochs: int = 10, lr: float = 0.001) -> tuple[list[float], list[float]]:
    """
    Train the HurricaneLSTM model.
    """
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters, lr)
    best_loss = 0.0

    for epoch in range(epochs):
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
            for X_test, y_test in test_loader:
                preds = model(X_test)
                loss = criterion(preds, y_test)
                test_loader += loss.item()
        
        test_loss /= len(test_loader)
        print(f"Epoch {epoch + 1}: Train = {train_loss:.4f}, Test = {test_loss:.4f}")

def main() -> None:
    """
    """
    hurdat2_path = DATA_DIR / "hurdat2.txt"
    parsed_hurdat2_csv = DATA_DIR / "parsed_hurdat2.csv"
    movement_features_csv = DATA_DIR / "movement_added.csv"

    df = parse_hurdat2(hurdat2_path, parsed_hurdat2_csv)
    df = add_movement_features(df, movement_features_csv)

    X, y = build_sequences(df)  # X: (samples, time_steps, features)
                                # y: (samples, 2)
    #print(X.shape)
    #print(y.shape)

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

    # sanity check
    print("X_train: ", X_train.shape)
    print("y_train: ", y_train.shape)

    model = HurricaneLSTM(
        input_dim=X_train.shape[-1],
        hidden_dim=64,
        layer_dim=2,
        output_dim=y_train.shape[-1]
    )

    training_loop(model, train_loader, test_loader)

if __name__ == "__main__":
    main()