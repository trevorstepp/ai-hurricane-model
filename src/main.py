from parse_hurdat2 import parse_hurdat2
from movement_features import add_movement_features
from sequences import build_sequences

from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"

def main():
    """
    """
    hurdat2_path = DATA_DIR / "hurdat2.txt"
    parsed_hurdat2_csv = DATA_DIR / "parsed_hurdat2.csv"
    movement_features_csv = DATA_DIR / "movement_added.csv"

    df = parse_hurdat2(hurdat2_path, parsed_hurdat2_csv)
    df = add_movement_features(df, movement_features_csv)

    X, y = build_sequences(df)

    # split data and normalize
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

if __name__ == "__main__":
    main()