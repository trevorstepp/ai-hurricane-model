from parse_hurdat2 import parse_hurdat2
from movement_features import add_movement_features
from sequences import build_sequences
from split_and_scale import split_and_scale_data

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

    X, y = build_sequences(df)  # (samples, time_steps, features)
    print(X.shape)
    print(y.shape)

    X_train, X_test, y_train, y_test, scaler_X, scaler_y = split_and_scale_data(X, y)

if __name__ == "__main__":
    main()