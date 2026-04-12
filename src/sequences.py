import pandas as pd
import numpy as np
import numpy.typing as npt

FEATURES = ["dlat", "dlon", "wind", "pressure"]

def build_sequences(df: pd.DataFrame, seq_len: int = 6) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Build LSTM-ready sequences for hurricane movement prediction.

    Each sample:
        X = past `seq_len` timesteps.
        y = next timestep movement (`dlat`, `dlon`)

    Parameters
    ----
    df : pd.DataFrame
        DataFrame containing
    
    seq_len : int
        The

    Returns
    ----
    """
    X, y = [], []

    # create sequences of length 'seq_len' to predict the next movement
    # discard 'lat' and 'lon' since we have 'dlat' and 'dlon'
    for storm_id, group in df.groupby("storm_id"):
        vals = group[FEATURES].values

        for i in range(len(vals) - seq_len):
            X_seq = vals[i:i+seq_len]
            y_seq = vals[i+seq_len]