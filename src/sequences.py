import pandas as pd
import numpy as np
import numpy.typing as npt

FEATURES = ["dlat", "dlon", "wind", "pressure"]

def build_sequences(df: pd.DataFrame, seq_len: int = 6) -> tuple[npt.NDArray, npt.NDArray]:
    """
    Build LSTM-ready sequences for hurricane movement prediction.

    Each sample consists of:
        X_seq: past `seq_len` timesteps of features
        y_seq: next timestep movement (dlat, dlon)

    Parameters
    ----
    df : DataFrame
        The DataFrame containing hurricane track data with required columns:
        [`storm_id`, `datetime`, `wind`, `pressure`, `dlat`, `dlon`]
    
    seq_len : int, default=6
        The number of past timesteps used to predict the next step.

    Returns
    ----
    X : ndarray

    y : ndarray
    """
    X, y = [], []

    # create sequences of length 'seq_len' to predict the next movement
    # discard 'lat' and 'lon' since we have 'dlat' and 'dlon'
    for storm_id, group in df.groupby("storm_id"):
        # re-sort by datetime just in case
        group = group.sort_values("datetime")
        vals = group[FEATURES].values

        for i in range(len(vals) - seq_len - 1):
            X_seq = vals[i:i+seq_len]
            y_seq = vals[i+seq_len][0:2]  # selects 'dlat' and 'dlon'

            X.append(X_seq)
            y.append(y_seq)
    
    return np.array(X), np.array(y)