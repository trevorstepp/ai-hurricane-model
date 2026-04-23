import numpy.typing as npt
from typing import NamedTuple
from sklearn.preprocessing import StandardScaler

class TrainTestData(NamedTuple):
    X_train: npt.NDArray
    X_test: npt.NDArray
    y_train: npt.NDArray
    y_test: npt.NDArray
    scaler_X: StandardScaler
    scaler_y: StandardScaler

def split_and_scale_data(X: npt.NDArray, y: npt.NDArray) -> TrainTestData:
    """
    Split input sequences and target values into training and testing datasets, 
    then apply standardization.

    Parameters
    ----
    X : ndarray
        Input features of shape (samples, time_steps, features)
    y : ndarray
        Target values of shape (samples, 2)
    
    Returns
    ----
    A named tuple with the following attributes:

    X_train : ndarray
        Scaled input sequence for training the model.
    X_test : ndarray
        Scaled input sequence for testing the model.
    y_train : ndarray
        Scaled target values for training the model.
    y_test : ndarray
        Scaled target values for testing the model.
    scaler_X : StandardScaler
        Scaler for X_train and X_test datasets.
    scaler_y : StandardScaler
        Scaler for y_train and y_test datasets.
    """
    # split data and normalize
    num_features = X.shape[-1]
    split_idx = int(len(X) * 0.8)
    X_train_split, X_test_split = X[:split_idx], X[split_idx:]
    y_train_split, y_test_split = y[:split_idx], y[split_idx:]

    scaler_X = StandardScaler()
    scaler_y = StandardScaler()

    # need to reshape X for scaler
    X_train_2d = X_train_split.reshape(-1, num_features)
    X_test_2d = X_test_split.reshape(-1, num_features)
    X_train_scaled = scaler_X.fit_transform(X_train_2d)
    X_test_scaled = scaler_X.transform(X_test_2d)

    # must reshape to original
    X_train = X_train_scaled.reshape(X_train_split.shape)
    X_test = X_test_scaled.reshape(X_test_split.shape)

    # no need to reshape y for scaler
    y_train = scaler_y.fit_transform(y_train_split)
    y_test = scaler_y.transform(y_test_split)

    return TrainTestData(X_train, X_test, y_train, y_test, scaler_X, scaler_y)