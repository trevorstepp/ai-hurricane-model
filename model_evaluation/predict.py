import numpy as np
import numpy.typing as npt
import torch.nn as nn
import torch
from sklearn.preprocessing import StandardScaler

def predict_one_step(
    model: nn.Module, 
    X_seq: npt.NDArray, 
    scaler_X: StandardScaler,
    scaler_y: StandardScaler
) -> npt.NDArray:
    """
    Predict the next timestep (dlat, dlon) given a sequence.

    Parameters
    ----
    model : nn.Module
        Trained PyTorch model that maps a sequence of features to the next movement (dlat, dlon).
    X_seq : ndarray
        Input sequence of shape (time_steps, num_features).
    scaler_X : StandardScaler
        Fitted scaler used to standardize input features.
    scaler_y : StandardScaler
        Fitted scaler used to standardize target values during training. Used with
        `inverse_transform` to unscale the prediction.
    
    Returns
    ----
    ndarray
        Predicted (dlat, dlon), shape (2,).
    """
    model.eval()

    num_features = X_seq.shape[-1]  # X_seq has shape (time_steps, features)
    # scale then need to reshape to 3 dimensions (what the model expects)
    X_2d = X_seq.reshape(-1, num_features)
    X_2d_scaled = scaler_X.transform(X_2d)
    X_scaled = X_2d_scaled.reshape(1, -1, num_features)

    X_tensor = torch.tensor(X_scaled, dtype=torch.float32)

    with torch.no_grad():
        pred = model(X_tensor).numpy()
    
    pred_unscaled = scaler_y.inverse_transform(pred)
    return pred_unscaled[0]