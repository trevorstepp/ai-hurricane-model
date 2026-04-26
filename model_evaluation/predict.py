import torch
import torch.nn as nn
from sklearn.preprocessing import StandardScaler

class PredictTrack():
    def __init__(self, model: nn.Module, scaler_X: StandardScaler, 
                 scaler_y: StandardScaler):
        self.model = model
        self.scaler_X = scaler_X
        self.scaler_y = scaler_y