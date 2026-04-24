import torch
import torch.nn as nn

class HurricaneLSTM(nn.Module):
    def __init__(self, input_dim, hidden_dim, layer_dim, output_dim):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, hidden_dim, layer_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, output_dim)

    def forward(self, x: torch.Tensor):
        """
        Forward pass of the LSTM model.

        Parameters
        ----
        x : torch.Tensor
            Input tensor of shape (batch_size, seq_len, input_dim).

        Returns
        ----
        torch.Tensor
            Predicted movement (dlat, dlon) of shape (batch_size, 2).
        """
        lstm_out, _ = self.lstm(x)
        last_timestep = lstm_out[:, -1, :]
        out = self.fc(last_timestep)
        return out