"""
Temporal Attention layer for BiLSTM-based sequence models.

Learns which timesteps in the input window are most important for prediction.
"""

import torch
import torch.nn as nn
import torch.nn.functional as F


class TemporalAttention(nn.Module):
    """
    Additive (Bahdanau-style) attention over timesteps.

    Input:  (batch, seq_len, hidden_dim)
    Output: (batch, hidden_dim)  — weighted sum over timesteps

    The attention weights are also returned for interpretability.
    """

    def __init__(self, hidden_dim):
        super().__init__()
        self.W = nn.Linear(hidden_dim, hidden_dim, bias=False)
        self.v = nn.Linear(hidden_dim, 1, bias=False)

    def forward(self, lstm_output):
        """
        Parameters
        ----------
        lstm_output : Tensor of shape (batch, seq_len, hidden_dim)

        Returns
        -------
        context : Tensor of shape (batch, hidden_dim)
        weights : Tensor of shape (batch, seq_len)
        """
        # Score each timestep
        energy = torch.tanh(self.W(lstm_output))   # (B, T, H)
        scores = self.v(energy).squeeze(-1)         # (B, T)

        # Softmax to get attention weights
        weights = F.softmax(scores, dim=1)          # (B, T)

        # Weighted sum
        context = torch.bmm(
            weights.unsqueeze(1),   # (B, 1, T)
            lstm_output             # (B, T, H)
        ).squeeze(1)                # (B, H)

        return context, weights
