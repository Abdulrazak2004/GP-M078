"""
Multi-task model architectures for Casing RUL Prediction.

All neural models share:
  - A backbone that maps (batch, window, n_features) → (batch, hidden_dim)
  - 4 task-specific output heads:
      Head 1: RUL (days)          — regression, clamped [0, 500]
      Head 2: Corrosion Rate (mpy) — regression, ReLU output
      Head 3: Wall Thickness (mm)  — regression, ReLU output
      Head 4: 60-Month Forecast    — 60 regression outputs (every 30 days for 5 years), ReLU

  Note: Corrosion Cause classification is trained as a separate model.

Architectures:
  0. NaiveBaseline    — linear extrapolation (not a neural net)
  1. SimpleLSTM       — 2-layer LSTM
  2. BiLSTMAttention  — 2-layer Bidirectional LSTM + Temporal Attention
  3. CNNLSTM          — Conv1D feature extraction → LSTM
"""

import numpy as np
import torch
import torch.nn as nn

from src.attention import TemporalAttention
from src.config import (
    LSTM_HIDDEN_1, LSTM_HIDDEN_2,
    DROPOUT_LSTM, DROPOUT_BILSTM, DROPOUT_HEAD,
    CNN_FILTERS_1, CNN_FILTERS_2, CNN_KERNEL,
    NUM_FORECAST_HORIZONS,
    RUL_CAP, WINDOW_SIZE,
    TRANSFORMER_D_MODEL, TRANSFORMER_NHEAD,
    TRANSFORMER_LAYERS, TRANSFORMER_DROPOUT,
)


# ============================================================================
# MULTI-TASK OUTPUT HEADS  (shared by all neural models)
# ============================================================================

class MultiTaskHeads(nn.Module):
    """Four task-specific output heads with physics-informed CR head.

    CR is instantaneous (NORSOK takes current conditions → CR), so the CR head
    uses raw features from the last timestep instead of temporal backbone output.
    RUL/WT/Forecast are cumulative and use temporal features from the backbone.

    Cause classification is trained separately — not part of this model.
    """

    def __init__(self, temporal_dim, raw_dim):
        super().__init__()

        # Head 1: RUL (regression) — uses temporal features, clamped to [0, RUL_CAP]
        self.head_rul = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_HEAD),
            nn.Linear(32, 1),
        )

        # Head 2: Corrosion Rate (regression) — physics-informed: uses RAW features
        # CR is instantaneous (NORSOK equation), no temporal dependency
        self.head_cr = nn.Sequential(
            nn.Linear(raw_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_HEAD),
            nn.Linear(64, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_HEAD),
            nn.Linear(32, 1),
            nn.ReLU(),  # rate >= 0
        )

        # Head 3: Wall Thickness (regression) — uses temporal features
        self.head_wt = nn.Sequential(
            nn.Linear(temporal_dim, 32),
            nn.ReLU(),
            nn.Dropout(DROPOUT_HEAD),
            nn.Linear(32, 1),
            nn.ReLU(),  # thickness >= 0
        )

        # Head 4: 60-Month Forecast — uses temporal features
        self.head_forecast = nn.Sequential(
            nn.Linear(temporal_dim, 64),
            nn.ReLU(),
            nn.Dropout(DROPOUT_HEAD),
            nn.Linear(64, NUM_FORECAST_HORIZONS),
            nn.ReLU(),  # thickness >= 0
        )

    def forward(self, temporal_features, raw_features):
        """
        Parameters
        ----------
        temporal_features : Tensor of shape (batch, temporal_dim) — from backbone
        raw_features : Tensor of shape (batch, raw_dim) — x[:, -1, :] last timestep

        Returns
        -------
        dict with keys: 'rul', 'cr', 'wt', 'forecast'
        """
        rul = self.head_rul(temporal_features).squeeze(-1)
        rul = rul.clamp(0.0, RUL_CAP)  # enforce [0, 500]
        return {
            "rul": rul,                                                    # (B,)
            "cr": self.head_cr(raw_features).squeeze(-1),                  # (B,)
            "wt": self.head_wt(temporal_features).squeeze(-1),             # (B,)
            "forecast": self.head_forecast(temporal_features),             # (B, 60)
        }


# ============================================================================
# MODEL 0: NAIVE BASELINE  (not a neural network)
# ============================================================================

class NaiveBaseline:
    """
    Linear extrapolation baseline.

    Uses the slope of Current_Thickness_mm over the input window to estimate
    when thickness will hit the failure threshold (3 mm or 50% of initial).

    Not a nn.Module — used only for evaluation comparison.
    """

    def __init__(self, thickness_feature_idx, initial_thickness=11.0,
                 failure_threshold_mm=3.0):
        self.thickness_idx = thickness_feature_idx
        self.initial_thickness = initial_thickness
        self.failure_threshold = max(failure_threshold_mm,
                                     0.5 * initial_thickness)

    def predict(self, X):
        """
        Parameters
        ----------
        X : np.ndarray of shape (N, window_size, n_features) — UNSCALED

        Returns
        -------
        rul_pred : np.ndarray of shape (N,)
        """
        thickness = X[:, :, self.thickness_idx]  # (N, W)
        window_size = thickness.shape[1]

        # Linear slope over window
        t_axis = np.arange(window_size, dtype=np.float32)
        t_mean = t_axis.mean()
        t_var = ((t_axis - t_mean) ** 2).sum()

        # Vectorized slope calculation
        thickness_mean = thickness.mean(axis=1, keepdims=True)  # (N, 1)
        cov = ((t_axis[None, :] - t_mean) * (thickness - thickness_mean)).sum(axis=1)
        slope = cov / max(t_var, 1e-8)  # mm/day  (N,)

        current_thickness = thickness[:, -1]  # (N,)

        # RUL = (current - threshold) / |slope|
        gap = current_thickness - self.failure_threshold
        rul = np.where(
            slope < -1e-8,
            gap / np.abs(slope),
            500.0  # no measurable decline → cap at 500 days
        )

        return np.clip(rul, 0.0, 500.0).astype(np.float32)


# ============================================================================
# MODEL 1: SIMPLE LSTM
# ============================================================================

class SimpleLSTM(nn.Module):
    """
    Two-layer stacked LSTM with LayerNorm → multi-task heads.
    """

    def __init__(self, n_features, hidden1=LSTM_HIDDEN_1,
                 hidden2=LSTM_HIDDEN_2, dropout=DROPOUT_LSTM):
        super().__init__()

        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True)
        self.norm1 = nn.LayerNorm(hidden1)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1, hidden2, batch_first=True)
        self.norm2 = nn.LayerNorm(hidden2)
        self.drop2 = nn.Dropout(dropout)

        self.heads = MultiTaskHeads(hidden2, n_features)

    def forward(self, x):
        """x: (batch, window, n_features) → dict of predictions"""
        raw_last = x[:, -1, :]  # (B, n_features) — for CR head

        out, _ = self.lstm1(x)
        out = self.drop1(self.norm1(out))
        out, _ = self.lstm2(out)
        out = self.drop2(self.norm2(out))

        # Take last timestep
        features = out[:, -1, :]  # (B, hidden2)
        return self.heads(features, raw_last)


# ============================================================================
# MODEL 2: BiLSTM + TEMPORAL ATTENTION
# ============================================================================

class BiLSTMAttention(nn.Module):
    """
    Two-layer Bidirectional LSTM + LayerNorm + Temporal Attention → multi-task heads.
    """

    def __init__(self, n_features, hidden1=LSTM_HIDDEN_1,
                 hidden2=LSTM_HIDDEN_2, dropout=DROPOUT_BILSTM):
        super().__init__()

        self.lstm1 = nn.LSTM(n_features, hidden1, batch_first=True,
                             bidirectional=True)
        self.norm1 = nn.LayerNorm(hidden1 * 2)
        self.drop1 = nn.Dropout(dropout)
        self.lstm2 = nn.LSTM(hidden1 * 2, hidden2, batch_first=True,
                             bidirectional=True)
        self.norm2 = nn.LayerNorm(hidden2 * 2)
        self.drop2 = nn.Dropout(dropout)

        # Attention over timesteps (hidden2 * 2 due to bidirectional)
        self.attention = TemporalAttention(hidden2 * 2)

        # Extra dropout before heads
        self.drop3 = nn.Dropout(DROPOUT_HEAD)

        self.heads = MultiTaskHeads(hidden2 * 2, n_features)

    def forward(self, x):
        """x: (batch, window, n_features) → dict of predictions"""
        raw_last = x[:, -1, :]  # (B, n_features) — for CR head

        out, _ = self.lstm1(x)
        out = self.drop1(self.norm1(out))
        out, _ = self.lstm2(out)
        out = self.drop2(self.norm2(out))

        # Attention-weighted sum over timesteps
        features, self._attn_weights = self.attention(out)  # (B, hidden2*2)
        features = self.drop3(features)

        return self.heads(features, raw_last)

    @property
    def attention_weights(self):
        """Access last-computed attention weights for interpretability."""
        return self._attn_weights


# ============================================================================
# MODEL 3: CNN-LSTM HYBRID
# ============================================================================

class CNNLSTM(nn.Module):
    """
    Conv1D → LSTM → multi-task heads.

    CNN captures local patterns (short-term trends),
    LSTM captures longer temporal dependencies.
    CR head uses raw last-timestep features (physics-informed).
    """

    def __init__(self, n_features, cnn1=CNN_FILTERS_1, cnn2=CNN_FILTERS_2,
                 kernel=CNN_KERNEL, lstm_hidden=LSTM_HIDDEN_1,
                 dropout=DROPOUT_LSTM):
        super().__init__()

        self.conv1 = nn.Conv1d(n_features, cnn1, kernel_size=kernel, padding=0)
        self.bn1 = nn.BatchNorm1d(cnn1)
        self.relu1 = nn.ReLU()
        self.conv2 = nn.Conv1d(cnn1, cnn2, kernel_size=kernel, padding=0)
        self.bn2 = nn.BatchNorm1d(cnn2)
        self.relu2 = nn.ReLU()
        self.pool = nn.MaxPool1d(kernel_size=2)

        self.lstm = nn.LSTM(cnn2, lstm_hidden, batch_first=True)
        self.norm = nn.LayerNorm(lstm_hidden)
        self.drop = nn.Dropout(dropout)

        self.heads = MultiTaskHeads(lstm_hidden, n_features)

    def forward(self, x):
        """x: (batch, window, n_features) → dict of predictions"""
        raw_last = x[:, -1, :]  # (B, n_features) — for CR head

        # Conv1d expects (B, C, L) — transpose from (B, L, C)
        out = x.transpose(1, 2)                        # (B, n_features, window)
        out = self.relu1(self.bn1(self.conv1(out)))     # (B, cnn1, window-2)
        out = self.relu2(self.bn2(self.conv2(out)))     # (B, cnn2, window-4)
        cnn_features = self.pool(out)                   # (B, cnn2, (window-4)//2)

        # Back to (B, L, C) for LSTM
        out = cnn_features.transpose(1, 2)
        out, _ = self.lstm(out)
        out = self.drop(self.norm(out))

        # Take last timestep
        lstm_features = out[:, -1, :]                   # (B, lstm_hidden)

        return self.heads(lstm_features, raw_last)


# ============================================================================
# MODEL 4: TRANSFORMER
# ============================================================================

class TransformerBackbone(nn.Module):
    """
    Transformer encoder with learnable positional embeddings → multi-task heads.

    Linear projection → positional embedding → TransformerEncoder → mean pool.
    """

    def __init__(self, n_features, d_model=TRANSFORMER_D_MODEL,
                 nhead=TRANSFORMER_NHEAD, num_layers=TRANSFORMER_LAYERS,
                 dropout=TRANSFORMER_DROPOUT, max_len=WINDOW_SIZE):
        super().__init__()

        # Linear projection: n_features → d_model
        self.input_proj = nn.Linear(n_features, d_model)

        # Learnable positional embedding (fixed window size)
        self.pos_embed = nn.Parameter(torch.randn(1, max_len, d_model) * 0.02)

        encoder_layer = nn.TransformerEncoderLayer(
            d_model=d_model,
            nhead=nhead,
            dim_feedforward=d_model * 4,
            dropout=dropout,
            batch_first=True,
            activation="gelu",
            norm_first=True,  # Pre-LN — more stable for deeper models
        )
        self.encoder = nn.TransformerEncoder(encoder_layer, num_layers=num_layers)

        self.norm = nn.LayerNorm(d_model)
        self.drop = nn.Dropout(dropout)

        self.heads = MultiTaskHeads(d_model, n_features)

    def forward(self, x):
        """x: (batch, window, n_features) → dict of predictions"""
        raw_last = x[:, -1, :]  # (B, n_features) — for CR head
        seq_len = x.size(1)

        # Project + add positional embeddings
        out = self.input_proj(x) + self.pos_embed[:, :seq_len, :]

        # Transformer encoder
        out = self.encoder(out)  # (B, L, d_model)

        # Global average pooling over timesteps
        features = self.norm(out.mean(dim=1))  # (B, d_model)
        features = self.drop(features)

        return self.heads(features, raw_last)


# ============================================================================
# MODEL FACTORY
# ============================================================================

def build_model(name, n_features):
    """
    Instantiate a model by name.

    Parameters
    ----------
    name : str
        One of 'SimpleLSTM', 'BiLSTMAttention', 'CNNLSTM'
    n_features : int
        Number of input features per timestep

    Returns
    -------
    nn.Module
    """
    models = {
        "SimpleLSTM": SimpleLSTM,
        "BiLSTMAttention": BiLSTMAttention,
        "CNNLSTM": CNNLSTM,
        "TransformerBackbone": TransformerBackbone,
    }
    if name not in models:
        raise ValueError(f"Unknown model: {name}. Choose from {list(models.keys())}")
    return models[name](n_features)
