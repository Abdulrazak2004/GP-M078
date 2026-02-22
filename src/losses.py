"""
Multi-task weighted loss for the Casing RUL Prediction pipeline.

Combines:
  - Huber loss for RUL, Corrosion Rate, Wall Thickness, Forecast (regression)
  - CrossEntropy loss for Corrosion Cause (classification)
  - Configurable per-task weights

Forecast loss ignores NaN targets (samples where look-ahead data is unavailable).
"""

import torch
import torch.nn as nn

from src.config import (
    HUBER_DELTA_RUL, HUBER_DELTA_CR, HUBER_DELTA_WT, HUBER_DELTA_FORECAST,
    LOSS_WEIGHT_RUL, LOSS_WEIGHT_CR, LOSS_WEIGHT_WT,
    LOSS_WEIGHT_CAUSE, LOSS_WEIGHT_FORECAST,
)


class MultiTaskLoss(nn.Module):
    """
    Weighted multi-task loss combining regression and classification objectives.

    Call signature matches model output dict + target tensors from CasingDataset.
    """

    def __init__(self,
                 w_rul=LOSS_WEIGHT_RUL,
                 w_cr=LOSS_WEIGHT_CR,
                 w_wt=LOSS_WEIGHT_WT,
                 w_cause=LOSS_WEIGHT_CAUSE,
                 w_forecast=LOSS_WEIGHT_FORECAST):
        super().__init__()

        self.w_rul = w_rul
        self.w_cr = w_cr
        self.w_wt = w_wt
        self.w_cause = w_cause
        self.w_forecast = w_forecast

        self.huber_rul = nn.HuberLoss(delta=HUBER_DELTA_RUL)
        self.huber_cr = nn.HuberLoss(delta=HUBER_DELTA_CR)
        self.huber_wt = nn.HuberLoss(delta=HUBER_DELTA_WT)
        self.huber_forecast = nn.HuberLoss(delta=HUBER_DELTA_FORECAST, reduction="none")
        self.ce_cause = nn.CrossEntropyLoss()

    def forward(self, preds, y_rul, y_cr, y_wt, y_cause, y_forecast):
        """
        Parameters
        ----------
        preds : dict from model forward()
            Keys: 'rul', 'cr', 'wt', 'cause', 'forecast'
        y_rul, y_cr, y_wt : Tensor (B,)
        y_cause : LongTensor (B,)
        y_forecast : Tensor (B, 60) — may contain NaN

        Returns
        -------
        total_loss : scalar Tensor
        loss_dict : dict of individual losses (detached, for logging)
        """
        loss_rul = self.huber_rul(preds["rul"], y_rul)
        loss_cr = self.huber_cr(preds["cr"], y_cr)
        loss_wt = self.huber_wt(preds["wt"], y_wt)
        loss_cause = self.ce_cause(preds["cause"], y_cause)

        # Forecast loss — replace NaN targets with 0 before loss, then mask
        valid_mask = ~torch.isnan(y_forecast)
        y_forecast_safe = y_forecast.clone()
        y_forecast_safe[~valid_mask] = 0.0
        forecast_loss_raw = self.huber_forecast(preds["forecast"], y_forecast_safe)
        if valid_mask.any():
            loss_forecast = (forecast_loss_raw * valid_mask.float()).sum() / valid_mask.float().sum()
        else:
            loss_forecast = torch.tensor(0.0, device=preds["rul"].device)

        total = (
            self.w_rul * loss_rul
            + self.w_cr * loss_cr
            + self.w_wt * loss_wt
            + self.w_cause * loss_cause
            + self.w_forecast * loss_forecast
        )

        loss_dict = {
            "rul": loss_rul.item(),
            "cr": loss_cr.item(),
            "wt": loss_wt.item(),
            "cause": loss_cause.item(),
            "forecast": loss_forecast.item(),
            "total": total.item(),
        }

        return total, loss_dict
