"""
Multi-task loss for the Casing RUL Prediction pipeline.

Implements Homoscedastic Task Uncertainty Weighting (Kendall et al., 2018):
    L_total = sum_i (1/(2*sigma_i^2) * L_i + log(sigma_i))

Each task learns its own variance (sigma) dynamically, so the model
automatically balances tasks with different scales (RUL ~500d vs CR ~10mpy).

Forecast loss ignores NaN targets (samples where look-ahead data is unavailable).
"""

import torch
import torch.nn as nn

from src.config import (
    HUBER_DELTA_RUL, HUBER_DELTA_CR, HUBER_DELTA_WT, HUBER_DELTA_FORECAST,
    HUBER_DELTA_PHYSICS, MPY_TO_MMDAY,
)


class MultiTaskLoss(nn.Module):
    """
    Uncertainty-weighted multi-task loss (Kendall et al., 2018).

    Learns a log-variance parameter per task. The loss for task i is:
        1/(2*exp(log_var_i)) * L_i + log_var_i / 2

    This automatically down-weights noisy tasks and up-weights clean ones.
    """

    def __init__(self):
        super().__init__()

        self.huber_rul = nn.HuberLoss(delta=HUBER_DELTA_RUL)
        self.huber_cr = nn.HuberLoss(delta=HUBER_DELTA_CR)
        self.huber_wt = nn.HuberLoss(delta=HUBER_DELTA_WT)
        self.huber_forecast = nn.HuberLoss(delta=HUBER_DELTA_FORECAST, reduction="none")
        self.huber_physics = nn.HuberLoss(delta=HUBER_DELTA_PHYSICS)

        # Learnable log-variance per task (initialized so initial weight ≈ 1.0)
        # log_var = 0 → sigma^2 = 1 → weight = 1/(2*1) = 0.5
        self.log_var_rul = nn.Parameter(torch.zeros(1))
        self.log_var_cr = nn.Parameter(torch.zeros(1))
        self.log_var_wt = nn.Parameter(torch.zeros(1))
        self.log_var_forecast = nn.Parameter(torch.zeros(1))
        self.log_var_physics = nn.Parameter(torch.zeros(1))

    def forward(self, preds, y_rul, y_cr, y_wt, y_forecast, y_wt_prev=None):
        """
        Parameters
        ----------
        preds : dict from model forward()
            Keys: 'rul', 'cr', 'wt', 'forecast'
        y_rul, y_cr, y_wt : Tensor (B,)
        y_forecast : Tensor (B, 60) — may contain NaN
        y_wt_prev : Tensor (B,), optional — wall thickness at t-1 for physics constraint

        Returns
        -------
        total_loss : scalar Tensor
        loss_dict : dict of individual losses (detached, for logging)
        """
        loss_rul = self.huber_rul(preds["rul"], y_rul)
        loss_cr = self.huber_cr(preds["cr"], y_cr)
        loss_wt = self.huber_wt(preds["wt"], y_wt)

        # Forecast loss — replace NaN targets with 0 before loss, then mask
        valid_mask = ~torch.isnan(y_forecast)
        y_forecast_safe = y_forecast.clone()
        y_forecast_safe[~valid_mask] = 0.0
        forecast_loss_raw = self.huber_forecast(preds["forecast"], y_forecast_safe)
        if valid_mask.any():
            loss_forecast = (forecast_loss_raw * valid_mask.float()).sum() / valid_mask.float().sum()
        else:
            loss_forecast = torch.tensor(0.0, device=preds["rul"].device)

        # Physics constraint: WT(t) ≈ WT(t-1) - CR(t) * MPY_TO_MMDAY
        if y_wt_prev is not None:
            wt_expected = y_wt_prev - preds["cr"] * MPY_TO_MMDAY
            loss_physics = self.huber_physics(preds["wt"], wt_expected)
        else:
            loss_physics = torch.tensor(0.0, device=preds["rul"].device)

        # Kendall uncertainty weighting: 1/(2*sigma^2) * L + log(sigma)
        # Using log_var = log(sigma^2), so sigma^2 = exp(log_var)
        # Weight = 1/(2*exp(log_var)), regularizer = log_var/2
        total = (
            torch.exp(-self.log_var_rul) * loss_rul + self.log_var_rul / 2
            + torch.exp(-self.log_var_cr) * loss_cr + self.log_var_cr / 2
            + torch.exp(-self.log_var_wt) * loss_wt + self.log_var_wt / 2
            + torch.exp(-self.log_var_forecast) * loss_forecast + self.log_var_forecast / 2
            + torch.exp(-self.log_var_physics) * loss_physics + self.log_var_physics / 2
        )

        loss_dict = {
            "rul": loss_rul.item(),
            "cr": loss_cr.item(),
            "wt": loss_wt.item(),
            "forecast": loss_forecast.item(),
            "physics": loss_physics.item(),
            "total": total.item(),
            # Log learned weights for monitoring
            "w_rul": torch.exp(-self.log_var_rul).item(),
            "w_cr": torch.exp(-self.log_var_cr).item(),
            "w_wt": torch.exp(-self.log_var_wt).item(),
            "w_forecast": torch.exp(-self.log_var_forecast).item(),
            "w_physics": torch.exp(-self.log_var_physics).item(),
        }

        return total, loss_dict
