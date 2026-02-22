"""
Corrosion Failure Index (CFI) — post-processing formula.

Combines model outputs into a single 0-100 risk score for field engineers.

CFI = 0.35 * WT_score + 0.25 * CR_score + 0.25 * RUL_score + 0.15 * Cause_score

Thresholds:
  0-25  : Green  (Safe)     — standard monitoring
  25-50 : Yellow (Watch)    — increased monitoring
  50-75 : Orange (Elevated) — inspect within 60 days
  75-100: Red    (Critical) — immediate inspection
"""

import numpy as np

from src.config import (
    CFI_WEIGHT_WT, CFI_WEIGHT_CR, CFI_WEIGHT_RUL, CFI_WEIGHT_CAUSE,
    CFI_CAUSE_SEVERITY,
    CFI_GREEN, CFI_YELLOW, CFI_ORANGE,
)


def nace_severity_scale(cr_mpy):
    """
    Map corrosion rate (mpy) to 0-100 severity score per NACE SP0775.

    < 1 mpy   → 0-20   (low)
    1-5 mpy   → 20-50  (moderate)
    5-10 mpy  → 50-75  (high)
    > 10 mpy  → 75-100 (severe)
    """
    cr = np.asarray(cr_mpy, dtype=np.float64)
    score = np.where(
        cr < 1,
        cr * 20.0,
        np.where(
            cr < 5,
            20.0 + (cr - 1.0) * 7.5,
            np.where(
                cr < 10,
                50.0 + (cr - 5.0) * 5.0,
                np.minimum(75.0 + (cr - 10.0) * 2.5, 100.0)
            )
        )
    )
    return score


def compute_cfi(rul_days, corr_rate_mpy, thickness_loss_pct, cause_probs):
    """
    Compute the Corrosion Failure Index (0-100).

    Parameters
    ----------
    rul_days : float or array
        Predicted remaining useful life in days.
    corr_rate_mpy : float or array
        Predicted corrosion rate in mils per year.
    thickness_loss_pct : float or array
        Percentage of wall thickness lost (0-100).
    cause_probs : array-like of shape (..., 6)
        Predicted probability distribution over 6 corrosion causes.

    Returns
    -------
    cfi : float or array
        Corrosion Failure Index (0-100).
    """
    rul = np.asarray(rul_days, dtype=np.float64)
    cr = np.asarray(corr_rate_mpy, dtype=np.float64)
    tl = np.asarray(thickness_loss_pct, dtype=np.float64)
    probs = np.asarray(cause_probs, dtype=np.float64)

    # Wall Thickness Loss Score (0-100): 50% loss = 100
    wt_score = np.minimum(tl / 50.0 * 100.0, 100.0)

    # Corrosion Rate Score (0-100): NACE SP0775 scale
    cr_score = nace_severity_scale(cr)

    # Remaining Life Score (0-100): 0 days = 100, 5+ years = 0
    rul_score = np.maximum(0.0, 100.0 - (rul / (5.0 * 365.0)) * 100.0)

    # Cause Severity Score (0-100): weighted by predicted class probabilities
    severity = np.array(CFI_CAUSE_SEVERITY, dtype=np.float64)
    cause_score = (probs * severity).sum(axis=-1)

    # Final CFI
    cfi = (
        CFI_WEIGHT_WT * wt_score
        + CFI_WEIGHT_CR * cr_score
        + CFI_WEIGHT_RUL * rul_score
        + CFI_WEIGHT_CAUSE * cause_score
    )

    return np.round(np.clip(cfi, 0.0, 100.0), 1)


def cfi_label(cfi_value):
    """Return the risk category label for a CFI value."""
    cfi = float(cfi_value)
    if cfi <= CFI_GREEN:
        return "Green (Safe)"
    elif cfi <= CFI_YELLOW:
        return "Yellow (Watch)"
    elif cfi <= CFI_ORANGE:
        return "Orange (Elevated)"
    else:
        return "Red (Critical)"
