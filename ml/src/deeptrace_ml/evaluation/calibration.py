"""Temperature scaling, threshold selection and the "uncertain" band — all fitted on val_calib.

Temperature scaling divides every logit by one scalar T > 0. It fixes over/under-confidence without
changing the ranking (ROC-AUC is identical before and after), so it can't "improve accuracy" by
accident — it only makes the probabilities honest.
"""

from __future__ import annotations

from dataclasses import asdict, dataclass
from typing import Any

import numpy as np
from scipy.optimize import minimize_scalar

from deeptrace_ml.evaluation.metrics import expected_calibration_error, sigmoid
from deeptrace_ml.serving.decision import verdict  # noqa: F401  (re-exported for callers)


@dataclass(frozen=True)
class CalibrationResult:
    temperature: float
    threshold: float
    target_fpr: float
    achieved_fpr_on_calib: float
    tpr_on_calib: float
    youden_threshold: float
    uncertainty_delta: float
    abstention_rate_on_calib: float
    errors_captured_by_band: float
    ece_before: float
    ece_after: float
    nll_before: float
    nll_after: float
    n_calib: int

    def to_dict(self) -> dict[str, Any]:
        return asdict(self)


def _nll(logits: np.ndarray, labels: np.ndarray, temperature: float) -> float:
    z = logits / temperature
    # numerically stable binary cross-entropy with logits
    return float(np.mean(np.maximum(z, 0) - z * labels + np.log1p(np.exp(-np.abs(z)))))


def fit_temperature(logits: np.ndarray, labels: np.ndarray) -> float:
    """Minimise NLL over log T (bounded to T in [0.05, 20])."""
    res = minimize_scalar(
        lambda log_t: _nll(logits, labels, float(np.exp(log_t))), bounds=(-3.0, 3.0), method="bounded"
    )
    if not res.success:
        raise RuntimeError(f"Temperature fit failed: {res.message}")
    return float(np.exp(res.x))


def threshold_for_fpr(probs: np.ndarray, labels: np.ndarray, target_fpr: float) -> float:
    """Smallest threshold t such that the share of REAL images with p >= t is <= target_fpr."""
    real = np.sort(probs[labels == 0])[::-1]
    if len(real) == 0:
        raise ValueError("Need real images to set an FPR-based threshold")
    k = int(np.floor(target_fpr * len(real)))  # number of real images we may flag
    if k >= len(real):
        return 0.0
    return float(np.nextafter(real[k], np.inf))


def youden_threshold(probs: np.ndarray, labels: np.ndarray) -> float:
    from sklearn.metrics import roc_curve

    fpr, tpr, thr = roc_curve(labels, probs)
    return float(np.clip(thr[int(np.argmax(tpr - fpr))], 0.0, 1.0))


def choose_uncertainty_delta(
    probs: np.ndarray,
    labels: np.ndarray,
    threshold: float,
    max_abstention: float,
    target_error_capture: float,
) -> tuple[float, float, float]:
    """Smallest delta whose band [t-d, t+d] captures `target_error_capture` of errors while abstaining
    on at most `max_abstention` of images. If the target can't be met, the widest band within the
    abstention budget is used. Returns (delta, abstention_rate, error_capture)."""
    preds = probs >= threshold
    errors = preds != labels.astype(bool)
    distance = np.abs(probs - threshold)
    best = (0.0, 0.0, 0.0)
    for delta in np.round(np.arange(0.0, 0.5001, 0.01), 2):
        band = distance < delta
        abstain = float(band.mean())
        if abstain > max_abstention:
            break
        capture = float((band & errors).sum() / errors.sum()) if errors.any() else 1.0
        best = (float(delta), abstain, capture)
        if capture >= target_error_capture:
            break
    return best


def calibrate(
    logits: np.ndarray,
    labels: np.ndarray,
    target_fpr: float = 0.05,
    max_abstention: float = 0.15,
    target_error_capture: float = 0.5,
) -> CalibrationResult:
    labels = labels.astype(int)
    if len(np.unique(labels)) < 2:
        raise ValueError("Calibration set needs both classes")
    temperature = fit_temperature(logits, labels)
    probs_before, probs = sigmoid(logits), sigmoid(logits / temperature)
    threshold = threshold_for_fpr(probs, labels, target_fpr)
    preds = probs >= threshold
    delta, abstention, capture = choose_uncertainty_delta(
        probs, labels, threshold, max_abstention, target_error_capture
    )
    return CalibrationResult(
        temperature=temperature,
        threshold=threshold,
        target_fpr=target_fpr,
        achieved_fpr_on_calib=float(preds[labels == 0].mean()),
        tpr_on_calib=float(preds[labels == 1].mean()),
        youden_threshold=youden_threshold(probs, labels),
        uncertainty_delta=delta,
        abstention_rate_on_calib=abstention,
        errors_captured_by_band=capture,
        ece_before=expected_calibration_error(labels, probs_before),
        ece_after=expected_calibration_error(labels, probs),
        nll_before=_nll(logits, labels, 1.0),
        nll_after=_nll(logits, labels, temperature),
        n_calib=int(len(labels)),
    )


def apply_calibration(logits: np.ndarray, temperature: float) -> np.ndarray:
    return sigmoid(logits / temperature)
