"""Serialization + math for the calibrated hallucination-risk probe.

This module is deliberately dependency-light and numpy-only at inference time
so the trained probe can be loaded and applied without scikit-learn:

- a standardizer (per-feature mean/std),
- a linear logistic probe (weights + bias) over the standardized features,
- a Platt / sigmoid calibrator (scale ``A`` + shift ``B``) mapping the raw
  probe log-odds to a calibrated hallucination-risk probability.

The forward path is:

    z          = (x - mean) / std                 # standardize
    raw_logit  = z @ weights + bias               # probe decision value
    raw_prob   = sigmoid(raw_logit)               # uncalibrated probe score
    cal_logit  = A * raw_logit + B                # Platt scaling
    p_risk     = sigmoid(cal_logit)               # calibrated risk

Everything needed to reproduce this is stored in a single ``.npz`` via
:func:`save_probe` and read back by :func:`load_probe` with
``allow_pickle=False`` (metadata is JSON-encoded into a 0-d unicode array).

The tool predicts *knowledge-absence risk* (KNOWN=0 vs UNKNOWN_REAL/FABRICATED=1),
not lie detection; see ``docs/risk-tool.md`` and the training script for rationale.
"""
from __future__ import annotations

import json
from dataclasses import dataclass, field
from pathlib import Path

import numpy as np

# Qualitative bands for the calibrated risk probability. Left-closed intervals:
#   LOW    : p < 0.25
#   MEDIUM : 0.25 <= p < 0.60
#   HIGH   : p >= 0.60
BAND_LOW_MAX = 0.25
BAND_MEDIUM_MAX = 0.60


def sigmoid(x: np.ndarray | float) -> np.ndarray | float:
    """Numerically stable logistic sigmoid."""
    x = np.asarray(x, dtype=np.float64)
    # np.where evaluates both branches; clip each to its valid domain so the
    # unused branch never overflows, then select. Result is exact for the
    # selected branch.
    with np.errstate(over="ignore", invalid="ignore"):
        pos = 1.0 / (1.0 + np.exp(-np.clip(x, 0, None)))
        neg = np.exp(np.clip(x, None, 0)) / (1.0 + np.exp(np.clip(x, None, 0)))
        out = np.where(x >= 0, pos, neg)
    return float(out) if out.ndim == 0 else out


def risk_band(p: float) -> str:
    """Map a calibrated risk probability to LOW / MEDIUM / HIGH."""
    if p < BAND_LOW_MAX:
        return "LOW"
    if p < BAND_MEDIUM_MAX:
        return "MEDIUM"
    return "HIGH"


@dataclass
class RiskProbe:
    """A trained + calibrated hallucination-risk probe (numpy-only inference)."""

    weights: np.ndarray          # (hidden,) probe coefficients on standardized features
    bias: float                  # probe intercept
    scaler_mean: np.ndarray      # (hidden,) feature means
    scaler_std: np.ndarray       # (hidden,) feature stds (never zero; see standardize)
    calibrator_a: float          # Platt scale applied to the raw probe logit
    calibrator_b: float          # Platt shift
    layer: int                   # residual-stream layer index used (prompt point)
    metadata: dict = field(default_factory=dict)

    # ---- forward path -------------------------------------------------------
    def standardize(self, x: np.ndarray) -> np.ndarray:
        return (np.asarray(x, dtype=np.float64) - self.scaler_mean) / self.scaler_std

    def raw_logit(self, x: np.ndarray) -> np.ndarray | float:
        z = self.standardize(x)
        logit = z @ self.weights + self.bias
        return float(logit) if np.ndim(logit) == 0 else logit

    def raw_score(self, x: np.ndarray) -> np.ndarray | float:
        """Uncalibrated probe probability, sigmoid(raw_logit)."""
        return sigmoid(self.raw_logit(x))

    def predict_proba(self, x: np.ndarray) -> np.ndarray | float:
        """Calibrated P(hallucination-risk) for one feature vector or a batch."""
        cal_logit = self.calibrator_a * np.asarray(self.raw_logit(x)) + self.calibrator_b
        return sigmoid(cal_logit)

    def band(self, x: np.ndarray) -> str:
        return risk_band(float(self.predict_proba(x)))

    # ---- serialization ------------------------------------------------------
    def save(self, path: str | Path) -> Path:
        return save_probe(self, path)


def save_probe(probe: RiskProbe, path: str | Path) -> Path:
    """Serialize a :class:`RiskProbe` to ``path`` (numpy ``.npz``, pickle-free)."""
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        path,
        weights=np.asarray(probe.weights, dtype=np.float64),
        bias=np.asarray(probe.bias, dtype=np.float64),
        scaler_mean=np.asarray(probe.scaler_mean, dtype=np.float64),
        scaler_std=np.asarray(probe.scaler_std, dtype=np.float64),
        calibrator_a=np.asarray(probe.calibrator_a, dtype=np.float64),
        calibrator_b=np.asarray(probe.calibrator_b, dtype=np.float64),
        layer=np.asarray(probe.layer, dtype=np.int64),
        # JSON string in a 0-d unicode array -> loads with allow_pickle=False.
        metadata=np.asarray(json.dumps(probe.metadata)),
    )
    # np.savez appends .npz if missing; normalize the returned path to match.
    if path.suffix != ".npz":
        path = path.with_suffix(path.suffix + ".npz") if path.suffix else path.with_suffix(".npz")
    return path


def load_probe(path: str | Path) -> RiskProbe:
    """Load a :class:`RiskProbe` written by :func:`save_probe`."""
    path = Path(path)
    with np.load(path, allow_pickle=False) as d:
        meta_raw = str(d["metadata"])
        return RiskProbe(
            weights=d["weights"].astype(np.float64),
            bias=float(d["bias"]),
            scaler_mean=d["scaler_mean"].astype(np.float64),
            scaler_std=d["scaler_std"].astype(np.float64),
            calibrator_a=float(d["calibrator_a"]),
            calibrator_b=float(d["calibrator_b"]),
            layer=int(d["layer"]),
            metadata=json.loads(meta_raw) if meta_raw else {},
        )


# ---------------------------------------------------------------------------
# Calibration helpers (used at train time; kept here so tests can exercise them
# without importing scikit-learn).
# ---------------------------------------------------------------------------
def brier_score(p: np.ndarray, y: np.ndarray) -> float:
    """Mean squared error between predicted probabilities and 0/1 labels."""
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    return float(np.mean((p - y) ** 2))


def reliability_bins(p: np.ndarray, y: np.ndarray, n_bins: int = 5) -> list[dict]:
    """Equal-width reliability table over [0, 1].

    Returns one dict per non-empty bin with the bin range, count, mean predicted
    probability, and empirical positive rate.
    """
    p = np.asarray(p, dtype=np.float64)
    y = np.asarray(y, dtype=np.float64)
    edges = np.linspace(0.0, 1.0, n_bins + 1)
    out: list[dict] = []
    for i in range(n_bins):
        lo, hi = edges[i], edges[i + 1]
        mask = (p >= lo) & (p < hi) if i < n_bins - 1 else (p >= lo) & (p <= hi)
        count = int(mask.sum())
        if count == 0:
            continue
        out.append(
            {
                "bin": [float(lo), float(hi)],
                "count": count,
                "mean_pred": float(p[mask].mean()),
                "empirical_rate": float(y[mask].mean()),
            }
        )
    return out
