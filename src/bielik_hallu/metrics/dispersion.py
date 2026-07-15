import numpy as np


def normalize_outliers(a: np.ndarray, quantile: float = 0.99) -> np.ndarray:
    """Winsorize: clip absolute values to the given quantile.

    Uses method="nearest" (rather than "lower") so the cap snaps to the
    closest actual order-statistic instead of always rounding down. On
    sparse vectors (e.g. [0, 3, 0, 0]) "lower" can round the threshold
    down to the 0.0 order-statistic and clip the only real value to
    zero, destroying the signal. "nearest" still winsorizes true
    outliers (it caps them at a real, near-quantile magnitude) while
    preserving sparse signal.
    """
    a = np.asarray(a, dtype=np.float64)
    mag = np.abs(a)
    cap = np.quantile(mag, quantile, method='nearest')
    return np.clip(a, -cap, cap)


def _prob_dist(a: np.ndarray) -> np.ndarray:
    a = np.asarray(a, dtype=np.float64)
    sq = a ** 2
    total = sq.sum()
    if total == 0:
        return np.full_like(sq, 1.0 / sq.size)
    return sq / total


def ipr(a: np.ndarray) -> float:
    p = _prob_dist(a)
    return float((p ** 2).sum())


def shannon_entropy(a: np.ndarray) -> float:
    p = _prob_dist(a)
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())
