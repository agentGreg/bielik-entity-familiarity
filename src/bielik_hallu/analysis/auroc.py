import numpy as np
from sklearn.metrics import roc_auc_score


def auroc(scores: np.ndarray, labels: np.ndarray) -> float:
    if len(np.unique(labels)) < 2:
        return float("nan")
    a = roc_auc_score(labels, scores)
    return float(max(a, 1.0 - a))


def auroc_per_layer(metric_by_layer: dict[int, np.ndarray], labels: np.ndarray) -> dict[int, float]:
    return {layer: auroc(scores, labels) for layer, scores in metric_by_layer.items()}
