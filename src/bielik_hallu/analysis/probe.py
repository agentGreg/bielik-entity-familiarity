import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import StratifiedKFold, cross_val_predict
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.metrics import roc_auc_score


def probe_auroc_per_layer(
    hidden_by_layer: dict[int, np.ndarray],
    labels: np.ndarray,
    n_splits: int = 5,
    seed: int = 0,
) -> dict[int, float]:
    labels = np.asarray(labels)
    minority = int(np.bincount(labels).min()) if len(labels) else 0
    effective_splits = min(n_splits, minority)

    if effective_splits < 2:
        return {layer: float("nan") for layer in hidden_by_layer}

    cv = StratifiedKFold(n_splits=effective_splits, shuffle=True, random_state=seed)
    result: dict[int, float] = {}
    for layer, X in hidden_by_layer.items():
        clf = make_pipeline(StandardScaler(), LogisticRegression(max_iter=1000))
        proba = cross_val_predict(clf, X, labels, cv=cv, method="predict_proba")[:, 1]
        a = roc_auc_score(labels, proba)
        result[layer] = float(max(a, 1.0 - a))
    return result
