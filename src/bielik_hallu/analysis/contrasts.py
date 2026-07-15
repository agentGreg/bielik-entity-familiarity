import numpy as np


def condition_contrast_labels(conditions, positive: str, negative: str):
    """Return (mask, labels) selecting only `positive`/`negative` conditions;
    labels=1 for positive, 0 for negative. `conditions` is a 1D array/sequence of strings."""
    conditions = np.asarray(conditions)
    mask = np.isin(conditions, [positive, negative])
    labels = (conditions[mask] == positive).astype(int)
    return mask, labels
