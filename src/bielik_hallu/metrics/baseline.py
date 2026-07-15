import numpy as np


def first_token_logprob_entropy(logits: np.ndarray) -> float:
    """Entropy of the softmax over the first answer-token logits (natural log)."""
    logits = np.asarray(logits, dtype=np.float64)
    logits = logits - logits.max()
    exp = np.exp(logits)
    p = exp / exp.sum()
    nz = p[p > 0]
    return float(-(nz * np.log(nz)).sum())
