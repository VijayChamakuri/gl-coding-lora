"""Bootstrap intervals, matching TrustGate's line-level resample.

`bootstrap_ci` is the same procedure as `trustgate.realgl.bootstrap_ci`:
2,000 resamples with seed 0, percentile 2.5 and 97.5. Subgroup intervals
resample within the subgroup.
"""

from __future__ import annotations

import numpy as np
from sklearn.metrics import f1_score


def bootstrap_ci(correct: np.ndarray, n: int = 2000, seed: int = 0) -> list[float] | None:
    correct = np.asarray(correct, dtype=float)
    if len(correct) == 0:
        return None
    rng = np.random.default_rng(seed)
    means = correct[rng.integers(0, len(correct), size=(n, len(correct)))].mean(axis=1)
    return [float(np.quantile(means, 0.025)), float(np.quantile(means, 0.975))]


def rank_labels(labels: list[str], means: list[float], sums: list[float], k: int = 3) -> dict:
    """Stable ranking: equal scores keep the incoming label order."""
    order_mean = sorted(range(len(labels)), key=lambda i: (-means[i], i))
    order_sum = sorted(range(len(labels)), key=lambda i: (-sums[i], i))
    return {
        "pred": labels[order_mean[0]],
        "top3": [labels[i] for i in order_mean[:k]],
        "pred_sum": labels[order_sum[0]],
        "top3_sum": [labels[i] for i in order_sum[:k]],
        "mean_logprob": means[order_mean[0]],
        "sum_logprob": sums[order_sum[0]],
    }


def summarize(correct: np.ndarray, top3: np.ndarray, seen: np.ndarray, y=None, pred=None) -> dict:
    correct = np.asarray(correct, dtype=float)
    top3 = np.asarray(top3, dtype=float)
    seen = np.asarray(seen, dtype=bool)
    new = ~seen

    def _rate(mask, values):
        if not mask.any():
            return None
        return float(values[mask].mean())

    out = {
        "n": int(len(correct)),
        "top1": float(correct.mean()) if len(correct) else None,
        "top1_ci": bootstrap_ci(correct),
        "top3": float(top3.mean()) if len(top3) else None,
        "top3_ci": bootstrap_ci(top3),
        "n_seen_merchants": int(seen.sum()),
        "n_new_merchants": int(new.sum()),
        "top1_seen_merchants": _rate(seen, correct),
        "top1_seen_ci": bootstrap_ci(correct[seen]) if seen.any() else None,
        "top1_new_merchants": _rate(new, correct),
        "top1_new_ci": bootstrap_ci(correct[new]) if new.any() else None,
        "top3_seen_merchants": _rate(seen, top3),
        "top3_new_merchants": _rate(new, top3),
        "top3_new_ci": bootstrap_ci(top3[new]) if new.any() else None,
    }
    if y is not None and pred is not None:
        out["macro_f1"] = float(f1_score(y, pred, average="macro", zero_division=0))
    return out
