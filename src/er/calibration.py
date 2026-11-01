"""Calibration and selective prediction.

H3 of the project: a matcher should say how sure it is, and abstain when it is
not. Two questions, two metrics:

* *Are the probabilities honest?* -> :func:`expected_calibration_error` (ECE,
  equal-width or equal-mass bins) plus Brier score and a reliability table for
  the paper figure.
* *What does abstaining cost?* -> :func:`selective_report`, which sweeps the
  review band and reports accuracy on the auto-accepted region, the fraction of
  pairs needing manual review, and the recall lost to review.

Published ER systems mostly report a threshold and stop; the calibration work
(Kamsteeg et al., 2025) reports ECE up to 0.055 with mild overconfidence, so
this module exists to make that measurable rather than asserted.
"""

from __future__ import annotations

import numpy as np
import pandas as pd


def expected_calibration_error(
    probs: np.ndarray, labels: np.ndarray, n_bins: int = 10, strategy: str = "uniform"
) -> float:
    """Mean absolute gap between predicted probability and observed frequency.

    Args:
        strategy: "uniform" (equal-width bins, the classic ECE) or "quantile"
            (equal-mass bins, more stable when scores crowd near 0 or 1 - the
            common case in ER, where most candidates are negatives).
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.size == 0:
        return float("nan")
    if strategy == "quantile":
        edges = np.quantile(probs, np.linspace(0, 1, n_bins + 1))
        edges[0], edges[-1] = -np.inf, np.inf
        edges = np.unique(edges)
    else:
        edges = np.linspace(0, 1, n_bins + 1)

    idx = np.clip(np.digitize(probs, edges[1:-1], right=False), 0, len(edges) - 2)
    total = 0.0
    for b in range(len(edges) - 1):
        m = idx == b
        if not m.any():
            continue
        total += m.sum() / len(probs) * abs(probs[m].mean() - labels[m].mean())
    return float(total)


def brier_score(probs: np.ndarray, labels: np.ndarray) -> float:
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if probs.size == 0:
        return float("nan")
    return float(np.mean((probs - labels) ** 2))


def reliability_table(probs: np.ndarray, labels: np.ndarray, n_bins: int = 10) -> pd.DataFrame:
    """Bin-level counts for the reliability diagram figure."""
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=float)
    edges = np.linspace(0, 1, n_bins + 1)
    rows = []
    for i in range(n_bins):
        m = (probs >= edges[i]) & (probs <= edges[i + 1] if i == n_bins - 1 else probs < edges[i + 1])
        rows.append(
            {
                "bin": f"[{edges[i]:.1f},{edges[i + 1]:.1f})",
                "count": int(m.sum()),
                "mean_prob": float(probs[m].mean()) if m.any() else float("nan"),
                "observed_rate": float(labels[m].mean()) if m.any() else float("nan"),
            }
        )
    return pd.DataFrame(rows)


def temperature_scale(
    logits: np.ndarray, labels: np.ndarray, lo: float = 0.05, hi: float = 20.0, steps: int = 200
) -> float:
    """Fit a single temperature by grid search on NLL.

    Temperature scaling is the cheapest published calibrator and is exactly what
    Kamsteeg et al. found effective for ER matchers, so it is the default here.
    """
    logits = np.asarray(logits, dtype=float)
    labels = np.asarray(labels, dtype=float)
    if logits.size == 0:
        return 1.0
    best_t, best_nll = 1.0, float("inf")
    for t in np.linspace(lo, hi, steps):
        p = 1.0 / (1.0 + np.exp(-logits / t))
        p = np.clip(p, 1e-9, 1 - 1e-9)
        nll = float(-np.mean(labels * np.log(p) + (1 - labels) * np.log(1 - p)))
        if nll < best_nll:
            best_t, best_nll = float(t), nll
    return best_t


def selective_report(
    probs: np.ndarray,
    labels: np.ndarray,
    lower: float = 0.35,
    upper: float = 0.65,
) -> dict[str, float]:
    """Report the three-way decision: accept / review / reject.

    Pairs with probability inside ``[lower, upper]`` go to manual review, so the
    auto-decided region is where errors must be controlled. The key paper number
    is the trade-off: how much accuracy is gained on the auto region, and how
    many pairs the reviewer must look at.

    Returns:
        coverage_auto: fraction decided automatically.
        precision_auto / recall_auto / f1_auto: quality of auto decisions.
        review_share: fraction routed to review.
        review_yield: precision inside the review band (how wasted the reviewer's
            time is; a low value means the band is doing useful work).
        errors_avoided: gold positives that would have been missed by a single
            threshold, recovered by routing them to review instead.
    """
    probs = np.asarray(probs, dtype=float)
    labels = np.asarray(labels, dtype=int)
    if probs.size == 0:
        return {}

    accept = probs >= upper
    review = (probs >= lower) & (probs < upper)
    reject = probs < lower

    tp = int((accept & (labels == 1)).sum())
    fp = int((accept & (labels == 0)).sum())
    fn_auto = int((~accept & ~review & (labels == 1)).sum()) + int((review & (labels == 1)).sum())
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + int((labels == 1).sum())) if (labels == 1).any() else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0

    rv = labels[review]
    return {
        "coverage_auto": float((accept | reject).sum() / len(probs)),
        "review_share": float(review.sum() / len(probs)),
        "precision_auto": p,
        "recall_auto": r,
        "f1_auto": f1,
        "review_yield": float(rv.mean()) if len(rv) else float("nan"),
        "errors_avoided": int((review & (labels == 1)).sum()),
        "tp_auto": tp,
        "fp_auto": fp,
        "fn_explained": fn_auto,
    }
