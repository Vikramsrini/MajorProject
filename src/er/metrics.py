"""Evaluation metrics for blocking, matching, and clustering.

Standard ER reporting is three-stage, and each stage needs its own metric:

1. blocking   -> :func:`blocking_quality` (TPR / recall, PPC, reduction ratio)
2. matching   -> :func:`pairwise_metrics` (micro P/R/F1, AUC-PR, Recall@K)
3. clustering -> :func:`bcubed` and :func:`entity_pairwise_f1`

Pair-level micro F1 is what the WDC/DL-Boat papers report, so it is the primary
number. Entity-level (BCubed) is reported alongside because a paper claim about
"duplicate resolution" is a clustering claim; the two can disagree, and when they
do, the reviewer will notice.
"""

from __future__ import annotations

import numpy as np
import pandas as pd

from .schema import PairTable, _key


# --------------------------------------------------------------------------- #
# blocking
# --------------------------------------------------------------------------- #
def blocking_quality(blocks_pairs: pd.DataFrame, gold_pairs: set[tuple[str, str]]) -> dict[str, float]:
    """Measure how many gold pairs a blocking stage preserved.

    Returns:
        tpr (a.k.a. pair recall): fraction of gold pairs that are candidates.
        ppc (pairs per comparison): fraction of candidates that are gold.
        n_candidates, recall_bound: the bound this stage puts on matching recall.
    """
    cand = {
        _key(str(l), str(r))
        for l, r in zip(blocks_pairs["record_id_l"], blocks_pairs["record_id_r"], strict=True)
    }
    n_gold = len(gold_pairs)
    hits = len(gold_pairs & cand)
    return {
        "tpr": hits / n_gold if n_gold else float("nan"),
        "ppc": hits / len(cand) if cand else float("nan"),
        "n_candidates": len(cand),
        "n_gold_pairs": n_gold,
        "recall_bound": hits / n_gold if n_gold else float("nan"),
    }


def gold_pairs_from_entities(entities: dict[str, list[str]]) -> set[tuple[str, str]]:
    """All pairs co-clustered by a gold entity id."""
    out: set[tuple[str, str]] = set()
    for members in entities.values():
        m = sorted(members)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                out.add((m[i], m[j]))
    return out


# --------------------------------------------------------------------------- #
# matching
# --------------------------------------------------------------------------- #
def _prf(tp: int, fp: int, fn: int) -> tuple[float, float, float]:
    p = tp / (tp + fp) if (tp + fp) else 0.0
    r = tp / (tp + fn) if (tp + fn) else 0.0
    f1 = 2 * p * r / (p + r) if (p + r) else 0.0
    return p, r, f1


def pairwise_metrics(pt: PairTable, threshold: float | None = None) -> dict[str, float]:
    """Threshold metrics, and threshold-free metrics when scores are present.

    Args:
        pt: pair table with ``label`` and (optionally) ``score``.
        threshold: if given, predict match when score >= threshold. If omitted
            and a ``pred`` column exists, that is used instead.
    """
    df = pt.labeled.frame
    if df.empty:
        return {"precision": 0.0, "recall": 0.0, "f1": 0.0, "tp": 0, "fp": 0, "fn": 0}

    y = df["label"].astype(int).to_numpy()
    if threshold is not None:
        if "score" not in df.columns:
            raise ValueError("threshold given but PairTable has no 'score' column")
        pred = (df["score"].astype(float).to_numpy() >= threshold).astype(int)
    elif "pred" in df.columns:
        pred = df["pred"].astype(int).to_numpy()
    else:
        raise ValueError("need a 'score' column (with threshold) or a 'pred' column")

    tp = int(((pred == 1) & (y == 1)).sum())
    fp = int(((pred == 1) & (y == 0)).sum())
    fn = int(((pred == 0) & (y == 1)).sum())
    tn = int(((pred == 0) & (y == 0)).sum())
    p, r, f1 = _prf(tp, fp, fn)

    out = {
        "precision": p,
        "recall": r,
        "f1": f1,
        "tp": tp,
        "fp": fp,
        "fn": fn,
        "tn": tn,
        "n_labeled": int(len(df)),
        "positives": int(y.sum()),
        "predicted_positives": int(pred.sum()),
    }

    if "score" in df.columns and len(np.unique(y)) == 2:
        out["auc_pr"] = _average_precision(y, df["score"].astype(float).to_numpy())
        out["roc_auc"] = _roc_auc(y, df["score"].astype(float).to_numpy())
    return out


def _average_precision(y: np.ndarray, s: np.ndarray) -> float:
    """Interpolated-free AP: sum of precision at each recall step (sklearn-free)."""
    order = np.argsort(-s, kind="mergesort")
    y_sorted = y[order]
    tp = np.cumsum(y_sorted)
    fp = np.cumsum(1 - y_sorted)
    recall = tp / max(1, y.sum())
    precision = tp / np.maximum(1, tp + fp)
    # AP = sum over cuts of (R_n - R_{n-1}) * P_n
    prev = np.concatenate(([0.0], recall[:-1]))
    return float(((recall - prev) * precision).sum())


def _roc_auc(y: np.ndarray, s: np.ndarray) -> float:
    order = np.argsort(s, kind="mergesort")
    y_sorted = y[order]
    ranks = np.empty(len(s), dtype=float)
    # average ranks for ties
    i = 0
    while i < len(s):
        j = i
        while j + 1 < len(s) and s[order[j + 1]] == s[order[i]]:
            j += 1
        ranks[order[i : j + 1]] = (i + j) / 2.0 + 1
        i = j + 1
    n_pos = int(y.sum())
    n_neg = len(y) - n_pos
    if n_pos == 0 or n_neg == 0:
        return float("nan")
    return float((ranks[y == 1].sum() - n_pos * (n_pos + 1) / 2) / (n_pos * n_neg))


def precision_at_recall(y: np.ndarray, s: np.ndarray, target_recall: float = 0.5) -> float:
    """Best precision achievable at >= ``target_recall``.

    This is the comparison that survives threshold re-optimisation. When two
    systems are each evaluated at their own F1-optimal threshold, a constraint
    that removes false positives looks neutral, because the system simply lowers
    its threshold and reads similar noise back in. Fixing recall and comparing
    precision removes that confound, which is why ER papers report PR curves.
    """
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    order = np.argsort(-s, kind="mergesort")
    ys = y[order]
    ss = s[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    recall = tp / n_pos
    precision = tp / np.maximum(1, tp + fp)
    ok = np.where(recall >= target_recall)[0]
    if len(ok) == 0:
        return float("nan")
    # the cut with the highest score threshold that still meets the recall
    i = ok[0]
    thr = ss[i]
    # ties at the same score must be resolved identically, else precision depends
    # on arbitrary ordering; recompute with a >= threshold cut.
    pred = s >= thr
    tp_t = int((pred & (y == 1)).sum())
    fp_t = int((pred & (y == 0)).sum())
    return tp_t / (tp_t + fp_t) if (tp_t + fp_t) else 0.0


def recall_at_threshold(y: np.ndarray, s: np.ndarray, thr: float) -> float:
    y = np.asarray(y, dtype=int)
    s = np.asarray(s, dtype=float)
    n_pos = int(y.sum())
    if n_pos == 0:
        return float("nan")
    return float((s >= thr)[y == 1].sum() / n_pos)


def best_threshold(pt: PairTable, metric: str = "f1") -> dict[str, float]:
    """Sweep all observed scores and return the threshold maximizing ``metric``.

    Used for the "oracle threshold" row in tables: it shows how much of a
    system's weakness is threshold choice rather than ranking quality, which is a
    legitimate and cheap ablation.
    """
    df = pt.labeled.frame
    if df.empty or "score" not in df.columns:
        return {"threshold": float("nan"), metric: float("nan")}
    scores = np.unique(df["score"].astype(float).to_numpy())
    cands = np.concatenate(([0.0], scores, [scores.max() + 1e-6 if len(scores) else 1.0]))
    best = (float("-inf"), float("nan"))
    for t in cands:
        m = pairwise_metrics(pt, threshold=float(t))
        if m[metric] > best[0]:
            best = (m[metric], float(t))
    return {"threshold": best[1], metric: best[0]}


def recall_at_k(
    pt: PairTable, k: int, entity_of: dict[str, str] | None = None
) -> float:
    """Fraction of gold pairs whose partner appears in a record's top-k by score.

    This is the retrieval metric that bounds everything downstream: if Recall@K is
    0.85 the matcher can never exceed 0.85 recall on that candidate set.
    """
    df = pt.frame
    if "score" not in df.columns or df.empty:
        return float("nan")

    # Build per-record neighbour lists from the candidate set (both directions).
    neighbour_lists: dict[str, list[tuple[float, str]]] = {}
    for l, r, s in zip(df["record_id_l"], df["record_id_r"], df["score"], strict=True):
        neighbour_lists.setdefault(str(l), []).append((float(s), str(r)))
        neighbour_lists.setdefault(str(r), []).append((float(s), str(l)))

    if entity_of is None and "label" in df.columns:
        # Fall back to pair-level Recall@K using labels as gold.
        top = {
            rid: {o for _, o in sorted(v, key=lambda x: -x[0])[:k]}
            for rid, v in neighbour_lists.items()
        }
        pos = df[df["label"] == 1]
        if pos.empty:
            return float("nan")
        hit = sum(
            1 for l, r in zip(pos["record_id_l"], pos["record_id_r"], strict=True)
            if r in top.get(str(l), set())
        )
        return hit / len(pos)

    top = {
        rid: {o for _, o in sorted(v, key=lambda x: -x[0])[:k]} for rid, v in neighbour_lists.items()
    }
    gold = {(rid, other) for rid, other in _entity_pairs(entity_of or {}) }
    if not gold:
        return float("nan")
    hit = sum(1 for rid, other in gold if other in top.get(rid, set()) or rid in top.get(other, set()))
    return hit / len(gold)


def _entity_pairs(entity_of: dict[str, str]) -> set[tuple[str, str]]:
    groups: dict[str, list[str]] = {}
    for rid, ent in entity_of.items():
        groups.setdefault(ent, []).append(rid)
    out: set[tuple[str, str]] = set()
    for members in groups.values():
        m = sorted(members)
        for i in range(len(m)):
            for j in range(i + 1, len(m)):
                out.add((m[i], m[j]))
    return out


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def bcubed(
    predicted_clusters: list[list[str]], gold_clusters: list[list[str]]
) -> dict[str, float]:
    """BCubed precision/recall/F1, the standard entity-level ER measure.

    Computed on the co-membership relation, so it is comparable to the pairwise
    numbers while describing a different (harder) property: transitive closure
    errors made by merging.
    """
    id2pred: dict[str, set[str]] = {}
    for c in predicted_clusters:
        s = set(c)
        for rid in s:
            id2pred[rid] = s
    id2gold: dict[str, set[str]] = {}
    for c in gold_clusters:
        s = set(c)
        for rid in s:
            id2gold[rid] = s

    ids = set(id2gold) | set(id2pred)
    if not ids:
        return {"bcubed_p": 0.0, "bcubed_r": 0.0, "bcubed_f1": 0.0}

    prec = rec = 0.0
    for rid in ids:
        p, g = id2pred.get(rid, {rid}), id2gold.get(rid, {rid})
        inter = len(p & g)
        prec += inter / len(p)
        rec += inter / len(g)
    # records that exist in gold but not predicted count as recall 0 / precision 0
    n = len(ids)
    P, R = prec / n, rec / n
    F = 2 * P * R / (P + R) if (P + R) else 0.0
    return {"bcubed_p": P, "bcubed_r": R, "bcubed_f1": F}


def entity_pairwise_f1(predicted_clusters: list[list[str]], gold_clusters: list[list[str]]) -> dict[str, float]:
    """Pairwise P/R/F1 over the transitive closures of predicted clusters.

    This is the metric that exposes over-merging: one wrong merge can add many
    false-positive pairs, which is exactly the failure Case B is about.
    """
    def pairs(clusters: list[list[str]]) -> set[tuple[str, str]]:
        out: set[tuple[str, str]] = set()
        for c in clusters:
            m = sorted(set(c))
            for i in range(len(m)):
                for j in range(i + 1, len(m)):
                    out.add((m[i], m[j]))
        return out

    pred, gold = pairs(predicted_clusters), pairs(gold_clusters)
    tp = len(pred & gold)
    p, r, f1 = _prf(tp, len(pred) - tp, len(gold) - tp)
    return {"entity_precision": p, "entity_recall": r, "entity_f1": f1}
