"""Turning matched pairs into resolved entities (the clustering stage).

The verified literature says this stage matters more than pairwise F1 suggests:
GraLMatch (EDBT 2025) and TransClean (IEEE Access 2025) both show that a few
false-positive *pairwise* edges can corrupt large clusters, and that transitive
consistency is a label-free way to find them. That is exactly Case B: an
"iPhone 15" - "iPhone 15 Plus" edge that is wrong does not just add one FP pair,
it merges two whole clusters.

Two clusterers are provided so the effect is measurable:

* :func:`connected_components` - the standard baseline (union-find over accepted
  edges). Simple, transitive by construction, vulnerable to bridge edges.
* :func:`transitive_cleanup` - iteratively drop edges that conflict with
  transitivity, before clustering. Label-free, so it can be evaluated on test data.
"""

from __future__ import annotations

from collections import defaultdict

import pandas as pd


class _UnionFind:
    def __init__(self, ids) -> None:
        self.parent = {i: i for i in ids}
        self.rank = {i: 0 for i in ids}

    def find(self, x: str) -> str:
        while self.parent[x] != x:
            self.parent[x] = self.parent[self.parent[x]]
            x = self.parent[x]
        return x

    def union(self, a: str, b: str) -> None:
        ra, rb = self.find(a), self.find(b)
        if ra == rb:
            return
        if self.rank[ra] < self.rank[rb]:
            ra, rb = rb, ra
        self.parent[rb] = ra
        if self.rank[ra] == self.rank[rb]:
            self.rank[ra] += 1

    def groups(self) -> list[list[str]]:
        out: dict[str, list[str]] = defaultdict(list)
        for x in list(self.parent):
            out[self.find(x)].append(x)
        return [sorted(v) for v in out.values()]


def connected_components(ids, edges: pd.DataFrame, score_col: str | None = None) -> list[list[str]]:
    """Union-find over accepted edges; singletons become their own cluster."""
    uf = _UnionFind(ids)
    for l, r in zip(edges["record_id_l"].astype(str), edges["record_id_r"].astype(str), strict=True):
        uf.union(l, r)
    return uf.groups()


def transitive_cleanup(
    ids,
    edges: pd.DataFrame,
    score_col: str = "score",
    max_iter: int = 5,
    violation_weight: float = 0.5,
) -> pd.DataFrame:
    """Drop weak edges that create transitivity violations (label-free).

    For a triangle where a-b and a-c are kept but b-c is absent, one of the two
    kept edges is suspect. The weaker one is dropped, but **only if its score is
    below ``violation_weight``**: when both edges are confident, the violation is
    more plausibly a *missing* edge (a true match the matcher never proposed) than
    a false merge, and dropping anyway would destroy recall on a guess. That gate
    is what separates a principled cleanup from arbitrary edge deletion, so it is
    a first-class parameter to ablate rather than an implementation detail.

    The earlier version of this function compared the score against
    ``violation_weight + 1e9``, which made the gate always true and silently
    turned the method into "drop every violable edge". Results computed before the
    fix are therefore not comparable to results after it.

    Returns the surviving edges, with
    ``attrs["transitive_cleanup_dropped"]`` recording the number removed.
    """
    edges = edges.drop_duplicates(subset=["record_id_l", "record_id_r"]).reset_index(drop=True)
    has_score = score_col in edges.columns
    triples = [
        (str(row["record_id_l"]), str(row["record_id_r"]), float(row[score_col]) if has_score else 1.0)
        for _, row in edges.iterrows()
    ]

    dropped: set[int] = set()
    for _ in range(max_iter):
        # Rebuild adjacency from the surviving edges and detect violations
        # against that single snapshot. Dropping edges while iterating both
        # crashed on deleted entries and made the outcome depend on traversal
        # order; batch-per-pass removal is deterministic and reproducible.
        adj: dict[str, dict[str, tuple[float, int]]] = {}
        for idx, (l, r, s) in enumerate(triples):
            if idx in dropped:
                continue
            adj.setdefault(l, {})[r] = (s, idx)
            adj.setdefault(r, {})[l] = (s, idx)

        proposals: list[tuple[float, int]] = []
        for a, nb in adj.items():
            if len(nb) < 2:
                continue
            for b in nb:
                for c in nb:
                    if b >= c:
                        continue
                    if c in adj.get(b, {}):
                        continue  # triangle closed: consistent
                    sb, ib = nb[b]
                    sc, ic = nb[c]
                    sus_score, sus_idx = (sb, ib) if sb <= sc else (sc, ic)
                    if sus_idx in dropped or sus_score >= violation_weight:
                        continue  # confident on both sides: do not guess
                    proposals.append((sus_score, sus_idx))

        if not proposals:
            break
        # deterministic application: weakest edge first, ties broken by edge index
        dropped.update(idx for _, idx in sorted(proposals) if idx not in dropped)

    out = edges[[i not in dropped for i in range(len(edges))]].reset_index(drop=True)
    out.attrs = {"transitive_cleanup_dropped": int(len(dropped))}
    return out


def cluster_stats(clusters: list[list[str]]) -> dict[str, float]:
    """Size distribution of resolved entities - over-merging shows up here."""
    sizes = [len(c) for c in clusters]
    if not sizes:
        return {"n_clusters": 0}
    return {
        "n_clusters": len(sizes),
        "mean_size": float(pd.Series(sizes).mean()),
        "max_size": int(max(sizes)),
        "singletons": int(sum(1 for s in sizes if s == 1)),
        "pct_in_large_clusters": float(sum(s for s in sizes if s > 10) / sum(sizes)),
    }
