"""Candidate generation (blocking).

Every matcher in this project runs on candidates produced here. Because a paper
must report *what the blocking stage threw away*, each method returns a
:class:`~er.schema.Blocks` with its own statistics, and
:func:`er.metrics.blocking_quality` turns gold labels into TPR (pair recall) and
reduction ratio. A matcher can never recall more positives than blocking passed
to it, so this stage bounds the whole system.

Implemented methods, from exact to approximate:

* :func:`all_pairs_block`          - quadratic reference, small datasets only.
* :func:`token_block`              - inverted-index on tokens (collection blocking).
* :func:`prefix_block`             - token-prefix filtering (PPJoin-style, simplified).
* :func:`minhash_block`            - LSH banding on token shingles; Jaccard-threshold
                                     approximation with sub-linear cost.
* :func:`faiss_block`              - ANN over dense text embeddings.
* :func:`sorted_neighborhood_block`- sliding window over a sorted key.
"""

from __future__ import annotations

import hashlib
import itertools
from collections import defaultdict
from collections.abc import Callable, Sequence

import numpy as np
import pandas as pd

from .preprocess import normalize
from .schema import Blocks, Dataset, pair_frame


def _dataset_tokens(ds: Dataset, lowercase_only: bool = False) -> list[set[str]]:
    texts = ds.df["text"].astype(str).tolist()
    if lowercase_only:
        return [{t.lower()} for t in texts]
    return [set(normalize(t).split()) for t in texts]


def _finalize(pairs: list[tuple[str, str]], ids: Sequence[str], method: str, extra: dict | None = None) -> Blocks:
    n = len(ids)
    total = n * (n - 1) // 2
    frame = pair_frame(pairs)
    # A candidate stage must never emit duplicates or self-links; enforcing this
    # here keeps every downstream metric well-defined.
    frame = frame.drop_duplicates().astype(str)
    frame = frame[frame["record_id_l"] != frame["record_id_r"]]
    stats = {
        "method": method,
        "n_records": n,
        "all_pairs": total,
        "n_candidates": int(len(frame)),
        "reduction_ratio": float(1 - len(frame) / total) if total else 0.0,
    }
    stats.update(extra or {})
    return Blocks(frame, method, stats)


def all_pairs_block(ds: Dataset, cap: int = 200_000) -> Blocks:
    """Every unordered pair. Reference point for reduction-ratio reporting."""
    ids = ds.record_ids
    total = len(ids) * (len(ids) - 1) // 2
    if total > cap:
        raise ValueError(
            f"all_pairs would generate {total} pairs (> cap {cap}); use a blocking method"
        )
    pairs = list(itertools.combinations(ids, 2))
    return _finalize(pairs, ids, "all_pairs")


def token_block(
    ds: Dataset,
    token_fn: Callable[[str], set[str]] | None = None,
    max_block_size: int = 100,
    min_tokens: int = 1,
) -> Blocks:
    """Inverted index over normalized tokens (Easton et al. / collection blocking).

    Records sharing a token become candidates. Blocks larger than
    ``max_block_size`` are skipped: they are dominated by very common tokens
    ("with", "for") whose pairs are almost all negative, and skipping them is
    what keeps the quadratic term bounded.

    Args:
        token_fn: overrides normalization (e.g. pass a brand-only tokenizer).
        max_block_size: skip tokens appearing in more records than this.
        min_tokens: records with fewer tokens are not indexed at all.
    """
    ids = ds.record_ids
    token_fn = token_fn or (lambda t: set(normalize(t).split()))
    rows = ds.df["text"].astype(str).tolist()

    index: dict[str, list[int]] = defaultdict(list)
    for i, text in enumerate(rows):
        toks = token_fn(text)
        if len(toks) < min_tokens:
            continue
        for tok in toks:
            index[tok].append(i)

    pairs: set[tuple[str, str]] = set()
    oversized = 0
    for postings in index.values():
        if len(postings) > max_block_size:
            oversized += 1
            continue
        for a, b in itertools.combinations(sorted(postings), 2):
            pairs.add((ids[a], ids[b]))

    return _finalize(
        sorted(pairs), ids, "token_block", {"max_block_size": max_block_size, "skipped_tokens": oversized}
    )


def prefix_block(ds: Dataset, prefix_len: int = 4, max_block_size: int = 150) -> Blocks:
    """Prefix-filter blocking: only compare records whose token prefixes overlap.

    Tokens are ordered by global document frequency so the prefix is the rare
    part of a title; two records describing the same product almost always share
    a rare token in the same position band. This is a simplified PPJoin - enough
    for high recall at low cost, without the length-filter guarantee.
    """
    ids = ds.record_ids
    tok_sets = _dataset_tokens(ds)

    dfreq: dict[str, int] = defaultdict(int)
    for toks in tok_sets:
        for t in toks:
            dfreq[t] += 1

    prefixes: dict[str, list[int]] = defaultdict(list)
    for i, toks in enumerate(tok_sets):
        if not toks:
            continue
        ordered = sorted(toks, key=lambda t: (dfreq[t], t))
        for tok in ordered[:prefix_len]:
            prefixes[tok].append(i)

    pairs: set[tuple[str, str]] = set()
    oversized = 0
    for postings in prefixes.values():
        if len(postings) > max_block_size:
            oversized += 1
            continue
        for a, b in itertools.combinations(sorted(postings), 2):
            pairs.add((ids[a], ids[b]))

    return _finalize(
        sorted(pairs), ids, "prefix_block", {"prefix_len": prefix_len, "skipped_tokens": oversized}
    )


def sorted_neighborhood_block(
    ds: Dataset, key_fn: Callable[[str], str] | None = None, window: int = 30
) -> Blocks:
    """Slide a fixed window over records sorted by a locality-sensitive key.

    Cost is O(n * window) instead of O(n^2). Recall depends on the key; the
    default is the rarest token, which groups by brand/model-family.
    """
    ids = ds.record_ids
    tok_sets = _dataset_tokens(ds)

    dfreq: dict[str, int] = defaultdict(int)
    for toks in tok_sets:
        for t in toks:
            dfreq[t] += 1

    if key_fn is None:

        def key_fn(text: str) -> str:  # noqa: D103
            toks = normalize(text).split()
            return min(toks, key=lambda t: (dfreq[t], t)) if toks else ""

    keys = [key_fn(t) for t in ds.df["text"].astype(str)]
    order = sorted(range(len(ids)), key=lambda i: (keys[i], ids[i]))

    pairs: set[tuple[str, str]] = set()
    for pos, i in enumerate(order):
        for j in order[pos + 1 : pos + window]:
            pairs.add((ids[i], ids[j]) if ids[i] <= ids[j] else (ids[j], ids[i]))

    return _finalize(sorted(pairs), ids, "sorted_neighborhood", {"window": window})


def _stable_hash32(key: str | bytes) -> int:
    """Deterministic 32-bit hash of a token or a MinHash band.

    Python's built-in ``hash()`` is salted per process, which would make MinHash
    signatures (and therefore blocking results) irreproducible between runs. A
    fixed cryptographic digest is used instead so every experiment run is exact.
    """
    raw = key.encode() if isinstance(key, str) else bytes(key)
    return int.from_bytes(hashlib.blake2b(raw, digest_size=4).digest(), "little")


def _minhash_signatures(sets: Sequence[set[str]], num_perm: int, seed: int) -> np.ndarray:
    """Standard MinHash with random hash functions (integer arithmetic mod 2^61-1)."""
    rng = np.random.default_rng(seed)
    merp = np.uint64((1 << 61) - 1)
    a = rng.integers(1, 1 << 61, size=num_perm, dtype=np.uint64)
    b = rng.integers(0, 1 << 61, size=num_perm, dtype=np.uint64)

    out = np.full((len(sets), num_perm), 0xFFFFFFFF, dtype=np.uint32)
    for i, s in enumerate(sets):
        if not s:
            continue
        vals = np.fromiter((_stable_hash32(t) for t in s), dtype=np.uint64, count=len(s))
        hv = ((a * vals[:, None] + b) % merp).astype(np.uint32)
        out[i] = hv.min(axis=0)
    return out


def minhash_block(
    ds: Dataset,
    threshold: float = 0.5,
    num_perm: int = 64,
    seed: int = 0,
    max_block_size: int = 200,
) -> Blocks:
    """LSH banding over MinHash signatures: approximate Jaccard blocking.

    Band size ``b`` and number of bands ``r`` are chosen from ``threshold`` so
    that the S-curve approximates "become a candidate iff Jaccard >= threshold".
    Cost is O(n * num_perm) plus candidate enumeration, so it scales to catalogs
    where token blocking is too dense.

    Args:
        threshold: target Jaccard similarity for candidate pairs.
        num_perm: MinHash permutations (accuracy of the estimate).
    """
    ids = ds.record_ids
    tok_sets = _dataset_tokens(ds)
    sigs = _minhash_signatures(tok_sets, num_perm, seed)

    # r*b = num_perm, choose b = ceil(log(1/t)/log(num_perm)) style standard fit.
    best = None
    for b in range(1, num_perm + 1):
        r = num_perm // b
        if r * b != num_perm or r < 1:
            continue
        # (1/t)^(1/r) approximates the band width that centres the S-curve.
        err = abs((1.0 / threshold) ** (1.0 / r) - b)
        if best is None or err < best[0]:
            best = (err, b, r)
    _, band_size, n_bands = best if best else (0.0, num_perm, 1)

    pairs: set[tuple[str, str]] = set()
    for start in range(0, num_perm, band_size):
        bands = sigs[:, start : start + band_size]
        # Bucket bands with a deterministic hash (see _stable_hash32 rationale).
        bucket = np.fromiter(
            (_stable_hash32(row.tobytes()) for row in bands), dtype=np.int64, count=len(bands)
        )
        groups: dict[int, list[int]] = defaultdict(list)
        for i, bk in enumerate(bucket):
            groups[bk].append(i)
        for members in groups.values():
            if len(members) > max_block_size:
                continue
            for a, c in itertools.combinations(sorted(members), 2):
                ta, tc = tok_sets[a], tok_sets[c]
                if not ta or not tc:
                    continue
                jac = len(ta & tc) / len(ta | tc)
                if jac >= threshold * 0.75:  # verify against a relaxed bound
                    pairs.add((ids[a], ids[c]))

    return _finalize(
        sorted(pairs),
        ids,
        "minhash_lsh",
        {"threshold": threshold, "num_perm": num_perm, "bands": n_bands, "band_size": band_size},
    )


def faiss_block(
    ds: Dataset,
    embeddings: np.ndarray,
    top_k: int = 20,
    device: str = "cpu",
) -> Blocks:
    """ANN candidate generation over dense embeddings (the proposed retrieval stage).

    Reported per-dataset as Recall@K: the fraction of gold pairs whose partner
    appears in the top-K neighbours. That number, not F1, is what tells a
    reviewer whether the retrieval stage is a safe bottleneck.
    """
    import faiss  # local import: keeps the rest of er.* usable without faiss

    ids = ds.record_ids
    emb = np.ascontiguousarray(embeddings, dtype="float32")
    norms = np.linalg.norm(emb, axis=1, keepdims=True)
    emb = emb / np.maximum(norms, 1e-9)  # cosine == inner product on L2 rows

    dim = emb.shape[1]
    n = emb.shape[0]
    if n < 1024:
        index = faiss.IndexFlatIP(dim)
    else:
        # HNSW keeps search sub-linear; built on the CPU and moved only if asked.
        index = faiss.IndexHNSWFlat(dim, 32, faiss.METRIC_INNER_PRODUCT)
        index.hnsw.efSearch = max(64, top_k * 4)
    index.add(emb)

    k = min(top_k + 1, n)
    _, nbrs = index.search(emb, k)
    pairs: set[tuple[str, str]] = set()
    for i in range(n):
        for j in nbrs[i]:
            j = int(j)
            if j < 0 or j == i:
                continue
            a, c = (i, j) if i < j else (j, i)
            pairs.add((ids[a], ids[c]))

    return _finalize(sorted(pairs), ids, "faiss_ann", {"top_k": top_k, "index": type(index).__name__})


def from_frame(ds: Dataset, frame: pd.DataFrame, method: str = "external") -> Blocks:
    """Wrap a published dataset's own train/test pair split as candidate blocks."""
    return _finalize(
        list(zip(frame["record_id_l"].astype(str), frame["record_id_r"].astype(str), strict=True)),
        ds.record_ids,
        method,
    )
