"""Pairwise similarity features.

Feature families, kept as separable blocks so the ablation study (Objective 4)
can remove one family at a time:

* ``fuzz_*``    - string-similarity features (RapidFuzz).
* ``tok_*``     - token overlap / Jaccard / length features.
* ``attr_*``    - attribute agreement from :mod:`er.preprocess`.
* ``num_*``     - numeric-spec agreement on the extracted capacity/quantity.
* ``emb_*``     - embedding cosine (supplied by the caller, see
  :mod:`er.models.embeddings`).

The feature matrix is always returned as a tidy DataFrame (named columns) rather
than an anonymous array. A research table must be able to say *which* feature
carried the signal.
"""

from __future__ import annotations

from collections.abc import Sequence
from typing import TYPE_CHECKING

import numpy as np
import pandas as pd
from rapidfuzz import distance as rf_distance
from rapidfuzz import fuzz

from .preprocess import (
    ATTRIBUTE_NAMES,
    DEFAULT_LEXICON,
    AttributeLexicon,
    agreement,
    combined_attributes,
    extract_attributes,
    normalize,
)

if TYPE_CHECKING:  # pragma: no cover
    from .schema import Dataset


def _jaccard(a: set[str], b: set[str]) -> float:
    if not a and not b:
        return 0.0
    return len(a & b) / max(1, len(a | b))


def _containment(a: set[str], b: set[str]) -> float:
    """Asymmetric overlap: how much of the shorter record is present in the longer.

    Handles the common truncation case ("iPhone 15" vs the full seller title)
    that Jaccard punishes but which is usually a true match.
    """
    if not a or not b:
        return 0.0
    small, big = (a, b) if len(a) <= len(b) else (b, a)
    return len(a & b) / len(small)


def _num_agreement(va, vb) -> tuple[float, float]:
    """Return (agree, known) for numeric attributes with unit tolerance."""
    if va in (None, "") or vb in (None, ""):
        return 0.0, 0.0
    try:
        fa = float(str(va).rstrip("gmabit ")) if str(va) else float("nan")
        fb = float(str(vb).rstrip("gmabit ")) if str(vb) else float("nan")
    except ValueError:
        fa = fb = float("nan")
    if np.isnan(fa) or np.isnan(fb):
        eq = 1.0 if str(va) == str(vb) else -1.0
        return eq, 1.0
    if fa == fb:
        return 1.0, 1.0
    rel = abs(fa - fb) / max(abs(fa), abs(fb), 1e-9)
    return (0.0 if rel > 0.5 else -1.0), 1.0  # far apart -> hard conflict; close -> soft


def feature_group(col: str) -> str:
    """Map a feature column name to its ablation group.

    Explicit prefix table instead of ``col.split("_")[0]``: the naive version
    silently dropped ``len_*``/``ntok_*`` because their prefix is not a group
    name, which would understate the feature baseline in the ablation table.
    """
    for prefix, group in (("fuzz", "fuzz"), ("attr", "attr"), ("num", "num"), ("emb", "emb")):
        if col.startswith(prefix):
            return group
    return "tok"  # tok_*, len_*, ntok_*


def build_features(
    left_rows: Sequence[str],
    right_rows: Sequence[str],
    ds: Dataset,
    lexicon: AttributeLexicon = DEFAULT_LEXICON,
    embeddings: np.ndarray | None = None,
    id_col: str = "record_id",
) -> pd.DataFrame:
    """Compute the full feature matrix for a list of candidate pairs.

    Args:
        left_rows: record ids for the left side of each pair.
        right_rows: record ids for the right side of each pair.
        ds: :class:`~er.schema.Dataset` holding the texts (duck-typed here to
            avoid an import cycle; needs ``.df`` indexed by ``id_col``).
        embeddings: optional ``[n_records, dim]`` float matrix aligned with the
            dataset row order; used to fill ``emb_cos``.
    """
    df = ds.df.set_index(id_col)
    texts = df["text"].astype(str).to_dict()
    order = list(df.index)
    pos = {rid: i for i, rid in enumerate(order)}

    attrs_cache: dict[str, dict] = {}

    def attrs_for(rid: str) -> dict:
        if rid not in attrs_cache:
            raw = df.loc[rid, "attrs"] if "attrs" in df.columns else None
            attrs_cache[rid] = combined_attributes(texts[rid], raw if isinstance(raw, dict) else None, lexicon)
        return attrs_cache[rid]

    rows = []
    for lid, rid in zip(left_rows, right_rows, strict=True):
        a, b = texts[lid], texts[rid]
        na, nb = normalize(a), normalize(b)
        ta, tb = set(na.split()), set(nb.split())

        feat: dict[str, float] = {
            # rapidfuzz >=3 exposes Jaro-Winkler and normalized Levenshtein on
            # the 0-1 scale via rapidfuzz.distance, not the 0-100 fuzz scale.
            "fuzz_jaro": rf_distance.JaroWinkler.similarity(na, nb),
            "fuzz_normlev": rf_distance.Levenshtein.normalized_similarity(na, nb),
            "fuzz_ratio": fuzz.ratio(na, nb) / 100.0,
            "fuzz_partial": fuzz.partial_ratio(na, nb) / 100.0,
            "fuzz_token_sort": fuzz.token_sort_ratio(na, nb) / 100.0,
            "fuzz_token_set": fuzz.token_set_ratio(na, nb) / 100.0,
            "fuzz_token_ratio": fuzz.token_ratio(na, nb) / 100.0,
            "tok_jaccard": _jaccard(ta, tb),
            "tok_contain": _containment(ta, tb),
            "tok_same": float(len(ta & tb)),
            "len_left": float(len(na)),
            "len_right": float(len(nb)),
            "len_ratio": len(na) / max(1, len(nb)),
            "ntok_left": float(len(ta)),
            "ntok_right": float(len(tb)),
        }

        feat.update(agreement(attrs_for(lid), attrs_for(rid)))

        aa, ab = attrs_for(lid), attrs_for(rid)
        cap_a, cap_b = aa.get("capacity"), ab.get("capacity")
        feat["num_capacity_agree"], feat["num_capacity_known"] = _num_agreement(cap_a, cap_b)
        qty_a, qty_b = aa.get("quantity"), ab.get("quantity")
        feat["num_qty_agree"], feat["num_qty_known"] = _num_agreement(qty_a, qty_b)

        if embeddings is not None:
            ia, ib = pos.get(str(lid)), pos.get(str(rid))
            if ia is not None and ib is not None:
                va, vb = embeddings[ia], embeddings[ib]
                denom = (np.linalg.norm(va) * np.linalg.norm(vb)) or 1e-9
                feat["emb_cos"] = float(va @ vb / denom)
            else:
                feat["emb_cos"] = 0.0

        rows.append(feat)

    out = pd.DataFrame(rows)
    if len(out) == 0:  # keep the column contract stable on empty input
        base = [
            "fuzz_jaro", "fuzz_normlev", "fuzz_ratio", "fuzz_partial", "fuzz_token_sort",
            "fuzz_token_set",
            "fuzz_token_ratio", "tok_jaccard", "tok_contain", "tok_same", "len_left",
            "len_right", "len_ratio", "ntok_left", "ntok_right", "num_capacity_agree",
            "num_capacity_known", "num_qty_agree", "num_qty_known",
        ]
        for n in ATTRIBUTE_NAMES:
            base += [f"attr_{n}", f"attr_{n}_known"]
        if embeddings is not None:
            base.append("emb_cos")
        out = pd.DataFrame(columns=sorted(base))
    return out
