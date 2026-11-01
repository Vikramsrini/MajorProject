"""Attribute-constrained matching (the proposed contribution, H2).

A matcher over similarity features will happily score "Apple iPhone 15 128GB" and
"Apple iPhone 15 Plus 128GB" at 0.95, because they differ by one token. This
layer puts an explicit, auditable constraint on top of any base matcher: when
two records state *contradicting* values for an attribute that identifies a
different product, the pair cannot be accepted, no matter how similar the text.

Why a cap instead of a hard reject
    The pair keeps its position in the ranking (score is capped to ``ceiling``)
    rather than being deleted. Deleting pairs would silently break Recall@K
    accounting and would make the constraint look better than it is by removing
    the pair from the denominator. Capping keeps every metric honest.

The interesting design question is *asymmetric* evidence, and it is exposed as a
mode so the paper can measure it rather than assume it:

    symmetric       veto only when BOTH sides state the attribute and they differ.
                    Safe, never blocks a truncated title.
    asymmetric      a variant marker present on one side and absent on the other
                    also counts as conflict ("Plus" vs nothing).
    asymmetric_subset
                    as above, but waived when the marker-free side's tokens are
                    fully contained in the other - i.e. plausibly a truncated
                    title of the same product rather than a different model.

Each mode is a different bet about real catalog behaviour, and each is measured
by the variant-slice evaluation in :mod:`er.experiment`.
"""

from __future__ import annotations

from collections.abc import Iterable, Sequence

import numpy as np
import pandas as pd

from ..preprocess import (
    DEFAULT_LEXICON,
    AttributeLexicon,
    combined_attributes,
    normalize,
    specifier_conflicts,
    specifier_map,
)
from ..schema import PairTable

#: Attributes that identify a distinct product when they conflict.
HARD_ATTRIBUTES: tuple[str, ...] = ("brand", "model_family", "variant", "capacity", "quantity")
#: Attributes that are advisory: colour/size variants are frequently mislabelled
#: by sellers, so they only count in an explicitly "strict" configuration.
SOFT_ATTRIBUTES: tuple[str, ...] = ("color", "year")


def _attr_sets(
    ids: Iterable[str], texts: Sequence[str], lexicon: AttributeLexicon
) -> dict[str, dict[str, str]]:
    return {str(i): extract_attributes(t, lexicon) for i, t in zip(ids, texts, strict=True)}


def hard_conflicts(
    a: dict[str, object],
    b: dict[str, object],
    hard: Sequence[str] = HARD_ATTRIBUTES,
    mode: str = "symmetric",
    subset_ok: bool = False,
    tokens_a: frozenset[str] = frozenset(),
    tokens_b: frozenset[str] = frozenset(),
) -> list[str]:
    """Return the names of attributes that forbid this merge.

    Args:
        a, b: extracted attribute dicts.
        hard: attribute names treated as identifying.
        mode: ``"symmetric"`` or ``"asymmetric"`` (see module docstring).
        subset_ok: with asymmetric mode, waive a one-sided ``variant`` conflict
            when one title's tokens contain the other's.
    """
    conflicts: list[str] = []
    for name in hard:
        va, vb = a.get(name), b.get(name)
        both = va not in (None, "") and vb not in (None, "")
        if both:
            if str(va) != str(vb):
                conflicts.append(name)
            continue
        if mode == "symmetric":
            continue
        # asymmetric: a one-sided variant/capacity marker is treated as evidence
        # that the records describe different products.
        one_sided = (va not in (None, "")) ^ (vb not in (None, ""))
        if one_sided and name in {"variant", "capacity", "quantity"}:
            if subset_ok:
                shorter, longer = (
                    (tokens_a, tokens_b) if len(tokens_a) <= len(tokens_b) else (tokens_b, tokens_a)
                )
                if shorter and shorter <= longer:
                    continue  # plausibly a truncated title, not a different model
            conflicts.append(f"{name}:one_sided")
    return conflicts


def specifier_maps(ids: Iterable[str], texts: Sequence[str]) -> dict[str, dict[str, set[str]]]:
    """Per-record unit-numeric specifier maps (cached by the matcher)."""
    return {str(i): specifier_map(t) for i, t in zip(ids, texts, strict=True)}


class AttributeConstrainedMatcher:
    """Wraps a base matcher and enforces attribute conflicts on its scores.

    Args:
        base: any object with ``score(pairs) -> ndarray``.
        ds: dataset the pairs refer to (used for attribute extraction).
        mode: constraint mode, see :func:`hard_conflicts`.
        ceiling: scores of conflicting pairs are capped to this value. Must be
            below the operating threshold, otherwise the constraint is inert.
        hard: attribute names to enforce.
        soft_extra: additionally enforce advisory attributes (colour, year).
        use_specifiers: also veto when both sides state different values for the
            same unit ("50mm" vs "85mm"). This is what carries the constraint
            onto datasets whose products are not phone-like; toggle it off to
            measure its contribution separately from the typed attributes.
    """

    def __init__(
        self,
        base: object,
        ds,
        mode: str = "asymmetric_subset",
        ceiling: float = 0.30,
        hard: Sequence[str] = HARD_ATTRIBUTES,
        soft_extra: bool = False,
        lexicon: AttributeLexicon = DEFAULT_LEXICON,
        use_specifiers: bool = True,
    ) -> None:
        self.base = base
        self.ds = ds
        self.mode = mode
        self.ceiling = float(ceiling)
        self.hard = tuple(hard) + (SOFT_ATTRIBUTES if soft_extra else ())
        self.lexicon = lexicon
        self.use_specifiers = use_specifiers
        self.last_stats: dict[str, float] = {}
        self._attrs: dict[str, dict[str, object]] | None = None
        self._toks: dict[str, frozenset[str]] | None = None
        self._specs: dict[str, dict[str, set[str]]] | None = None

    def bind(self, ds) -> AttributeConstrainedMatcher:
        if hasattr(self.base, "bind"):
            self.base = self.base.bind(ds)  # type: ignore[attr-defined]
        self.ds = ds
        self._attrs = None
        self._toks = None
        self._specs = None
        return self

    def _ensure(self) -> None:
        if self._attrs is None:
            df = self.ds.df
            ids = df["record_id"].astype(str)
            texts = df["text"].astype(str)
            stated = df["attrs"] if "attrs" in df.columns else [None] * len(df)
            # Inferred AND stated, merged: many benchmarks ship only brand/price
            # as columns, and taking the stated dict alone would leave variant
            # and capacity invisible - disabling the contribution in practice.
            self._attrs = {
                str(i): combined_attributes(t, raw if isinstance(raw, dict) else None, self.lexicon)
                for i, t, raw in zip(ids, texts, stated, strict=True)
            }
            self._toks = {str(i): frozenset(normalize(t).split()) for i, t in zip(ids, texts, strict=True)}
            self._specs = specifier_maps(ids, texts)

    def _conflicts_for(self, l: str, r: str) -> list[str]:
        """All reasons this pair may not be merged (attribute + identifier)."""
        conflicts = hard_conflicts(
            self._attrs[l],
            self._attrs[r],
            self.hard,
            mode=("symmetric" if self.mode == "symmetric" else "asymmetric"),
            subset_ok=(self.mode == "asymmetric_subset"),
            tokens_a=self._toks[l],
            tokens_b=self._toks[r],
        )
        if self.use_specifiers:
            conflicts = conflicts + specifier_conflicts(self._specs[l], self._specs[r])
        return conflicts

    def score(self, pairs: pd.DataFrame) -> np.ndarray:
        self._ensure()
        raw = np.asarray(self.base.score(pairs), dtype=float)  # type: ignore[attr-defined]
        out = raw.copy()
        n_conflict = 0
        per_attr: dict[str, int] = {}
        for idx, (l, r) in enumerate(
            zip(pairs["record_id_l"].astype(str), pairs["record_id_r"].astype(str), strict=True)
        ):
            conflicts = self._conflicts_for(l, r)
            if conflicts:
                out[idx] = min(out[idx], self.ceiling)
                n_conflict += 1
                for c in conflicts:
                    # Group specifier conflicts by kind, not by unit, so the
                    # reporting table stays readable on large catalogs.
                    key = "specifier" if c.startswith("specifier:") else c
                    per_attr[key] = per_attr.get(key, 0) + 1
        self.last_stats = {
            "n_pairs": len(pairs),
            "n_constrained": n_conflict,
            "share_constrained": n_conflict / len(pairs) if len(pairs) else 0.0,
            "per_attribute": per_attr,
        }
        return out

    def fit(self, train: PairTable) -> AttributeConstrainedMatcher:
        if hasattr(self.base, "fit"):
            self.base = self.base.fit(train)  # type: ignore[attr-defined]
        return self

    def conflicts_frame(self, pairs: pd.DataFrame) -> pd.DataFrame:
        """Expose which attribute caused each veto - needed for the case tables."""
        self._ensure()
        rows = []
        for l, r in zip(pairs["record_id_l"].astype(str), pairs["record_id_r"].astype(str), strict=True):
            c = hard_conflicts(
                self._attrs[l],
                self._attrs[r],
                self.hard,
                mode=("symmetric" if self.mode == "symmetric" else "asymmetric"),
                subset_ok=(self.mode == "asymmetric_subset"),
                tokens_a=self._toks[l],
                tokens_b=self._toks[r],
            )
            rows.append({"record_id_l": l, "record_id_r": r, "conflicts": ";".join(c)})
        return pd.DataFrame(rows)
