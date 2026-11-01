"""Core data model for entity resolution experiments.

Everything in this project is expressed in terms of three objects:

* ``Dataset``      - a set of records, each with an optional gold entity id.
* ``Blocks``       - candidate pairs proposed by a blocking (indexing) stage.
* ``PairTable``    - scored/labeled candidate pairs with features and metadata.

Keeping these as plain dataclasses with CSV round-tripping lets every stage of
the pipeline be cached, inspected and reproduced independently, which is what a
research evaluation needs: each stage must be measurable on its own.
"""

from __future__ import annotations

import hashlib
import json
from collections.abc import Iterable, Iterator, Sequence
from dataclasses import dataclass, field
from pathlib import Path

import pandas as pd

#: Columns every record table must contain.
REQUIRED_COLUMNS = ("record_id", "text")


def _key(a: str, b: str) -> tuple[str, str]:
    """Canonical order for an unordered pair so the same pair never appears twice."""
    return (a, b) if a <= b else (b, a)


@dataclass
class Dataset:
    """A collection of records from one or more sources.

    Attributes:
        name: identifier used in result tables.
        df: DataFrame with ``record_id``, ``text`` and optional ``source``,
            ``gold_entity``, and free-form attribute columns (``attrs``).
    """

    name: str
    df: pd.DataFrame

    def __post_init__(self) -> None:
        missing = [c for c in REQUIRED_COLUMNS if c not in self.df.columns]
        if missing:
            raise ValueError(f"dataset {self.name!r} missing columns: {missing}")
        self.df = self.df.reset_index(drop=True)
        if self.df["record_id"].duplicated().any():
            raise ValueError("record_id values must be unique")
        if "source" not in self.df.columns:
            self.df["source"] = self.name
        if "attrs" not in self.df.columns:
            self.df["attrs"] = None

    # ------------------------------------------------------------------ access
    @property
    def record_ids(self) -> list[str]:
        return self.df["record_id"].astype(str).tolist()

    @property
    def sources(self) -> list[str]:
        return sorted(self.df["source"].unique().tolist())

    @property
    def has_gold_entities(self) -> bool:
        return "gold_entity" in self.df.columns and self.df["gold_entity"].notna().any()

    @property
    def n(self) -> int:
        return len(self.df)

    def text(self, record_id: str) -> str:
        row = self.df.loc[self.df["record_id"].astype(str) == str(record_id)]
        if row.empty:
            raise KeyError(f"unknown record_id {record_id!r}")
        return str(row["text"].iloc[0])

    def entities(self) -> dict[str, list[str]]:
        """Map gold entity -> member record ids (for entity-level evaluation)."""
        if not self.has_gold_entities:
            raise ValueError("dataset has no gold_entity column")
        return {
            str(ent): group["record_id"].astype(str).tolist()
            for ent, group in self.df.groupby("gold_entity")
        }

    def fingerprint(self) -> str:
        """Stable hash of the record set, embedded in every result row.

        This makes accidental comparisons across different data versions visible
        in result tables instead of silently corrupting a paper's numbers.
        """
        h = hashlib.sha256()
        for rid, text in zip(
            self.df["record_id"].astype(str), self.df["text"].astype(str), strict=True
        ):
            h.update(f"{rid}\x1f{text}\x1e".encode())
        return h.hexdigest()[:16]

    # -------------------------------------------------------------------- i/o
    def to_csv(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        out = self.df.copy()
        if out["attrs"].map(lambda v: v is not None).any():
            out["attrs"] = out["attrs"].map(
                lambda v: json.dumps(v) if isinstance(v, dict) else None
            )
        out.to_csv(path, index=False)

    @classmethod
    def from_csv(cls, path: Path | str, name: str | None = None) -> Dataset:
        path = Path(path)
        df = pd.read_csv(path, dtype={"record_id": str})
        if "attrs" in df.columns:
            df["attrs"] = df["attrs"].map(
                lambda v: json.loads(v) if isinstance(v, str) and v.strip().startswith("{") else None
            )
        return Dataset(name or path.stem, df)


@dataclass
class Blocks:
    """Candidate pairs produced by a blocking stage.

    Attributes:
        pairs: DataFrame with ``record_id_l``/``record_id_r`` and optional
            ``score`` (similarity used for ranking) and ``method`` columns.
        stats: bookkeeping (pairs generated, all-pairs count, reduction ratio).
    """

    pairs: pd.DataFrame
    method: str
    stats: dict = field(default_factory=dict)

    @property
    def n_candidates(self) -> int:
        return len(self.pairs)

    def as_set(self) -> set[tuple[str, str]]:
        return {
            _key(str(l), str(r))
            for l, r in zip(self.pairs["record_id_l"], self.pairs["record_id_r"], strict=True)
        }

    def to_parquet_or_csv(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.pairs.to_csv(path, index=False)
        (path.with_suffix(path.suffix + ".meta.json")).write_text(
            json.dumps({"method": self.method, "stats": self.stats}, indent=2)
        )


@dataclass
class PairTable:
    """Candidate pairs joined with gold labels, features and model scores.

    The canonical column contract for the whole evaluation stack:
    ``record_id_l``, ``record_id_r``, ``label`` (0/1, NaN if unlabeled),
    ``score`` (matcher probability, NaN if not scored), then feature columns.
    """

    frame: pd.DataFrame
    meta: dict = field(default_factory=dict)

    def __post_init__(self) -> None:
        need = {"record_id_l", "record_id_r"}
        if not need.issubset(self.frame.columns):
            raise ValueError(f"PairTable missing {need - set(self.frame.columns)}")
        self.frame["record_id_l"] = self.frame["record_id_l"].astype(str)
        self.frame["record_id_r"] = self.frame["record_id_r"].astype(str)

    @property
    def labeled(self) -> PairTable:
        if "label" not in self.frame.columns:
            return PairTable(self.frame.iloc[0:0], self.meta)
        return PairTable(self.frame[self.frame["label"].notna()], self.meta)

    @property
    def positives(self) -> int:
        if "label" not in self.frame.columns:
            return 0
        return int((self.frame["label"] == 1).sum())

    def keys(self) -> list[tuple[str, str]]:
        return [
            _key(l, r)
            for l, r in zip(self.frame["record_id_l"], self.frame["record_id_r"], strict=True)
        ]

    def join_scores(self, scores: Iterable[float] | None = None, **columns) -> PairTable:
        """Return a copy with extra columns aligned positionally.

        Scalars are broadcast so run metadata (e.g. the fitted temperature) can
        travel with every row of the scored table.
        """
        out = self.frame.copy()
        if scores is not None:
            out["score"] = list(scores)
        n = len(out)
        for name, values in columns.items():
            if isinstance(values, (int, float, str, bool)) or values is None:
                out[name] = [values] * n
            else:
                vals = list(values)
                if len(vals) == 1 and n != 1:
                    vals = vals * n
                if len(vals) != n:
                    raise ValueError(f"column {name!r} has {len(vals)} values for {n} rows")
                out[name] = vals
        return PairTable(out, self.meta)

    def to_csv(self, path: Path | str) -> None:
        path = Path(path)
        path.parent.mkdir(parents=True, exist_ok=True)
        self.frame.to_csv(path, index=False)
        (path.with_suffix(path.suffix + ".meta.json")).write_text(json.dumps(self.meta, indent=2))


def all_pairs(ids: Sequence[str]) -> Iterator[tuple[str, str]]:
    """Enumerate every unordered pair; only used as the quadratic reference point."""
    for i in range(len(ids)):
        for j in range(i + 1, len(ids)):
            yield _key(ids[i], ids[j])


def pair_frame(pairs: Iterable[tuple[str, str]], **columns: Sequence) -> pd.DataFrame:
    """Build a candidate DataFrame with canonical ordering applied."""
    rows = [_key(p[0], p[1]) for p in pairs]
    df = pd.DataFrame(rows, columns=["record_id_l", "record_id_r"])
    for name, values in columns.items():
        df[name] = list(values)
    return df
