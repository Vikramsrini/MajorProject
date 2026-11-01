"""Dataset loaders.

Currently implements the loader the first paper depends on: the WDC Products
benchmark (Peeters, Der, Bizer; EDBT 2024) in its pair-wise format.

Two naming traps found while wiring this up, both encoded here so nobody has to
rediscover them (see docs/DATASETS.md):

1. **Corner-case ratio is the prefix, not the ``cc`` token.** ``80pair.zip``
   contains ``wdcproducts80cc20rnd000un_*.json.gz``. The ``80`` is the
   corner-case share; ``cc20`` is the *complementary* random-negative share.
   Verified against the benchmark paper's Table 1 row counts: train_large has
   19,835 rows for 80%, 19,607 for 50%, 19,015 for 20%.
2. ``_gs`` files are the test splits (gold standard); ``_rnd{000,050,100}un``
   encodes 0/50/100% unseen products in the test set.

The benchmark ships its own train/valid/test pair splits, and offers are
disjoint across splits by construction, so loaders return the published pairs
rather than re-blocking them. Re-deriving a split here would silently leak.
"""

from __future__ import annotations

import gzip
import json
import re
from pathlib import Path

import pandas as pd

from .schema import Dataset, PairTable

# --------------------------------------------------------------------------- #
# WDC Products (pair-wise)
# --------------------------------------------------------------------------- #
_WDC_FILE = re.compile(
    r"wdcproducts(?P<cc>\d{2})cc\d{2}rnd(?P<unseen>\d{3})un_(?P<split>train|valid|gs)"
    r"(?:_(?P<size>small|medium|large))?\.json\.gz$"
)


def wdc_split_of(path: Path | str) -> dict:
    """Parse a WDC Products file name into its variant coordinates.

    ``corner_cases`` and ``unseen`` are returned as ints so callers filter with
    ``== 80`` rather than a string form; the raw names mix ``000``/``050``/``100``
    and ``80pct``-style labels, which is how a filtering bug first appears.
    """
    name = Path(path).name
    m = _WDC_FILE.search(name)
    if not m:
        raise ValueError(f"unrecognised WDC Products file name: {name}")
    split = m["split"]
    if split == "gs":
        split = "test"
    return {
        "corner_cases": int(m["cc"]),  # prefix number = corner-case share
        "unseen": int(m["unseen"]),  # 0 | 50 | 100
        "split": split,
        "dev_size": m["size"] or "n/a",
    }


def _serialize(row: pd.Series, side: str, include_description: bool = True) -> str:
    """Flatten one offer into the single text field the pipeline consumes.

    Title first (it carries brand/model/variant), then description when present.
    Keeping this in one place matters: a changed serialization changes every
    similarity number, so it must be a deliberate, logged decision.
    """
    title = str(row.get(f"title_{side}", "") or "")
    brand = str(row.get(f"brand_{side}", "") or "")
    parts = [title]
    if include_description:
        desc = str(row.get(f"description_{side}", "") or "")
        if desc and desc.lower() not in {"nan", "none"}:
            parts.append(desc)
    text = " ".join(p for p in parts if p.strip())
    if brand and brand.lower() not in {"nan", "none"} and brand.lower() not in text.lower():
        text = f"{brand} {text}"
    return re.sub(r"\s+", " ", text).strip()


def _clean_str(val) -> str | None:
    """Return a stripped string, or None for the many empty/NaN schema.org fields."""
    if val is None:
        return None
    s = str(val).strip()
    if not s or s.lower() in {"nan", "none", "null"}:
        return None
    return s


def _attrs(row: pd.Series, side: str) -> dict[str, object]:
    """Structured attributes as stated by the source (not inferred)."""
    out: dict[str, object] = {}
    brand = _clean_str(row.get(f"brand_{side}"))
    if brand:
        out["brand"] = brand.lower()
    price = _clean_str(row.get(f"price_{side}"))
    cur = _clean_str(row.get(f"priceCurrency_{side}"))
    if price:
        try:
            out["price"] = float(price)
        except ValueError:
            pass
    if cur:
        # WDC stores an upper-case 3-letter currency, but some rows carry the
        # literal string "NAN"/"nan" which must not become a comparable value.
        out["currency"] = cur.upper() if len(cur) == 3 and cur.isalpha() else None
    return out


def load_wdc_products_pairwise(
    path: Path | str,
    include_description: bool = True,
) -> tuple[Dataset, PairTable]:
    """Load one WDC Products pair-wise split file.

    Returns:
        dataset: unique offers (``record_id``, ``text``, ``source``,
            ``gold_entity`` = cluster_id, ``attrs`` = stated fields).
        pairs: the published candidate pairs with gold ``label`` plus the
            ``is_hard_negative`` flag, which identifies the benchmark's own
            similar-but-different negatives - the exact population the
            attribute-constraint contribution targets.
    """
    path = Path(path)
    rows = []
    with gzip.open(path, "rt") as fh:
        for line in fh:
            rows.append(json.loads(line))
    raw = pd.DataFrame(rows)

    meta = wdc_split_of(path)
    prefix = f"{path.stem}"

    records: dict[str, dict] = {}
    for side in ("left", "right"):
        sub = raw[raw[f"id_{side}"].notna()]
        for _, row in sub.iterrows():
            rid = str(row[f"id_{side}"])
            if rid in records:
                continue
            records[rid] = {
                "record_id": rid,
                "text": _serialize(row, side, include_description),
                # Keep the original fields as well as the flattened text: a
                # Ditto-style cross-encoder must serialise per field ([COL] name
                # [VAL] value) to be comparable with the published numbers, and
                # error analysis reads far better with the real title present.
                "title": _clean_str(row.get(f"title_{side}")) or "",
                "description": _clean_str(row.get(f"description_{side}")) or "",
                "source": f"{prefix}_{side}",
                "gold_entity": str(row[f"cluster_id_{side}"]),
                "attrs": _attrs(row, side),
            }
    ds = Dataset(f"wdc_products_{meta['corner_cases']}_{meta['unseen']}_{meta['split']}", pd.DataFrame(records.values()))

    pairs = pd.DataFrame(
        {
            "record_id_l": raw["id_left"].astype(str),
            "record_id_r": raw["id_right"].astype(str),
            "label": raw["label"].astype(int),
            "is_hard_negative": raw.get("is_hard_negative", pd.Series([None] * len(raw))).astype(object),
            "pair_id": raw.get("pair_id"),
        }
    )
    # De-duplicate pair ids defensively: a duplicated labeled pair would weight
    # one gold decision twice in micro-F1.
    pairs = pairs.drop_duplicates(subset=["record_id_l", "record_id_r"]).reset_index(drop=True)

    pt = PairTable(
        pairs,
        meta={
            "dataset": ds.name,
            "file": path.name,
            "fingerprint": ds.fingerprint(),
            **meta,
            "n_pairs": int(len(pairs)),
            "positives": int(pairs["label"].sum()),
            "negatives": int((pairs["label"] == 0).sum()),
            "hard_negatives": int(pairs["is_hard_negative"].fillna(False).astype(bool).sum()),
        },
    )
    return ds, pt


def wdc_index(data_dir: Path | str) -> pd.DataFrame:
    """Inventory every WDC Products pair-wise file present on disk.

    Matched on file *content* name pattern, not directory name: the files are
    ``wdcproducts80cc20rnd000un_gs.json.gz`` inside ``80pair/``, so a ``*pair*``
    glob would find nothing.
    """
    data_dir = Path(data_dir)
    rows = []
    for p in sorted(data_dir.rglob("*.json.gz")):
        try:
            meta = wdc_split_of(p)
        except ValueError:
            continue
        rows.append({"path": str(p), **meta, "mb": round(p.stat().st_size / 1e6, 2)})
    return pd.DataFrame(rows)


# --------------------------------------------------------------------------- #
# generic CSV pair tables (synthetic data, custom annotations)
# --------------------------------------------------------------------------- #
def load_pair_csv(records_csv: Path | str, pairs_csv: Path | str) -> tuple[Dataset, PairTable]:
    ds = Dataset.from_csv(records_csv)
    pairs = pd.read_csv(pairs_csv, dtype={"record_id_l": str, "record_id_r": str})
    return ds, PairTable(pairs, meta={"records": ds.name, "n_pairs": int(len(pairs))})
