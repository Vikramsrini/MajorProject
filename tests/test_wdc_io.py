"""Integration tests against the real WDC Products files (skipped if not downloaded).

These lock in the two naming traps in docs/DATASETS.md: the corner-case ratio is
the file-name prefix, and the ``_gs`` files are the test splits. If the files are
not on disk the tests skip rather than silently pass, so a fresh clone still
runs the suite.
"""

from __future__ import annotations

from pathlib import Path

import pytest

from er.io import load_wdc_products_pairwise, wdc_index, wdc_split_of

DATA = Path(__file__).resolve().parents[1] / "data" / "raw"

needs_data = pytest.mark.skipif(
    not DATA.exists() or not list(DATA.rglob("*.json.gz")), reason="WDC Products not downloaded"
)


def test_split_of_corner_case_is_the_prefix():
    """80cc20 means 80% corner cases (cc20 = complementary random share)."""
    m = wdc_split_of("wdcproducts80cc20rnd050un_gs.json.gz")
    assert m == {"corner_cases": 80, "unseen": 50, "split": "test", "dev_size": "n/a"}


def test_split_of_train_medium():
    m = wdc_split_of("wdcproducts20cc80rnd000un_train_medium.json.gz")
    assert m == {"corner_cases": 20, "unseen": 0, "split": "train", "dev_size": "medium"}


def test_split_of_rejects_other_names():
    with pytest.raises(ValueError):
        wdc_split_of("something_else.json.gz")


@needs_data
def test_row_counts_match_the_benchmark_paper():
    """Table 1 of the WDC Products paper - the check that the naming is right."""
    idx = wdc_index(DATA)
    expected = {80: 19835, 50: 19607, 20: 19015}
    for cc, want in expected.items():
        row = idx[(idx.corner_cases == cc) & (idx.split == "train") & (idx.dev_size == "large")]
        if row.empty:
            continue
        _, pt = load_wdc_products_pairwise(row.iloc[0]["path"])
        assert pt.meta["n_pairs"] == want, f"corner {cc}: got {pt.meta['n_pairs']} want {want}"


@needs_data
def test_test_splits_are_4500_pairs_with_500_positives():
    idx = wdc_index(DATA)
    for _, row in idx[idx.split == "test"].iterrows():
        _, pt = load_wdc_products_pairwise(row["path"])
        assert pt.meta["n_pairs"] == 4500
        assert pt.positives == 500


@needs_data
def test_records_have_no_nan_leakage():
    idx = wdc_index(DATA)
    row = idx[(idx.corner_cases == 80) & (idx.split == "train") & (idx.dev_size == "medium")].iloc[0]
    ds, _ = load_wdc_products_pairwise(row["path"])
    assert ds.df["text"].astype(str).str.strip().ne("").all()
    for attrs in ds.df["attrs"]:
        if isinstance(attrs, dict):
            assert "nan" not in {str(v).lower() for v in attrs.values()}
