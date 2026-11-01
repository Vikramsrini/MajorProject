"""Offline tests for the cross-encoder path.

Deliberately no network and no model downloads: these run in CI and on a laptop. They
cover the parts that break silently — serialisation order, the early-stopping criterion,
and whether the CLI/system wiring actually reaches the new matcher. The training loop
itself is exercised by `scripts/run_wdc.py --with-crossencoder --max-steps 20` (the
notebook's smoke gate), which cannot be a unit test because it downloads weights.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

from er.models.crossencoder import (
    CrossEncoderMatcher,
    _max_f1_at_best_threshold,
    serialize_record,
)


def test_serialisation_uses_ditto_field_markers_in_order():
    rec = {
        "title": "Canon EF 100mm F2.8 L IS USM Macro",
        "description": "true macro 1:1",
        "attrs": {"brand": "canon"},
        "text": "ignored because fields exist",
    }
    s = serialize_record(rec)
    assert s.startswith("[COL] title [VAL] Canon")
    # order is part of the contract: the published baselines serialise title first
    assert s.index("[COL] description") < s.index("[COL] brand")
    assert "ignored" not in s


def test_serialisation_falls_back_to_flat_text():
    rec = {"title": None, "description": "", "attrs": None, "text": "plain text only"}
    assert serialize_record(rec) == "[COL] text [VAL] plain text only"


def test_serialisation_skips_blank_and_nan_fields():
    rec = {"title": "Real Title", "description": float("nan"), "attrs": {"brand": "nan"}, "text": "x"}
    s = serialize_record(rec)
    assert "[COL] title [VAL] Real Title" in s
    assert "description" not in s
    assert "brand" not in s  # "nan" must not be serialised as a real brand


def test_early_stopping_criterion_finds_perfect_separation():
    y = np.array([0, 1, 1, 0])
    s = np.array([0.1, 0.9, 0.8, 0.2])
    assert _max_f1_at_best_threshold(y, s) == pytest.approx(1.0)


def test_early_stopping_criterion_handles_degenerate_input():
    assert _max_f1_at_best_threshold(np.array([0, 0]), np.array([0.4, 0.6])) == 0.0
    assert _max_f1_at_best_threshold(np.array([], dtype=int), np.array([])) == 0.0


def test_early_stopping_criterion_ignores_broken_tie_ordering():
    """Identical scores cannot separate anything, so F1 must reflect the best achievable cut."""
    y = np.array([1, 0, 0, 0])
    s = np.full(4, 0.5)
    # accepting everything gives P=0.25/R=1.0 -> F1=0.4; the criterion must not exceed that
    assert _max_f1_at_best_threshold(y, s) == pytest.approx(0.4)


class _CountingMatcher:
    """Minimal matcher that records how often it is fitted/scored."""

    def __init__(self, score: float = 0.5) -> None:
        self.fit_calls = 0
        self.score_calls = 0
        self._s = score

    def bind(self, ds):  # noqa: ANN001
        return self

    def fit(self, train):  # noqa: ANN001
        self.fit_calls += 1
        return self

    def score(self, pairs):  # noqa: ANN001
        self.score_calls += 1
        return np.full(len(pairs), self._s, dtype=float)


def test_constrained_system_reuses_fitted_base_without_refit():
    """A constrained variant must not retrain its base.

    Two reasons: a cross-encoder retrain doubles GPU cost, and a fresh random
    initialisation would put seed noise into the comparison that the constraint is
    supposed to isolate. This test pins the behaviour.
    """
    from er.experiment import SystemSpec, run_system
    from er.schema import Dataset, PairTable, pair_frame

    ids = [str(i) for i in range(6)]
    ds = Dataset("t", pd.DataFrame({"record_id": ids, "text": [f"item {i}" for i in ids]}))
    frame = pair_frame([("0", "1"), ("2", "3"), ("4", "5")])
    frame["label"] = [1, 1, 0]
    pt = PairTable(frame)

    base = _CountingMatcher(score=0.8)
    spec = SystemSpec(
        "constrained", "classifier", {}, constraints={"mode": "symmetric", "use_specifiers": True}
    )
    rows, scored, fitted = run_system(spec, ds, pt, pt, pt, base_matcher=base)
    assert base.fit_calls == 0, "the base matcher must not be refitted"
    assert base.score_calls > 0, "the base must still be scored"
    assert fitted is not base and hasattr(fitted, "score"), "returned matcher is the wrapper"
    assert rows and scored.frame["score"].notna().all()


def test_specs_expose_crossencoder_systems_and_dispatch():
    """The CLI wiring must reach CrossEncoderMatcher without touching the network."""
    import importlib.util
    from pathlib import Path

    script = Path(__file__).resolve().parents[1] / "scripts" / "run_wdc.py"
    spec = importlib.util.spec_from_file_location("run_wdc_probe", script)
    mod = importlib.util.module_from_spec(spec)
    assert spec and spec.loader
    spec.loader.exec_module(mod)

    names = [s.name for s in mod.specs(False, True, "roberta-base", 1, 5)]
    assert "s8_crossencoder" in names and "s9_crossencoder_constraint" in names
    # default invocation must not silently include the expensive systems
    assert "s8_crossencoder" not in [s.name for s in mod.specs(False)]

    ce = [s for s in mod.specs(False, True, "roberta-base", 1, 5) if s.name == "s8_crossencoder"][0]
    from er.experiment import build_matcher
    from er.preprocess import DEFAULT_LEXICON
    from er.schema import Dataset
    import pandas as pd

    ds = Dataset("t", pd.DataFrame({"record_id": ["1", "2"], "text": ["a", "b"]}))
    matcher = build_matcher(ce, ds, DEFAULT_LEXICON)
    assert isinstance(matcher, CrossEncoderMatcher)
    assert matcher.max_steps == 5  # the smoke-test budget must survive the plumbing
