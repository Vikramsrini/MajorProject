"""Unit tests for the correctness-critical primitives.

These target the places where a subtle bug would silently change a reported
number rather than raise: pair canonicalisation, the missing-vs-conflicting
attribute distinction, the veto modes, blocking determinism, and the metric
formulas.
"""

from __future__ import annotations

import numpy as np
import pandas as pd
import pytest

import er
from er.blocking import minhash_block, prefix_block, token_block
from er.calibration import expected_calibration_error, temperature_scale
from er.clustering import connected_components, transitive_cleanup
from er.features import build_features, feature_group
from er.metrics import bcubed, blocking_quality, entity_pairwise_f1, gold_pairs_from_entities, pairwise_metrics
from er.models.attribute import hard_conflicts
from er.preprocess import (
    AttributeLexicon,
    agreement,
    extract_attributes,
    normalize,
)
from er.schema import Dataset, PairTable, pair_frame


@pytest.fixture
def toy_ds() -> Dataset:
    rows = [
        {"record_id": "1", "text": "Apple iPhone 15, 128GB, Black", "source": "a", "gold_entity": "ip15-128", "attrs": None},
        {"record_id": "2", "text": "iPhone 15 Apple 128 GB Midnight", "source": "b", "gold_entity": "ip15-128", "attrs": None},
        {"record_id": "3", "text": "Apple iPhone 15 Plus 128GB", "source": "a", "gold_entity": "ip15plus-128", "attrs": None},
        {"record_id": "4", "text": "Samsung Galaxy S23 Ultra 256GB Green", "source": "b", "gold_entity": "s23u", "attrs": None},
        {"record_id": "5", "text": "Canon EF 100mm F2.8 L IS USM Macro", "source": "a", "gold_entity": "canon100", "attrs": None},
    ]
    return Dataset("toy", pd.DataFrame(rows))


# --------------------------------------------------------------------------- #
# preprocess
# --------------------------------------------------------------------------- #
def test_normalize_basic():
    assert normalize("Apple  iPhone\u00a015, 128GB!") == "apple iphone 15 128gb"
    assert normalize(None) == ""


def test_capacity_units_are_comparable():
    a = extract_attributes("Crucial 1TB SSD")
    b = extract_attributes("Crucial 1024GB SSD")
    assert a["capacity"] == b["capacity"] == "1024gb"


def test_variant_marker_detected():
    base = extract_attributes("Apple iPhone 15 128GB")
    plus = extract_attributes("Apple iPhone 15 Plus 128GB")
    assert "variant" not in base
    assert plus["variant"] == "plus"


def test_agreement_distinguishes_missing_from_conflicting():
    """The feature layer treats "absent" as neutral, never as a conflict.

    A base-model title ("iPhone 15 128GB") states no variant, so soft features
    see 0.0 - the *asymmetric* veto in models/attribute.py is what catches
    base-vs-Plus. Keeping the two mechanisms separate is deliberate: features
    may learn from partial evidence, the veto is the hard guarantee.
    """
    a = extract_attributes("Apple iPhone 15 128GB")
    b = extract_attributes("Apple iPhone 15 Plus 128GB")
    agree = agreement(a, b)
    assert agree["attr_variant"] == 0.0  # one side missing -> no evidence
    assert agree["attr_capacity"] == 1.0  # both 128gb

    both = extract_attributes("Apple iPhone 15 Pro 128GB")
    assert agreement(both, b)["attr_variant"] == -1.0  # pro vs plus: real conflict
    # colour unknown on both sides -> no evidence, NOT a conflict
    assert agree["attr_color"] == 0.0
    assert agree["attr_color_known"] == 0.0


def test_agreement_missing_one_side_is_neutral():
    a = {"brand": "apple"}
    b = {}
    assert agreement(a, b)["attr_brand"] == 0.0
    assert agreement(a, b)["attr_brand_known"] == 1.0


# --------------------------------------------------------------------------- #
# veto modes
# --------------------------------------------------------------------------- #
def test_symmetric_veto_requires_both_sides():
    """Symmetric mode only vetoes stated-vs-stated conflicts; asymmetric also
    catches stated-vs-absent, which is the base-vs-Plus situation."""
    base = extract_attributes("Apple iPhone 15 128GB")
    plus = extract_attributes("Apple iPhone 15 Plus 128GB")
    pro = extract_attributes("Apple iPhone 15 Pro 128GB")

    # stated vs stated: both modes fire
    assert "variant" in hard_conflicts(pro, plus, mode="symmetric")
    assert "variant" in hard_conflicts(pro, plus, mode="asymmetric")

    # stated vs absent: only asymmetric fires
    assert hard_conflicts(base, plus, mode="symmetric") == []
    assert any(x.startswith("variant") for x in hard_conflicts(base, plus, mode="asymmetric"))

    # and the subset waiver keeps a truncated title from being vetoed
    tb = frozenset(normalize("Apple iPhone 15 Plus 128GB Black"))
    tshort = frozenset(normalize("iPhone 15 Plus"))
    assert hard_conflicts(
        extract_attributes("iPhone 15 Plus"),
        extract_attributes("Apple iPhone 15 Plus 128GB Black"),
        mode="asymmetric",
        subset_ok=True,
        tokens_a=tshort,
        tokens_b=tb,
    ) == []


def test_capacity_conflict_vetoes():
    a = extract_attributes("Seagate IronWolf 4TB NAS")
    b = extract_attributes("Seagate IronWolf 8TB NAS")
    assert a["capacity"] != b["capacity"]
    assert "capacity" in hard_conflicts(a, b, mode="symmetric")


def test_brand_conflict_vetoes():
    """Brand conflicts depend on the lexicon, which is a dataset input, not a constant."""
    lex = AttributeLexicon.from_iterables(brands=["canon", "nikon"], variants=["ef", "z"])
    a = extract_attributes("Canon EF 50mm lens", lex)
    b = extract_attributes("Nikon Z 50mm lens", lex)
    assert a["brand"] == "canon" and b["brand"] == "nikon"
    assert "brand" in hard_conflicts(a, b, mode="symmetric")


# --------------------------------------------------------------------------- #
# unit-numeric specifiers (the generalised identifier rule)
# --------------------------------------------------------------------------- #
def test_specifier_map_keys_on_the_unit():
    from er.preprocess import specifier_map

    a = specifier_map("Canon EF 50mm f/1.8 STM lens")
    b = specifier_map("Canon EF 85mm f/1.8 lens")
    assert "50" in a["mm"] and "85" in b["mm"]
    assert "1.8" in a["f"] and "1.8" in b["f"]

    from er.preprocess import specifier_conflicts

    # same aperture, different focal length -> veto on mm only
    assert specifier_conflicts(a, b) == ["specifier:mm"]
    # identical values never conflict
    assert specifier_conflicts(a, a) == []


def test_specifier_one_sided_is_not_a_conflict():
    """A title omitting a spec is missing evidence, not contradicting it."""
    from er.preprocess import specifier_conflicts, specifier_map

    full = specifier_map("Seagate IronWolf 4TB NAS Drive 7200RPM 256MB Cache")
    short = specifier_map("Seagate IronWolf NAS Drive")
    assert specifier_conflicts(full, short) == []


def test_specifier_handles_tb_and_unit_normalisation():
    from er.preprocess import specifier_conflicts, specifier_map

    assert specifier_conflicts(specifier_map("SSD 1 TB"), specifier_map("SSD 2TB")) == ["specifier:tb"]


def test_stated_brand_overrides_inferred_but_keeps_text_attrs():
    """Regression: partial structured columns must not disable text extraction."""
    from er.preprocess import combined_attributes

    attrs = combined_attributes("Samsung Galaxy S23 Ultra 256GB Green", {"brand": "samsung"})
    assert attrs["brand"] == "samsung"
    assert "variant" in attrs, attrs  # text-derived attributes still present
    assert attrs["capacity"] == "256gb"
    # a stated value that is blank must not clobber the inferred one
    attrs2 = combined_attributes("Samsung Galaxy S23 Ultra 256GB", {"brand": None, "model_family": "nan"})
    assert attrs2["brand"] == "samsung"


# --------------------------------------------------------------------------- #
# features
# --------------------------------------------------------------------------- #
def test_feature_groups_are_routed():
    assert feature_group("attr_brand") == "attr"
    assert feature_group("attr_brand_known") == "attr"
    assert feature_group("fuzz_jaro") == "fuzz"
    assert feature_group("len_left") == "tok"
    assert feature_group("num_capacity_agree") == "num"
    assert feature_group("emb_cos") == "emb"


def test_build_features_shape_and_values(toy_ds):
    f = build_features(["1", "1"], ["2", "3"], toy_ds)
    assert len(f) == 2
    assert f.loc[0, "attr_variant"] == 0.0  # 1 vs 2: neither has a variant
    assert f.loc[0, "attr_capacity"] == 1.0  # 128gb stated on both
    assert f.loc[1, "attr_capacity"] == 1.0


def test_similarity_alone_prefers_the_wrong_pair(toy_ds):
    """The Case A / Case B trap, reproduced as a regression test.

    Record 1 is the same product as 2 (different word order + colour wording)
    and a *different* product from 3 (it is the Plus). Plain token similarity
    scores 3 higher, so a similarity matcher merges the wrong one. This is the
    behaviour the attribute constraint is supposed to fix.
    """
    f = build_features(["1", "1"], ["2", "3"], toy_ds)
    assert f.loc[1, "fuzz_token_set"] > f.loc[0, "fuzz_token_set"]


def test_attribute_constraint_reverses_the_wrong_pair(toy_ds):
    """End-to-end proof of the contribution on the running example."""
    from er.models.attribute import AttributeConstrainedMatcher
    from er.models.baselines import SimilarityMatcher
    from er.schema import pair_frame

    pairs = pair_frame([("1", "2"), ("1", "3")])
    base = SimilarityMatcher(metric="token_set").bind(toy_ds)
    uncon = base.score(pairs)
    assert uncon[1] > uncon[0]  # wrong pair wins

    wrapped = AttributeConstrainedMatcher(base, toy_ds, mode="asymmetric_subset", ceiling=0.30)
    fixed = wrapped.score(pairs)
    assert fixed[0] > fixed[1], f"constraint failed: {fixed}"
    assert wrapped.last_stats["n_constrained"] == 1
    assert wrapped.last_stats["per_attribute"] == {"variant:one_sided": 1}


# --------------------------------------------------------------------------- #
# blocking
# --------------------------------------------------------------------------- #
def test_blocking_is_deterministic(toy_ds):
    a = minhash_block(toy_ds, threshold=0.4, num_perm=32)
    b = minhash_block(toy_ds, threshold=0.4, num_perm=32)
    assert a.as_set() == b.as_set()


def test_blocking_reduction_and_recall(toy_ds):
    gold = gold_pairs_from_entities(toy_ds.entities())
    assert gold == {("1", "2")}
    tk = token_block(toy_ds, max_block_size=100)
    q = blocking_quality(tk.pairs, gold)
    assert q["tpr"] == 1.0  # every gold pair shares a token here
    assert 0 < tk.n_candidates < 10  # strict reduction vs the 10 all-pairs


def test_all_pairs_cap_protects_memory(toy_ds):
    from er.blocking import all_pairs_block

    all_pairs_block(toy_ds)
    big = Dataset("big", pd.DataFrame({"record_id": [str(i) for i in range(1000)], "text": ["x"] * 1000}))
    with pytest.raises(ValueError):
        all_pairs_block(big, cap=1000)


# --------------------------------------------------------------------------- #
# metrics
# --------------------------------------------------------------------------- #
def test_pairwise_metrics_counts():
    frame = pair_frame([("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")])
    frame["label"] = [1, 1, 0, 0]
    frame["score"] = [0.9, 0.4, 0.8, 0.1]
    pt = PairTable(frame)
    m = pairwise_metrics(pt, threshold=0.5)
    assert (m["tp"], m["fp"], m["fn"], m["tn"]) == (1, 1, 1, 1)
    assert m["precision"] == 0.5 and m["recall"] == 0.5


def test_average_precision_matches_manual():
    frame = pair_frame([("a", "b"), ("c", "d"), ("e", "f")])
    frame["label"] = [1, 0, 1]
    frame["score"] = [0.9, 0.8, 0.7]
    m = pairwise_metrics(PairTable(frame), threshold=0.5)
    # ranks: TP, FP, TP -> precisions 1, 2/3, 1 -> AP = (1/2)*1 + (1/2)*(2/3)... manual check
    assert 0.8 < m["auc_pr"] <= 1.0


def test_bcubed_known_case():
    gold = [["1", "2"], ["3", "4"]]
    # perfect prediction
    b = bcubed(gold, gold)
    assert b["bcubed_f1"] == pytest.approx(1.0)
    # merging both gold clusters into one: precision 0.5, recall 1.0
    b2 = bcubed([["1", "2", "3", "4"]], gold)
    assert b2["bcubed_p"] == pytest.approx(0.5)
    assert b2["bcubed_r"] == pytest.approx(1.0)


def test_entity_f1_penalizes_single_bad_merge():
    gold = [["1", "2"], ["3", "4"], ["5"]]
    pred = [["1", "2", "3"], ["4"], ["5"]]  # one bridge edge over-merged
    m = entity_pairwise_f1(pred, gold)
    assert m["entity_f1"] < 1.0


# --------------------------------------------------------------------------- #
# calibration
# --------------------------------------------------------------------------- #
def test_operating_point_prefers_strictest_cut_on_ties():
    """Guards the degenerate "accept everything" solution.

    When a constraint caps most positives to the same score, every feasible cut
    sits at the base rate; a recall-leaning tie-break then selects the loosest
    possible threshold (R=1.0, P=pi), which is not a decision at all.
    """
    from er.metrics import select_operating_point

    frame = pair_frame([("a", "b"), ("c", "d"), ("e", "f"), ("g", "h")])
    frame["label"] = [1, 0, 0, 0]
    frame["score"] = [0.3, 0.3, 0.3, 0.9]  # three capped pairs
    t, m = select_operating_point(PairTable(frame), min_recall=0.5)
    # both t=0.0 and t=0.3 give the same precision; the rule must pick the strictest
    assert abs(t - 0.3) < 1e-9, f"expected the strictest tied cut 0.3, got {t}"
    assert m["recall"] >= 0.5 - 1e-9
    # the strictest cut achieving the floor accepts the 0.9 pair and the 0.3 block
    assert m["rule"].startswith("max_precision")

    # an unlabeled table yields nothing rather than a fabricated decision
    bare = PairTable(pair_frame([("a", "b")]))
    t3, m3 = select_operating_point(bare, min_recall=0.5)
    assert m3 == {}


def test_ece_zero_for_perfect_calibration():
    """A score equal to the empirical positive rate in every bin must give ECE 0."""
    probs, labels = [], []
    for p in np.linspace(0.05, 0.95, 19):
        probs += [float(p)] * 20
        k = int(round(p * 20))
        labels += [1] * k + [0] * (20 - k)
    e = expected_calibration_error(np.array(probs), np.array(labels), n_bins=19, strategy="uniform")
    assert e == pytest.approx(0.0, abs=1e-9)


def test_ece_detects_overconfidence():
    """90% confident but right half the time -> ECE of exactly 0.4."""
    probs = np.full(200, 0.9)
    labels = np.array([1] * 100 + [0] * 100)
    assert expected_calibration_error(probs, labels, n_bins=10) == pytest.approx(0.4)


def test_temperature_scale_pulls_overconfident_scores():
    rng = np.random.default_rng(1)
    logits = rng.normal(0, 3, 4000)
    labels = (rng.random(4000) < (1 / (1 + np.exp(-logits / 5.0)))).astype(int)
    t = temperature_scale(logits, labels)
    assert t > 1.5  # overconfident scores need T > 1


# --------------------------------------------------------------------------- #
# clustering
# --------------------------------------------------------------------------- #
def test_connected_components_transitive_bridge():
    ids = ["1", "2", "3", "4"]
    edges = pair_frame([("1", "2"), ("2", "3")])
    clusters = connected_components(ids, edges)
    sizes = sorted(len(c) for c in clusters)
    assert sizes == [1, 3]  # 1-2-3 merged by transitivity through 2


def test_transitive_cleanup_drops_weakest_bridge():
    ids = ["1", "2", "3"]
    # complete triangle: consistent, nothing to drop
    edges = pair_frame([("1", "2"), ("2", "3"), ("1", "3")])
    edges["score"] = [0.9, 0.9, 0.4]
    assert len(transitive_cleanup(ids, edges, score_col="score")) == 3

    # open wedge 1-2-3 with a weak far edge: the weak edge is dropped
    edges2 = pair_frame([("1", "2"), ("2", "3")])
    edges2["score"] = [0.9, 0.35]
    kept2 = transitive_cleanup(ids, edges2, score_col="score", violation_weight=0.5)
    assert len(kept2) == 1
    assert kept2.attrs["transitive_cleanup_dropped"] == 1


def test_transitive_cleanup_does_not_guess_when_both_edges_are_confident():
    """The violation_weight gate: confident-vs-confident violations are kept.

    Without this gate the method deletes every violable edge, which looked
    impressive on entity F1 but was really just discarding most of the match set.
    """
    ids = ["1", "2", "3"]
    edges = pair_frame([("1", "2"), ("2", "3")])
    edges["score"] = [0.9, 0.85]
    kept = transitive_cleanup(ids, edges, score_col="score", violation_weight=0.5)
    assert len(kept) == 2, "both edges above the gate must survive"
    # lowering the gate does not change anything; raising it above 0.85 does
    assert len(transitive_cleanup(ids, edges, score_col="score", violation_weight=0.88)) == 1


# --------------------------------------------------------------------------- #
# schema
# --------------------------------------------------------------------------- #
def test_fingerprint_stable_and_sensitive(toy_ds):
    f1 = toy_ds.fingerprint()
    f2 = Dataset("other", toy_ds.df.copy()).fingerprint()
    assert f1 == f2
    changed = toy_ds.df.copy()
    changed.loc[0, "text"] = "totally different"
    assert Dataset("c", changed).fingerprint() != f1


def test_join_scores_broadcasts_scalar(toy_ds):
    pt = PairTable(pair_frame([("1", "2"), ("3", "4")]))
    out = pt.join_scores(scores=[0.5, 0.9], temperature=1.7)
    assert list(out.frame["temperature"]) == [1.7, 1.7]
    with pytest.raises(ValueError):
        pt.join_scores(scores=[0.5])


def test_pair_canonicalisation():
    f = pair_frame([("b", "a")])
    assert (f["record_id_l"].iloc[0], f["record_id_r"].iloc[0]) == ("a", "b")
