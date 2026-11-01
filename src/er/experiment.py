"""Experiment runner: fit -> calibrate -> evaluate, with slices and ablations.

The protocol this implements is the one the WDC Products paper uses, so results
are comparable with published Ditto/RoBERTa/Magellan numbers rather than being a
private metric:

1. fit the matcher on the *train* pairs only;
2. fit the temperature on the *valid* pairs only (never on test);
3. evaluate on the *_gs* test pairs, reported overall and sliced by
   ``is_hard_negative`` and by the unseen-product share of the split.

Slicing is the point. Aggregate F1 hides exactly the behaviour the project is
about: a constraint that trades 1 point of overall F1 for a 30-point drop in
hard-negative false merges is a good result, and a single-number table would
call it a regression.

Everything is written as tidy rows (one row per system x dataset x slice) so
that a paper table is a ``pivot_table`` away, and every row carries the dataset
fingerprint and git-style config dump for provenance.
"""

from __future__ import annotations

import json
import sys
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import average_precision_score

from .calibration import brier_score, expected_calibration_error, reliability_table, temperature_scale, selective_report
from .clustering import cluster_stats, connected_components
from .features import build_features
from .metrics import (
    pairwise_metrics,
    precision_at_recall,
    select_operating_point,
)
from .models.attribute import AttributeConstrainedMatcher, HARD_ATTRIBUTES
from .preprocess import DEFAULT_LEXICON, AttributeLexicon
from .schema import Dataset, PairTable

#: logit with clamping; used to convert probabilities to a temperature-tunable scale.
def _logit(p: np.ndarray, eps: float = 1e-6) -> np.ndarray:
    p = np.clip(np.asarray(p, dtype=float), eps, 1 - eps)
    return np.log(p / (1 - p))


def _sigmoid(z: np.ndarray, t: float) -> np.ndarray:
    return 1.0 / (1.0 + np.exp(-np.asarray(z, dtype=float) / max(t, 1e-6)))


@dataclass
class SystemSpec:
    """A named system in the comparison table."""

    name: str
    kind: str  # similarity | classifier | embedding
    base: dict = field(default_factory=dict)
    constraints: dict | None = None  # None = unconstrained
    group: str = "core"

    def describe(self) -> str:
        parts = [f"{self.kind}:{self.name}"]
        if self.constraints:
            parts.append(f"constraints({self.constraints.get('mode')})")
        return " + ".join(parts)


def build_matcher(spec: SystemSpec, ds: Dataset, lexicon: AttributeLexicon, embeddings=None):
    """Instantiate the matcher described by a spec, unconstrained then wrapped."""
    if spec.kind == "similarity":
        from .models.baselines import SimilarityMatcher

        m = SimilarityMatcher(metric=spec.base.get("metric", "token_set"))
        m = m.bind(ds)
    elif spec.kind == "wordcooc":
        from .models.baselines import WordCooccurrenceMatcher

        m = WordCooccurrenceMatcher(ds)
    elif spec.kind == "crossencoder":
        from .models.crossencoder import CrossEncoderMatcher

        kwargs = {k: v for k, v in spec.base.items() if k != "model"}
        m = CrossEncoderMatcher(ds, model_name=spec.base.get("model", "roberta-base"), **kwargs)
    elif spec.kind == "classifier":
        from .models.baselines import ClassifierMatcher

        m = ClassifierMatcher(
            ds,
            feature_groups=set(spec.base.get("groups", ["fuzz", "tok", "attr", "num"])),
            model=spec.base.get("model", "gbdt"),
            lexicon=lexicon,
            embeddings=embeddings,
            random_state=int(spec.base.get("seed", 0)),
        )
    elif spec.kind == "embedding":
        from .models.encoders import EmbeddingMatcher

        m = EmbeddingMatcher(ds, model_name=spec.base.get("model", "sentence-transformers/all-MiniLM-L6-v2"))
    else:
        raise ValueError(f"unknown kind {spec.kind!r}")

    if spec.constraints:
        m = AttributeConstrainedMatcher(
            m,
            ds,
            mode=spec.constraints.get("mode", "asymmetric_subset"),
            ceiling=float(spec.constraints.get("ceiling", 0.30)),
            hard=spec.constraints.get("hard", HARD_ATTRIBUTES),
            soft_extra=bool(spec.constraints.get("soft_extra", False)),
            lexicon=lexicon,
            use_specifiers=bool(spec.constraints.get("use_specifiers", True)),
        )
    return m


def run_system(
    spec: SystemSpec,
    ds: Dataset,
    train: PairTable,
    valid: PairTable,
    test: PairTable,
    lexicon: AttributeLexicon = DEFAULT_LEXICON,
    embeddings: np.ndarray | None = None,
    review_band: tuple[float, float] | None = None,
    review_delta: float = 0.10,
    ref_threshold: float | None = None,
    min_recall: float = 0.5,
    base_matcher: object | None = None,
    seed: int | None = None,
) -> tuple[list[dict], PairTable, object]:
    """Fit, calibrate and evaluate one system; returns tidy rows + scored pairs + matcher.

    Args:
        ref_threshold: an external operating point (usually the unconstrained
            baseline's chosen threshold). When set, an ``all_at_ref_threshold``
            slice is emitted so the constraint's effect is separable from the
            threshold sliding to compensate for it.
        base_matcher: an **already fitted** matcher to constrain, instead of building
            and fitting a fresh base. Two reasons this matters for a cross-encoder:
            retraining costs a full fine-tuning run per constrained variant, and a
            second random initialisation would mix seed noise into the very
            comparison the constraint is supposed to isolate. The returned matcher is
            the fitted one, so callers can pass it on.
    """
    t0 = time.time()
    if base_matcher is not None and spec.constraints:
        # wrap the shared, already-fitted base; do NOT refit it
        c = spec.constraints
        matcher = AttributeConstrainedMatcher(
            base_matcher,
            ds,
            mode=c.get("mode", "asymmetric_subset"),
            ceiling=float(c.get("ceiling", 0.30)),
            hard=c.get("hard", HARD_ATTRIBUTES),
            soft_extra=bool(c.get("soft_extra", False)),
            lexicon=lexicon,
            use_specifiers=bool(c.get("use_specifiers", True)),
        )
    else:
        matcher = build_matcher(spec, ds, lexicon, embeddings)
        matcher.fit(train)

    # --- operating point + calibration, chosen on VALID only ----------------
    # A hard constraint's benefit is invisible at an F1-optimal threshold,
    # because the threshold slides down and reads comparable noise back in. All
    # systems therefore use the same precision-forward rule (maximise precision
    # subject to recall >= min_recall), and the veto ceiling is tuned rather than
    # hard-coded. Neither the threshold nor the ceiling touches test data.
    y_valid = valid.labeled.frame["label"].astype(int).to_numpy()
    ceiling_used = None
    n_effective_valid = None
    valid_op: dict[str, float] = {}
    if spec.constraints and hasattr(matcher, "ceiling"):
        base_valid = np.asarray(matcher.base.score(valid.labeled.frame), dtype=float)  # type: ignore[attr-defined]
        grid = list(spec.constraints.get("ceiling_grid", [0.05, 0.15, 0.25, 0.35, 0.45]))
        trials = []
        for c in grid:
            capped = np.minimum(base_valid, c)
            t_c = temperature_scale(_logit(capped), y_valid) if len(capped) else 1.0
            cal = _sigmoid(_logit(capped), t_c)
            thr, vm = select_operating_point(valid.join_scores(scores=cal), min_recall=min_recall)
            if not vm or thr != thr:
                continue
            # Count pairs the cap actually moves across the boundary. An inert
            # ceiling would otherwise win by being indistinguishable from the
            # unconstrained system, and the table would describe no constraint.
            uncapped = _sigmoid(_logit(base_valid), t_c)
            effective = int(((cal < thr) & (uncapped >= thr)).sum())
            ap = average_precision_score(y_valid, cal)
            trials.append((ap, float(c), float(thr), t_c, effective, vm))
        if not trials:
            raise RuntimeError("no ceiling candidate produced a usable operating point")
        # Rank by a threshold-free criterion (valid AP), preferring ceilings that
        # bind; operating-point metrics are reported separately, not used to rank.
        best_ap = max(t[0] for t in trials)
        cands = [t for t in trials if abs(t[0] - best_ap) < 1e-12]
        binding = [t for t in cands if t[4] > 0]
        ap, ceiling_used, chosen, temp, n_effective_valid, valid_op = max(
            binding or cands, key=lambda x: x[4]
        )
        if not binding:
            print(
                f"  warning: {spec.name}: no ceiling in {grid} binds at its operating point "
                f"(the recall floor forces capped pairs back in)",
                file=sys.stderr,
            )
        matcher.ceiling = ceiling_used
        s_valid_raw = np.minimum(base_valid, ceiling_used)
    else:
        s_valid_raw = np.asarray(matcher.score(valid.labeled.frame), dtype=float)
        temp = temperature_scale(_logit(s_valid_raw), y_valid) if len(s_valid_raw) else 1.0
        chosen, valid_op = select_operating_point(
            valid.join_scores(scores=_sigmoid(_logit(s_valid_raw), temp)), min_recall=min_recall
        )

    # --- test scoring -------------------------------------------------------
    s_test_raw = np.asarray(matcher.score(test.labeled.frame), dtype=float)
    s_test = _sigmoid(_logit(s_test_raw), temp)

    # Score and slice on the labeled subset only: mixing an unlabeled candidate
    # into the scored table would silently change every denominator in the table.
    scored = test.labeled.join_scores(scores=s_test, score_raw=s_test_raw, temperature=temp)
    fit_seconds = time.time() - t0
    n_test = len(test.labeled.frame)
    try:
        thr = float(matcher.threshold)  # only similarity matchers carry one
    except AttributeError:
        thr = float("nan")

    # Review band tied to the actual operating point: the pairs the decision
    # throws away are exactly the pairs a reviewer could recover. A fixed
    # [0.35, 0.65] band would describe a policy unrelated to the threshold in use.
    if review_band is None:
        review_band = (max(0.0, float(chosen) - review_delta), float(chosen))

    # Detect an accept-everything operating point from VALID (never test), so the
    # diagnostic cannot leak and so no table row can be misread as a matcher score.
    _vr = valid_op.get("recall")
    degenerate = bool(_vr is not None and _vr == _vr and _vr >= 0.999)

    rows: list[dict] = []
    base_common = {
        "system": spec.name,
        "kind": spec.kind,
        # An accept-everything operating point is not a decision. Flag it so no
        # table can be read as "this matcher scored 0.111 precision" when what
        # actually happened is that the recall floor left no feasible threshold.
        "degenerate_operating_point": degenerate,
        "group": spec.group,
        "constraints": (spec.constraints or {}).get("mode", "none"),
        "dataset": test.meta.get("dataset"),
        "split": test.meta.get("split", "test"),
        "corner_cases": test.meta.get("corner_cases"),
        "unseen": test.meta.get("unseen"),
        "dev_size": train.meta.get("dev_size"),
        "n_test_pairs": n_test,
        "positives": int(test.labeled.frame["label"].sum()),
        "temperature": temp,
        "native_threshold": thr,
        "seed": seed,
        "ceiling_used": ceiling_used,
        "veto_binds_valid": n_effective_valid,
        "op_rule": valid_op.get("rule"),
        "review_lower": review_band[0],
        "review_upper": review_band[1],
        "min_recall": min_recall,
        "valid_precision": valid_op.get("precision"),
        "valid_recall": valid_op.get("recall"),
        "valid_f1": valid_op.get("f1"),
        "fit_seconds": round(fit_seconds, 2),
        "fingerprint": test.meta.get("fingerprint"),
        "config": json.dumps(asdict(spec), default=str),
    }
    if spec.constraints:
        base_common["n_constrained"] = matcher.last_stats.get("n_constrained")
        base_common["share_constrained"] = matcher.last_stats.get("share_constrained")

        # An *effective* veto is a pair whose un-capped score was above the
        # ceiling, i.e. one the constraint actually moved across the decision
        # boundary. Comparing raw vs calibrated scores would measure the
        # temperature instead of the constraint, so the base matcher is scored
        # again, uncapped, to recover the counterfactual.
        tdf = scored.frame
        base_raw = np.asarray(matcher.base.score(tdf), dtype=float)  # type: ignore[attr-defined]
        ceiling = float(matcher.ceiling)
        vetoed_mask = (base_raw > ceiling + 1e-9) & (tdf["score_raw"].to_numpy() <= ceiling + 1e-9)
        vetoed = tdf[vetoed_mask]
        n_vetoed = len(vetoed)
        if n_vetoed:
            y_v = vetoed["label"].astype(int)
            hard_v = (
                vetoed["is_hard_negative"].fillna(False).astype(bool)
                if "is_hard_negative" in vetoed.columns
                else pd.Series(False, index=vetoed.index)
            )
            rows.append(
                {
                    **base_common,
                    "slice": "veto_diagnostics",
                    "threshold": chosen,
                    "n_test_pairs": n_vetoed,
                    # a veto is useful when it blocks a non-match, harmful when it
                    # blocks a true match: this is the constraint's own precision
                    "veto_precision": float((y_v == 0).mean()),
                    "veto_true_matches": int((y_v == 1).sum()),
                    "veto_hard_negatives": int(hard_v.sum()),
                    "veto_recall_on_hard_neg": float(
                        hard_v[(y_v == 0)].sum() / max(1, int((tdf["label"] == 0).sum()))
                    ),
                    "per_attribute": json.dumps(matcher.last_stats.get("per_attribute", {})),
                }
            )

    def add(slice_name: str, pt: PairTable, thr: float) -> None:
        m = pairwise_metrics(pt, threshold=thr)
        row = dict(base_common)
        row.update(
            {
                "slice": slice_name,
                "threshold": thr,
                "precision": m["precision"],
                "recall": m["recall"],
                "f1": m["f1"],
                "auc_pr": m.get("auc_pr", float("nan")),
                "roc_auc": m.get("roc_auc", float("nan")),
                "fp": m["fp"],
                "fn": m["fn"],
                "tp": m["tp"],
            }
        )
        if thr == thr and len(pt.labeled.frame):
            probs = pt.labeled.frame["score"].astype(float).to_numpy()
            lab = pt.labeled.frame["label"].astype(int).to_numpy()
            row["p_at_r50"] = precision_at_recall(lab, probs, 0.50)
            row["p_at_r60"] = precision_at_recall(lab, probs, 0.60)
            row["ece"] = expected_calibration_error(probs, lab, n_bins=10, strategy="quantile")
            row["brier"] = brier_score(probs, lab)
            sel = selective_report(probs, lab, lower=review_band[0], upper=review_band[1])
            row.update({f"sel_{k}": v for k, v in sel.items()})
        rows.append(row)

    if ref_threshold is not None and ref_threshold == ref_threshold:
        # same operating point as the unconstrained reference: isolates the effect
        # of the constraint from the effect of sliding the threshold
        add("all_at_ref_threshold", scored, float(ref_threshold))

    # operating point was selected on calibrated VALID scores above (no test peeking)
    scored = scored.join_scores(threshold=chosen)
    add("all", scored, chosen)

    # hard-negative slice: the benchmark's own similar-but-different pairs.
    df = scored.frame
    hard = df[df["is_hard_negative"].fillna(False).astype(bool)] if "is_hard_negative" in df.columns else df.iloc[0:0]
    if len(hard):
        hard_pos = hard[hard["label"] == 1]
        hard_neg = hard[hard["label"] == 0]
        rows.append(
            {
                **base_common,
                "slice": "hard_negative_only",
                "threshold": chosen,
                "n_test_pairs": int(len(hard)),
                "positives": int(len(hard_pos)),
                "fp": int((hard_neg["score"].to_numpy() >= chosen).sum()),
                "fn": int((hard_pos["score"].to_numpy() < chosen).sum()),
                "tp": int((hard_pos["score"].to_numpy() >= chosen).sum()),
                "false_merge_rate": float((hard_neg["score"].to_numpy() >= chosen).mean()) if len(hard_neg) else float("nan"),
                "hard_recall": float((hard_pos["score"].to_numpy() >= chosen).mean()) if len(hard_pos) else float("nan"),
            }
        )
        # matched-but-should-not-be: the Case B population
        easy_neg = df[(df["label"] == 0) & ~df.index.isin(hard.index)]
        rows.append(
            {
                **base_common,
                "slice": "easy_negative_only",
                "threshold": chosen,
                "n_test_pairs": int(len(easy_neg)),
                "false_merge_rate": float((easy_neg["score"].to_numpy() >= chosen).mean()) if len(easy_neg) else float("nan"),
            }
        )

    # top-similarity slice: pairs a pure-similarity matcher would wrongly merge.
    sim_feats = build_features(
        df["record_id_l"], df["record_id_r"], ds, lexicon=lexicon
    )
    hi = sim_feats["fuzz_token_set"] >= 0.9
    if hi.any():
        sub = PairTable(df[hi.values].reset_index(drop=True), scored.meta)
        add("high_similarity_subset", sub, chosen)

    # Entity-level resolution, measured on the nodes the scored pairs cover.
    # WDC assigns each offer to exactly one split, so a gold cluster's train-split
    # members are not reachable from test pairs at all: clustering over the union
    # of splits would report a recall floor that is an artefact of the split
    # design rather than a property of the system.
    nodes = sorted(set(df["record_id_l"].astype(str)) | set(df["record_id_r"].astype(str)))
    accepted = df[df["score"].to_numpy() >= chosen][["record_id_l", "record_id_r"]]
    have_gold = "gold_entity" in ds.df.columns and ds.df["gold_entity"].notna().any()
    gold_ids = set(nodes)
    gmap = (
        ds.df[ds.df["record_id"].astype(str).isin(gold_ids) & ds.df["gold_entity"].notna()]
        if have_gold
        else ds.df.iloc[0:0]
    )
    gold_groups = (
        [sorted(g["record_id"].astype(str).tolist()) for _, g in gmap.groupby("gold_entity")]
        if have_gold
        else []
    )
    if gold_groups:
        from .metrics import entity_pairwise_f1

        clusters = connected_components(nodes, accepted)
        row = dict(base_common)
        row.update({"slice": "entity_level", "threshold": chosen, "n_nodes": len(nodes)})
        row.update(entity_pairwise_f1([c for c in clusters if len(c) > 1], gold_groups))
        row.update(cluster_stats(clusters))
        rows.append(row)

    return rows, scored, matcher


def reliability_rows(pt: PairTable, n_bins: int = 10) -> pd.DataFrame:
    """Reliability-diagram data for the calibration figure."""
    df = pt.labeled.frame
    if "score" not in df.columns:
        return pd.DataFrame()
    return reliability_table(df["score"].astype(float).to_numpy(), df["label"].astype(int).to_numpy(), n_bins)


def gold_groups_for_nodes(ds: Dataset, nodes: set[str]) -> list[list[str]]:
    """Gold clusters restricted to the records actually present in a node set.

    WDC Products assigns each offer to exactly one split, so a gold cluster's
    train-split members are unreachable from test pairs. Measuring clustering
    quality over all records would report a recall floor that is an artefact of
    the split design instead of a property of the system.
    """
    if "gold_entity" not in ds.df.columns:
        return []
    sub = ds.df[ds.df["record_id"].astype(str).isin(nodes) & ds.df["gold_entity"].notna()]
    return [sorted(g["record_id"].astype(str).tolist()) for _, g in sub.groupby("gold_entity")]


def clustering_sweep(
    ds: Dataset,
    scored: PairTable,
    thresholds: np.ndarray | None = None,
    use_cleanup: bool = False,
    violation_weight: float = 1.0,
    label: str = "",
) -> pd.DataFrame:
    """Sweep the acceptance threshold and measure the *entity-level* result.

    The point of this table: the threshold that maximises pairwise F1 is usually
    far from the one that maximises entity-level F1, because pairwise metrics
    count each edge once while clustering pays for a single wrong bridge edge
    many times over (it merges two whole clusters). Reporting only the pairwise
    optimum hides exactly the failure that matters downstream.

    Args:
        use_cleanup: apply :func:`er.clustering.transitive_cleanup` before
            clustering, to measure whether it recovers entity precision.
    """
    from .clustering import connected_components, transitive_cleanup
    from .metrics import entity_pairwise_f1

    df = scored.frame
    nodes = sorted(set(df["record_id_l"].astype(str)) | set(df["record_id_r"].astype(str)))
    gold = gold_groups_for_nodes(ds, set(nodes))
    if not gold:
        return pd.DataFrame()

    if thresholds is None:
        thresholds = np.unique(np.round(np.quantile(df["score"].astype(float), np.linspace(0.3, 0.99, 25)), 4))

    rows = []
    for thr in thresholds:
        accepted = df[df["score"].astype(float) >= thr][["record_id_l", "record_id_r", "score"]]
        n_before = int(len(accepted))
        n_dropped = 0
        if use_cleanup and len(accepted):
            accepted = transitive_cleanup(
                nodes, accepted, score_col="score", violation_weight=violation_weight
            )
            n_dropped = int(accepted.attrs.get("transitive_cleanup_dropped", 0))
        clusters = connected_components(nodes, accepted)
        m = entity_pairwise_f1([c for c in clusters if len(c) > 1], gold)
        m.update(cluster_stats(clusters))
        pw = pairwise_metrics(PairTable(df.assign(_p=(df["score"].astype(float) >= thr).astype(int))[
            ["record_id_l", "record_id_r", "label", "_p"]].rename(columns={"_p": "pred"})))
        rows.append(
            {
                "system": label,
                "threshold": float(thr),
                "cleanup": use_cleanup,
                "violation_weight": violation_weight if use_cleanup else None,
                "n_edges_before_cleanup": n_before,
                "n_accepted_edges": int(len(accepted)),
                "n_edges_dropped": n_dropped,
                "pairwise_precision": pw["precision"],
                "pairwise_recall": pw["recall"],
                "pairwise_f1": pw["f1"],
                **m,
            }
        )
    return pd.DataFrame(rows)



def case_table(pt: PairTable, ds: Dataset, n: int = 20, kind: str = "worst") -> pd.DataFrame:
    """Human-readable cases for qualitative error analysis.

    Papers are won and lost on these tables: a reviewer wants to see the actual
    titles behind a false merge.

    Args:
        kind: ``"worst"`` = largest score/label disagreements first;
            ``"false_merges"`` = predicted matches that are actually negatives.
    """
    df = pt.frame.copy()
    if "label" not in df.columns or "score" not in df.columns:
        return pd.DataFrame()
    d = df.dropna(subset=["label"]).copy()
    d["label"] = d["label"].astype(int)

    thr = d.get("threshold")
    t = float(thr.iloc[0]) if thr is not None and len(thr) else 0.5
    pred = d["score"].astype(float) >= t
    d["decision"] = np.where(pred & (d["label"] == 0), "false_merge", np.where(~pred & (d["label"] == 1), "false_split", "correct"))

    if kind == "false_merges":
        d = d[d["decision"] == "false_merge"]
    else:
        d = d.assign(_gap=(d["score"] - d["label"]).abs()).sort_values("_gap", ascending=False)

    out = d[["record_id_l", "record_id_r", "label", "score", "decision"]].head(n).copy()
    texts = ds.df.set_index(ds.df["record_id"].astype(str))["text"]
    out["text_l"] = texts.reindex(out["record_id_l"]).to_numpy()
    out["text_r"] = texts.reindex(out["record_id_r"]).to_numpy()
    return out.reset_index(drop=True)
