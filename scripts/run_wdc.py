#!/usr/bin/env python3
"""Run the main experiment: does an attribute constraint reduce false merges?

This is the experiment that produces the paper's central table. Systems compared
on the WDC Products benchmark (pair-wise), fit on train, calibrated on valid,
reported on the published test (gs) split:

    s1  token-set similarity           - string baseline
    s2  classifier (fuzz+tok only)     - similarity features, no attributes
    s3  classifier (+attr features)    - attributes as *soft* features
    s4  s3 + attribute constraint      - attributes as *hard* constraints (ours)
    s5  SBERT cosine                   - embedding baseline
    s6  SBERT + attribute constraint    - ours on top of embeddings

Reported per system, overall and sliced by hard negatives:
micro P/R/F1, AUC-PR, false-merge rate on hard negatives, ECE, review workload.

Usage:
    python scripts/run_wdc.py --data data/raw --corner 80 --unseen 000 [--with-embeddings]
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er.experiment import SystemSpec, run_system, reliability_rows  # noqa: E402
from er.io import load_wdc_products_pairwise, wdc_index  # noqa: E402
from er.schema import Dataset, PairTable  # noqa: E402

import er  # noqa: E402


def find_files(data_dir: Path, corner: int, unseen: int) -> dict[str, Path]:
    """Locate the train/valid/test files for one benchmark variant.

    WDC ships train/valid only for ``rnd000un``; the unseen dimension exists in
    the test (gs) files. So the same train/valid pair serves every unseen level,
    which is what makes the seen-vs-unseen columns of the paper table comparable.
    """
    idx = wdc_index(data_dir)
    if idx.empty:
        raise SystemExit(f"no WDC Products files under {data_dir}; run scripts/download_wdc.py")
    idx = idx[idx["corner_cases"] == int(corner)]
    out = {}
    for split in ("train", "valid", "test"):
        sub = idx[idx["split"] == split]
        if split == "test":
            sub = sub[sub["unseen"] == int(unseen)]
        else:
            sub = sub[sub["dev_size"] != "n/a"]
        if sub.empty:
            raise SystemExit(f"missing {split} file for corner={corner} unseen={unseen}")
        # default to medium development set when several are present
        pick = sub[sub["dev_size"] == "medium"]
        out[split] = Path((pick if not pick.empty else sub).iloc[0]["path"])
    return out


def specs(with_embeddings: bool, with_crossencoder: bool = False, model: str = "roberta-base",
          epochs: int = 3, max_steps: int | None = None, seed: int = 0) -> list[SystemSpec]:
    s = [
        SystemSpec("s1_similarity_tokenset", "similarity", {"metric": "token_set"}, group="baseline"),
        SystemSpec("s1b_similarity_jaro", "similarity", {"metric": "jaro_winkler"}, group="baseline"),
        SystemSpec("s1c_wordcooc_svm", "wordcooc", {}, group="baseline"),
        SystemSpec(
            "s1d_wordcooc_constraint",
            "wordcooc",
            {},
            constraints={"mode": "asymmetric_subset", "ceiling": 0.30, "use_specifiers": True, "ref": "s1c_wordcooc_svm"},
            group="ours",
        ),
        SystemSpec("s2_classifier_nofeatattr", "classifier", {"groups": ["fuzz", "tok", "num"]}, group="baseline"),
        SystemSpec("s3_classifier_featattr", "classifier", {"groups": ["fuzz", "tok", "num", "attr"]}, group="ours"),
        # typed attribute veto only (no unit-numeric specifiers)
        SystemSpec(
            "s4a_typed_veto_asym",
            "classifier",
            {"groups": ["fuzz", "tok", "num", "attr"]},
            constraints={"mode": "asymmetric_subset", "ceiling": 0.30, "use_specifiers": False},
            group="ablation",
        ),
        # symmetric only: stated-vs-stated
        SystemSpec(
            "s4b_typed_veto_sym",
            "classifier",
            {"groups": ["fuzz", "tok", "num", "attr"]},
            constraints={"mode": "symmetric", "ceiling": 0.30, "use_specifiers": False},
            group="ablation",
        ),
        # full constraint: typed attributes + unit-numeric specifiers
        SystemSpec(
            "s5_full_constraint",
            "classifier",
            {"groups": ["fuzz", "tok", "num", "attr"]},
            constraints={"mode": "asymmetric_subset", "ceiling": 0.30, "use_specifiers": True},
            group="ours",
        ),
    ]
    if with_embeddings:
        s += [
            SystemSpec("s6_sbert_cosine", "embedding", {"model": "sentence-transformers/all-MiniLM-L6-v2"}, group="baseline"),
            SystemSpec(
                "s7_sbert_full_constraint",
                "embedding",
                {"model": "sentence-transformers/all-MiniLM-L6-v2"},
                constraints={"mode": "asymmetric_subset", "ceiling": 0.30, "use_specifiers": True},
                group="ours",
            ),
        ]
    if with_crossencoder:
        ce_base = {
            "model": model,
            "epochs": epochs,
            "seed": seed,
        }
        if max_steps:
            ce_base["max_steps"] = max_steps
        s += [
            SystemSpec("s8_crossencoder", "crossencoder", dict(ce_base), group="baseline"),
            SystemSpec(
                "s9_crossencoder_constraint",
                "crossencoder",
                dict(ce_base),
                constraints={
                    "mode": "asymmetric_subset",
                    "ceiling": 0.30,
                    "use_specifiers": True,
                    "ref": "s8_crossencoder",
                },
                group="ours",
            ),
        ]
    # Every stochastic component reads this from the spec, so one seed value flows
    # through the classifier, the embedding PCA and any fine-tuned model.
    for spec in s:
        spec.base.setdefault("seed", seed)
    return s


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--corner", type=int, default=80, help="corner-case share: 80|50|20")
    ap.add_argument("--unseen", type=int, default=0, help="unseen product share in test: 0|50|100")
    ap.add_argument("--dev-size", default="medium", help="train/valid development-set size")
    ap.add_argument("--out", default="results")
    ap.add_argument("--with-embeddings", action="store_true")
    ap.add_argument("--with-crossencoder", action="store_true",
                    help="fine-tune a cross-encoder (GPU recommended; use --max-steps to smoke-test)")
    ap.add_argument("--model", default="roberta-base", help="encoder for the cross-encoder")
    ap.add_argument("--epochs", type=int, default=3)
    ap.add_argument("--max-steps", type=int, default=None,
                    help="cap optimiser steps (smoke test); leaves limited training otherwise")
    ap.add_argument("--no-description", action="store_true", help="serialize title only")
    ap.add_argument("--tag-suffix", default="",
                    help="suffix for output filenames, e.g. '_ce' so GPU runs do not overwrite CPU rows")
    ap.add_argument("--seeds", default="0",
                    help="comma-separated seeds, e.g. '0,1,2'; each row records the seed that produced it")
    ap.add_argument("--train-subsample", type=float, default=1.0,
                    help="fraction of TRAIN pairs resampled per seed (e.g. 0.9). Required for "
                         "meaningful variance: our deterministic learners give byte-identical "
                         "results across seeds, so only resampling the training data produces "
                         "run-to-run spread that reflects generalisation, not optimiser noise")
    ap.add_argument("--only-systems", default=None,
                    help="comma-separated system names to run (default: all); use with a tag suffix")
    args = ap.parse_args()

    data_dir = Path(args.data)
    out_dir = Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    files = find_files(data_dir, args.corner, args.unseen)
    if args.dev_size != "medium":
        idx = wdc_index(data_dir)
        for split in ("train", "valid"):
            sub = idx[
                (idx.corner_cases == args.corner) & (idx.split == split) & (idx.dev_size == args.dev_size)
            ]
            if not sub.empty:
                files[split] = Path(sub.iloc[0]["path"])

    inc_desc = not args.no_description
    ds_tr, tr = load_wdc_products_pairwise(files["train"], inc_desc)
    ds_va, va = load_wdc_products_pairwise(files["valid"], inc_desc)
    ds_te, te = load_wdc_products_pairwise(files["test"], inc_desc)

    # one Dataset spanning every record any split references: attributes and
    # embeddings are per-record, and train/test records are disjoint in WDC.
    all_records = pd.concat([ds_tr.df, ds_va.df, ds_te.df], ignore_index=True).drop_duplicates("record_id")
    ds = Dataset(f"wdc{args.corner}cc_{args.unseen}un_all", all_records)

    print(f"records={ds.n}  train={len(tr.frame)} valid={len(va.frame)} test={len(te.frame)} "
          f"test_positives={int(te.labeled.frame['label'].sum())} "
          f"test_hard_negatives={te.meta.get('hard_negatives')}")

    embeddings = None
    if args.with_embeddings:
        from er.models.encoders import encode_texts

        texts = ds.df["text"].astype(str).tolist()
        embeddings = encode_texts(texts, fingerprint=ds.fingerprint())
        print(f"embeddings: {embeddings.shape}")

    rows: list[dict] = []
    scored_store: dict[str, object] = {}
    fitted: dict[str, object] = {}
    chosen_thr: dict[str, float] = {}
    # The unconstrained systems are the operating-point reference: constrained
    # variants are additionally scored at the reference threshold so a reviewer
    # can see the constraint's effect without the threshold sliding to hide it.
    REFERENCE = "s3_classifier_featattr"
    EMB_REFERENCE = "s6_sbert_cosine"
    # Seeds are a recorded dimension, not a silent rerun: every row carries the seed
    # that produced it, so variance can be reported instead of assumed away.
    seeds = [int(x) for x in str(args.seeds).split(",") if x.strip()]
    plan = [
        (sd, sp)
        for sd in seeds
        for sp in specs(
            args.with_embeddings, args.with_crossencoder, args.model, args.epochs,
            args.max_steps, seed=sd,
        )
    ]
    cur_seed: int | None = None
    for seed, spec in plan:
        if seed != cur_seed:
            # fitted weights and reference thresholds are per-seed; carrying them across
            # seeds would let one seed's operating point leak into another's rows.
            fitted.clear()
            chosen_thr.clear()
            cur_seed = seed
            if len(seeds) > 1:
                print(f"\n===== seed {seed} ({len(seeds)} seeds) =====")
        if args.only_systems:
            wanted = {x.strip() for x in args.only_systems.split(",") if x.strip()}
            if spec.name not in wanted:
                continue
        ref = None
        base_matcher = None
        if spec.constraints:
            # each constrained system is compared at the operating point of ITS
            # own unconstrained base matcher (wordcooc vs wordcooc, sbert vs sbert)
            ref_name = spec.constraints.get("ref")
            if ref_name is None:
                ref_name = EMB_REFERENCE if spec.kind == "embedding" else REFERENCE
            ref = chosen_thr.get(ref_name)
            base_matcher = fitted.get(ref_name)
            if ref is None:
                print(f"  note: reference {ref_name} not run yet; skipping ref-threshold slice", file=sys.stderr)
            if base_matcher is not None:
                print(f"  reusing fitted base from {ref_name} (no refit)")
        # Deterministic learners (similarity, SVM, GBDT, SBERT) are invariant to the
        # random seed, so resampling the training pairs is the only way to obtain
        # variance that means anything. See docs/FINDINGS.md.
        tr_fit = tr
        if args.train_subsample < 1.0:
            n = max(2, int(round(args.train_subsample * len(tr.labeled.frame))))
            sub = tr.labeled.frame.sample(n=n, random_state=seed).reset_index(drop=True)
            tr_fit = PairTable(sub, {**tr.meta, "subsample": args.train_subsample, "n_fit_pairs": n})
        try:
            r, scored, fitted_matcher = run_system(
                spec, ds, tr_fit, va, te, embeddings=embeddings, ref_threshold=ref,
                base_matcher=base_matcher, seed=seed,
            )
            rows.extend(r)
            scored_store[spec.name] = scored
            fitted[spec.name] = fitted_matcher
            alls = [x for x in r if x["slice"] == "all"]
            if alls:
                d = alls[0]
                chosen_thr[spec.name] = float(d["threshold"])
                extra = ""
                if d.get("p_at_r50") == d.get("p_at_r50"):
                    extra = f" P@R50={d['p_at_r50']:.4f}"
                vd = [x for x in r if x["slice"] == "veto_diagnostics"]
                vtxt = (
                    f" vetoed={vd[0]['n_test_pairs']:.0f} vetoP={vd[0]['veto_precision']:.3f}"
                    if vd
                    else ""
                )
                print(
                    f"  {spec.name:30s} F1={d['f1']:.4f} P={d['precision']:.4f} R={d['recall']:.4f} "
                    f"AUC-PR={d.get('auc_pr', float('nan')):.4f}{extra}{vtxt}"
                )
        except Exception as exc:  # keep the sweep alive; a failed row is visible, not silent
            print(f"  {spec.name:38s} FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
            rows.append({"system": spec.name, "slice": "all", "error": f"{type(exc).__name__}: {exc}"})

    res = pd.DataFrame(rows)
    tag = f"wdc{args.corner}cc_{args.unseen}un_dev{args.dev_size}" + args.tag_suffix
    res_path = out_dir / f"{tag}_results.csv"
    res.to_csv(res_path, index=False)
    print(f"\nwrote {res_path} ({len(res)} rows)")

    # the headline comparison table
    keep = ["system", "group", "slice", "threshold", "precision", "recall", "f1", "auc_pr",
            "p_at_r50", "p_at_r60", "false_merge_rate", "veto_precision", "veto_true_matches",
            "n_constrained", "ece", "sel_review_share", "sel_precision_auto",
            "entity_precision", "entity_recall", "entity_f1"]
    cols = [c for c in keep if c in res.columns]
    table = res[res["slice"].isin(
        ["all", "all_at_ref_threshold", "hard_negative_only", "easy_negative_only",
         "high_similarity_subset", "veto_diagnostics", "entity_level"]
    )][cols]
    table.to_csv(out_dir / f"{tag}_table.csv", index=False)
    print("\n=== results ===")
    with pd.option_context("display.width", 200, "display.max_columns", 30, "display.float_format", "{:.4f}".format):
        print(table.to_string(index=False))

    # Persist the matching artifacts FIRST: a failure in the clustering sweep
    # below must not cost the scored tables of a multi-minute run.
    for name, scored in scored_store.items():
        scored.to_csv(out_dir / f"{tag}_scored_{name}.csv")
    for name in ("s3_classifier_featattr", "s5_full_constraint"):
        if name in scored_store:
            rel = reliability_rows(scored_store[name])
            if not rel.empty:
                rel.insert(0, "system", name)
                rel.to_csv(out_dir / f"{tag}_reliability_{name}.csv", index=False)

    # --- clustering stage: the pairwise-optimal threshold is not the entity-optimal one
    from er.experiment import clustering_sweep

    for name in [n for n in ("s3_classifier_featattr", "s5_full_constraint") if n in scored_store]:
        sc = scored_store[name]
        for cleanup, vw in [(False, 0.5), (True, 0.4), (True, 0.6), (True, 0.8)]:
            try:
                sweep = clustering_sweep(ds, sc, use_cleanup=cleanup, violation_weight=vw, label=name)
            except Exception as exc:  # one bad sweep config must not lose the rest of the run
                print(f"{name} cleanup={vw} SWEEP FAILED: {type(exc).__name__}: {exc}", file=sys.stderr)
                continue
            if sweep.empty:
                continue
            tagname = f"{name}_{'clean' + str(vw) if cleanup else 'naive'}"
            sweep.to_csv(out_dir / f"{tag}_clustering_{tagname}.csv", index=False)
            best = sweep.loc[sweep["entity_f1"].idxmax()]
            pw_best = sweep.loc[sweep["pairwise_f1"].idxmax()]
            print(
                f"{name} cleanup={vw if cleanup else 'naive '}: "
                f"pairwise-opt thr={pw_best['threshold']:.3f} -> entity F1={pw_best['entity_f1']:.4f} "
                f"(max cl={pw_best['max_size']:.0f}) | best entity F1={best['entity_f1']:.4f} "
                f"@thr={best['threshold']:.3f} (P={best['entity_precision']:.3f} R={best['entity_recall']:.3f} "
                f"dropped={best['n_edges_dropped']:.0f})"
            )
    from er.experiment import case_table

    for name in ("s3_classifier_featattr", "s5_full_constraint"):
        if name in scored_store:
            ct = case_table(scored_store[name], ds, n=15, kind="false_merges")
            if not ct.empty:
                ct.to_csv(out_dir / f"{tag}_false_merges_{name}.csv", index=False)
                print(f"\n{name}: {len(ct)} false-merge cases -> {out_dir / f'{tag}_false_merges_{name}.csv'}")

    meta = {
        "dataset": ds.name,
        "fingerprint": ds.fingerprint(),
        "files": {k: str(v) for k, v in files.items()},
        "counts": tr.meta | {"test": te.meta},
        "python": sys.version.split()[0],
        "er_version": er.__version__,
        "with_embeddings": args.with_embeddings,
    }
    (out_dir / f"{tag}_run_meta.json").write_text(json.dumps(meta, indent=2, default=str))
    print(f"wrote {out_dir / f'{tag}_run_meta.json'}")


if __name__ == "__main__":
    from er.schema import Dataset  # noqa: F401  (used above via name)

    main()
