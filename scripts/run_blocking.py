#!/usr/bin/env python3
"""Blocking evaluation: candidate recall versus comparisons actually made.

This is the one experiment the matching pipeline never exercised, because WDC
Products ships its own candidate pairs. It answers a different and necessary
question: *if we had to build the candidate set ourselves, what would each
blocking method cost, and what would it throw away?*

Gold is defined catalog-wide: every intra-cluster pair among the offers of a
variant, including train-to-test pairs that the published pair tables never
contain. That is the honest "resolve the whole catalog" task (the same framing as
WDC Block), and it is deliberately harder than the published splits.

The output is a Pareto table per method: pair recall (TPR) against the number of
comparisons, plus the reduction ratio against the all-pairs baseline. A matcher
can never beat the TPR of its blocking stage, so this table bounds everything
downstream and belongs in the paper next to the matching results.

Usage:
    python scripts/run_blocking.py --data data/raw --corner 80 --out results
"""

from __future__ import annotations

import argparse
import math
import sys
from pathlib import Path

import pandas as pd

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

from er import blocking as blk  # noqa: E402
from er.io import load_wdc_products_pairwise, wdc_index  # noqa: E402
from er.metrics import blocking_quality, gold_pairs_from_entities  # noqa: E402
from er.schema import Dataset  # noqa: E402


def build_corpus(data_dir: Path, corner: int) -> Dataset:
    """Union of every offer appearing in any split of one corner-case level."""
    idx = wdc_index(data_dir)
    idx = idx[idx.corner_cases == corner]
    frames = []
    for _, row in idx.iterrows():
        ds, _ = load_wdc_products_pairwise(row["path"])
        frames.append(ds.df)
    all_records = pd.concat(frames, ignore_index=True).drop_duplicates("record_id")
    return Dataset(f"wdc{corner}cc_catalog", all_records)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data", default="data/raw")
    ap.add_argument("--corner", type=int, default=80)
    ap.add_argument("--out", default="results")
    args = ap.parse_args()

    data_dir, out_dir = Path(args.data), Path(args.out)
    out_dir.mkdir(parents=True, exist_ok=True)

    ds = build_corpus(data_dir, args.corner)
    gold = gold_pairs_from_entities(ds.entities())
    n = ds.n
    all_pairs = n * (n - 1) // 2
    print(f"corpus: {n} offers, {len(ds.entities())} product clusters, {len(gold)} gold pairs")
    print(f"all-pairs baseline: {all_pairs:,} comparisons\n")

    rows: list[dict] = []

    # NOTE the `key=value` defaults: `lambda: f(bs)` would close over the
    # *variable*, so every config in a group would run with the final loop value
    # (this silently produced identical rows for all four token_block settings).
    configs = [
        ("token_block", lambda bs=bs: blk.token_block(ds, max_block_size=bs)) for bs in (25, 50, 100, 250)
    ] + [
        ("prefix_block", lambda pl=pl: blk.prefix_block(ds, prefix_len=pl)) for pl in (2, 4, 6)
    ] + [
        ("sorted_neighborhood", lambda w=w: blk.sorted_neighborhood_block(ds, window=w)) for w in (10, 30, 60)
    ] + [
        ("minhash_lsh", lambda t=t: blk.minhash_block(ds, threshold=t)) for t in (0.3, 0.5, 0.7)
    ]

    for name, fn in configs:
        try:
            b = fn()
        except Exception as exc:
            print(f"  {name} FAILED: {exc}", file=sys.stderr)
            continue
        q = blocking_quality(b.pairs, gold)
        rows.append(
            {
                "method": b.method,
                "params": {
                    k: v
                    for k, v in b.stats.items()
                    if k not in {"method", "n_records", "all_pairs", "n_candidates", "reduction_ratio"}
                },
                "n_records": n,
                "all_pairs": all_pairs,
                "n_candidates": q["n_candidates"],
                "reduction_ratio": b.stats["reduction_ratio"],
                "pair_recall_tpr": q["tpr"],
                "ppc": q["ppc"],
                "missed_gold_pairs": q["n_gold_pairs"] - int(round(q["tpr"] * q["n_gold_pairs"])),
            }
        )
        print(
            f"  {b.method:20s} cands={q['n_candidates']:>10,}  "
            f"reduction={b.stats['reduction_ratio']:.4f}  TPR={q['tpr']:.4f}  PPC={q['ppc']:.4f}"
        )

    df = pd.DataFrame(rows)
    if df.empty:
        raise SystemExit("no blocking method produced results")
    df["recall_loss_pct"] = (1 - df["pair_recall_tpr"]) * 100
    path = out_dir / f"blocking_wdc{args.corner}cc.csv"
    df.to_csv(path, index=False)
    print(f"\nwrote {path}")

    # the number a reviewer actually wants: recall lost per order of magnitude of work saved
    print("\n=== recall bound by cost band ===")
    df["log10_cost"] = df["n_candidates"].map(lambda x: int(round(math.log10(max(x, 1)))))
    print(
        df.groupby("log10_cost")
        .agg(min_cands=("n_candidates", "min"), best_tpr=("pair_recall_tpr", "max"))
        .sort_index()
        .to_string()
    )


if __name__ == "__main__":
    main()
