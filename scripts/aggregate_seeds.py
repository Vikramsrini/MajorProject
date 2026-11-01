#!/usr/bin/env python3
"""Aggregate multi-seed runs and test the deltas between systems.

Every number in the paper was originally a single run. This script turns the seed
dimension into something reportable:

1. **Seed variance** — mean and standard deviation per (variant, system, slice, metric).
   Reported so a reader can see whether a delta is larger than run-to-run noise.
2. **Variant-level paired test** — the unit of generalisation here is the *benchmark
   variant* (nine settings), not the seed. For each system we take its mean across seeds
   in each variant, pair the nine variants against a reference system, and run a
   Wilcoxon signed-rank test. With n=9 that is a weak test, so we also report the raw win
   count: "9/9" is more persuasive than a p-value computed on nine points.

Writes `results/seed_summary.csv`, `results/variant_tests.csv`, and a compact LaTeX table.

Usage:
    python scripts/aggregate_seeds.py --results results --reference s3_classifier_featattr
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import numpy as np
import pandas as pd

METRICS = ["f1", "auc_pr", "precision", "recall", "ece", "entity_f1", "false_merge_rate"]

LABELS = {
    "s1_similarity_tokenset": "Token-set similarity",
    "s1b_similarity_jaro": "Jaro-Winkler",
    "s1c_wordcooc_svm": "Word co-occurrence + SVM",
    "s1d_wordcooc_constraint": "Word co-occ. + constraint",
    "s2_classifier_nofeatattr": "GBDT, no attribute features",
    "s3_classifier_featattr": "GBDT + attribute features",
    "s4a_typed_veto_asym": "GBDT + typed veto (asym.)",
    "s4b_typed_veto_sym": "GBDT + typed veto (sym.)",
    "s5_full_constraint": "GBDT + typed + specifiers",
    "s6_sbert_cosine": "Sentence-BERT cosine",
    "s7_sbert_full_constraint": "SBERT + constraint",
    "s8_crossencoder": "Fine-tuned cross-encoder",
    "s9_crossencoder_constraint": "Cross-encoder + constraint",
}


def load(results: Path, include_ce: bool, pattern: str = "*_results.csv") -> pd.DataFrame:
    files = [
        f
        for f in sorted(glob.glob(str(results / pattern)))
        if "_smoke" not in f and (include_ce or "_ce_" not in f)
    ]
    if not files:
        raise SystemExit(f"no result files under {results}")
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    if "seed" not in df.columns:
        df["seed"] = 0
    df["variant"] = (
        df.corner_cases.astype(int).astype(str) + "/" + df.unseen.astype(int).astype(str)
    )
    return df


def seed_summary(df: pd.DataFrame) -> pd.DataFrame:
    metrics = [m for m in METRICS if m in df.columns]
    keys = ["variant", "system", "slice"]
    g = df.groupby(keys, dropna=False)[metrics]
    out = g.agg(["mean", "std", "count"])
    out.columns = [f"{m}_{stat}" for m, stat in out.columns]
    out = out.reset_index()
    out["seed_std_f1"] = out.get("f1_std", np.nan)
    return out


def variant_tests(df: pd.DataFrame, reference: str) -> pd.DataFrame:
    """Paired test across variants, using each system's seed-mean per variant."""
    metrics = [m for m in ("f1", "auc_pr", "entity_f1") if m in df.columns]
    base = df[df.slice == "all"]
    ent = df[df.slice == "entity_level"][["variant", "system", "entity_f1"]]
    work = base.merge(ent, on=["variant", "system"], how="left", suffixes=("", "_ent"))
    pivot = {m: work.pivot_table(index="variant", columns="system", values=m, aggfunc="mean")
             for m in metrics}
    if reference not in pivot["f1"].columns:
        raise SystemExit(f"reference {reference!r} not present in results")

    rows = []
    for m in metrics:
        p = pivot[m]
        for system in p.columns:
            if system == reference:
                continue
            pair = p[[system, reference]].dropna()
            if len(pair) < 5:
                continue
            delta = pair[system] - pair[reference]
            p_value = np.nan
            try:
                from scipy.stats import wilcoxon

                if (delta != 0).any():
                    p_value = float(wilcoxon(pair[system], pair[reference]).pvalue)
            except Exception:  # noqa: BLE001 - scipy optional / degenerate input
                pass
            rows.append(
                {
                    "metric": m,
                    "system": system,
                    "reference": reference,
                    "n_variants": int(len(pair)),
                    "mean_delta": float(delta.mean()),
                    "std_delta": float(delta.std(ddof=1)),
                    "wins": int((delta > 0).sum()),
                    "losses": int((delta < 0).sum()),
                    "wilcoxon_p": p_value,
                }
            )
    return pd.DataFrame(rows)


def latex_table(summary: pd.DataFrame, tests: pd.DataFrame, out: Path) -> None:
    """Compact table: seed-mean per system, with the variant-level win count vs reference."""
    a = summary[summary.slice == "all"]
    per_sys = a.groupby("system").agg(
        f1_mean=("f1_mean", "mean"), f1_std=("f1_mean", "std"),
        ap_mean=("auc_pr_mean", "mean"), ap_std=("auc_pr_mean", "std"),
        ece_mean=("ece_mean", "mean"),
    )
    order = [s for s in LABELS if s in per_sys.index]
    rows = []
    for s in order:
        r = per_sys.loc[s]
        t_ap = tests[(tests.metric == "auc_pr") & (tests.system == s)]
        win = f"{int(t_ap.iloc[0]['wins'])}/{int(t_ap.iloc[0]['n_variants'])}" if len(t_ap) else "--"
        rows.append(
            " & ".join(
                [
                    LABELS[s],
                    f"{r.f1_mean:.3f}\\,$\\pm$\\,{r.f1_std:.3f}",
                    f"{r.ap_mean:.3f}\\,$\\pm$\\,{r.ap_std:.3f}",
                    f"{r.ece_mean:.3f}",
                    win,
                ]
            )
            + r" \\"
        )
    body = (
        r"\begin{tabular}{lrrrr}" + "\n" + r"\toprule" + "\n"
        + r" & pair F1 & AP & ECE & AP wins vs ref. \\" + "\n" + r"\midrule" + "\n"
        + "\n".join(rows) + "\n" + r"\bottomrule" + "\n" + r"\end{tabular}" + "\n"
    )
    cap = (
        "Seed-averaged results (mean over seeds, $\\pm$ standard deviation across the nine "
        "benchmark variants; not the seed spread, which is reported in "
        "\\texttt{results/seed\\_summary.csv}). The last column counts variants in which the "
        "system beats the reference on AP --- a paired test on nine points has little power, "
        "so the win count is the more honest statistic."
    )
    tex = (
        "\\begin{table}[t]\n\\centering\n"
        f"\\caption{{{cap}}}\n\\label{{tab:seeds}}\n\\small\n" + body + "\\end{table}\n"
    )
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(tex)
    print(f"  wrote {out}")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables/table_seeds.tex")
    ap.add_argument("--reference", default="s3_classifier_featattr")
    ap.add_argument("--include-ce", action="store_true", help="include single-seed cross-encoder runs")
    ap.add_argument("--pattern", default="*_results.csv", help="glob for result files, e.g. '*_seed_results.csv'")
    args = ap.parse_args()

    results = Path(args.results)
    df = load(results, args.include_ce, args.pattern)
    seeds = sorted(df.seed.dropna().unique().tolist())
    print(f"loaded {len(df)} rows | seeds: {seeds} | variants: {df.variant.nunique()}")

    summary = seed_summary(df)
    summary.to_csv(results / "seed_summary.csv", index=False)
    print(f"  wrote {results/'seed_summary.csv'} ({len(summary)} rows)")

    tests = variant_tests(df, args.reference)
    tests.to_csv(results / "variant_tests.csv", index=False)
    print(f"  wrote {results/'variant_tests.csv'} ({len(tests)} comparisons)")

    latex_table(summary, tests, Path(args.out))

    # the numbers a reader will actually look for
    if len(seeds) > 1:
        spread = summary[summary.slice == "all"].groupby("system")["f1_std"].mean()
        print(f"\nmean within-seed F1 std across systems: {spread.mean():.4f}")
        print(f"largest: {spread.idxmax()} {spread.max():.4f}")
    key = tests[(tests.metric == "auc_pr")]
    if len(key):
        print("\nAP delta vs reference (wins/9, mean delta):")
        for _, r in key.sort_values("mean_delta", ascending=False).iterrows():
            print(f"  {r.system:30s} {int(r.wins)}/9  {r.mean_delta:+.4f}")


if __name__ == "__main__":
    main()
