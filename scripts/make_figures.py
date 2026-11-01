#!/usr/bin/env python3
"""Generate the paper figures from result artifacts.

Every figure reads a CSV produced by another script, so a figure can never
disagree with a table: the same run is the single source of truth. Regenerate
with:

    python scripts/run_wdc.py ... && python scripts/run_blocking.py ... && python scripts/make_figures.py

Figures written to figures/ (vector PDF for the paper, PNG for slides):

  fig_blocking_pareto     candidate cost vs pair recall (the recall bound)
  fig_pr_curves           precision-recall per system
  fig_threshold_entity    threshold vs entity-level P/R/F1 (the operating-point result)
  fig_reliability         reliability diagram, raw vs temperature-scaled
"""

from __future__ import annotations

import argparse
import sys
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
import pandas as pd  # noqa: E402
from sklearn.metrics import average_precision_score  # noqa: E402

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))

plt.rcParams.update({"figure.dpi": 130, "font.size": 9, "axes.grid": True, "grid.alpha": 0.3})


def save(fig: plt.Figure, out: Path, name: str) -> None:
    out.mkdir(parents=True, exist_ok=True)
    fig.tight_layout()
    fig.savefig(out / f"{name}.pdf", bbox_inches="tight")
    fig.savefig(out / f"{name}.png", bbox_inches="tight")
    plt.close(fig)
    print(f"  wrote figures/{name}.pdf/.png")


def fig_blocking_pareto(results: Path, out: Path, corner: int = 80) -> None:
    p = results / f"blocking_wdc{corner}cc.csv"
    if not p.exists():
        print(f"  skip {p.name} (missing)")
        return
    df = pd.read_csv(p)
    fig, ax = plt.subplots(figsize=(4.6, 3.2))
    markers = {"token_block": "o", "prefix_block": "s", "minhash_lsh": "^", "sorted_neighborhood": "D"}
    for method, g in df.groupby("method"):
        g = g.sort_values("n_candidates")
        ax.loglog(
            g["n_candidates"], g["pair_recall_tpr"] * 100, marker=markers.get(method, "x"),
            lw=1.2, ms=4, label=method.replace("_", " "),
        )
        for _, r in g.iterrows():
            ax.annotate(f"{r['n_candidates'] / 1e3:.0f}k", (r["n_candidates"], r["pair_recall_tpr"] * 100),
                        fontsize=6, xytext=(2, -6), textcoords="offset points")
    ax.axhline(100, ls="--", lw=0.8, c="k", alpha=0.6)
    ax.set_xlabel("comparisons actually made (log)")
    ax.set_ylabel("pair recall / TPR (%)")
    ax.set_title(f"WDC Products {corner}%: blocking bounds all downstream recall")
    ax.legend(fontsize=7, loc="lower right")
    save(fig, out, "fig_blocking_pareto")


def _pr_curve(y, s):
    order = np.argsort(-np.asarray(s), kind="mergesort")
    ys = np.asarray(y)[order]
    tp = np.cumsum(ys)
    fp = np.cumsum(1 - ys)
    rec = tp / max(1, int(ys.sum()))
    prec = tp / np.maximum(1, tp + fp)
    return rec, prec


def fig_pr_curves(results: Path, out: Path, tag: str) -> None:
    files = sorted(results.glob(f"{tag}_scored_*.csv"))
    if not files:
        print(f"  skip PR curves ({tag} not found)")
        return
    fig, axes = plt.subplots(1, 2, figsize=(7.4, 3.1))
    for f in files:
        name = f.name.replace(f"{tag}_scored_", "").removesuffix(".csv")
        df = pd.read_csv(f)
        if "score" not in df.columns or "label" not in df.columns:
            continue
        rec, prec = _pr_curve(df["label"].astype(int), df["score"].astype(float))
        ap = average_precision_score(df["label"].astype(int).to_numpy(), df["score"].astype(float).to_numpy())
        for ax, mask, title in (
            (axes[0], np.ones(len(df), bool), "all test pairs"),
            (axes[1], df.get("is_hard_negative", pd.Series(False)).fillna(False).astype(bool).to_numpy(), "hard negatives only"),
        ):
            if not mask.any():
                continue
            sub = df[mask]
            r, p = _pr_curve(sub["label"].astype(int), sub["score"].astype(float))
            if int(sub["label"].sum()) == 0:
                continue
            ax.plot(r * 100, p * 100, lw=1.2, label=f"{name} (AP={average_precision_score(sub['label'].astype(int).to_numpy(), sub['score'].astype(float).to_numpy()) * 100:.1f})")
    for ax, t in zip(axes, ("all test pairs", "similar-but-different pairs")):
        ax.set_xlabel("recall (%)")
        ax.set_xlim(0, 100)
        ax.set_ylim(0, 100)
        ax.set_title(t, fontsize=8)
    axes[0].set_ylabel("precision (%)")
    axes[0].legend(fontsize=6, loc="lower left")
    fig.suptitle(f"Precision-recall, {tag}", fontsize=9)
    save(fig, out, "fig_pr_curves")


def fig_threshold_entity(results: Path, out: Path, tag: str, system: str) -> None:
    files = sorted(results.glob(f"{tag}_clustering_{system}_*.csv"))
    if not files:
        print(f"  skip threshold/entity figure ({system})")
        return
    fig, ax = plt.subplots(figsize=(4.8, 3.2))
    colors = {"naive": "tab:red", "clean0.4": "tab:blue", "clean0.6": "tab:green", "clean0.8": "tab:purple"}
    for f in files:
        kind = f.name.split(f"_clustering_{system}_")[1].removesuffix(".csv")
        df = pd.read_csv(f).sort_values("threshold")
        ax.plot(df["threshold"], df["entity_f1"], marker="o", ms=3, lw=1.2,
                color=colors.get(kind), label=f"{kind} (best={df['entity_f1'].max():.3f})")
    # mark the pairwise-F1-optimal threshold of the naive clustering
    naive = [f for f in files if f.name.endswith("_naive.csv")]
    if naive:
        d = pd.read_csv(naive[0])
        i = d["pairwise_f1"].idxmax()
        ax.axvline(d.loc[i, "threshold"], ls="--", c="k", lw=0.9)
        ax.annotate(
            f"pairwise-optimal\nentity F1={d.loc[i, 'entity_f1']:.3f}",
            (d.loc[i, "threshold"], d["entity_f1"].max() * 0.35), fontsize=6.5, ha="left",
        )
        j = d["entity_f1"].idxmax()
        ax.annotate(
            f"entity-optimal thr={d.loc[j, 'threshold']:.2f}",
            (d.loc[j, "threshold"], d.loc[j, "entity_f1"]), fontsize=6.5, ha="right", va="bottom",
        )
    ax.set_xlabel("acceptance threshold on pair score")
    ax.set_ylabel("entity-level pairwise F1")
    ax.set_title(f"Operating-point mismatch: {system.replace('_', ' ')}", fontsize=8.5)
    ax.legend(fontsize=6.5)
    save(fig, out, "fig_threshold_entity")


def fig_reliability(results: Path, out: Path, tag: str, system: str) -> None:
    p = results / f"{tag}_reliability_{system}.csv"
    if not p.exists():
        print(f"  skip reliability ({system})")
        return
    df = pd.read_csv(p)
    df = df[df["count"] > 0]
    fig, ax = plt.subplots(figsize=(3.4, 3.2))
    centres = np.array([float(b.strip("[]()").split(",")[0]) + 0.05 for b in df["bin"]])
    ax.plot([0, 1], [0, 1], ls="--", c="k", lw=0.8, label="perfect")
    ax.bar(centres, df["observed_rate"], width=0.085, alpha=0.45, label="observed")
    ax.plot(centres, df["mean_prob"], marker="o", ms=3, lw=1.2, label="predicted")
    ax.set_xlabel("predicted probability of match")
    ax.set_ylabel("observed match rate")
    ax.set_title(f"Reliability: {system}", fontsize=8.5)
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    ax.legend(fontsize=6.5)
    save(fig, out, "fig_reliability")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="figures")
    ap.add_argument("--tag", default="wdc80cc_0un_devmedium")
    ap.add_argument("--system", default="s3_classifier_featattr")
    args = ap.parse_args()

    results, out = Path(args.results), Path(args.out)
    print("figures:")
    fig_blocking_pareto(results, out)
    fig_pr_curves(results, out, args.tag)
    fig_threshold_entity(results, out, args.tag, args.system)
    fig_reliability(results, out, args.tag, args.system)


if __name__ == "__main__":
    main()
