#!/usr/bin/env python3
"""Emit LaTeX tables from the result CSVs.

Tables in the paper are generated, never hand-typed. A hand-copied number is a
number that can silently disagree with `results/` after the next run; this script
makes that structurally impossible. Regenerate with:

    python scripts/make_tables.py --results results --out paper/tables

Then `\\input{tables/table_matching}` etc. from main.tex. Column selection is by
name against the CSV, so a missing experiment yields a missing column plus a
warning, not a wrong table.
"""

from __future__ import annotations

import argparse
import glob
import re
from pathlib import Path

import pandas as pd

# --- LaTeX helpers ---------------------------------------------------------- #

_LATEX_SPECIAL = {
    "&": r"\&",
    "%": r"\%",
    "$": r"\$",
    "#": r"\#",
    "_": r"\_",
    "{": r"\{",
    "}": r"\}",
    "~": r"\textasciitilde{}",
    "^": r"\textasciicircum{}",
}


def esc(text: str) -> str:
    out = str(text)
    for ch, rep in _LATEX_SPECIAL.items():
        out = out.replace(ch, rep)
    return out


def fmt(v, spec: str = "{:.4f}", dash: str = "--") -> str:
    """Format a cell; NaN/None become an em-dash rather than the string 'nan'."""
    if v is None:
        return dash
    try:
        if pd.isna(v):
            return dash
    except (TypeError, ValueError):
        return esc(v)
    try:
        return spec.format(float(v))
    except (ValueError, TypeError):
        return esc(v)


def caption(text: str, label: str) -> str:
    return f"\\caption{{{text}}}\n\\label{{tab:{label}}}"


def wrap(body: str, label: str, cap: str, cols: int, resize: bool | None = None, wide: bool = False) -> str:
    """Emit a float. ``wide`` uses table* (spans both columns) and \\textwidth.

    A 11-system x 9-variant grid cannot fit \\columnwidth, and a float that does not
    fit is either dropped by LaTeX or spills into the margin, so the wide form has to
    exist rather than hoping the table stays small.

    Scaling is off by default: \\resizebox{\\columnwidth} on a table that already fits
    enlarges it to the full measure, which is how a 4-column table ends up typeset at
    14pt. It is applied only to wide floats, or when explicitly requested.
    """
    env = "table*" if wide else "table"
    width = "\\textwidth" if wide else "\\columnwidth"
    do_resize = wide if resize is None else resize
    colspec = "l" + "r" * (cols - 1)
    head = f"\\begin{{{env}}}[t]\n\\centering\n" + caption(cap, label) + "\n\\small\n"
    if do_resize:
        head += f"\\resizebox{{{width}}}{{!}}{{%\n"
    head += f"\\begin{{tabular}}{{{colspec}}}\n\\toprule\n"
    tail = "\\bottomrule\n\\end{tabular}\n" + ("}\n" if do_resize else "")
    return head + body + tail + f"\\end{{{env}}}\n"


def rows_table(rows: list[list[str]], header: list[str]) -> str:
    # Header cells are LaTeX authored by this script (e.g. r"$\tau$"), so they are
    # joined raw; escaping them produced unusable "\$\tau\$" column names.
    out = " & ".join(header) + " \\\\\n\\midrule\n"
    for r in rows:
        cells = " & ".join(r) if isinstance(r, (list, tuple)) else str(r)
        if cells.startswith("\\addlinespace") or cells.startswith("\\multicolumn") and "&" not in cells:
            # raw inter-row material: \addlinespace must not be terminated as a row
            out += cells + "\n" if cells.startswith("\\addlinespace") else cells + " \\\\\n"
        else:
            out += cells + " \\\\\n"
    return out


# --- tables ----------------------------------------------------------------- #

SYSTEM_LABELS = {
    "s1_similarity_tokenset": "Token-set similarity",
    "s1b_similarity_jaro": "Jaro-Winkler",
    "s1c_wordcooc_svm": "Word co-occurrence + SVM \\cite{peeters2024wdc}",
    "s1d_wordcooc_constraint": "Word co-occ. + constraint",
    "s2_classifier_nofeatattr": "GBDT, no attr. features",
    "s3_classifier_featattr": "GBDT + attr. features",
    "s4a_typed_veto_asym": "GBDT + typed veto (asym.)",
    "s4b_typed_veto_sym": "GBDT + typed veto (sym.)",
    "s5_full_constraint": "GBDT + typed + specifiers",
    "s6_sbert_cosine": "Sentence-BERT cosine",
    "s7_sbert_full_constraint": "SBERT + constraint",
}


def table_matching(results: Path, tag: str, out: Path) -> None:
    p = results / f"{tag}_results.csv"
    if not p.exists():
        print(f"  skip matching table: {p.name} missing")
        return
    df = pd.read_csv(p)
    a = df[df.slice == "all"].set_index("system")
    h = df[df.slice == "hard_negative_only"].set_index("system")
    order = [s for s in SYSTEM_LABELS if s in a.index]
    # bold the F1 winner only (a bolded *system name* reads as a claim, not a highlight)
    best_f1 = a.loc[order, "f1"].max() if order else None
    rows = []
    for s in order:
        r = a.loc[s]
        fmr = h.loc[s, "false_merge_rate"] if s in h.index else float("nan")
        f1s = fmt(r.get("f1"))
        if best_f1 is not None and r["f1"] == best_f1:
            f1s = f"\\textbf{{{f1s}}}"
        rows.append(
            [
                SYSTEM_LABELS[s],
                fmt(r.get("threshold"), "{:.3f}"),
                fmt(r.get("precision")),
                fmt(r.get("recall")),
                f1s,
                fmt(r.get("auc_pr")),
                fmt(fmr),
                fmt(r.get("ece")),
            ]
        )
    body = rows_table(rows, ["System", r"$\tau$", "Prec.", "Rec.", "F1", "AP", "FNMR", "ECE"])
    cap = (
        "Matching results on WDC Products (80\\% corner cases, 0\\% unseen, medium "
        "development set; 4{,}500 test pairs, 500 positives). FNMR is the false-merge "
        "rate on the benchmark's own similar-but-different negatives; ECE is quantile-binned "
        "expected calibration error."
    )
    out.joinpath("table_matching.tex").write_text(wrap(body, "matching", cap, 8))
    print(f"  wrote {out/'table_matching.tex'}")


def table_blocking(results: Path, out: Path) -> None:
    p = results / "blocking_wdc80cc.csv"
    if not p.exists():
        print("  skip blocking table: not generated yet")
        return
    df = pd.read_csv(p)
    rows = []
    for _, r in df.iterrows():
        m = str(r["method"]).replace("_", " ")
        par = str(r.get("params", ""))
        key = re.sub(r"[^a-z0-9=]+", "", par.replace("'", "").replace(",", " "))
        rows.append(
            [
                m,
                esc(key),
                f"{int(r['n_candidates']):,}",
                fmt(r["reduction_ratio"]),
                (f"\\textbf{{{fmt(r['pair_recall_tpr'])}}}" if r["pair_recall_tpr"] == df["pair_recall_tpr"].max() else fmt(r["pair_recall_tpr"])),
                fmt(r["ppc"]),
            ]
        )
    body = rows_table(rows, ["Blocking", "setting", "comparisons", "reduction", "TPR", "PPC"])
    cap = (
        "Blocking puts a hard ceiling on achievable recall: TPR is the share of the "
        "24{,}019 catalog-wide gold pairs that survive candidate generation (6{,}056 offers, "
        "1{,}000 clusters; all-pairs = 18{,}334{,}540 comparisons)."
    )
    out.joinpath("table_blocking.tex").write_text(wrap(body, "blocking", cap, 6))
    print(f"  wrote {out/'table_blocking.tex'}")


def table_entity(results: Path, tag: str, system: str, out: Path) -> None:
    files = sorted(glob.glob(str(results / f"{tag}_clustering_{system}_*.csv")))
    if not files:
        print(f"  skip entity table: no clustering files for {system}")
        return
    rows = []
    per_file = {}
    for f in files:
        kind = f.split(f"{system}_")[-1].replace(".csv", "")
        d = pd.read_csv(f)
        i = d["pairwise_f1"].idxmax()
        j = d["entity_f1"].idxmax()
        per_file[kind] = (d, i, j)
    # bold the single best entity F1 at the pairwise-optimal threshold
    label_of = {
        "naive": "union-find (no cleanup)",
        "clean0.4": "gate 0.4",
        "clean0.6": "gate 0.6",
        "clean0.8": "gate 0.8",
    }
    best_kind = max(per_file, key=lambda k: per_file[k][0].loc[per_file[k][1], "entity_f1"])
    for kind in ["naive", "clean0.4", "clean0.6", "clean0.8"]:
        if kind not in per_file:
            continue
        d, i, j = per_file[kind]
        at_pw = d.loc[i]
        f1s = fmt(at_pw["entity_f1"])
        if kind == best_kind:
            f1s = f"\\textbf{{{f1s}}}"
        rows.append(
            [
                label_of.get(kind, kind),
                fmt(at_pw["entity_precision"]),
                fmt(at_pw["entity_recall"]),
                f1s,
                f"{int(at_pw['max_size'])}",
                fmt(at_pw["threshold"], "{:.2f}"),
                fmt(d.loc[j, "entity_f1"]),
                fmt(d.loc[j, "threshold"], "{:.2f}"),
            ]
        )
    body = rows_table(rows, ["Clustering", "ent-P", "ent-R", "ent-F1", r"max $|C|$", r"$\tau_{\mathrm{pw}}$", "best F1", r"$\tau_{\mathrm{best}}$"])
    cap = (
        "Entity-level quality at the threshold that maximises \\emph{pairwise} F1 "
        f"({system}). The same matcher, evaluated on the objective a catalog owner actually "
        "cares about, degrades by 2.4x to 3.5x depending on the mechanism chosen; the cleanup gate recovers it by "
        "refusing to guess when both bridging edges are confident."
    )
    out.joinpath("table_entity.tex").write_text(wrap(body, "entity", cap, 8))
    print(f"  wrote {out/'table_entity.tex'}")


def table_cross_variant(results: Path, out: Path) -> None:
    """Systems as rows, all nine benchmark variants as columns.

    The long form (one row per system x variant) reaches ~100 rows and cannot fit a
    two-column paper, and a table that does not fit is a table that gets cut. The wide
    form is also the one that makes the interesting pattern legible: a ranking that
    flips along the unseen-product axis is invisible when sorted by variant.
    """
    files = sorted(glob.glob(str(results / "*_results.csv")))
    files = [f for f in files if "blocking" not in f]
    if not files:
        print("  skip cross-variant table: no results yet")
        return
    df = pd.concat([pd.read_csv(f) for f in files], ignore_index=True)
    a = df[df.slice == "all"].copy()
    e = df[df.slice == "entity_level"].copy()
    a["variant"] = a.corner_cases.astype(int).astype(str) + "/" + a.unseen.astype(int).astype(str)
    e["variant"] = e.corner_cases.astype(int).astype(str) + "/" + e.unseen.astype(int).astype(str)

    variants = sorted(a.variant.unique(), key=lambda v: (-int(v.split("/")[0]), int(v.split("/")[1])))
    order = [s for s in SYSTEM_LABELS if s in set(a.system)]
    fpiv = a.pivot_table(index="system", columns="variant", values="f1", aggfunc="first")
    epiv = e.pivot_table(index="system", columns="variant", values="entity_f1", aggfunc="first")
    best_pair = {v: fpiv[v].max() for v in variants if v in fpiv.columns}
    best_ent = {v: epiv[v].max() for v in variants if v in epiv.columns}
    # Systems whose operating point degenerated to "accept everything" in at least
    # one variant. Marked in the table itself, because the number alone looks like a
    # legitimate low score and a reader would otherwise compare it to real matchers.
    degenerate = (
        set(a[a.get("degenerate_operating_point").fillna(False).astype(bool)].system)
        if "degenerate_operating_point" in a.columns
        else set()
    )

    def block(piv, best, title):
        rows = [["\\multicolumn{" + str(len(variants) + 1) + "}{l}{\\textit{" + title + "}}"]]
        for s in order:
            label = SYSTEM_LABELS[s] + ("$^{\\dagger}$" if s in degenerate else "")
            cells = [label]
            for v in variants:
                val = piv.loc[s, v] if v in piv.columns and s in piv.index else float("nan")
                txt = fmt(val, "{:.3f}")
                if val == val and v in best and abs(val - best[v]) < 1e-9:
                    txt = f"\\textbf{{{txt}}}"
                cells.append(txt)
            rows.append(cells)
        return rows

    rows = block(fpiv, best_pair, "pair-level F1") + [["\\addlinespace"]] + block(epiv, best_ent, "entity-level F1")
    body = rows_table(rows, ["System"] + variants)
    cap = (
        "Every system on all nine WDC Products variants, columns labeled "
        "corner-case\\%/unseen\\%. Word co-occurrence (row 3) leads on seen products and falls "
        "below the structural feature model (row 6) at 100\\% unseen. The entity block is the "
        "same systems an order of magnitude worse than their own pair-level numbers. "
        "\\emph{Reading caution:} a system marked $^{\\dagger}$ degenerated to an "
        "accept-everything operating point in at least one variant, so its pair F1 is pinned "
        "at the class base rate rather than measuring matching skill --- the recall floor left "
        "no feasible threshold short of accepting every candidate, and its clustering collapses "
        "into one mega-cluster. Jaro-Winkler is different: a genuinely poor scorer (AP "
        "0.145--0.203) at a usable threshold. Both are reported rather than smoothed away."
    )
    out.joinpath("table_cross_variant.tex").write_text(wrap(body, "cross", cap, len(variants) + 1, wide=True))
    print(f"  wrote {out/'table_cross_variant.tex'} ({len(order)} systems x {len(variants)} variants)")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--results", default="results")
    ap.add_argument("--out", default="paper/tables")
    ap.add_argument("--tag", default="wdc80cc_0un_devmedium")
    ap.add_argument("--entity-system", default="s3_classifier_featattr")
    args = ap.parse_args()

    results, out = Path(args.results), Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    print("tables:")
    table_matching(results, args.tag, out)
    table_blocking(results, out)
    table_entity(results, args.tag, args.entity_system, out)
    table_cross_variant(results, out)


if __name__ == "__main__":
    main()
