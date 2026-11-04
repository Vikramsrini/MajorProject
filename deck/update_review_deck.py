#!/usr/bin/env python3
"""Bring the zeroth-review deck in line with what the project actually measured.

The original deck proposed a multimodal framework: CLIP image embeddings, adaptive
text-image weighting, and an attribute constraint as a hard veto (H2). The work
dropped the multimodal stage (no licence-clean image benchmark) and *refuted* H2 on a
strong matcher. Presenting the original slides at the review would show claims the
experiments have since overturned, so this script rewrites the affected slides in
place.

It edits the slide XML directly rather than regenerating the deck, so the existing
theme, layout, typography and every untouched slide survive exactly as they were.

Usage:
    python deck/update_review_deck.py IN.pptx OUT.pptx

Every number written here is traceable to docs/FINDINGS.md sections 10-12 and
paper/main.tex. If a number changes there, change it here - the deck is not the
source of truth.
"""

from __future__ import annotations

import html
import re
import shutil
import sys
import zipfile
from pathlib import Path

# ---------------------------------------------------------------- XML helpers

SP_RE = re.compile(r"<p:sp>.*?</p:sp>", re.S)
P_RE = re.compile(r"<a:p>.*?</a:p>", re.S)
T_RE = re.compile(r"<a:t>(.*?)</a:t>", re.S)


def _find_shape(xml: str, name: str) -> tuple[int, int]:
    """Return (start, end) of the <p:sp> whose cNvPr name is `name`."""
    for m in SP_RE.finditer(xml):
        if re.search(r'name="%s"' % re.escape(name), m.group(0)):
            return m.start(), m.end()
    raise KeyError(f"shape {name!r} not found")


def set_runs(xml: str, name: str, texts: list[str]) -> str:
    """Replace the text of each run in a shape, keeping the run's formatting."""
    s, e = _find_shape(xml, name)
    block = xml[s:e]
    runs = list(T_RE.finditer(block))
    if len(runs) != len(texts):
        raise ValueError(f"{name}: shape has {len(runs)} runs, got {len(texts)} texts")
    for m, t in zip(reversed(runs), reversed(texts)):
        block = block[: m.start(1)] + html.escape(t, quote=False) + block[m.end(1):]
    return xml[:s] + block + xml[e:]


def set_paragraphs(xml: str, name: str, texts: list[str]) -> str:
    """Replace a shape's paragraphs, cloning the first paragraph's formatting."""
    s, e = _find_shape(xml, name)
    block = xml[s:e]
    paras = list(P_RE.finditer(block))
    if not paras:
        raise ValueError(f"{name}: no paragraphs")
    template = paras[0].group(0)
    if len(T_RE.findall(template)) != 1:
        raise ValueError(f"{name}: first paragraph is not a single run")
    new = "\n".join(
        T_RE.sub(lambda m: "<a:t>" + html.escape(t, quote=False) + "</a:t>", template, count=1)
        for t in texts
    )
    return xml[:s] + block[: paras[0].start()] + new + block[paras[-1].end():] + xml[e:]


def split_run(xml: str, name: str, texts: list[str], sizes: list[int]) -> str:
    """Replace a shape's single run with several runs at different sizes.

    Used for status labels, where "H2" stays at the label size and the status word
    sits beside it smaller, so the pair still fits on one line.
    """
    s, e = _find_shape(xml, name)
    block = xml[s:e]
    runs = list(re.finditer(r"<a:r>.*?</a:r>", block, re.S))
    if len(runs) != 1:
        raise ValueError(f"{name}: expected one run, found {len(runs)}")
    r = runs[0].group(0)
    rpr = re.search(r"<a:rPr[^>]*>.*?</a:rPr>|<a:rPr[^>]*/>", r, re.S).group(0)
    parts = []
    for text, size in zip(texts, sizes):
        rp = re.sub(r'sz="\d+"', f'sz="{size}"', rpr)
        parts.append(f"<a:r>{rp}<a:t>{html.escape(text, quote=False)}</a:t></a:r>")
    return xml[:s] + block[: runs[0].start()] + "".join(parts) + block[runs[0].end():] + xml[e:]


def widen(xml: str, name: str, cx: int) -> str:
    """Set a shape's width, keeping its position (status labels need one line)."""
    s, e = _find_shape(xml, name)
    block = re.sub(r'(<a:off x="\d+" y="\d+"/><a:ext cx=")\d+(")', rf"\g<1>{cx}\g<2>", xml[s:e], count=1)
    return xml[:s] + block + xml[e:]


def recolour(xml: str, name: str, old: str, new: str) -> str:
    """Recolour the fill inside one shape (for status labels)."""
    s, e = _find_shape(xml, name)
    block = xml[s:e].replace(f'val="{old}"', f'val="{new}"')
    return xml[:s] + block + xml[e:]


# ---------------------------------------------------------------- table helper

TR_RE = re.compile(r"<a:tr\b.*?</a:tr>", re.S)
TC_RE = re.compile(r"<a:tc>.*?</a:tc>", re.S)


def set_table(xml: str, rows: list[list[str]]) -> str:
    """Rewrite a single table's cells."""
    tbl = re.search(r"<a:tbl>.*?</a:tbl>", xml, re.S)
    if not tbl:
        raise ValueError("no table")
    block = tbl.group(0)
    trs = list(TR_RE.finditer(block))
    if len(trs) != len(rows):
        raise ValueError(f"table has {len(trs)} rows, got {len(rows)}")
    for m, row in zip(reversed(trs), reversed(rows)):
        rblock = m.group(0)
        tcs = list(TC_RE.finditer(rblock))
        if len(tcs) != len(row):
            raise ValueError(f"row has {len(tcs)} cells, got {len(row)}")
        for c, text in zip(reversed(tcs), reversed(row)):
            cblock = T_RE.sub(
                lambda mm: "<a:t>" + html.escape(text, quote=False) + "</a:t>",
                c.group(0), count=1,
            )
            rblock = rblock[: c.start()] + cblock + rblock[c.end():]
        block = block[: m.start()] + rblock + block[m.end():]
    return xml[: tbl.start()] + block + xml[tbl.end():]


# ---------------------------------------------------------------- slide edits

RED, GREEN, AMBER, TEAL = "B4231F", "1E7B4F", "B7791F", "0E9DBF"


def slide1(x: str) -> str:
    return set_runs(
        x, "Text 1",
        ["Attribute-Aware Product Entity Resolution: A Reproducible Study on the "
         "WDC Products Benchmark"],
    )


def slide6(x: str) -> str:
    x = set_runs(x, "Text 7", ["Scored at the wrong threshold"])
    x = set_runs(
        x, "Text 8",
        ["Matchers are reported at the F1-optimal pairwise threshold, while the "
         "deliverable is a set of resolved entities."],
    )
    x = set_runs(x, "Text 10", ["H1: Report entity-level quality, not only pairwise F1"])
    x = set_runs(x, "Text 12", ["Constraints applied without auditing"])
    x = set_runs(
        x, "Text 13",
        ["Attribute rules are assumed to remove false positives, but what they block "
         "is rarely measured."],
    )
    x = set_runs(x, "Text 15", ["H2: Quantify veto precision on a strong matcher"])
    x = set_runs(x, "Text 17", ["Binary decisions without uncertainty"])
    x = set_runs(
        x, "Text 18",
        ["Accept/reject outputs give no basis for routing ambiguous pairs to review."],
    )
    x = set_runs(x, "Text 20", ["H3: Calibration and selective acceptance"])
    return x


def slide7(x: str) -> str:
    x = set_runs(
        x, "Text 7",
        ["Build a reproducible pipeline with preprocessing and attribute extraction; "
         "regenerate every table from the result files."],
    )
    x = set_runs(
        x, "Text 12",
        ["Compare against string similarity, SBERT-only, GBDT, and fine-tuned "
         "cross-encoder matchers using precision, recall, and F1."],
    )
    return set_runs(
        x, "Text 22",
        ["Quantify each component: hard constraint vs soft features, typed vs full "
         "rule, and seed resampling."],
    )


def slide8(x: str) -> str:
    x = set_runs(
        x, "Text 14",
        ["Sentence-BERT embeddings; fine-tuned RoBERTa cross-encoder"],
    )
    x = set_runs(x, "Text 25", ["Attribute matching"])
    return set_runs(x, "Text 26", ["Attribute-aware scoring with calibration"])


def slide9(x: str) -> str:
    x = set_runs(x, "Text 7", ["Preprocessing", "Normalise text and units"])
    x = set_runs(x, "Text 10", ["Encoders", "SBERT + RoBERTa cross-encoder"])
    x = set_runs(x, "Text 16", ["Matching model", "Attribute-aware scoring + calibration"])
    x = set_runs(x, "Text 19", ["Confidence estimation", "Accept, reject, or route"])
    return x


def slide10(x: str) -> str:
    x = set_runs(x, "Text 0", ["Technologies Used"])
    rows = [
        ["Technology", "Role in the pipeline", "Why it is selected", "Category"],
        ["PyTorch", "Training and fine-tuning the matcher",
         "Standard framework for transformer training", "Core research"],
        ["Sentence Transformers", "Text embeddings for titles and specs",
         "Strong sentence-level similarity out of the box", "Core research"],
        ["Transformers (RoBERTa)", "Fine-tuned cross-encoder matcher",
         "Pre-trained language model at published-baseline parity", "Core research"],
        ["FAISS", "Top-K candidate generation",
         "Efficient nearest-neighbour search at scale", "Core research"],
        ["scikit-learn GBDT", "Gradient-boosted attribute matcher",
         "Strong, interpretable tabular baseline", "Core research"],
        ["Python", "Primary development language",
         "Ecosystem for ML, data processing, evaluation", "Implementation"],
        ["pytest", "Regression tests on the pipeline",
         "Locks results to the code that produced them", "Implementation"],
        ["pandas / NumPy", "Data handling and result tables",
         "Tables generated from result CSVs, never hand-typed", "Implementation"],
    ]
    return set_table(x, rows)


def slide11(x: str) -> str:
    x = set_runs(x, "Text 0", ["Hypotheses and What the Experiments Found"])

    # H1 - entity-level reporting
    x = split_run(x, "Text 4", ["H1", "  SUPPORTED"], [2800, 1400])
    x = recolour(x, "Text 4", TEAL, GREEN)
    x = widen(x, "Text 4", 3017520)
    x = set_runs(x, "Text 5", ["Base quality dominates repair"])
    x = set_runs(
        x, "Text 6",
        ["The fine-tuned cross-encoder lifts entity-level F1 from 0.240 to 0.629 "
         "(+0.389, 9/9 variants); the best clustering mechanism adds +0.230."],
    )
    x = set_runs(x, "Text 7", ["Matching is worth 1.7x the best post-hoc repair."])

    # H2 - attribute constraint
    x = split_run(x, "Text 9", ["H2", "  REFUTED"], [2800, 1400])
    x = recolour(x, "Text 9", TEAL, RED)
    x = widen(x, "Text 9", 3017520)
    x = set_runs(x, "Text 10", ["Attribute constraint as a hard veto"])
    x = set_runs(
        x, "Text 11",
        ["On the cross-encoder, AP is lower in 9/9 variants (mean -0.054) and F1 at "
         "a shared threshold falls from 0.640 to 0.384 at 80%/0%."],
    )
    x = set_runs(x, "Text 12",
                 ["3 seeds at 80%/0%: AP lower 3/3 (mean -0.071); veto precision 0.273."])

    # H3 - calibration
    x = split_run(x, "Text 14", ["H3", "  PARTLY"], [2800, 1400])
    x = recolour(x, "Text 14", TEAL, AMBER)
    x = widen(x, "Text 14", 3017520)
    x = set_runs(x, "Text 15", ["Calibrated confidence"])
    x = set_runs(
        x, "Text 16",
        ["Expected calibration error is 0.027 for the cross-encoder against 0.39 for "
         "raw string similarity, so selective acceptance is usable."],
    )
    x = set_runs(x, "Text 17", ["Review-band workload not yet measured."])

    return set_runs(
        x, "Text 18",
        ["Status as of 4 Nov 2026. Seed variance covers 1 of 9 cross-encoder variants; "
         "the rest are single-seed. No state-of-the-art claim."],
    )


def slide12(x: str) -> str:
    x = set_runs(x, "Text 0", ["Measured Results"])
    x = set_runs(
        x, "Text 4",
        ["Measured on WDC Products, all nine variants. No state-of-the-art claim."],
    )

    # row 1 - supported
    x = set_runs(x, "Text 7", ["Protocol reproduces the published baselines"])
    x = set_runs(
        x, "Text 8",
        ["Word co-occurrence F1 51.52 vs 52.66; RoBERTa 0.7102 vs 0.7218."],
    )

    # row 2 - supported
    x = set_runs(x, "Text 11", ["Base matcher beats post-hoc cleanup"])
    x = set_runs(
        x, "Text 12",
        ["Entity F1 +0.389 for the matcher vs +0.230 for the best cleanup rule."],
    )

    # row 3 - refuted
    x = set_runs(x, "Text 14", ["\u00d7"])
    x = recolour(x, "Shape 13", TEAL, RED)
    x = set_runs(x, "Text 15", ["Hard attribute constraint lowers ranking"])
    x = set_runs(
        x, "Text 16",
        ["AP lower 9/9; veto precision 0.281 — most blocks were genuine."],
    )

    # row 4 - supported
    x = set_runs(x, "Text 19", ["Same evidence helps as soft features"])
    x = set_runs(
        x, "Text 20",
        ["AP +0.019 in 9/9 variants; hard-negative false merges down 1.9 points."],
    )

    # row 5 - pending
    x = set_runs(x, "Text 22", ["\u2013"])
    x = recolour(x, "Shape 21", TEAL, AMBER)
    x = set_runs(x, "Text 23", ["Seed robustness partly established"])
    x = set_runs(
        x, "Text 24",
        ["3 seeds at 80%/0%; the other 8 variants are single-seed."],
    )

    x = set_runs(x, "Text 26", ["Status and next steps"])
    x = set_paragraphs(
        x, "Text 27",
        [
            "Scope: text-only. The image stage was dropped for lack of a licence-clean "
            "benchmark.",
            "Done: CPU grid, 3-seed CPU runs, cross-encoder grid, 80%/0% on 3 seeds.",
            "Next: 8 remaining cross-encoder variants on Kaggle; six-page cut; "
            "full read-through.",
            "Open: guide sign-off on workshop vs main track, and clearance to publish "
            "results from scraped offers.",
        ],
    )
    return x


def slide13(x: str) -> str:
    rows = [
        ["Metric", "What it measures"],
        ["Pairwise F1", "Harmonic mean of precision and recall"],
        ["Average precision", "Threshold-free ranking quality"],
        ["Entity-level F1", "Quality of the resolved product clusters"],
        ["Calibration error", "Agreement between confidence and accuracy (ECE)"],
        ["Veto precision", "Share of blocked pairs that were true non-matches"],
        ["Recall@K", "True matches among the top-K candidates"],
    ]
    x = set_table(x, rows)
    x = set_paragraphs(
        x, "Text 5",
        [
            "String similarity (token-set, Jaro-Winkler)",
            "Word co-occurrence + SVM; SBERT cosine",
            "GBDT on attribute features; RoBERTa cross-encoder",
        ],
    )
    x = set_paragraphs(
        x, "Text 8",
        [
            "Hard constraint vs the same evidence as soft features",
            "Typed veto vs the full constraint",
            "3 seeds × 90% training resample",
            "Own-threshold vs shared-threshold F1",
        ],
    )
    return set_runs(
        x, "Text 9",
        ["Thresholds are selected on validation only. F1-optimal and precision-first "
         "operating points are both reported; no target is set by hindsight."],
    )


def slide14(x: str) -> str:
    x = set_runs(x, "Text 4", ["Oct 2026"])
    x = set_runs(x, "Text 5", ["Pipeline, CPU grid, 3-seed resampling"])
    x = set_runs(x, "Text 7", ["Nov 2026"])
    x = set_runs(x, "Text 8", ["Cross-encoder grid; 80%/0% on 3 seeds"])
    x = set_runs(x, "Text 10", ["Dec 2026"])
    x = set_runs(x, "Text 11", ["Remaining 8 variants; six-page cut"])
    x = set_runs(x, "Text 13", ["Jan 2027"])
    x = set_runs(x, "Text 14", ["Read-through, guide sign-off, PDF checks"])
    x = set_runs(x, "Text 16", ["31 Jan 2027"])
    return set_runs(x, "Text 17", ["IJCNN 2027 submission"])


EDITORS = {
    1: slide1, 6: slide6, 7: slide7, 8: slide8, 9: slide9,
    10: slide10, 11: slide11, 12: slide12, 13: slide13, 14: slide14,
}


def main() -> None:
    if len(sys.argv) != 3:
        raise SystemExit(__doc__)
    src, dst = Path(sys.argv[1]), Path(sys.argv[2])

    with zipfile.ZipFile(src) as z:
        names = z.namelist()
        blobs = {n: z.read(n) for n in names}

    for n, fn in EDITORS.items():
        key = f"ppt/slides/slide{n}.xml"
        xml = blobs[key].decode("utf8")
        blobs[key] = fn(xml).encode("utf8")
        print(f"  edited slide {n}")

    dst.unlink(missing_ok=True)
    with zipfile.ZipFile(dst, "w", zipfile.ZIP_DEFLATED) as z:
        for n in names:
            z.writestr(n, blobs[n])
    print(f"wrote {dst}")


if __name__ == "__main__":
    main()
