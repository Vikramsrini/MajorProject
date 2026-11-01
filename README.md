# Attribute-Constrained, Confidence-Aware Entity Resolution for Product Catalogs

B.Tech Major Project - Department of Computer Science and Engineering, SRM Institute of Science and
Technology. Academic year 2026-27. Target: an IEEE conference paper.

> **Status: research implemented, measured, and currently mid-iteration.**
> The pipeline runs end-to-end on the real WDC Products benchmark and produces the tables in
> `results/`. Read `docs/FINDINGS.md` before repeating any claim from the zeroth-review deck: two of
> the three original hypotheses did not survive contact with the data.

---

## 1. Problem

The same product is listed by many sellers with different titles, specifications and prices. Entity
resolution (ER) decides which records refer to the same real-world product and merges them into one
entity. Two cases define the difficulty:

- **Case A - same product, different wording.** `Apple iPhone 15, 128GB, Black` vs
  `iPhone 15 Apple 128 GB Midnight`. Partial token overlap, colour word differs. Exact and
  token-overlap matching miss it.
- **Case B - different product, similar wording.** `Apple iPhone 15, 128GB` vs
  `Apple iPhone 15 Plus 128GB`. Surface similarity is *higher* than Case A, yet merging them is wrong.

A system that fixes Case A by loosening thresholds breaks Case B. That tension is the project.

## 2. Contributions (as measured, not as hoped)

1. **Protocol validated against the benchmark's own published baseline.** Our Word co-occurrence +
   LinearSVC reproduces the WDC Products paper's F1 52.66 as **51.52** on the same split, which makes the
   rest of our tables comparable to published numbers. It also shows our feature classifier (36.90) sits
   *below* that baseline, so no state-of-the-art claim is made anywhere in this project.
2. **Pairwise-vs-entity operating-point mismatch.** The threshold that maximises *pairwise* F1 is far
   from the one that maximises *entity-level* F1, because one wrong bridge edge merges two whole
   clusters while pairwise metrics charge it once. Measured: entity F1 **0.108** with a 50-record mega-cluster
   at the pairwise-optimal threshold, versus **0.262** at the entity-optimal threshold. Standard ER
   reporting hides this.
3. **Attribute- and specifier-constrained scoring - measured and mostly refuted as a hard veto.** The
   constraint layer caps a pair's score when records state contradictory identity attributes (brand, model
   family, variant, capacity, quantity) or contradictory unit-qualified numeric specifiers (`50mm` vs
   `85mm`, `f/1.4` vs `f/1.8`). Reported with **veto precision** (how often it blocks a genuine match), which
   the literature rarely publishes. Result: at a fixed operating point it moves precision by **+0.5 points**,
   because every recall-respecting threshold re-admits the capped pairs - the constraint cancels itself. As
   *soft features* the same evidence does help (AP 0.323 → 0.354).
4. **Calibrated, selective resolution.** Raw similarity is nowhere near calibrated (ECE **0.39** for
   token-set/Jaro-Winkler, versus 0.07-0.12 for trained matchers after temperature scaling), which is the
   strongest justification in this project for treating confidence as a first-class output. The
   accept/review/reject band is tied to the actual operating point, so its cost is reviewer workload, not a
   free accuracy win.
5. **A reproducible, honest toolkit** with fixed splits, dataset fingerprints on every result row, 39 tests,
   and negative results written down in `docs/FINDINGS.md` rather than deleted.

## 3. Layout

```
src/er/
  schema.py         Dataset / Blocks / PairTable contracts, fingerprints, CSV I/O
  preprocess.py     normalisation, typed attribute extraction, unit-numeric specifier maps
  blocking.py       token / prefix / sorted-neighbourhood / MinHash-LSH / FAISS candidate generation
  features.py       fuzz + token + attribute + numeric + embedding feature families
  models/
    baselines.py    SimilarityMatcher, ClassifierMatcher (GBDT/logistic), WordCooccurrenceMatcher
    encoders.py     Sentence Transformer bi-encoder with disk-cached embeddings
    attribute.py    AttributeConstrainedMatcher - the hard-constraint layer
  metrics.py        pairwise micro P/R/F1, AP, ROC-AUC, precision@recall, Recall@K, BCubed, entity F1
  calibration.py    ECE, Brier, reliability table, temperature scaling, selective-report
  clustering.py     union-find, gated transitive cleanup (reproduction, not novelty)
  experiment.py     fit -> calibrate -> evaluate with slices, veto diagnostics, clustering sweep
  io.py             WDC Products loader (and the two filename traps, in code)
scripts/
  download_wdc.py   fetch the benchmark (verified URLs)
  run_wdc.py        the main experiment: system sweep + slices + clustering sweep
  run_blocking.py   blocking Pareto: pair recall vs comparisons, catalog-wide gold
  make_figures.py   paper figures, generated only from the CSVs above
tests/
  test_core.py      33 unit tests on the correctness-critical primitives
  test_wdc_io.py    6 integration tests re-checking counts against the benchmark paper
docs/
  LITERATURE_SURVEY.md  2023-2026 only, every citation URL-resolved, plus negative results
  DATASETS.md           dataset/licence audit and why multimodal was dropped
  FINDINGS.md           what was measured, including what failed
  RESEARCH_PLAN.md      paper plan, venue, timeline, risks
data/raw/               benchmark files (git-ignored)
results/                generated tables (git-ignored)
```

## 4. Run it

```bash
git clone https://github.com/Vikramsrini/MajorProject.git
cd MajorProject
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .[ml]                       # adds sentence-transformers, faiss, torch
python scripts/download_wdc.py --dest data/raw
python scripts/run_wdc.py --data data/raw --corner 80 --unseen 0 --out results
python scripts/run_wdc.py --data data/raw --corner 80 --unseen 100 --out results --with-embeddings
pytest -q
```

**On GPU (Kaggle):** open `notebooks/finetune_wdc_colab.ipynb` — it clones this repo, runs a
20-step smoke test, then fine-tunes the cross-encoder (see `docs/COMPUTE.md` for the time
budget: ~30-50 min for all nine variants on one seed).

Runs on an Apple M2 / 16 GB laptop: the gradient-boosting systems take a couple of minutes, the
embedding systems add model download plus one encode pass (cached to `~/.cache/er-research`).

**What is in the repo vs. regenerated:** `results/` holds the summary CSVs and `figures/` the
paper plots, so a clone can rebuild every table and compile the paper. The 45 MB per-pair score
dumps and the 124 MB benchmark download are excluded (`scripts/run_wdc.py` and
`scripts/download_wdc.py` recreate them).

## 5. Deliberate research-integrity choices

- **No number is reported that the code did not produce.** Where the deck anticipated a result, the
  doc says so and points at the measurement that replaced it.
- **Thresholds are chosen on the validation split only.** ECE and review-band figures use the same
  fitted temperature as the decision.
- **Published baseline results are reused, not re-run**, so we inherit their protocol (WDC Products
  reports Magellan, RoBERTa, Ditto, HierGAT and R-SupCon on exactly our splits).
- **Negative results are kept in the repo** (`docs/FINDINGS.md`), because the failure of the typed
  attribute veto is the finding that led to the specifier rule.
- **Multimodal claims were removed** rather than softened, once the licence audit showed no usable
  image benchmark.
- **The Amazon ML Challenge 2026 corpus is not used**: it is competition data with no verified reuse
  licence. It remains an inspiration, not a dataset.
