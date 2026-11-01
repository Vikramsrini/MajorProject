# Research Plan

## 1. One-sentence claim (candidate)

> On a public product-matching benchmark, the operating point that maximises pairwise matching F1
> yields entity-level resolution that is catastrophically over-merged, and a combination of
> contradiction-constrained scoring and calibrated selective acceptance recovers entity-level quality
> at a measurable review cost.

If the measured recovery is small, the honest fallback claim is the *diagnosis* plus the *negative
results* (typed attribute vetoes being nearly inert), which is still a publishable empirical
contribution at a mid-tier IEEE venue - but the paper must then be framed as a study, not a method.

## 2. Why this is worth publishing

From `LITERATURE_SURVEY.md`:

- The benchmark paper itself (WDC Products, EDBT 2024) shows every published matcher degrades on
  corner cases and unseen products; nobody in the reviewed set reports **entity-level** quality for
  these variants.
- The clustering-stage problem is explicitly flagged by others (GraLMatch EDBT 2025; TransClean IEEE
  Access 2025; OpenSanctions Pairs 2026: "attention must shift to blocking/clustering/review") but is
  not yet measured as a **threshold-selection** failure on product data.
- Calibration in ER is barely developed (UncertaiNLP 2025 reports ECE only), and no reviewed paper
  ties a review band to downstream cluster quality.

This is a small, defensible slice of an active problem, computable on one laptop in hours.

## 3. Experimental design

**Dataset.** WDC Products pair-wise, all 3 corner-case levels x 3 unseen levels (9 test sets), medium
development set for the headline table and small/large for the label-efficiency axis.

**Systems (fixed set, one protocol each).**

| id | system | role |
|---|---|---|
| s1, s1b | token-set / Jaro-Winkler similarity | no-ML baselines, shows raw-similarity calibration is hopeless |
| s2 | GBDT on fuzz+token+numeric features | feature baseline comparable to Magellan-style systems |
| s3 | s2 + typed attribute *features* | attributes as soft evidence |
| s4a / s4b | s3 + typed veto (asymmetric / symmetric) | isolates the veto from the features |
| s5 | s3 + typed veto + specifier veto | **our full constraint** |
| s6 / s7 | SBERT cosine, with and without s5's constraint | does the constraint transfer to a strong embedding matcher? |

**Protocol.** Fit on train pairs, temperature + threshold chosen on valid pairs, evaluated on `_gs`
test pairs. One seed reported for the table; 5 seeds for the variance of the headline row.

**Reported per slice (all / hard-negative / easy-negative / high-similarity / entity-level):**
micro P, R, F1; AP (AUC-PR); precision at matched recall (0.5, 0.6); false-merge rate on hard
negatives; **veto precision**; ECE; review share and auto-accept precision; **entity P/R/F1** and
cluster-size distribution.

**Ablations.** constraint off / typed-only / +specifier; symmetric vs asymmetric vs subset-waived;
`violation_weight` in {0.4, 0.6, 0.8}; title-only vs title+description serialisation; development-set
size small/medium/large.

**Significance.** Paired bootstrap over test pairs (10k resamples) for the F1 and entity-F1 deltas;
report confidence intervals rather than bare "improvement".

## 4. Paper skeleton (IEEE 2-column, 10 pages incl. refs)

1. Introduction - Case A/B figure, the operating-point claim, bullet contributions.
2. Related work - LLM matchers, calibration, multi-source clustering, benchmarks. Cite the 2023-2026
   set; explicitly cite TransClean/GraLMatch so our reuse of transitive consistency is not mistaken
   for a novelty claim.
3. Method - pipeline figure; constraint definition (typed + specifier); calibration and selective
   acceptance; entity-level clustering. Keep equations minimal (cap rule, ECE, BCubed).
4. Setup - dataset variants, protocol, hyperparameters, hardware, code-release statement.
5. Results - main table; the pairwise-vs-entity threshold figure; ablations; label-efficiency.
6. Analysis - veto precision breakdown, false-merge case study (real titles), error taxonomy.
7. Limitations - text-only, English-ish web offers, single benchmark family, no LLM matcher in the
   loop, cleanup is a reproduction.
8. Conclusion + future work (multimodal blocked on data; LLM cascades; WDC Block for scale).

Figures needed: (1) PR curves with the entity-optimal and pairwise-optimal points marked;
(2) threshold vs entity P/R/F1 sweep; (3) reliability diagram before/after temperature scaling;
(4) case-study table; (5) pipeline diagram.

## 5. Venue and dates

Target **IEEE International Conference on Big Data (IEEE BigData)**, since ER/blocking work appears
there (a verified example: "Evaluating Blocking Biases in Entity Matching", IEEE BigData 2024).

- IEEE BigData 2026 has already closed for the year (full papers 21 Aug 2026; conference 14-17 Dec
  2026, Phoenix), so the realistic target is the **2027 cycle**, historically with an August deadline
  and a notification in October.
- Backups with IEEE branding and friendlier acceptance rates: **IEEE ICDE Workshops** (a verified
  Mannheim EM paper appeared at IEEE ICDE Workshops 2025), IEEE ICMLA, ICDM, and the conference's own
  undergraduate/REU consortium track if a student is first author.
- **Action item:** confirm the 2027 dates on the official site when the call is posted; do not plan
  around a remembered date.

Secondary use of the same work: the department's first/second/third reviews need progress evidence,
which `results/*.csv` and the figures already provide.

## 6. Timeline (from 1 Nov 2026)

| Phase | Dates | Output |
|---|---|---|
| A. Toolkit + first measurement | done (Nov) | pipeline, 38 tests, 9-variant results |
| B. Finish sweeps + seeds + bootstrap | Nov-Dec 2026 | all tables, variance, CI |
| C. Improve the constraint | Jan 2027 | specifier tuning, per-category lexicons; decide whether the method story survives |
| D. Optional LLM baseline | Feb 2027 | small GPT-4o-mini/locally-quantised comparison on a subsample, with cost reported |
| E. Writing + figures | Mar-Apr 2027 | full draft for internal review |
| F. Internal review, revisions, submission | May-Jul 2027 | polished submission |

## 7. Risks

| Risk | Likelihood | Mitigation |
|---|---|---|
| Constraint gains too small to carry a method paper | Medium-high (currently the F1 delta at fixed threshold is ~0) | Keep the diagnosis + measurement framing; add the LLM-cascade comparison so the paper is about operating points, not one rule |
| Reviewers say "clustering cleanup is TransClean" | High | Cite it in the abstract-level contribution list; report our version only as a gated reproduction |
| Only one benchmark family | Medium | Add billiger.de Products (2026) and Abt-Buy/Amazon-Google from the Apache-2.0 Ditto repo |
| No LLM baseline -> looks dated | Medium | Add a zero-shot LLM row even if only on a subsample, and report cost |
| Licence/publication objection to scraped offers | Low-medium | Use the public benchmark files only, cite WDC, redistribute nothing scraped |
| Compute/time on student hardware | Low | Everything measured so far runs in minutes on an M2/16GB laptop |

## 8. Definition of done

- All 9 test variants x 7 systems table, with entity-level metrics and CIs, reproducible by one
  command from a fresh clone.
- A public code release with the exact data fingerprint recorded in every result row.
- Every citation URL resolves; no citation to anything in the survey's negative-results list.
- `FINDINGS.md` contains at least one *negative* result we did not plan for (it already does).
