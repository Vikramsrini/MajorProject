# Findings (measured)

All numbers here were produced by `scripts/run_wdc.py` / `scripts/run_blocking.py` and are stored in
`results/`. Dataset: **WDC Products**, pair-wise, 80% corner cases, test split with 0% unseen products,
medium development set (train 6,000 / valid 3,500 / test 4,500 pairs; 500 positives; 3,000
benchmark-flagged hard negatives). Date: 1 Nov 2026.


> **Read Section 9 first.** Sections 1-5 were written from the single 80%/0%-unseen cell. Re-testing every
> claim across all nine benchmark variants (Section 9) confirmed some and **retracted others**. Where the two
> disagree, Section 9 wins.

Read this before reusing any claim from the zeroth-review deck: **two of its three hypotheses did not
survive measurement**, and the deck's anticipated benefits are not results.

---

## 0. Protocol validation first (this is what makes the rest credible)

The benchmark paper reports **Word (Co-)Occurrence + LinearSVM at F1 52.66** on this exact split
(medium / 80% corner cases / seen). Our reproduction:

| | F1 |
|---|---|
| Published Word-Cooc | 52.66 |
| Our `s1c_wordcooc_svm` | **51.52** |

1.14 points apart. Splits, serialisation and metric handling are therefore consistent with the
published protocol, so other rows in this file are comparable to published numbers.

**Corollary that matters:** our own feature-based classifier (`s3`, F1 36.90) is *15 points worse* than
this published simple baseline. Any "improvement" measured on top of s3 is therefore measured on a weak
base, and must not be advertised as state-of-the-art. The strong base is Word-Cooc here; the published
transformer numbers (RoBERTa 72.05, Ditto 72.08, R-SupCon 67.21) require fine-tuning, which we have not
done.

Also: raw Sentence Transformer cosine (`s6`, AP 0.2176) is *worse than token-set similarity* (AP 0.2709)
here. Zero-shot cosine over `title + description` is diluted by seller boilerplate. It is **not** the
Ditto number and must never be compared to it in the paper.

## 1. Matching results

| system | threshold | P | R | F1 | AP | hard-neg false-merge rate | ECE |
|---|---|---|---|---|---|---|---|
| s1 token-set similarity | 0.548 | 0.2560 | 0.4940 | 0.3372 | 0.2709 | 0.2390 | **0.3913** |
| s1b Jaro-Winkler | 0.507 | 0.1426 | 0.4140 | 0.2121 | 0.1449 | 0.3687 | **0.3951** |
| **s1c word co-occurrence (published baseline)** | 0.653 | **0.5226** | 0.5080 | **0.5152** | **0.4766** | **0.0770** | 0.1154 |
| s1d s1c + constraint | 0.361 | 0.3727 | 0.6880 | 0.4835 | 0.4447 | 0.1923 | 0.0524 |
| s2 GBDT, no attribute features | 0.448 | 0.2988 | 0.4320 | 0.3532 | 0.3229 | 0.1687 | 0.0897 |
| s3 s2 + attribute features | 0.460 | 0.3267 | 0.4240 | 0.3690 | 0.3539 | 0.1457 | 0.0898 |
| s4a s3 + typed veto (asymmetric) | 0.452 | 0.3267 | 0.4240 | 0.3690 | 0.3467 | 0.1457 | 0.0702 |
| s4b s3 + typed veto (symmetric) | 0.452 | 0.3267 | 0.4240 | 0.3690 | 0.3478 | 0.1457 | 0.0774 |
| s5 s3 + typed veto + specifiers | 0.452 | 0.3267 | 0.4240 | 0.3690 | 0.3664 | 0.1457 | 0.0671 |
| s6 SBERT cosine | 0.521 | 0.2223 | 0.4540 | 0.2985 | 0.2176 | 0.2637 | 0.4030 |
| s7 s6 + constraint | 0.208 | 0.1112 | 1.0000 | 0.2001 | 0.2281 | 0.9997 | 0.2901 |

### What this says

- **Attribute evidence helps as soft features**: s2 → s3 is AP 0.3229 → 0.3539 (+3.1 points) and
  hard-negative false merges 16.9% → 14.6%. This is a real, if modest, gain.
- **The same evidence as a hard veto helps almost nothing, and can hurt.** At s3's own operating point
  (threshold 0.460, identical for s4a/s4b/s5): precision +0.0046, recall −0.008, FP 437 → 420. On top of
  the *strong* base it is clearly harmful: s1c → s1d is AP 0.4766 → 0.4447 and entity-level F1 0.3258 →
  0.1354.
- **Raw similarity is nowhere near calibrated** (ECE 0.39 for s1/s1b, versus 0.07-0.12 for trained
  matchers). This is the strongest justification for the calibration half of the project: a similarity
  score is not a probability, and any "confidence" claimed without scaling is fabricated.
- **s7 is a measured failure, kept deliberately.** Capping SBERT cosines moved 3,321 of 4,500 test pairs
  (74%); the operating-point rule then collapsed to "accept everything" (R = 1.0000, threshold 0.208),
  and the entity stage produced **one cluster containing all 1,000 products** (entity F1 0.0020). A
  constraint that fires on three quarters of candidates is not a constraint, it is noise.

## 2. Why the veto is nearly inert: it cannot bind at a recall-respecting operating point

Every constrained system logged `warning: no ceiling in [0.05 … 0.45] binds at its operating point`, with
`ceiling_used = 0.45` and `veto_binds_valid = 0`. The mechanism is structural:

1. The veto caps conflicting pairs to a ceiling.
2. Many *true* matches also conflict (truncated titles, missing specifications), so capping costs recall.
3. To satisfy the recall floor the threshold must then sit at or below the ceiling, which re-admits exactly
   the pairs the veto had blocked. The constraint cancels itself.

Veto precision confirms the recall cost — of the pairs the veto *effectively* blocked, this share were
genuine matches:

| system | effective vetoes | veto precision | true matches destroyed |
|---|---|---|---|
| s1d wordcooc + constraint | 596 | 0.691 | 184 |
| s4a typed veto asymmetric | 273 | 0.696 | 83 |
| s4b typed veto symmetric | 63 | 0.619 | 24 |
| s5 typed + specifiers | 367 | 0.749 | 92 |
| s7 SBERT + specifiers | 3,321 | 0.918 | 273 |

The rule that fires most is not the one that discriminates: `variant:one_sided` fires 1,398 times and
`quantity:one_sided` 1,004 times, i.e. mostly "one title is shorter", not "these are different products".
Symmetric mode (stated-vs-stated conflicts only) fires 63 times instead of 273 and is correspondingly less
destructive. **One-sided absence is weak evidence, and we measured that rather than assumed it.**

A methodological trap found here: thresholds are **not comparable across systems** once each system gets
its own temperature. s7's decision function is monotone in raw score, yet at s6's *calibrated* threshold it
accepted *more* false positives (794 → 922). Use threshold-free metrics (AP) or compare in raw-score space.

## 3. The result that does hold: the pairwise operating point is wrong for entity resolution

Entity-level resolution of the same records, clustering by naive transitive closure, swept over thresholds
(`results/*_clustering_*.csv`). At the threshold that maximises **pairwise** F1:

| configuration | entity P | entity R | **entity F1** | largest cluster |
|---|---|---|---|---|
| naive union-find | 0.058 | 0.780 | **0.1080** | 50 |
| + gated transitive cleanup, gate 0.4 | 0.077 | 0.680 | 0.1389 | 36 |
| + gate 0.6 | 0.290 | 0.388 | 0.3316 | 10 |
| + gate 0.8 | 0.491 | 0.314 | **0.3829** | 4 |
| *best entity threshold, naive* | 0.243 | 0.284 | 0.2618 | 11 (at thr 0.588) |

With the constrained matcher the same sweep reaches entity F1 **0.4000** (gate 0.8).

Three conclusions, each defensible:

1. **The pairwise-F1-optimal threshold is a bad decision for entity resolution**: 0.1080 entity F1 there
   versus 0.2618 at the entity-optimal threshold, and 0.3829 after cleanup at that *same* bad threshold.
   One bridging false edge merges two whole product families, so a single wrong merge costs many pairs —
   invisible to pairwise reporting.
2. **Over-merging is directly visible in cluster size** (largest cluster 50 → 4), a diagnostic that pairwise
   tables cannot express.
3. **The cleanup gate is not free**: entity precision rises to 0.491 while recall falls to 0.314, and 1,080
   accepted edges are reduced to 315. It buys precision with recall, so it must be reported as a trade, and
   it is a *reproduction* of published ideas (TransClean, IEEE Access 2025; GraLMatch, EDBT 2025), not novelty.

Note the ordering: choosing the **stronger base matcher** (wordcooc, entity F1 0.3258 naive) beats every
matching-stage intervention we tried (s3+s5+cleanup: 0.4000 is the exception, but it is a
different-matcher comparison and must be aligned before any claim). Base-matcher quality dominates;
post-hoc constraints are second-order.

## 4. Blocking: the recall bound (catalog-wide, 80% variant)

The published pair tables are not a blocking experiment, so gold was defined catalog-wide: all intra-cluster
pairs among the variant's **6,056 offers / 1,000 product clusters = 24,019 gold pairs**; all-pairs cost
18,334,540 comparisons.

| method | params | comparisons | reduction | **pair recall (TPR)** | PPC |
|---|---|---|---|---|---|
| token_block | max_block_size 25 | 205,585 | 0.9888 | 0.6812 | 0.0796 |
| token_block | 50 | 544,304 | 0.9703 | 0.8653 | 0.0382 |
| token_block | 100 | 1,255,241 | 0.9315 | 0.9700 | 0.0186 |
| token_block | 250 | 2,916,224 | 0.8409 | **0.9957** | 0.0082 |
| prefix_block | prefix_len 2 | 7,777 | 0.9996 | 0.1345 | 0.4155 |
| prefix_block | 4 | 30,954 | 0.9983 | 0.3025 | 0.2347 |
| prefix_block | 6 | 76,806 | 0.9958 | 0.4340 | 0.1357 |
| sorted_neighborhood | window 10 | 54,459 | 0.9970 | 0.0675 | 0.0298 |
| sorted_neighborhood | 30 | 175,189 | 0.9904 | 0.0780 | 0.0107 |
| sorted_neighborhood | 60 | 355,534 | 0.9806 | 0.0907 | 0.0061 |
| minhash_lsh | threshold 0.3 | 33,453 | 0.9982 | 0.3752 | 0.2694 |
| minhash_lsh | 0.5 | 10,104 | 0.9994 | 0.1628 | 0.3870 |
| minhash_lsh | 0.7 | 3,650 | 0.9998 | 0.0726 | 0.4778 |

- **Blocking puts a hard ceiling on recall.** token_block at 2.9M comparisons preserves 99.6% of gold pairs;
  at 206k comparisons it already loses 32%. Whatever the matcher, entity recall can never exceed these numbers.
- Cheapest useful operating point: prefix_block(pl=2) or minhash(0.7) at ~3-8k comparisons keep only 7-13% of
  gold pairs. MinHash at 0.3 (33k comparisons, TPR 0.375) is the best cost/quality point in the LSH family.
- **Sorted-neighbourhood is the weakest method here** (TPR 0.07-0.09) because the "rarest token" key splits
  product families across the sort order. Kept in the table because reporting a method that underperforms is
  how a benchmark earns trust.

## 5. Selective acceptance: what the review band actually buys

The band is defined as `[threshold - 0.10, threshold)` --- exactly the pairs the automatic decision
rejects but which sit nearest the boundary, i.e. the ones a reviewer could recover. Yield is the share of
reviewed pairs that are genuine matches. The test-set base rate is 500/4,500 = **0.111**, which is the
number a band must beat to be worth any human time at all.

| system | threshold | review share | band yield | positives recovered |
|---|---|---|---|---|
| s1c word co-occurrence | 0.653 | 0.030 | **0.287** (2.6x base) | 39 |
| s3 GBDT + attributes | 0.460 | 0.095 | 0.221 (2.0x base) | 94 |
| s5 s3 + full constraint | 0.452 | 0.076 | 0.199 (1.8x base) | 68 |
| s1 token-set similarity | 0.548 | **0.592** | **0.095** (below base) | 252 |

Three conclusions:

1. **Calibration is what makes selective acceptance meaningful, not cosmetic.** The raw-similarity band
   captures 59% of all pairs and its yield (0.095) is *below* the base rate: a reviewer working through it
   would do worse than picking pairs at random. The trained, temperature-scaled matchers produce bands with
   1.8-2.6x the base rate at 3-10% workload. This is the strongest evidence in the project for H3.
2. **The band is small and cheap for good matchers.** 3.0% of pairs (135 of 4,500) recovers 39 missed
   positives for word co-occurrence. That is a reportable operating recommendation, not a hypothetical.
3. **The constraint slightly *reduces* band yield** (s3 0.221 -> s5 0.199) while also reducing workload
   (0.095 -> 0.076): it moves some recoverable pairs out of the band entirely. Consistent with the
   precision-forward reading of Section 2 --- it is a lever, not a free improvement.

## 6. Bugs found by the data (kept because they are the interesting part)

Each of these would have silently corrupted a paper table:

1. `violation_weight` compared against `+ 1e9`, so the "gate" never gated: the cleanup dropped *every*
   violable edge, and entity F1 looked 3.5x better than it is.
2. `transitive_cleanup` deleted edges while iterating a snapshot of the graph → `KeyError`, and before
   crashing it made the result depend on traversal order. Now violations are detected per pass against one
   snapshot and applied synchronously.
3. Operating-point tie-break preferred the loosest cut, collapsing a capped system into "accept everything"
   (s7). Now prefers the strictest tied cut, and the rule name is recorded per row.
4. Entity-level metrics were computed over the union of all splits. WDC assigns each offer to exactly one
   split, so test pairs can *never* reconnect a cluster's train-split members: the reported recall floor was
   a split artefact, not a system property. Now measured over the nodes the scored pairs actually cover.
5. `len_*`/`ntok_*` features were dropped by an ablation filter that keyed on the name prefix, understating
   the feature baseline.
6. `rapidfuzz` ≥3 moved Jaro-Winkler to `rapidfuzz.distance` (0-1 scale, not 0-100).
7. Stated attributes and inferred attributes were mutually exclusive, so any dataset shipping a `brand` column
   silently lost variant/capacity detection — i.e. the contribution switched itself off.
8. A closure in a config list captured the loop variable by reference, so all four token-blocking settings
   ran at `max_block_size=250`, producing four identical rows that looked like a sweep.
9. MinHash used Python's salted `hash()`, which differs per process → blocking was irreproducible across runs.
10. The benchmark's own filename trap: `wdcproducts80cc20…` means **80% corner cases**, not 20% (proved by
    row counts 19,835/19,607/19,015 vs the paper's Table 1). Reading it wrongly inverts the difficulty axis.

## 7. What this changes in the paper

| Zeroth-review claim | Status after measurement |
|---|---|
| H1 adaptive text+image weighting | **Dropped.** No licence-clean multimodal ER benchmark exists (`DATASETS.md` §4). |
| H2 attribute-aware similarity reduces false merges | **Mostly refuted as a hard constraint; survives as soft features.** Typed-attribute veto: +0.5 precision points at fixed threshold. Specifier rules do improve AP (+1.3) but destroy true matches. Must be reported as a precision/recall trade, not a win. |
| H3 confidence-aware resolution | **Strengthened and measured.** Raw similarity ECE 0.39 vs 0.07-0.12 after training + temperature scaling; and only calibrated matchers produce a review band that beats the base rate (0.20-0.29 vs 0.111), while the similarity band sits *below* it (0.095). See Section 5. |
| "Blocking needed for scale" (was a contribution) | Remains an engineering requirement, now with a measured recall ceiling table. |
| **New primary contribution** | The pairwise-vs-entity objective mismatch: the F1-optimal threshold produces entity F1 0.108 with 50-record mega-clusters, and the levers that actually matter are base-matcher quality and clustering-stage precision, not scoring constraints. |

**Do not** claim state-of-the-art: our best matcher (0.5152) is the published simple baseline, and published
fine-tuned transformers reach 0.67-0.73 on the same split.

## 8. Next measurements (in priority order)

1. **Fine-tune a real base matcher** (Ditto-style, or at least logistic over our features on the large dev
   set) so the constraint study sits on a competitive matcher; published numbers give the target.
2 Re-run the full 3 × 3 variant grid (corner cases × unseen products) — running now; the unseen axis is
   where the literature says every method breaks, and it is our strongest available claim.
3. Multiple seeds + paired bootstrap CIs; every number above is currently a single run.
4. Precision-at-recall curve (not just P@R50) for s3 vs s5 vs s1c vs s1d — the correct way to show a
   precision/recall trade without arguing about thresholds.
5. ~~Review-band reporting~~ **done** (Section 5): band tied to the operating point, yield compared against
   the base rate. Next refinement: sweep the band width rather than fixing it at 0.10.

## 9. Grid-wide re-test: which claims survived (supersedes 1-5)

All nine variants (corner cases 20/50/80 x unseen 0/50/100, medium development set) were run with the same
protocol. Each claim was re-tested as "in how many cells does the sign hold?" rather than from the one
convenient cell. `verdict()` logic: mean delta and win count over the grid.

### Confirmed (9/9 cells)

| claim | evidence across the grid |
|---|---|
| Attribute evidence helps **as features** | AP s2 -> s3 improves in **9/9**, mean **+0.019** (range +0.004 to +0.032); F1 improves 7/9, mean +0.015 |
| Attribute features reduce hard-negative false merges | lower FMR in **9/9**, mean **-1.9 points** |
| Raw similarity is badly uncalibrated | ECE(s1) - ECE(s3) = **+0.30 in 9/9** (mean 0.382 vs 0.082; SBERT cosine 0.403) |
| Calibrated matchers produce informative review bands | band yield above the 0.111 base rate in **9/9**: word co-occurrence mean +0.209 (2.9x base), GBDT +0.115 (2.0x base) |
| The veto ceiling never binds | `veto_binds_valid = 0` and `ceiling_used = 0.45` in **every cell**, for every constrained system |
| Cleanup improves entity F1 at the pairwise-optimal threshold | improves in **18/18** (system x variant) cells; largest cluster 14-68 -> 4-6 every time |
| The constraint destroys entity quality on a strong base | entity F1 s1c -> s1d falls in all cells, e.g. 0.393 -> 0.054 (20%/0%), 0.405 -> 0.087 (50%/0%), 0.326 -> 0.135 (80%/0%) |

### Retracted: claims that were single-cell artefacts

| claim as first written | grid verdict | correction |
|---|---|---|
| "Specifier rules improve AP (+1.3 points)" (s3 -> s5) | **3/9, mean -0.002** | No effect. The +1.3 was the one cell where it happened to help. The constraint changes *decisions* only when the threshold is forced elsewhere; it does not improve ranking. |
| "On the strong base the constraint is clearly harmful (AP 0.477 -> 0.445)" | **6/9 positive, mean +0.006** | Its AP effect is noise. It is harmful at the *entity* level (see above), not at the scoring level. |
| "An uncalibrated similarity band is worse than random (yield 0.095 < 0.111)" | **above base in 6/9, mean +0.036** | Not worse than random in general. The robust statement is about **workload**: the similarity band swallows 32.7% of all pairs at ~1.15x base rate, while calibrated matchers give 3-10% workload at 2.0-2.9x base rate. The band is useless because it is huge, not because it is inverted. |
| Word-Cooc reproduction "1.14 points apart" | mean **+0.024** at our operating point | See the stronger validation below. |

### Stronger protocol validation than originally claimed

At the SVM's own decision boundary (rather than our precision-forward operating point), our word-co-occurrence
baseline tracks the published numbers across **all nine cells** with mean bias **-0.001** and maximum absolute
deviation **4.5 points** - unbiased. Our reported F1 sits ~2.2 points lower than published **by construction**,
because our operating-point rule deliberately trades F1 for precision. That is a protocol statement, not a
deficit: the model is reproduced, the decision rule is ours and is disclosed.

### The new robustness finding worth its own subsection

Base-matcher choice and robustness trade off against each other, and the direction is not obvious:

| at 0% unseen | at 100% unseen |
|---|---|
| word co-occurrence **0.515 / 0.533 / 0.548** | word co-occurrence **0.273 / 0.281 / 0.320** |
| GBDT + attribute features 0.369 / 0.447 / 0.455 | GBDT + attribute features **0.414 / 0.438 / 0.472** |

(for 80% / 50% / 20% corner cases). Word co-occurrence wins on seen products by 9-15 points and **loses on
unseen products by 14-16 points**, because a bag-of-words co-occurrence model leans on product-specific
vocabulary that unseen offers do not reuse, while structural similarity and attribute features transfer.
Ranking a matcher on one split therefore inverts on another - which is the same lesson as the operating-point
mismatch, one level up. This is now a headline-eligible result and it was invisible in the single-cell view.

### The recall floor can be infeasible, and that must be visible in the table

For `s7_sbert_full_constraint` the operating-point rule returned **recall = 1.000 and precision = 0.1112
in all three embedding variants** --- precision equal to the test base rate (500/4,500 = 0.1111). That is not
a matcher; it is the rule having no feasible cut. Cause: the constraint caps 74% of test pairs, and because a
large share of the capped pairs are positives, no threshold above the capped block reaches the 0.5 recall
floor, so the only feasible cut is the one that admits everything. Two consequences recorded here:

* `run_system` now emits `degenerate_operating_point`, computed from **validation** recall (never test), so a
  row of this kind cannot be misread as a score.
* A recall floor is a hard constraint on the constraint. Any future veto mechanism must either cap a
  smaller, higher-precision set (e.g. stated-vs-stated conflicts only) or route capped pairs to the review
  band instead of rejecting them --- which is where H3 and H2 should have been combined from the start.

The distinction matters for reading Table 4: Jaro-Winkler is a genuinely weak scorer (AP 0.145-0.203) at a
usable threshold, while s7 is a usable scorer at an unusable operating point. Reporting both numbers in the
same column without that note would invite exactly the wrong comparison.
