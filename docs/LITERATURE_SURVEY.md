# Literature Survey: Entity Resolution / Entity Matching (2023 - 2026)

**Scope.** Every paper below was published or first made public in **2023, 2024, 2025 or 2026**, as
requested. Pre-2023 work is deliberately excluded from the survey table and appears only in
"Foundational background" at the bottom.

**Verification.** Each row carries a URL. arXiv identifiers were resolved directly against
`arxiv.org/abs/<id>` and the returned titles matched the claim; venue/DOI statements come from the
paper's own comments field, the ACM/IEEE/ACL landing page, or Crossref. Anything that could not be
resolved is listed in section 7 as a **negative result** rather than cited.

**Date of survey:** 1 November 2026.

---

## 1. Where the field actually went (2023-2026)

Four things changed since the pre-2023 transformer era that the zeroth-review deck described:

1. **LLMs replaced fine-tuning as the default starting point.** Zero-/few-shot prompting of hosted
   models now matches or beats fine-tuned PLMs, especially on *unseen* entities - the exact regime
   where classic matchers collapse.
2. **Cost became a first-class metric.** A large share of recent work is about *how to not call the
   LLM 10 million times*: batch prompting, distillation into small models, blocking-batched RAG,
   budgeted graph query selection, cascades.
3. **Evaluation shifted from pairwise F1 toward the harder axes**: unseen products, corner cases,
   cross-dataset transfer, calibration, and end-to-end (blocking → matching → clustering) quality.
4. **Clustering/review got promoted.** Multiple 2025-2026 papers argue that pairwise matching is
   close to solved relative to what comes after it - false-positive edges corrupting clusters, and
   routing ambiguous pairs to humans. This is where our measured results also point.

## 2. LLM-based entity matching

| Paper | Authors | Year / Venue | ID | Method | Evidence |
|---|---|---|---|---|---|
| Entity Matching using Large Language Models | Peeters, Steiner, Bizer | 2023; publ. EDBT 2025 | arXiv:2310.11244 | Zero-/few-shot prompting and fine-tuning of hosted + open LLMs vs fine-tuned PLMs; prompt-sensitivity analysis; best LLMs need little/no training data and are more robust to unseen entities | [abs](https://arxiv.org/abs/2310.11244) |
| Fine-tuning Large Language Models for Entity Matching | Steiner, Peeters, Bizer | 2024; IEEE ICDE Workshops 2025, doi:10.1109/ICDEW67478.2025.00006 | arXiv:2409.08185 | Fine-tunes LLMs with/without LLM-generated explanations; LLM-based example selection/generation. Mixed for large models; hurts cross-domain transfer | [abs](https://arxiv.org/abs/2409.08185) |
| AnyMatch: Efficient Zero-Shot EM with a Small Language Model | Zhang, Groth, Calixto, Schelter | 2024 | arXiv:2409.04073 | GPT-2-scale model fine-tuned with data-selection tricks (hard-pair filter, attribute-level examples, label balance); 2nd-best overall F1, ~4 orders of magnitude fewer parameters and ~3,899x lower inference cost than GPT-4 | [abs](https://arxiv.org/abs/2409.04073) |
| Match, Compare, or Select? | Wang et al. | COLING 2025, pp. 96-109 | arXiv:2405.16884 | Compares binary-match vs compare vs select prompting; *select* is superior; ComEM exploits interactions among candidate records | [ACL](https://aclanthology.org/2025.coling-main.8/) |
| Entity Matching using Decision Models | Steiner, Peeters, Bizer | 2026 (Oct) | arXiv:2610.34626 | Decision models emit a probability per answer option (no generation); cascade routes low-confidence pairs to an LLM ensemble; +1.9-4.0 F1 points; probabilities well-calibrated for the strongest models only | [abs](https://arxiv.org/abs/2610.34626) |
| CaRL-EM: Cost-Aware RL for EM with LLMs | Guo, Klein, Huang | 2026, ACL 2026 main | arXiv:2609.01195 | RL controller chooses operator (match/compare/select/decide) and model capacity under a cost budget | [abs](https://arxiv.org/abs/2609.01195) |
| Labeling Training Data for Entity Matching Using LLMs | Steiner, Bizer | 2026 (Jun) | arXiv:2606.28823 | LLM teacher labels pairs for PLM students; students need 34-459x less inference time than direct LLM matching; labeling cost reported per training set | [abs](https://arxiv.org/abs/2606.28823) |
| DistillER: Knowledge Distillation in ER with LLMs | Zeakis, Papadakis, Skoutas, Koubarakis | 2026 (Feb) | arXiv:2602.05452 | Distillation design space (data selection x teacher elicitation x algorithm) without gold labels; SFT on noisy LLM labels is the best strategy | [abs](https://arxiv.org/abs/2602.05452) |
| Beyond Scale and Generation | Zhang et al. | 2026 (Jul) | arXiv:2607.24688 | Controlled factorial study: bi/cross/generative x Qwen3 variants x sizes, 1,215 fine-tuning runs. Cross-encoders consistently beat bi-encoders; generative is *not* universally better | [abs](https://arxiv.org/abs/2607.24688) |
| KcMF: Knowledge-compliant EM/SM with fine-tuning-free LLMs | Xu, Li, Chen, Shou | 2024 | arXiv:2410.12480 | Pseudo-code task decomposition + dataset/example-as-knowledge + ensembling to curb hallucination; +17.93% mean F1 over non-LLM competitors | [abs](https://arxiv.org/abs/2410.12480) |
| Automatic End-to-End Data Integration using LLMs | Steiner, Bizer | 2026, Beyond SQL Workshop @ IEEE ICDE 2026 | arXiv:2603.10547 | LLM-built pipelines approximate human-designed ones at ~$10 per case study | [abs](https://arxiv.org/abs/2603.10547) |

## 3. Cost, scaling and blocking

| Paper | Year / Venue | ID | Relevance to us | Evidence |
|---|---|---|---|---|
| BATCHER: Cost-Effective In-Context Learning for ER (design-space) | 2023; **IEEE ICDE 2024**, doi:10.1109/ICDE60146.2024.00284 | arXiv:2312.03987 | Batched prompting cuts API cost; our laptop-cost accounting follows this framing | [abs](https://arxiv.org/abs/2312.03987) |
| CE-RAG4EM: Cost-Efficient RAG for EM, blocking-based batching | 2026 (Feb); accepted ACM SIGMOD 2027 | arXiv:2602.05708 | Blocking amortises retrieval cost; biggest gains on ambiguous/numerically sparse attributes - the same population we target | [abs](https://arxiv.org/abs/2602.05708) |
| LLM-CER: In-context Clustering-based ER | 2025 (Jun); SIGMOD 2026 | arXiv:2506.02509 | Clusters sets of records directly; up to +10% F-measure and up to 5x fewer API calls. Supports our "cluster quality, not pair quality" framing | [abs](https://arxiv.org/abs/2506.02509) |
| Alper: Adaptive Graph Refinement + Label Propagation with LLMs | 2026 (May) | arXiv:2605.25814 | Unifies matching+clustering; budget-constrained selection of expensive pair queries; beats cascaded blocking-matching-clustering pipelines | [abs](https://arxiv.org/abs/2605.25814) |
| HyperBlocker: accelerating rule-based blocking on GPUs | 2024; PVLDB 18(2) | arXiv:2410.04349 | >=6.8x/9.1x faster than CPU-distributed/GPU solvers; >=30% end-to-end ER speed-up. Useful as our blocking-cost reference | [abs](https://arxiv.org/abs/2410.04349) |
| NLSHBlock: Neural LSH for Entity Blocking | 2024, SIAM SDM 2024 | arXiv:2401.18064 | Learned LSH; the direct ancestor of our MinHash-LSH baseline | [abs](https://arxiv.org/abs/2401.18064) |
| Pre-Trained Embeddings for ER: an Experimental Analysis | 2023, PVLDB 16(9):2225-2238 | arXiv:2304.12329 | 12 embedding models measured for blocking *and* matching incl. scalability; justifies reporting blocking recall separately | [abs](https://arxiv.org/abs/2304.12329) |
| Pre-trained LMs for Entity Blocking: a Reproducibility Study | 2024, NAACL 2024 | doi:10.18653/v1/2024.naacl-long.483 | Cautionary tale on blocking-evaluation protocol; we adopt its fixed-split discipline | [ACL](https://aclanthology.org/2024.naacl-long.483/) |
| **Evaluating Blocking Biases in Entity Matching** | 2024, **IEEE BigData 2024**, doi:10.1109/BigData62323.2024.10825531 | arXiv:2409.16410 | Blocking can be systematically unfair. **Also our venue precedent: ER work does get into IEEE BigData.** | [abs](https://arxiv.org/abs/2409.16410) |
| The Battleship Approach to Low-Resource EM | 2023, Proc. ACM Mod. Data (SIGMOD) | arXiv:2311.15685, doi:10.1145/3626711 | Low-label regime; our small-development-set runs live here | [abs](https://arxiv.org/abs/2311.15685) |
| Ground Truth Inference for Weakly Supervised EM | 2023, SIGMOD 2023 | arXiv:2211.06975 | Transitivity-aware labelling - conceptually adjacent to our transitive cleanup | [abs](https://arxiv.org/abs/2211.06975) |

## 4. Calibration, confidence and review

This is the thinnest and most opportunity-rich area, and the one our H3 sits in.

| Paper | Year / Venue | ID | Finding | Evidence |
|---|---|---|---|---|
| Confidence Calibration in LLM-Based Entity Matching | 2025, UncertaiNLP @ EMNLP | arXiv:2509.19557 | Reports ECE 0.0043-0.0552 for ER matchers with mild overconfidence; temperature scaling cuts ECE by up to 23.83%. **Direct precedent for our calibration numbers.** | [abs](https://arxiv.org/abs/2509.19557) |
| On Leveraging LLMs for Enhancing ER: a Cost-efficient Approach | 2024 (preprint) | arXiv:2401.03426 | Uncertainty-reduction: initialise clusters, query the LLM only on the most valuable pairs | [abs](https://arxiv.org/abs/2401.03426) |
| BoostER: Leveraging LLMs for Enhancing ER (demo) | 2024, The Web Conf | arXiv:2403.06434 | Interactive/boosting style ER assistance | [abs](https://arxiv.org/abs/2403.06434) |
| Can We Trust LLM Self-Explanations for ER? | 2026 (Jun) | arXiv:2606.01210 | 3 LLMs x 10 datasets: self-explanations are unstable and weakly faithful. **Argues against explanation-based confidence.** | [abs](https://arxiv.org/abs/2606.01210) |
| Entity Profile Generation and Reasoning with LLMs for Entity Alignment | 2025, Findings of EMNLP | aclanthology 2025.findings-emnlp.1093 | Confidence-threshold *rejection* of low-confidence alignments; Hits@1 0.993 on DBP15K(EN-FR) vs 0.990 prior SOTA. Same reject-option design as our review band | [ACL](https://aclanthology.org/2025.findings-emnlp.1093/) |

**Gap we can still claim:** calibrated, *selective* ER where the abstention band is evaluated jointly
against entity-level resolution quality and review workload. Existing calibration work reports ECE on
pairwise scores; it does not tie the band to downstream cluster quality. We do not claim exclusivity
until a final search before submission.

## 5. Multi-source ER and the clustering stage

| Paper | Year / Venue | ID | Finding | Evidence |
|---|---|---|---|---|
| GraLMatch: Matching Groups of Entities with Graphs and Language Models | 2024; EDBT 2025, doi:10.48786/EDBT.2025.75 | arXiv:2406.15015 | Multi-source group matching; graph properties detect/remove false-positive pairwise edges; limited pairwise FPs corrupt large groups | [abs](https://arxiv.org/abs/2406.15015) + [EDBT PDF](https://openproceedings.org/2025/conf/edbt/paper-224.pdf) |
| TransClean: Finding False Positives in Multi-Source EM via Transitive Consistency | 2025, **IEEE Access**, doi:10.1109/ACCESS.2025.3632400 | arXiv:2506.04006 | **Label-free iterative removal of FPs using transitivity.** Detects most FPs while removing few TPs | [abs](https://arxiv.org/abs/2506.04006) |
| OpenSanctions Pairs | 2026 (Feb) | arXiv:2603.11051 | 755,540 expert-labelled pairs, 1M entities, 293 sources, 45 jurisdictions, multilingual scripts. Rule matcher 91.3% F1, GPT-4o 99.0%; LLMs fail on cross-script transliteration. **Explicitly says attention must shift to blocking/clustering/review** | [abs](https://arxiv.org/abs/2603.11051) |
| AgenticER: The Next Frontier in Entity Resolution (vision) | 2026 (Jul), VLDB submission | arXiv:2607.27435 | Agentic, decision-theoretic ER; no empirical baseline yet | [abs](https://arxiv.org/abs/2607.27435) |

**Consequence for us (important).** Transitive-consistency FP removal is **already published**
(TransClean 2025; GraLMatch 2025). Our `transitive_cleanup` therefore cannot be claimed as novel.
It is a *reproduction with a stricter gate*, used to make the operating-point argument measurable.
We must cite both and position our contribution as the **pairwise-vs-entity objective mismatch plus
the attribute/specifier constraint layer**, not the cleanup itself.

## 6. Benchmarks 2023-2026

| Benchmark | Year / Venue | Size | Notes | Evidence |
|---|---|---|---|---|
| **WDC Products** (Peeters, Der, Bizer) | 2023 (arXiv); EDBT 2024 | 11,715 offers / 2,162 products; 27 variants | Three evaluation dimensions: corner cases (20/50/80%), unseen products (0/50/100%), development-set size. Baselines: Magellan 31-41 F1, Ditto 55-84, RoBERTa 64-87, R-SupCon up to 89. **Our primary dataset.** | [abs](https://arxiv.org/abs/2301.09521), [data](https://webdatacommons.org/largescaleproductcorpus/wdc-products/) |
| Billiger.de Products (bilingual) | 2026 (Sep) | 13,730 DE offers + EN translations | Harder than WDC Products; ~half the offers in difficult categories (clothing, furniture); adds cross-language pairs. Candidate *second* dataset | [abs](https://arxiv.org/abs/2609.37713) |
| MaDI-Bench (end-to-end data integration) | 2026 (Jun) | - | Tests integration steps jointly, not matcher-in-isolation | [abs](https://arxiv.org/abs/2606.30371) |
| WDC Block | 2023 | up to 200B candidate pairs, 3,259 e-shops | Dedicated blocking benchmark if we need scale evidence beyond our own blocking runs | [site](https://webdatacommons.org/largescaleproductcorpus/wdc-block/) |

## 7. Negative results (searched, could not verify - do not cite)

Searched via arXiv API, Crossref, OpenAlex, Semantic Scholar, GitHub and web search; none resolved:

- **"Falcon" / "Satellite" transformer blocking by Stek & Nauta** - not found. (An unrelated
  *Falcon: Scaling up Hands-off Crowdsourced Entity Matching*, Deng et al., SIGMOD **2017**, does exist
  and is out of scope.)
- **"Sealow"**, **"AnyER"**, **"MultiAtMixER"**, **"PEL" prompt-enhanced learning for EM**,
  **"PromptDistillation"** (Rueckhaus/Peeters/Bizer) - not found under those names.
- **LUMI / LUMI2** (supposed Amazon Science multimodal entity-matching library) - the GitHub org listing
  contains no such repository; treat any prior mention of LUMI as unverified.

If a reviewer or supervisor produces a DOI for any of these, re-check and add; as of today they are
not citable.

## 8. Multimodal ER: the honest status

Verified multimodal work in scope is **adjacent, not record-level product matching**:

- **MMoE: Multi-level Mixture of Experts for Multimodal Entity Linking**, Hu et al., **KDD 2025**
  (arXiv:2507.07108) - mention linking in text, not e-commerce offers.
- **Juno: Cross-Modal Entity Matching for Visually Rich Documents**, Sarkhel & Nandi, 2023
  (arXiv:2303.00720) - invoices/forms, not product catalogs.
- The **Heterogeneity in Entity Matching survey** (Moslemi et al., Data & Knowledge Engineering 2025,
  arXiv:2508.08076) lists multimodal matching as a *future direction*.

And the dataset audit (see `DATASETS.md`) found **no 2023-2026 multimodal product-matching benchmark
with images and a licence clean enough for publication**. Consequence: the adaptive-multimodal
hypothesis from the zeroth-review deck cannot currently be tested, and the deck's citation of CLIP as
enabling a multimodal contribution is unsupported by any benchmark we can actually use. **H1 is
demoted to future work.**

## 9. What this survey changed in our plan

| Zeroth-review claim | Post-survey status |
|---|---|
| H1 adaptive text+image weighting | **Dropped from the first paper.** No usable licensed image benchmark (sec. 8). |
| H2 attribute-aware similarity | **Kept, reframed.** Typed-attribute veto alone is nearly inert on WDC Products (see `FINDINGS.md`); the working signal is unit-numeric specifier disagreement. Measured, not assumed. |
| H3 confidence-aware resolution | **Kept and strengthened.** Precedent exists for ECE on ER (2509.19557) but not for tying the review band to entity-level quality. |
| Novelty of transitive-cleanup clustering | **Withdrawn.** Published (TransClean 2506.04006, GraLMatch 2406.15015). Reused as a reproduction. |
| Quadratic-cost blocking as a contribution | **Downgraded to an engineering requirement.** Blocking is a mature sub-field (2304.12329, 2410.04349, 2401.18064). |
| New headline contribution | **Pairwise-vs-entity operating-point mismatch**: the threshold that maximises pairwise F1 is far from the one that maximises entity-level F1, and this is invisible in standard reporting. Measured on WDC Products in `FINDINGS.md`. |

## 10. Foundational pre-2023 background (context only, outside the requested range)

Kept because they define terminology and are expected citations, but explicitly *not* part of the
2023+ survey: Mudgal et al. SIGMOD 2018 (DL EM design space); Li et al. Ditto PVLDB 2021; Peeters &
Bizer JointBERT PVLDB 2021 and R-SupCon WWW 2022; Radford et al. CLIP ICML 2021; Reimers & Gurevych
Sentence-BERT EMNLP 2019; Johnson et al. FAISS IEEE TBD 2019; Konda et al. Magellan VLDB 2016;
Christophides et al. ACM CSUR 2020 and Papadakis et al. ACM CSUR 2020 (blocking/filtering survey).
