# Research progress: where we are and how to resume

**Read this first on any machine.** Update the "Current stage" line and tick boxes as you go.
Last updated: 2 Nov 2026.

## Current stage

> **Stage 2: quick test on Colab (not started).**
> Next action: upload `notebooks/er_full_pipeline_colab.ipynb` to Colab (T4 GPU), set `QUICK = True`,
> Run all (~20 min). If any cell fails, fix it before the full run.

## Stage tracker

| # | Stage | Status | Output / where |
|---|---|---|---|
| 0 | Analyse the original project; find weaknesses | done | notes in this file, section "Decisions" |
| 1 | Build the full experiment notebook and the v2 paper draft | done | `notebooks/er_full_pipeline_colab.ipynb`, `paper/main.tex` |
| 2 | **Quick test** on Colab T4 (`QUICK = True`) | **not started** | must finish without errors |
| 3 | Full run on Colab Pro L4 (`QUICK = False`, ~7-8 h, resumable) | not started | `er_results_full_v1.zip` (Google Drive `MyDrive/er_runs/full_v1`) |
| 4 | Import results: `python scripts/import_results.py er_results_full_v1.zip` | not started | `paper/generated/` |
| 5 | Read `paper/generated/PAPER_READINESS.md`; decide paper framing with guide | not started | which claims passed |
| 6 | Compile in Overleaf; read every auto-selected sentence; add one real error example | not started | PDF with no red text |
| 7 | Guide review, revisions, page-limit cut | not started | |
| 8 | Turnitin/iThenticate check, departmental clearance, AI-use disclosure | not started | |
| 9 | Submit (target: IEEE BigData 2027 or an IEEE ICDE workshop; confirm dates) | not started | |

Partial progress inside stage 3 is safe: the notebook caches every finished model run on Google Drive and
skips it when you "Run all" again. Do not change `MATCHERS`, `CRITERIA` or model settings mid-run.

## Resume on a new laptop

```bash
git clone https://github.com/Vikramsrini/MajorProject.git
cd MajorProject
# experiments run on Colab, not locally: upload notebooks/er_full_pipeline_colab.ipynb there
# after the run, locally:
python scripts/import_results.py path/to/er_results_full_v1.zip
# then upload paper/ to Overleaf and compile
```

Google Drive (`MyDrive/er_runs/<RUN_NAME>/`) holds all run outputs and cached scores; sign in to the same
Google account in Colab to continue a run.

## What exists (v2, 2 Nov 2026)

- **Notebook** `notebooks/er_full_pipeline_colab.ipynb`: one notebook for everything.
  - Stage A (GPU): scores 8 matchers (token-set, word co-occurrence, GBDT, fine-tuned bi-encoder, RoBERTa and
    DeBERTa-v3 cross-encoders, Qwen2.5 and Phi-3.5 zero-shot LLMs) on 9 WDC Products variants + Abt-Buy,
    Amazon-Google, Walmart-Amazon.
  - Stage B (CPU): entity-level threshold chosen on validation, hard veto and veto precision, soft conflict
    calibrator, conflict-aware review routing, bootstrap CIs, Wilcoxon tests, published-parity check.
  - Writes tables, figures, `FINDINGS_AUTO.md`, `PAPER_READINESS.md` and `latex/numbers.tex`.
- **Paper** `paper/main.tex`: complete IEEE draft. Every number is `\R{key}` from `generated/numbers.tex`;
  claim wording switches automatically via `\IfClaim{B..R}`. Red text = not yet filled. Old draft kept in
  `paper/main_v1_old.tex`.
- **Base paper** `docs/base_paper/Peeters2024_WDC_Products_EDBT.pdf` (WDC Products, EDBT 2024).
- **Import script** `scripts/import_results.py`.

## Decisions already made (do not reopen without the guide)

- Base paper: Peeters, Der & Bizer, *WDC Products*, EDBT 2024.
- Success criteria are fixed in the notebook's config cell (`CRITERIA`) **before** running. Do not change
  them after seeing results.
- RoBERTa-base is the primary strong matcher (published numbers exist); DeBERTa-v3 is the replication.
- All architectures are reported; none is dropped after seeing results.
- Published reference (Table 3 of the base paper, medium/80%/seen): Word-Cooc 52.66, RoBERTa 72.18,
  Ditto 74.07, R-SupCon 79.99.

## Open items

- [ ] Fill author names and guide in `paper/main.tex`.
- [ ] Verify the 11 new references marked in `paper/refs.bib` against publisher pages.
- [ ] Edit the AI-use disclosure in the paper's Acknowledgment (required by IEEE).
- [ ] Fix outdated published numbers in `docs/FINDINGS.md` section 0 and `docs/COMPUTE.md` (see above).
- [ ] Guide: venue, workshop acceptable?, publication clearance for scraped web data, author order.
- [ ] The paper draft is public in this repo: tell the guide/checker that a GitHub match is your own work.

## Known code issues (do not affect the notebook's results)

- `src/er/calibration.py:139` `recall_auto` uses `tp / (tp + n_pos)`; should be `tp / n_pos`.
- `src/er/models/attribute.py:55` `_attr_sets` calls undefined `extract_attributes` (dead code).
- `src/er/models/attribute.py:219` `conflicts_frame` omits specifier conflicts.
- Cross-encoder early-stopping slice is not grouped by product cluster (slight leak into the stopping signal).
