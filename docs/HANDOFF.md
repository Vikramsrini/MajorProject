# Handoff: where the project stands

> **Superseded for current status by [`PROGRESS.md`](../PROGRESS.md)** (v2 notebook and paper, 2 Nov 2026).
> This file describes the earlier v1 pipeline and is kept for history.

Start here. This file is the single entry point for picking up the work. Last updated 2 Nov 2026.

## 1. The goal

Publish an IEEE paper on entity resolution for product matching, using the WDC Products benchmark.
Target: IJCNN 2027 (deadline 31 Jan 2027, 6 pages including references). Fallback: a workshop, or IEEE BigData 2027 (about Aug 2027).

The paper's current claim is a negative result with a mechanism. A hard attribute constraint (veto) lowers ranking quality, and the reason is that its precision collapses on a strong matcher.

## 2. What is done

| item | status | where |
|---|---|---|
| Benchmark loader, 9-variant CPU grid | done | `src/er/`, `scripts/run_wdc.py` |
| Cross-encoder grid, 1 seed, 8 epochs (`ce_roberta-e8`) | done | `results/*_ce_roberta-e8_results.csv` (git-ignored, see section 5) |
| 3-seed × 90% resample, CPU systems (s1c, s1d, s3, s4a, s5) | done | `results/*_seed_results.csv` (git-ignored); summary in `docs/FINDINGS.md` §12 |
| 3-seed cross-encoder, variant 80/0 | done | Kaggle output only (see section 4) |
| 3-seed cross-encoder, other 8 variants | **not done** | run on Kaggle |
| Paper draft `paper/main.tex` | drafted, over 6 pages | needs cut and read-through |
| Tables and figures | generated from results | `scripts/make_tables.py`, `scripts/make_figures.py` |
| Notebook for Kaggle GPU runs | regenerated with seed settings | `scripts/make_notebook.py` → `notebooks/finetune_wdc_colab.ipynb` |

## 3. Key results so far

Full detail and every correction are in `docs/FINDINGS.md`. Use those numbers, not older ones in this file.

- **Protocol check:** word co-occurrence reproduces the published baseline (F1 51.52 vs 52.66). Cross-encoder at 80/0 reaches F1 0.7102 vs published 0.7218, at its F1-optimal threshold, single seed.
- **Base quality beats post-hoc cleanup:** entity F1 +0.389 from the cross-encoder vs +0.230 from the best clustering mechanism. Single seed for the cross-encoder.
- **Typed veto on the GBDT matcher (3 seeds, 90% resample):** lowers AP in **9/9** variants, mean −0.021, Wilcoxon p = 0.004. This is the most robust lightweight-matcher result.
- **Full constraint on the GBDT matcher:** no detectable AP effect (4/9, mean −0.002, within seed noise).
- **Cross-encoder with constraint, 80/0, 3 seeds:** AP lower on **3/3** seeds (mean −0.071), entity F1 lower on 3/3 (mean −0.048), F1 at the shared threshold lower on 3/3 (0.640 → 0.384). Veto precision 0.273 (0.222–0.306). Per seed, 161–198 blocked pairs were genuine matches.
- **Threshold slide:** the constraint's F1 gain at its own threshold (0.685 vs 0.640) is an artefact of the threshold moving. Do not report it as an improvement.

**Still single-seed:** the full cross-encoder grid apart from 80/0, and all of the cross-encoder claims in the paper.

## 4. The Kaggle run in progress

Command the notebook runs for each variant:

```
python scripts/run_wdc.py --data data/raw --corner C --unseen U \
  --with-crossencoder --only-systems s8_crossencoder,s9_crossencoder_constraint \
  --model roberta-base --epochs 8 --seeds 0,1,2 --train-subsample 0.9 \
  --tag-suffix _ce_roberta-e8_seed --out results
```

Status:
- 80/0: finished, 3 seeds. Numbers above. The CSV `wdc80cc_0un_devmedium_ce_roberta-e8_seed_results.csv` exists **only in the Kaggle session**. It is not in this repo, so the 80/0 numbers come from the run log. Recover the CSV before reporting them as final.
- 80/50, 80/100, 50/0, 50/50, 50/100, 20/0, 20/50, 20/100: **not run.** Each takes about 25 minutes for three seeds.
- The earlier session stopped mid-run and lost the variant in progress. Only finished variants survive.

**Rule for every Kaggle session:** right after a variant finishes, copy its CSV to `/kaggle/working/`, which is the only folder that survives the session and can be downloaded. The notebook writes to `results/` under the session's own working copy, which is lost when the session ends.

## 5. How to reproduce

```
git clone https://github.com/Vikramsrini/MajorProject.git
cd MajorProject
python3.11 -m venv .venv && source .venv/bin/activate
pip install -e .[ml]
python scripts/download_wdc.py --dest data/raw
python scripts/run_wdc.py --data data/raw --corner 80 --unseen 0 --out results   # CPU systems
python scripts/aggregate_seeds.py --pattern "*_seed_results.csv" --reference s3_classifier_featattr
```

- `data/` is git-ignored. `scripts/download_wdc.py` fetches the benchmark into `data/raw/`.
- `results/` is git-ignored because it is large and regenerable. Commit only the summaries recorded in `docs/FINDINGS.md`, and the notebook config.
- Cross-encoder GPU runs need Kaggle or Colab. Setup and the cost estimate are in `docs/COMPUTE.md`.

## 6. Open decisions and tasks

1. **Finish the 3-seed grid on Kaggle** (8 variants left). Recover the 80/0 CSV first.
2. **Aggregate and update the paper.** Replace the single-seed "9/9" wording with seed-level counts. Add the F1-optimal results beside the precision-first ones.
3. **Six-page cut and full read-through.** Install a LaTeX distribution first, because `pdflatex` is not installed on the laptop. Then compile and count pages.
4. **Guide questions (not yet answered):**
   - Is a workshop acceptable, or must it be main track?
   - Is there departmental clearance to publish results from scraped schema.org offers?
5. **Decide the venue.** IJCNN 31 Jan 2027 is the current target, with a workshop as the fallback.

## 7. Known issues

- `docs/COMPUTE.md` still says 3 epochs and the old single-seed procedure. The current run uses 8 epochs and 3 seeds. Follow this file, not that one, until it is updated.
- The Kaggle `FutureWarning` messages about `fillna` downcasting are harmless. They do not affect results.
- The cross-encoder constraint's ceiling never binds at the operating point (no ceiling in 0.05–0.45 binds). The constraint's effect is entirely through the recall floor. The paper should say so.
- `docs/FINDINGS.md` §9 may still hold single-seed AP numbers that disagree with the grid. Check it against the paper before submission.
