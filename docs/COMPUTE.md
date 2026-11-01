# Compute plan: where each job should run

**Decision, short version:** train the cross-encoder on **Kaggle**, not Colab.

| | Kaggle | Colab (free) |
|---|---|---|
| Session length | 9 h (12 h verified-ID) | ~90 min before reclaim, often less |
| GPU | 2× T4 (16 GB each) or P100 | 1× T4 |
| Weekly quota | ~30 GPU h/week | variable, frequently exhausted by mid-week |
| Internet | toggle per notebook (must be ON to fetch the model + data) | on |
| Persistence | `/kaggle/working` survives within a session; "Output" tab keeps files after | `/content` dies; mount Drive |
| Background execution | no | no (Pro only) |

Our workload is **many short jobs, not one long job** (nine variants × ~15–20 min). Colab's
90-minute ceiling plus disconnect-on-idle is the wrong shape for that; Kaggle's 9-hour
windows let one session finish six to eight variants. Use Colab only as a fallback when the
Kaggle weekly quota is gone — the notebook is written to work on both.

## What runs where

| job | where | why |
|---|---|---|
| Blocking sweep, features, GBDT matchers, calibration, clustering | **laptop (CPU/M-series)** | already done: 9 variants in ~4 min each, no GPU needed |
| Sentence-BERT encode (one pass, cached) | laptop (MPS) or GPU | 3,496 records took seconds on MPS |
| **Cross-encoder fine-tuning** | **Kaggle GPU** | the only genuinely GPU-bound step |
| Table/figure generation | laptop | must stay reproducible without a GPU |

## Cost estimate

Measured locally on this machine (Apple M2, MPS), then extrapolated to a T4. The data volume
is the reason this is cheap: **6,000 training pairs per variant** is a small dataset, so a
fine-tuning run is minutes, not hours.

Step count is fixed and verifiable from the data: 6,000 pairs × 0.88 (early-stopping slice
removed) = 5,280 → 165 steps/epoch at batch 32 → **495 steps for 3 epochs**.

| stage | per variant | basis |
|---|---|---|
| model + tokenizer load | 20–40 s | ~20 s measured; first run also downloads ~500 MB |
| training, 495 steps | 1.5–3.5 min | sequence length matters more than `max_len` (padding is to batch max, not 256) |
| scoring valid + test (~3 passes) | 20–45 s | ~8,000 pair-forward passes |
| tables + clustering sweep | <10 s | no model involved |
| **per variant** | **~3–5 min** | |

- **Full grid, 1 seed: ~30–50 min.**
- **3 seeds: ~1.5–2.5 h.**
- First run adds ~1–2 min of downloads.

Practical consequence: the whole grid fits **comfortably inside one 9-hour Kaggle session**,
so in practice the resume logic should never be needed. The earlier estimate in this document
(3 GPU-hours) was pessimistic — it assumed per-variant retraining of the base for the
constrained variant, which is now eliminated (see below).

## Why the constrained variant costs nothing extra

`s9_crossencoder_constraint` **reuses the fitted weights from `s8_crossencoder`** rather than
building and training its own. Two reasons, one of which is about time and one about validity:

- retraining would roughly double the run;
- worse, a fresh random initialisation would put seed noise inside the very comparison the
  constraint is meant to isolate, so a small effect could be an artefact of initialisation
  rather than of the constraint.

The runner logs `reusing fitted base from s8_crossencoder (no refit)` when this happens, and
`tests/test_crossencoder.py::test_constrained_system_reuses_fitted_base_without_refit` fails
if it ever stops happening.

## Calibrating the estimate from your own run

The notebook prints a line per training step:

```
step   200/495  loss 0.4021  es_f1 0.6834  (74s)
```

After the first 200 steps you have a real seconds/step figure. Multiply by 495 for the
training time of the remaining variants; the extrapolation above is a prediction, that number
is a measurement. If steps are 3× slower than predicted, you are on a slow T4 with long
sequences — reduce `max_len` to 160 (the median description is 32 words) or switch to
`distilroberta-base` and label the row honestly.


## Procedure

1. Push the repo to GitHub (or upload a zip) and open
   `notebooks/finetune_wdc_colab.ipynb`.
2. Set `GITHUB_URL` (or `ZIP_PATH`) in cell 1.
3. Configure: `MODEL="roberta-base"`, `EPOCHS=3`, `MAX_STEPS=None`,
   `SUFFIX="_ce_roberta-base"`.
4. **Run the smoke test cell.** It trains 20 steps on a tiny random RoBERTa. It cannot
   produce a meaningful score — constant probabilities are the correct outcome — it exists to
   prove serialisation, tokenisation, the training loop and CSV writing work on that machine.
   If it fails, the full run fails identically, only an hour later.
5. Run **one** variant and check the printed `s8_crossencoder` F1 before launching all nine.
   A 20-step random model gives F1 ≈ 0.20; the fine-tuned model should land far above that.
   If it does not exceed the word co-occurrence baseline (0.515 on 80%/0%), stop — the
   serialisation or the label alignment is wrong, and more compute will not fix it.
6. Run the loop. It is resumable: a variant whose output CSV exists is skipped, so a
   disconnect costs at most one variant.

## Sanity targets

| setting | published (WDC Products, medium dev) | what we expect |
|---|---|---|
| 80% corner / 0% unseen | RoBERTa 72.18, Ditto 73.93 | 0.68–0.74 |
| 80% corner / 100% unseen | RoBERTa 70.13, Ditto 69.43 | 0.62–0.70 |
| 20% corner / 0% unseen | RoBERTa 83.68 | 0.78–0.84 |

Landing materially *below* these means a bug, not a hard dataset. Check, in order:
field serialisation actually contains `[COL] title [VAL]`, labels are `1` for same
`cluster_id`, and the tokenizer is truncating from the right at 256 tokens.

## Bringing results back

```bash
# unzip the notebook's output into the repo
unzip -o ce_results.zip -d results/
python scripts/make_tables.py      # merges <variant>_<suffix>_results.csv automatically
python scripts/make_figures.py --tag wdc80cc_0un_devmedium
```

`table_matching` merges suffixed files and lets the base (CPU) file win for any system
present in both, so a GPU run can never duplicate a row. `results/` is git-ignored: commit
the **model config and training history** (`metrics/…/config.json`, `history.csv`) and record
the numbers in `docs/FINDINGS.md`, so provenance is the notebook plus the config rather than
a screenshot.

## What the GPU run is actually for

Not to make the paper look modern. It answers a specific open question:

> Does the attribute-constraint failure generalise, or was it an artefact of a weak base
> matcher?

On the GBDT base (F1 0.369) the veto never binds. On a fine-tuned cross-encoder (F1 ≈ 0.72)
the score distribution is different — far more mass above any reasonable ceiling — so the
same constraint may bind hard, and then the prompt's original hypothesis (H2) becomes
testable for the first time. Either outcome is publishable: if it binds and destroys recall,
the negative result is confirmed on a strong base; if it binds and helps, we have the result
the zeroth review hoped for.

## Guardrails

- **Never** report a GPU-run number without the checkpoint config next to it (seed, epochs,
  model name, batch size are all in `config.json`).
- **Never** tune the threshold on test. The runner already enforces valid-only selection; if a
  notebook cell changes that, it invalidates every number in the table.
- **Never** compare `distilroberta-base` numbers against published `roberta-base` baselines
  without saying so in the caption.
- If the quota runs out mid-grid, report the variants that finished and state the missing ones
  explicitly. A partial grid with a stated gap is fine; a silently partial grid is fabrication.
