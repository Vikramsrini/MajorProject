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

Measured from the smoke run on this machine (Apple M2, MPS): a 20-step run of a tiny
`RobertaForSequenceClassification` with batch 32 took 7 s of training, plus ~30 s of
inference over the validation and test splits.

For `roberta-base` at `max_len=256`, batch 32: ~188 steps/epoch on the 6,000-pair training
set, 3 epochs ≈ 564 steps. On a T4 that is roughly 0.3–0.4 s/step → **~10–12 min training**,
plus **~3–5 min inference** (validation and test are each scored twice: once unconstrained,
once through the capped variant). Budget **15–20 min per variant**, so:

- 1 variant, 1 seed ≈ 20 min
- 9 variants, 1 seed ≈ **3 GPU-hours**
- 9 variants, 3 seeds ≈ **9 GPU-hours** → about a third of a weekly Kaggle quota

`distilroberta-base` is roughly 2× faster. Report it as a separate row; never silently
substitute it for the `roberta-base` numbers that the published baselines use.

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
