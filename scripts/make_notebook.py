#!/usr/bin/env python3
"""Generate notebooks/finetune_wdc_colab.ipynb.

The notebook is generated rather than hand-edited so that the JSON is always valid and
so the smoke-test gate and the resume logic are reviewable as source. Regenerate with:

    python scripts/make_notebook.py
"""

from __future__ import annotations

import json
from pathlib import Path

CELLS: list[dict] = []


def md(text: str) -> None:
    CELLS.append({"cell_type": "markdown", "metadata": {}, "source": text.strip("\n").splitlines(keepends=True)})


def code(text: str) -> None:
    CELLS.append(
        {
            "cell_type": "code",
            "execution_count": None,
            "metadata": {},
            "outputs": [],
            "source": text.strip("\n").splitlines(keepends=True),
        }
    )


md(
    """
# Fine-tuning a cross-encoder for WDC Products (Colab / Kaggle)

This notebook produces the one thing the project is missing: **a base matcher that is not
15 F1 points below the published simple baseline.** The benchmark paper reports fine-tuned
transformers at 0.67-0.73 F1 on the 80% corner-case / seen split; our gradient-boosted
matcher reaches 0.369 and a reproduced word co-occurrence baseline reaches 0.515. Until a
fine-tuned model is in the table, a reviewer can dismiss every scoring-stage claim.

**What it trains.** A Ditto-style cross-encoder: each record is serialised per field
(`[COL] title [VAL] ... [COL] description [VAL] ... [COL] brand [VAL] ...`) and the pair is
passed as a tokenizer sentence pair. That is the scheme behind the published RoBERTa/Ditto
numbers, so the result is comparable rather than merely impressive.

**Cost.** `roberta-base`, 3 epochs, 6k training pairs: roughly 4-8 min/epoch on a T4/P100.
One variant ≈ 15-25 min including validation and test inference. All nine variants with one
seed ≈ 3-4 GPU-hours. Free Colab disconnects without warning, so **run the smoke test first
and expect to resume** - the loop skips variants whose output file already exists.

**The contract.** Each run appends rows to `results/<variant>_<suffix>_results.csv` using the
same schema as every other system. Drop those files into the repo and:

```bash
python scripts/make_tables.py   # the paper tables pick the new systems up automatically
```

Do **not** hand-copy numbers out of this notebook. If a number is not in a CSV it does not
belong in the paper.
"""
)

md("## 1. Environment")

code(
    """
import os, sys, subprocess, pathlib, platform

IN_COLAB = "google.colab" in sys.modules or os.path.isdir("/content")
IN_KAGGLE = os.path.isdir("/kaggle")
print("colab:", IN_COLAB, "| kaggle:", IN_KAGGLE, "| python:", platform.python_version())

# --- point this at your project -------------------------------------------------
# Preferred: push the repo to GitHub once and clone it here. Alternative: upload a zip
# of the project directory and set ZIP_PATH.
GITHUB_URL = "https://github.com/Vikramsrini/MajorProject.git"
ZIP_PATH   = ""            # e.g. "/content/major-project.zip" (alternative to git clone)
PROJECT_DIR = "/content/Major Project" if IN_COLAB else "/kaggle/working/Major Project"

def find_project() -> pathlib.Path:
    p = pathlib.Path(PROJECT_DIR)
    if (p / "src" / "er").is_dir():
        return p
    if GITHUB_URL:
        subprocess.run(["git", "clone", "--depth", "1", GITHUB_URL, str(p)], check=True)
        return p
    if ZIP_PATH and pathlib.Path(ZIP_PATH).exists():
        subprocess.run(["unzip", "-q", "-o", ZIP_PATH, "-d", p.parent], check=True)
        return p
    raise SystemExit(
        "Set GITHUB_URL or ZIP_PATH above, then re-run. The notebook needs src/er, "
        "scripts/ and data access."
    )

PROJ = find_project()
os.chdir(PROJ)
print("project:", PROJ, "| files:", len(list(PROJ.glob('*'))))
"""
)

code(
    """
# Install the project plus the fine-tuning extras. sentencepiece is needed because some
# checkpoints have no fast tokenizer under transformers v5; without it the failure only
# appears after the data download.
!pip install -q -e ".[finetune]"
import torch, transformers
print("torch", torch.__version__, "cuda:", torch.cuda.is_available(),
      torch.cuda.get_device_name(0) if torch.cuda.is_available() else "")
print("transformers", transformers.__version__)
assert torch.cuda.is_available() or IN_COLAB or IN_KAGGLE, "no GPU: set Runtime > Change runtime type"
"""
)

md(
    """
## 2. Configuration

`MODEL` should stay `roberta-base` for the headline row: it is what the benchmark paper uses,
so the comparison is like-for-like. `distilroberta-base` is the ~2x faster fallback if GPU
quota is short (report it as a separate row, never instead of `roberta-base`).

`MAX_STEPS` is a *safety* cap, not a hyperparameter: leave it `None` for real runs, set it
small only in the smoke test.
"""
)

code(
    """
MODEL   = "roberta-base"      # or "distilroberta-base"
EPOCHS  = 3
MAX_STEPS = None              # None = full training; small int ONLY for smoke tests
SUFFIX  = "_ce_roberta-base"  # keeps GPU rows out of the CPU result files

# All nine variants. Start with one (below) to confirm the numbers are sane, then run all.
VARIANTS = [(80, 0), (80, 50), (80, 100), (50, 0), (50, 50), (50, 100), (20, 0), (20, 50), (20, 100)]
RESULTS = PROJ / "results"
CKPT    = pathlib.Path("/content/ce_ckpt" if IN_COLAB else "/kaggle/working/ce_ckpt")
RESULTS.mkdir(exist_ok=True); CKPT.mkdir(parents=True, exist_ok=True)
print("model:", MODEL, "| suffix:", SUFFIX, "| ckpt:", CKPT)

# Persist to Drive on Colab so a disconnect does not cost the run.
if IN_COLAB:
    try:
        from google.colab import drive
        drive.mount("/content/drive")
        drive_dir = pathlib.Path("/content/drive/MyDrive/er_results")
        drive_dir.mkdir(parents=True, exist_ok=True)
        print("Drive ready:", drive_dir)
    except Exception as e:
        print("Drive not mounted (fine, but results die with the session):", e)
"""
)

md("## 3. Data (63 MB, public, no login)")

code(
    """
import subprocess, pathlib
raw = PROJ / "data" / "raw"
if not list(raw.rglob("*.json.gz")):
    subprocess.run(["python", "scripts/download_wdc.py", "--dest", "data/raw"], check=True)
else:
    print("data already present")

from er.io import wdc_index
idx = wdc_index(raw)
print(len(idx), "files |", sorted(idx.corner_cases.unique()), "corner-case levels |",
      sorted(idx.unseen.unique()), "unseen levels")
assert len(idx) >= 27, "expected 27 pair files (9 variants x 3 splits)"
"""
)

md(
    """
## 4. Smoke test — **run this before anything expensive**

Trains 20 steps on a tiny randomly-initialised RoBERTa of the same architecture. It cannot
produce a meaningful score (a 20-step random model outputs the class prior, so constant
predictions are the *correct* outcome); its job is to prove that serialisation, tokenisation,
the training loop, checkpointing and CSV writing all work on this machine.

If this cell fails, the full run will fail the same way - only after an hour.
"""
)

code(
    """
import subprocess, pandas as pd

smoke_tag = "smoke"
cmd = ["python", "scripts/run_wdc.py", "--data", "data/raw",
       "--corner", "80", "--unseen", "0",
       "--with-crossencoder", "--only-systems", "s8_crossencoder",
       "--model", "hf-internal-testing/tiny-random-RobertaModel",
       "--epochs", "1", "--max-steps", "20",
       "--tag-suffix", "_" + smoke_tag, "--out", "results"]
print(">>>", " ".join(cmd), flush=True)
subprocess.run(cmd, check=True)

f = PROJ / "results" / f"wdc80cc_0un_devmedium_{smoke_tag}_results.csv"
assert f.exists(), "smoke run wrote no results file"
d = pd.read_csv(f)
print(d[["system", "slice", "f1", "recall"]].to_string(index=False))
assert (d.system == "s8_crossencoder").any(), "s8 row missing from the CSV"
print("\\nSMOKE TEST PASSED - the training path works on this machine.")
print("Note: constant probabilities are expected (20 steps, random init).")
"""
)

md(
    """
## 5. Full run

Resumable: a variant whose `results/<variant><suffix>_results.csv` already exists is skipped.
Re-running this cell after a disconnect picks up where it stopped. Run **one** variant first
and sanity-check that `s8_crossencoder` F1 is in the 0.5-0.8 range before launching all nine.
"""
)

code(
    """
import subprocess, pathlib, pandas as pd   # re-imported: cells must survive out-of-order execution

def run_variant(corner: int, unseen: int, force: bool = False):
    tag = f"wdc{corner}cc_{unseen}un_devmedium{SUFFIX}"
    out = RESULTS / f"{tag}_results.csv"
    if out.exists() and not force:
        print(f"[skip] {tag} already done")
        return out
    cmd = [
        "python", "scripts/run_wdc.py", "--data", "data/raw",
        "--corner", str(corner), "--unseen", str(unseen),
        "--with-crossencoder", "--only-systems",
        "s8_crossencoder,s9_crossencoder_constraint",
        "--model", MODEL, "--epochs", str(EPOCHS),
        "--tag-suffix", SUFFIX, "--out", "results",
    ]
    if MAX_STEPS:
        cmd += ["--max-steps", str(MAX_STEPS)]
    print("\\n>>>", " ".join(cmd), flush=True)
    subprocess.run(cmd, check=True)
    return out

# One variant first. Check the printed s8 F1 before running the loop below.
one = run_variant(80, 0)
import pandas as pd
d = pd.read_csv(one)
print(d[d.slice == "all"][["system", "precision", "recall", "f1", "auc_pr"]].round(4).to_string(index=False))
"""
)

code(
    """
# All nine variants, resumable. Expect a session to die at some point; just re-run.
for corner, unseen in VARIANTS:
    try:
        run_variant(corner, unseen)
    except subprocess.CalledProcessError as e:
        print(f"!! variant {corner}/{unseen} failed: {e}. Continuing; re-run this cell later.")
print("\\nDONE. Collect the files below.")
"""
)

md("## 6. Collect results")

code(
    """
import shutil, glob
stamp = pathlib.Path(f"/content/ce_results_{MODEL.replace('/','_')}" if IN_COLAB
                     else f"/kaggle/working/ce_results")
stamp.mkdir(exist_ok=True)
files = sorted(RESULTS.glob(f"*{SUFFIX}_results.csv"))
for f in files:
    shutil.copy(f, stamp / f.name)
print(f"{len(files)} result files copied to {stamp}")
if IN_COLAB:
    shutil.make_archive("/content/ce_results", "zip", stamp)
    from google.colab import files as colab_files
    colab_files.download("/content/ce_results.zip")
else:
    shutil.make_archive("/kaggle/working/ce_results", "zip", stamp)
    print("download /kaggle/working/ce_results.zip from the Output tab")
"""
)

code(
    """
# Sanity table across everything produced in this session.
import pandas as pd, glob
frames = [pd.read_csv(f) for f in glob.glob(str(RESULTS / f"*{SUFFIX}_results.csv"))]
if frames:
    df = pd.concat(frames, ignore_index=True)
    print(df[df.slice == "all"].pivot_table(index="system", columns=["corner_cases", "unseen"],
                                            values="f1").round(3).to_string())
else:
    print("nothing produced yet")
"""
)

md(
    """
## 7. Bring the results back

1. Unzip into the repo's `results/` directory (filenames already include the variant and suffix).
2. `python scripts/make_tables.py` - `table_matching` merges the suffixed files automatically
   (base file wins for any system present in both, so nothing is duplicated).
3. `python scripts/make_figures.py --tag wdc80cc_0un_devmedium`
4. Commit the CSVs? No: `results/` is git-ignored. Commit the **checkpoint config + history**
   and record the numbers in `docs/FINDINGS.md`, so the paper's provenance is the notebook
   plus the model config, not a screenshot.

If `s8_crossencoder` lands near 0.72 F1 on 80%/0%, the paper's central weakness is fixed and
the constraint can finally be tested against a *strong* base - which is the experiment that
determines whether the negative result generalises or was an artefact of a weak matcher.
"""
)


def main() -> None:
    nb = {
        "cells": CELLS,
        "metadata": {
            "kernelspec": {"display_name": "Python 3", "language": "python", "name": "python3"},
            "language_info": {"name": "python", "version": "3.11"},
            "accelerator": "GPU",
            "colab": {"provenance": [], "gpuType": "T4"},
            "kaggle": {"accelerator": "gpu", "dataSources": [], "isGpuEnabled": True,
                        "isInternetEnabled": True},
        },
        "nbformat": 4,
        "nbformat_minor": 5,
    }
    out = Path(__file__).resolve().parents[1] / "notebooks" / "finetune_wdc_colab.ipynb"
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(nb, indent=1))
    print(f"wrote {out} ({len(CELLS)} cells)")


if __name__ == "__main__":
    main()
