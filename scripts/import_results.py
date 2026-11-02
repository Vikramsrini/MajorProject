#!/usr/bin/env python3
"""Copy a finished notebook run into paper/generated/ so paper/main.tex compiles with real results.

The Colab notebook (notebooks/er_full_pipeline_colab.ipynb) ends by producing er_results_<RUN>.zip. Its
latex/ folder holds numbers.tex, every table_*.tex and every fig_*.pdf the paper reads. Usage:

    python scripts/import_results.py ~/Downloads/er_results_full_v1.zip
    python scripts/import_results.py path/to/unzipped/run_folder

Then compile paper/main.tex. Anything still red in the PDF is a number or claim the run did not produce.
"""

from __future__ import annotations

import argparse
import shutil
import sys
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
DEST = ROOT / "paper" / "generated"


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("source", help="er_results_<RUN>.zip or an unzipped run folder")
    args = ap.parse_args()

    src = Path(args.source).expanduser()
    tmp = None
    if src.suffix == ".zip":
        tmp = Path(tempfile.mkdtemp())
        with zipfile.ZipFile(src) as zf:
            bad = [n for n in zf.namelist() if ".." in n or n.startswith("/")]
            if bad:
                raise SystemExit(f"refusing to extract unsafe entries: {bad}")
            zf.extractall(tmp)
        src = tmp
    latex = next((p for p in [src / "latex", *src.glob("*/latex")] if p.is_dir()), None)
    if latex is None or not (latex / "numbers.tex").exists():
        raise SystemExit(f"no latex/numbers.tex under {src}: is this a finished notebook run?")

    DEST.mkdir(parents=True, exist_ok=True)
    for old in DEST.glob("*"):
        if old.is_file():
            old.unlink()
    copied = []
    for f in sorted(latex.iterdir()):
        if f.suffix in {".tex", ".pdf"}:
            shutil.copy2(f, DEST / f.name)
            copied.append(f.name)
    run = latex.parent
    for name in ("PAPER_READINESS.md", "FINDINGS_AUTO.md", "run_meta.json"):
        if (run / name).exists():
            shutil.copy2(run / name, DEST / name)
            copied.append(name)
    print(f"copied {len(copied)} files to {DEST}")

    readiness = DEST / "PAPER_READINESS.md"
    if readiness.exists():
        print("\n" + readiness.read_text(encoding="utf-8"))
    if tmp:
        shutil.rmtree(tmp, ignore_errors=True)


if __name__ == "__main__":
    sys.exit(main())
