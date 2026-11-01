#!/usr/bin/env python3
"""Download the WDC Products benchmark (pair-wise splits) into data/raw.

Sizes (official): 80pair.zip 21.5MB, 50pair.zip 21.6MB, 20pair.zip 20MB.
The multi-class variants (80multi/50multi/20multi) and PDC2020-C (4.3GB) are
optional and not fetched by default.

Source: https://webdatacommons.org/largescaleproductcorpus/wdc-products/
"""

from __future__ import annotations

import argparse
import io
import shutil
import sys
import urllib.request
import zipfile
from pathlib import Path

BASE = "https://data.dws.informatik.uni-mannheim.de/largescaleproductcorpus/data/wdc-products"
PAIR_ZIPS = ["80pair.zip", "50pair.zip", "20pair.zip"]
UA = {"User-Agent": "Mozilla/5.0 (research; entity-resolution benchmark download)"}


def fetch(url: str, dest: Path, timeout: int = 120) -> None:
    req = urllib.request.Request(url, headers=UA)
    with urllib.request.urlopen(req, timeout=timeout) as r:
        data = r.read()
    dest.write_bytes(data)
    print(f"  {dest.name}: {len(data) / 1e6:.1f} MB")


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dest", default="data/raw", help="destination directory")
    ap.add_argument("--only", default=None, help="download a single zip, e.g. 80pair.zip")
    args = ap.parse_args()

    dest = Path(args.dest)
    dest.mkdir(parents=True, exist_ok=True)
    zips = [args.only] if args.only else PAIR_ZIPS
    for name in zips:
        zip_path = dest / name
        if not zip_path.exists():
            print(f"downloading {name}")
            fetch(f"{BASE}/{name}", zip_path)
        else:
            print(f"already present: {zip_path}")
        outdir = dest / name.removesuffix(".zip")
        if not outdir.exists():
            with zipfile.ZipFile(io.BytesIO(zip_path.read_bytes())) as zf:
                bad = [n for n in zf.namelist() if ".." in n or n.startswith("/")]
                if bad:
                    raise SystemExit(f"refusing to extract unsafe entries: {bad}")
                zf.extractall(outdir)
            n = len(list(outdir.rglob("*.json.gz")))
            print(f"  extracted {n} files -> {outdir}")

    print("\nDone. Verify with: python scripts/run_wdc.py --data data/raw --corner 80 --unseen 000")


if __name__ == "__main__":
    sys.exit(main())
