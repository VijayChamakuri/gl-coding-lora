#!/usr/bin/env python3
"""Check the labelled export, or rebuild it from the data.gov.uk files.

    python scripts/prepare_data.py --check
    python scripts/prepare_data.py --download
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glcode.dataset import YEAR_COUNTS, load_lines, sha256_file
from glcode.paths import REFERENCE, REFERENCE_SHA256


def check() -> None:
    digest = sha256_file(REFERENCE)
    if digest != REFERENCE_SHA256:
        raise SystemExit(f"hash mismatch for {REFERENCE}: {digest}")
    df = load_lines()
    counts = df.statement_year.value_counts().sort_index().to_dict()
    print(f"rows {len(df)} accounts {df.gl_account.nunique()} merchants {df.merchant_key.nunique()}")
    print("years", counts)
    if counts != YEAR_COUNTS:
        raise SystemExit(f"year counts differ from the TrustGate export: {counts}")
    print("reference ok", digest)


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--check", action="store_true")
    parser.add_argument("--download", action="store_true", help="fetch CKAN files and compare to the reference export")
    args = parser.parse_args()
    if not args.check and not args.download:
        args.check = True
    if args.check:
        check()
    if args.download:
        from glcode.prepare import run_download

        report = run_download()
        print(json.dumps(report, indent=2))
        if not report["comparison"]["content_match"]:
            raise SystemExit("rebuild did not match the reference export; see data/processed/rebuild_report.json")


if __name__ == "__main__":
    main()
