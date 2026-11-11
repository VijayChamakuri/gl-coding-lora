#!/usr/bin/env python3
"""Score an MLX LoRA adapter on the TrustGate rolling folds.

Requires Apple Silicon and mlx-lm. The Linux VM cannot run this script.
"""

from __future__ import annotations

import argparse
import json
import platform
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glcode.mlx_data import TEST_YEARS


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", required=True)
    parser.add_argument("--model", required=True)
    parser.add_argument("--adapter-root", required=True)
    parser.add_argument("--output", required=True)
    parser.add_argument("--tag", required=True)
    parser.add_argument("--fallback-from", default=None)
    args = parser.parse_args()
    if platform.system() != "Darwin" or platform.machine() != "arm64":
        raise SystemExit(
            f"MLX scoring was not run: this machine is {platform.system()} {platform.machine()}, not Apple Silicon."
        )
    years = [int(part) for part in args.folds.split(",") if part.strip()]
    unknown = [year for year in years if year not in TEST_YEARS]
    if unknown:
        raise SystemExit(f"unknown folds {unknown}")
    from glcode.mlx_eval import run

    run(
        {
            "folds": years,
            "model_name": args.model,
            "adapter_root": args.adapter_root,
            "output": args.output,
            "tag": args.tag,
            "fallback_from": args.fallback_from,
        }
    )


if __name__ == "__main__":
    main()
