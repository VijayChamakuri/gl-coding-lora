#!/usr/bin/env python3
"""LoRA/QLoRA fine-tune on the TrustGate rolling split.

CPU fold that this repository measured (no GPU on the build machine):

    python scripts/train_lora.py --config configs/cpu_fold2018.json

Full rolling run on a rented CUDA GPU. QLoRA is used when bitsandbytes
imports; otherwise the script trains float16/bfloat16 LoRA. Dollar cost was
not measured here: multiply the seconds field in train_manifest.json by the
provider's price.

    python scripts/train_lora.py --config configs/gpu_rolling.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glcode.train import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-steps", type=int, default=None, help="stop after this many optimizer steps (probes only)")
    parser.add_argument("--output-dir", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.max_steps is not None:
        cfg["max_steps"] = args.max_steps
    if args.output_dir is not None:
        cfg["output_dir"] = args.output_dir
    run(cfg)


if __name__ == "__main__":
    main()
