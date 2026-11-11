#!/usr/bin/env python3
"""Score a prompted Qwen model, or a LoRA adapter, on the rolling split.

    python scripts/evaluate.py --config configs/eval_zeroshot_2018.json
    python scripts/evaluate.py --config configs/eval_fewshot_2018.json
    python scripts/evaluate.py --config configs/eval_lora_2018.json

Predictions are appended to results/predictions/<tag>.jsonl so a stopped run
can be resumed with the same config. A partial file is not a full-fold result;
the JSON output says complete=false.
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glcode.evaluate import run


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", required=True)
    parser.add_argument("--max-eval", type=int, default=None, help="override the config sample size; 0 means the whole fold")
    parser.add_argument("--tag", default=None)
    args = parser.parse_args()
    cfg = json.loads(Path(args.config).read_text())
    if args.max_eval is not None:
        cfg["max_eval"] = args.max_eval
    if args.tag is not None:
        cfg["tag"] = args.tag
        cfg["output"] = f"results/probes/{args.tag}.json"
    run(cfg)


if __name__ == "__main__":
    main()
