#!/usr/bin/env python3
"""Write mlx-lm chat JSONL for one or more TrustGate folds.

    python scripts/export_mlx_data.py --folds 2018
    python scripts/export_mlx_data.py --folds 2018,2019,2020,2021,2022
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

from glcode.dataset import TEST_YEARS
from glcode.mlx_data import MAX_SEQ_LENGTH, PRIMARY_MODEL, export_folds, model_slug, write_lora_config
from glcode.paths import ROOT


def _warn_if_over_seq_length(dest: Path, years: list[int]) -> None:
    import os

    os.environ.setdefault("HF_HOME", str(ROOT / ".cache" / "huggingface"))
    try:
        from transformers import AutoTokenizer
    except ImportError:
        print(f"transformers is not installed; did not check that chats fit in {MAX_SEQ_LENGTH} tokens")
        return
    tokenizer = AutoTokenizer.from_pretrained("Qwen/Qwen2.5-0.5B-Instruct")
    longest = 0
    for year in years:
        for name in ("train.jsonl", "valid.jsonl"):
            for line in (dest / f"fold_{year}" / name).read_text().splitlines():
                messages = json.loads(line)["messages"]
                ids = tokenizer.apply_chat_template(messages, tokenize=True, add_generation_prompt=False)
                if hasattr(ids, "input_ids"):
                    ids = ids.input_ids
                if hasattr(ids, "tolist"):
                    ids = ids.tolist()
                if ids and isinstance(ids[0], list):
                    ids = ids[0]
                longest = max(longest, len(ids))
    print(f"longest training chat: {longest} tokens (cap {MAX_SEQ_LENGTH})")
    if longest > MAX_SEQ_LENGTH:
        raise SystemExit(
            f"a training chat is {longest} tokens, above max_seq_length {MAX_SEQ_LENGTH}; "
            "mlx-lm would truncate the account name"
        )


def parse_folds(text: str) -> list[int]:
    years = [int(part) for part in text.split(",") if part.strip()]
    unknown = [year for year in years if year not in TEST_YEARS]
    if unknown or not years:
        raise SystemExit(f"folds must be a subset of {TEST_YEARS}, got {text}")
    return years


def main() -> None:
    parser = argparse.ArgumentParser()
    parser.add_argument("--folds", default="2018")
    parser.add_argument("--model", default=PRIMARY_MODEL)
    args = parser.parse_args()
    years = parse_folds(args.folds)
    metas = export_folds(years, ROOT / "data" / "mlx")
    _warn_if_over_seq_length(ROOT / "data" / "mlx", years)
    written = []
    for meta in metas:
        year = meta["fold"]
        data_dir = ROOT / "data" / "mlx" / f"fold_{year}"
        adapter = ROOT / "results" / "mlx_adapters" / f"fold_{year}" / model_slug(args.model)
        config_path = ROOT / "results" / "mlx_configs" / f"fold_{year}_{model_slug(args.model)}.yaml"
        write_lora_config(config_path, args.model, data_dir, adapter, meta["iters"])
        written.append({"fold": year, "config": str(config_path), "train_rows": meta["train_rows"], "iters": meta["iters"]})
    print(json.dumps(written, indent=2))


if __name__ == "__main__":
    main()
