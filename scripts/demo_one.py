#!/usr/bin/env python3
"""Score one 2018 purchase with the committed 0.5B adapter.

The weights are results/adapters/fold_2018/adapter_model.safetensors.
The 4-bit MLX adapter is not in git. That path needs `make mlx` first.
This script prints the live ranking. It does not write a result file.
"""

from __future__ import annotations

import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

CARD_ID = "2018-01_f41af6e3.csv:10"
ADAPTER = Path("results/adapters/fold_2018")


def main() -> None:
    weights = ADAPTER / "adapter_model.safetensors"
    if not weights.exists():
        raise SystemExit(f"missing adapter weights at {weights}")
    from glcode.dataset import candidate_accounts, iter_folds, load_lines
    from glcode.evaluate import _load
    from glcode.metrics import rank_labels
    from glcode.prompts import user_message
    from glcode.score import completion_logprobs

    df = load_lines()
    train = test = None
    for year, train_fold, test_fold in iter_folds(df):
        if year == 2018:
            train, test = train_fold, test_fold
            break
    if test is None:
        raise SystemExit("2018 fold was not found")
    row = test[test.card_line_id == CARD_ID]
    if row.empty:
        raise SystemExit(f"{CARD_ID} is not in the 2018 fold")
    item = row.iloc[0]
    accounts = candidate_accounts(train)
    prompt = user_message(item, accounts)
    print(prompt)
    print("---")
    tokenizer, model = _load("Qwen/Qwen2.5-0.5B-Instruct", ADAPTER)
    sums, means = completion_logprobs(model, tokenizer, prompt, accounts)
    ranked = rank_labels(accounts, means, sums)
    print(f"card_line_id: {item.card_line_id}")
    print(f"booked: {item.gl_account}")
    print("top3: " + " | ".join(ranked["top3"]))
    print(f"top1: {ranked['pred']}")


if __name__ == "__main__":
    main()
