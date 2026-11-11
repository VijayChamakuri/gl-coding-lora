#!/usr/bin/env python3
"""Print the fold-2018 language-model rows from measured eval JSON.

Does not invent values. Exits non-zero if an expected file is missing or incomplete.
"""

from __future__ import annotations

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[1]))

FILES = [
    ("Qwen2.5-0.5B zero-shot", "results/eval_qwen05_zeroshot_2018.json"),
    ("Qwen2.5-0.5B 4-shot", "results/eval_qwen05_fewshot4_2018.json"),
    ("Qwen2.5-0.5B LoRA, 1 epoch", "results/eval_qwen05_lora_2018.json"),
]


def pct(value) -> str:
    if value is None:
        return "—"
    return f"{100 * value:.1f}%"


def ci(pair) -> str:
    if not pair:
        return "—"
    return f"{pct(pair[0]).rstrip('%')}–{pct(pair[1])}"


def row(label: str, metrics: dict, n: int) -> str:
    return (
        f"| {label} | {n:,} | {pct(metrics['top1'])} | {ci(metrics['top1_ci'])} | "
        f"{pct(metrics['top3'])} | {ci(metrics['top3_ci'])} | "
        f"{pct(metrics['top1_new_merchants'])} | {ci(metrics['top1_new_ci'])} |"
    )


def main() -> None:
    missing = [path for _, path in FILES if not Path(path).exists()]
    if missing:
        raise SystemExit("missing " + ", ".join(missing))
    print("| Model | n | Top-1 | 95% CI | Top-3 | Top-3 95% CI | New merchants | New-merchant 95% CI |")
    print("|---|---:|---:|---:|---:|---:|---:|---:|")
    for label, path in FILES:
        report = json.loads(Path(path).read_text())
        if not report.get("complete"):
            raise SystemExit(f"{path} is not complete")
        if not report.get("measured"):
            raise SystemExit(f"{path} is not marked measured")
        metrics = report["metrics"]
        print(row(label, metrics, report["n_scored"]))
        paired = report["paired_merchant_lookup"]
        print(row(label + " — paired merchant lookup", paired, report["n_scored"]))
        secondary = report["metrics_sum_logprob"]
        print(
            f"<!-- {label} sum-logprob top1 {secondary['top1']:.6f} "
            f"top3 {secondary['top3']:.6f} -->"
        )
        fmt = report["format_check"]
        print(
            f"<!-- {label} format {fmt['exact_account']}/{fmt['n']} exact, "
            f"{fmt['in_candidate_list']}/{fmt['n']} in the account list, error={fmt['error']} -->"
        )
        print(
            f"<!-- {label} seconds {report['seconds']} per_example {report['seconds_per_example']} "
            f"shots {report['shots']} adapter {report['use_adapter']} -->"
        )


if __name__ == "__main__":
    main()
