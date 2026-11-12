#!/usr/bin/env python3
"""Draw the README charts from measured result files.

Reads results/*.json, the fold-2018 prediction JSONL files, and, when it is
still on disk, the gitignored MLX training log. Writes docs/img/*.png and
results/mlx_train_curve_2018.json. Does not fit a model and does not fill a
missing metric.
"""

from __future__ import annotations

import csv
import gzip
import json
import re
from collections import Counter
from pathlib import Path

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt

ROOT = Path(__file__).resolve().parents[1]
RESULTS = ROOT / "results"
IMG = ROOT / "docs" / "img"
REFERENCE = ROOT / "data" / "reference" / "stockport_labelled_lines.csv.gz"
TRAIN_LOG = RESULTS / "mlx_fold_2018_qwen25-15b-instruct-4bit.log"
TRAIN_CURVE = RESULTS / "mlx_train_curve_2018.json"

NAVY = "#1f4e79"
CLAY = "#b8612e"
INK = "#1c1917"
GRID = "#e7e5e4"


def load_json(path: Path) -> dict:
    return json.loads(path.read_text())


def load_jsonl(path: Path) -> list[dict]:
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def load_reference() -> dict[str, dict]:
    opener = gzip.open if REFERENCE.suffix == ".gz" or REFERENCE.read_bytes()[:2] == b"\x1f\x8b" else open
    with opener(REFERENCE, "rt", newline="") as handle:
        return {row["card_line_id"]: row for row in csv.DictReader(handle)}


def parse_train_log(path: Path) -> dict:
    text = path.read_text(errors="replace")
    text = re.sub(r"\x1b\[[0-9;]*m", "", text)
    points = []
    val_points = []
    for raw in text.replace("\r", "\n").splitlines():
        line = raw.strip()
        val = re.match(r"^(\d+)\s+val\s+([0-9.]+)\s+([0-9.]+)s$", line)
        if val:
            val_points.append(
                {
                    "iter": int(val.group(1)),
                    "val_loss_printed": val.group(2),
                    "val_seconds_printed": val.group(3),
                }
            )
            continue
        train = re.match(
            r"^(\d+)\s+([0-9.]+)\s+[▼▲]\s+([0-9.]+)\s+([0-9.]+k)$",
            line,
        )
        if train:
            points.append(
                {
                    "iter": int(train.group(1)),
                    "train_loss_printed": train.group(2),
                    "tok_per_s_printed": train.group(3),
                    "tokens_printed": train.group(4),
                }
            )
    stat = path.stat()
    return {
        "source": str(path.relative_to(ROOT)),
        "note": "Values are the strings mlx-lm printed. They are not full-precision losses.",
        "train_points": points,
        "val_points": val_points,
        "log_birth_ns": stat.st_birthtime,
        "log_mtime_ns": stat.st_mtime,
    }


def style(ax) -> None:
    ax.set_facecolor("white")
    ax.spines["top"].set_visible(False)
    ax.spines["right"].set_visible(False)
    ax.spines["left"].set_color("#a8a29e")
    ax.spines["bottom"].set_color("#a8a29e")
    ax.tick_params(colors=INK, labelsize=9)
    ax.yaxis.grid(True, color=GRID, linewidth=0.8)
    ax.set_axisbelow(True)


def save(fig, name: str) -> None:
    IMG.mkdir(parents=True, exist_ok=True)
    path = IMG / name
    fig.savefig(path, dpi=140, bbox_inches="tight", facecolor="white")
    plt.close(fig)
    print(f"wrote {path.relative_to(ROOT)}")


def fold_models() -> list[dict]:
    specs = [
        ("Zero-shot 0.5B", RESULTS / "eval_qwen05_zeroshot_2018.json", "metrics"),
        ("4-shot 0.5B", RESULTS / "eval_qwen05_fewshot4_2018.json", "metrics"),
        ("LoRA 0.5B CPU", RESULTS / "eval_qwen05_lora_2018.json", "metrics"),
        ("LoRA 1.5B MLX", RESULTS / "eval_mlx_2018.json", "metrics"),
        ("Merchant lookup", RESULTS / "eval_mlx_2018.json", "paired_merchant_lookup"),
    ]
    rows = []
    for label, path, key in specs:
        report = load_json(path)
        metrics = report[key]
        rows.append(
            {
                "label": label,
                "n": metrics["n"],
                "top1": metrics["top1"],
                "top1_ci": metrics["top1_ci"],
                "top3": metrics["top3"],
                "top3_ci": metrics["top3_ci"],
                "top1_seen": metrics["top1_seen_merchants"],
                "top1_new": metrics["top1_new_merchants"],
                "top1_new_ci": metrics["top1_new_ci"],
                "n_new": metrics["n_new_merchants"],
                "n_seen": metrics["n_seen_merchants"],
            }
        )
    baselines = load_json(RESULTS / "baselines.json")
    tfidf = next(
        row
        for row in baselines["baselines"]["per_year"]
        if row["model"] == "tfidf_logreg" and row["year"] == 2018
    )
    rows.append(
        {
            "label": "TF-IDF 2018",
            "n": tfidf["n"],
            "top1": tfidf["accuracy"],
            "top1_ci": None,
            "top3": tfidf["top3"],
            "top3_ci": None,
            "top1_seen": None,
            "top1_new": tfidf["accuracy_new_merchants"],
            "top1_new_ci": None,
            "n_new": None,
            "n_seen": None,
            "ci_note": "results/baselines.json per_year has no bootstrap interval for this row",
        }
    )
    return rows


def chart_top1(rows: list[dict]) -> None:
    fig, ax = plt.subplots(figsize=(8.4, 4.4))
    style(ax)
    xs = list(range(len(rows)))
    heights = [row["top1"] for row in rows]
    ax.bar(xs, heights, color=NAVY, width=0.72, zorder=2)
    err_x, err_y, err_low, err_high = [], [], [], []
    for x, row in zip(xs, rows):
        if not row["top1_ci"]:
            continue
        err_x.append(x)
        err_y.append(row["top1"])
        err_low.append(row["top1"] - row["top1_ci"][0])
        err_high.append(row["top1_ci"][1] - row["top1"])
    if err_x:
        ax.errorbar(
            err_x,
            err_y,
            yerr=[err_low, err_high],
            fmt="none",
            ecolor=INK,
            elinewidth=1,
            capsize=3,
            zorder=3,
        )
    ax.set_xticks(xs, [row["label"] for row in rows], rotation=18, ha="right")
    ax.set_ylim(0, 0.85)
    ax.set_ylabel("Top-1")
    ax.set_title("Fold 2018 top-1. Caps are stored 95% intervals")
    save(fig, "top1-ci.png")


def chart_new_seen(rows: list[dict]) -> None:
    usable = [row for row in rows if row["top1_seen"] is not None]
    fig, ax = plt.subplots(figsize=(8.2, 4.4))
    style(ax)
    xs = list(range(len(usable)))
    width = 0.36
    ax.bar([x - width / 2 for x in xs], [row["top1_seen"] for row in usable], width=width, color=NAVY, label="Seen merchant")
    ax.bar([x + width / 2 for x in xs], [row["top1_new"] for row in usable], width=width, color=CLAY, label="New merchant")
    ax.set_xticks(xs, [row["label"] for row in usable], rotation=18, ha="right")
    ax.set_ylim(0, 1)
    ax.set_ylabel("Top-1")
    ax.set_title("Fold 2018 seen and new merchants")
    ax.legend(frameon=False)
    save(fig, "new-vs-seen.png")


def topk_from_jsonl(path: Path) -> dict[int, float]:
    rows = load_jsonl(path)
    hits = {1: 0, 2: 0, 3: 0}
    for row in rows:
        ranked = row["top3"]
        for k in (1, 2, 3):
            if row["y"] in ranked[:k]:
                hits[k] += 1
    n = len(rows)
    return {k: hits[k] / n for k in (1, 2, 3)} | {"n": n}


def chart_topk() -> dict:
    series = {
        "Zero-shot 0.5B": topk_from_jsonl(RESULTS / "predictions/qwen05_zeroshot_2018.jsonl"),
        "4-shot 0.5B": topk_from_jsonl(RESULTS / "predictions/qwen05_fewshot4_2018.jsonl"),
        "LoRA 0.5B CPU": topk_from_jsonl(RESULTS / "predictions/qwen05_lora_2018.jsonl"),
        "LoRA 1.5B MLX": topk_from_jsonl(RESULTS / "predictions/mlx_2018.jsonl"),
    }
    fig, ax = plt.subplots(figsize=(6.4, 4.2))
    style(ax)
    colors = [NAVY, "#5b7c99", CLAY, "#7c2d12"]
    for (label, rates), color in zip(series.items(), colors):
        ax.plot([1, 2, 3], [rates[1], rates[2], rates[3]], marker="o", color=color, label=label)
    ax.set_xticks([1, 2, 3])
    ax.set_ylim(0, 1)
    ax.set_xlabel("k")
    ax.set_ylabel("True account inside top k")
    ax.set_title("Top-k from stored top-3 lists (k = 1, 2, 3)")
    ax.legend(frameon=False, fontsize=8)
    save(fig, "topk.png")
    return series


def chart_loss(curve: dict) -> None:
    iters = [point["iter"] for point in curve["train_points"]]
    losses = [float(point["train_loss_printed"]) for point in curve["train_points"]]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style(ax)
    ax.plot(iters, losses, color=NAVY, linewidth=1.4)
    if curve["val_points"]:
        ax.scatter(
            [point["iter"] for point in curve["val_points"]],
            [float(point["val_loss_printed"]) for point in curve["val_points"]],
            color=CLAY,
            s=28,
            zorder=3,
            label="Printed validation loss",
        )
        ax.legend(frameon=False)
    ax.set_xlabel("Iteration")
    ax.set_ylabel("Loss printed by mlx-lm")
    ax.set_title("MLX 1.5B training loss, as printed")
    save(fig, "train-loss.png")


def chart_confusion() -> dict:
    rows = load_jsonl(RESULTS / "predictions/mlx_2018.jsonl")
    support = Counter(row["y"] for row in rows)
    labels = [name for name, _count in support.most_common(6)]
    matrix = [[0 for _ in labels] for _ in labels]
    index = {name: i for i, name in enumerate(labels)}
    outside = 0
    for row in rows:
        if row["y"] not in index:
            continue
        if row["pred"] not in index:
            outside += 1
            continue
        matrix[index[row["y"]]][index[row["pred"]]] += 1
    fig, ax = plt.subplots(figsize=(7.6, 6.2))
    ax.imshow(matrix, cmap="Blues")
    ax.set_xticks(range(len(labels)), labels, rotation=30, ha="right", fontsize=8)
    ax.set_yticks(range(len(labels)), labels, fontsize=8)
    ax.set_xlabel("Predicted")
    ax.set_ylabel("Booked account")
    ax.set_title("MLX 1.5B, six most common 2018 accounts")
    peak = max(max(row) for row in matrix) or 1
    for i in range(len(labels)):
        for j in range(len(labels)):
            value = matrix[i][j]
            color = "white" if value > peak * 0.45 else INK
            ax.text(j, i, str(value), ha="center", va="center", color=color, fontsize=8)
    fig.tight_layout()
    save(fig, "confusion.png")
    return {"labels": labels, "matrix": matrix, "predicted_outside_these_six": outside, "support": support.most_common(6)}


def examples() -> list[dict]:
    reference = load_reference()
    rows = load_jsonl(RESULTS / "predictions/mlx_2018.jsonl")
    rows = sorted(rows, key=lambda row: row["card_line_id"])

    def pack(row: dict) -> dict:
        source = reference[row["card_line_id"]]
        return {
            "card_line_id": row["card_line_id"],
            "merchant_raw": source["merchant_raw"],
            "department": source["department"],
            "service_category": source["service_category"],
            "amount": source["amount"],
            "y": row["y"],
            "pred": row["pred"],
            "top3": row["top3"],
            "seen_merchant": row["seen_merchant"],
            "lookup": row["lookup"],
        }

    picked = []
    for predicate, why in (
        (lambda row: row["pred"] == row["y"], "first correct row in card_line_id order"),
        (lambda row: row["pred"] != row["y"] and row["seen_merchant"], "first seen-merchant error in card_line_id order"),
        (lambda row: row["pred"] != row["y"] and not row["seen_merchant"], "first new-merchant error in card_line_id order"),
    ):
        match = next(row for row in rows if predicate(row))
        item = pack(match)
        item["why"] = why
        picked.append(item)
    return picked


def main() -> None:
    rows = fold_models()
    chart_top1(rows)
    chart_new_seen(rows)
    topk = chart_topk()
    if TRAIN_LOG.exists():
        curve = parse_train_log(TRAIN_LOG)
        TRAIN_CURVE.write_text(json.dumps(curve, indent=2) + "\n")
        print(f"wrote {TRAIN_CURVE.relative_to(ROOT)} points={len(curve['train_points'])} val={len(curve['val_points'])}")
    elif TRAIN_CURVE.exists():
        curve = load_json(TRAIN_CURVE)
    else:
        curve = None
        print("no training log and no curve JSON; skipped loss chart")
    if curve and curve["train_points"]:
        chart_loss(curve)
    confusion = chart_confusion()
    payload = {
        "models": rows,
        "topk": topk,
        "confusion_labels": confusion["labels"],
        "confusion_matrix": confusion["matrix"],
        "predicted_outside_these_six": confusion["predicted_outside_these_six"],
        "support": confusion["support"],
        "examples": examples(),
        "train_curve_points": None if not curve else len(curve["train_points"]),
        "train_curve_first": None if not curve else curve["train_points"][:2],
        "train_curve_last": None if not curve else curve["train_points"][-2:],
        "val_points": None if not curve else curve["val_points"],
    }
    out = Path("/tmp/mlx-run/figure-notes.json")
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(payload, indent=2) + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
