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

ACCENT = "#0f766e"
ACCENT_SOFT = "#5eead4"
GREY = "#a8a29e"
GREY_DARK = "#57534e"
GREY_LIGHT = "#d6d3d1"
INK = "#1c1917"
PAPER = "#fafaf9"
GRID = "#e7e5e4"
HEADLINE = "LoRA 0.5B CPU"
NAVY = ACCENT
CLAY = GREY_DARK


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


def is_headline(label: str) -> bool:
    return label == HEADLINE


def pct(value: float) -> str:
    from decimal import Decimal, ROUND_HALF_UP

    quant = (Decimal(str(value)) * Decimal(100)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
    return f"{quant}%"


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
    colors = [ACCENT if is_headline(row["label"]) else GREY for row in rows]
    ax.bar(xs, heights, color=colors, width=0.72, zorder=2)
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
    seen_colors = [ACCENT if is_headline(row["label"]) else GREY_DARK for row in usable]
    new_colors = [ACCENT_SOFT if is_headline(row["label"]) else GREY_LIGHT for row in usable]
    ax.bar([x - width / 2 for x in xs], [row["top1_seen"] for row in usable], width=width, color=seen_colors, label="Seen merchant")
    ax.bar([x + width / 2 for x in xs], [row["top1_new"] for row in usable], width=width, color=new_colors, label="New merchant")
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
    greys = ["#78716c", "#a8a29e", "#d6d3d1"]
    grey_i = 0
    for label, rates in series.items():
        color = ACCENT if is_headline(label) else greys[grey_i]
        if not is_headline(label):
            grey_i += 1
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
    ax.plot(iters, losses, color=GREY_DARK, linewidth=1.4)
    if curve["val_points"]:
        ax.scatter(
            [point["iter"] for point in curve["val_points"]],
            [float(point["val_loss_printed"]) for point in curve["val_points"]],
            color=GREY,
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
    ax.imshow(matrix, cmap="Greys")
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


def short_label(label: str) -> str:
    return {
        "Zero-shot 0.5B": "Zero-shot",
        "4-shot 0.5B": "4 examples",
        "LoRA 0.5B CPU": "LoRA 0.5B",
        "LoRA 1.5B MLX": "LoRA 1.5B, 4-bit",
        "Merchant lookup": "Merchant lookup",
        "TF-IDF 2018": "TF-IDF",
    }.get(label, label)


def chart_scoreboard(rows: list[dict]) -> None:
    fig, axes = plt.subplots(1, 2, figsize=(11.2, 4.6), facecolor="white")
    panels = (("top1", "Top-1, same 3,387 lines"), ("top1_new", "New merchants only"))
    for ax, (key, title) in zip(axes, panels):
        style(ax)
        ordered = list(reversed(rows))
        ys = list(range(len(ordered)))
        colors = [ACCENT if is_headline(row["label"]) else GREY for row in ordered]
        ax.barh(ys, [row[key] * 100 for row in ordered], color=colors, height=0.72)
        ax.set_yticks(ys, [short_label(row["label"]) for row in ordered])
        ax.set_xlim(0, 100)
        ax.set_xlabel("Percent")
        ax.set_title(title)
        for y, row in zip(ys, ordered):
            ax.text(row[key] * 100 + 1.2, y, pct(row[key]), va="center", fontsize=8, color=INK)
    fig.tight_layout()
    save(fig, "scoreboard.png")


def chart_progression(rows: list[dict]) -> None:
    wanted = ["Zero-shot 0.5B", "4-shot 0.5B", "LoRA 0.5B CPU"]
    picked = [next(row for row in rows if row["label"] == name) for name in wanted]
    fig, ax = plt.subplots(figsize=(7.2, 4.2))
    style(ax)
    xs = list(range(len(picked)))
    ax.bar(xs, [row["top1"] * 100 for row in picked], color=[GREY, GREY_DARK, ACCENT], width=0.68)
    ax.set_xticks(xs, [short_label(row["label"]) for row in picked])
    ax.set_ylim(0, 100)
    ax.set_ylabel("Top-1 percent")
    ax.set_title("Ablation: zero-shot, four examples, then LoRA")
    for x, row in zip(xs, picked):
        ax.text(x, row["top1"] * 100 + 2, pct(row["top1"]), ha="center", fontsize=10, color=INK)
    save(fig, "progression.png")


def _tile(fig, x, y, w, h, value, caption) -> None:
    box = plt.Rectangle((x, y), w, h, transform=fig.transFigure, facecolor="white", edgecolor=GREY_LIGHT, linewidth=1.2, zorder=2)
    fig.patches.append(box)
    fig.text(x + 0.02, y + h * 0.58, value, fontsize=14, color=ACCENT, fontweight="bold", zorder=3)
    fig.text(x + 0.02, y + h * 0.22, caption, fontsize=8, color=GREY_DARK, zorder=3)


def chart_hero(rows: list[dict]) -> None:
    lora = next(row for row in rows if is_headline(row["label"]))
    lookup = next(row for row in rows if row["label"] == "Merchant lookup")
    fig = plt.figure(figsize=(12.2, 3.8), facecolor=PAPER)
    fig.text(0.05, 0.74, "GL-LoRA", fontsize=32, color=INK, fontweight="bold")
    fig.text(0.05, 0.58, "Which account should this card purchase hit?", fontsize=14, color=GREY_DARK)
    _tile(fig, 0.05, 0.12, 0.28, 0.34, f"{pct(lora['top1_new'])} vs {pct(lookup['top1_new'])}", "New merchants, LoRA vs lookup")
    _tile(fig, 0.36, 0.12, 0.28, 0.34, "M2 MacBook Air, 8GB", "QLoRA config")
    _tile(fig, 0.67, 0.12, 0.28, 0.34, f"{lora['n']:,}", "2018 card lines scored")
    save(fig, "hero.png")


def chart_flow() -> None:
    report = load_json(RESULTS / "eval_mlx_2018.json")
    prompt = report["example_prompt"]
    picked = examples()[0]
    accounts = [line[2:] for line in prompt.splitlines() if line.startswith("- ")]
    fig = plt.figure(figsize=(12.2, 5.1), facecolor=PAPER)
    fig.text(0.04, 0.94, "One real purchase", fontsize=16, color=INK, fontweight="bold")
    fig.text(0.04, 0.86, "The card line", fontsize=11, color=ACCENT, fontweight="bold")
    fig.text(
        0.04,
        0.80,
        f"Merchant: {picked['merchant_raw']}\nDepartment: {picked['department']}\nService: {picked['service_category']}\nAmount GBP: {picked['amount']}",
        fontsize=9,
        color=INK,
        va="top",
        family="monospace",
    )
    fig.text(0.38, 0.86, "Account list in the prompt", fontsize=11, color=ACCENT, fontweight="bold")
    fig.text(0.38, 0.80, "\n".join(accounts), fontsize=8, color=INK, va="top", family="monospace", linespacing=1.25)
    fig.text(0.70, 0.86, "Saved top 3", fontsize=11, color=ACCENT, fontweight="bold")
    y = 0.78
    for i, name in enumerate(picked["top3"], start=1):
        booked = name == picked["y"]
        fig.text(
            0.70,
            y,
            f"{i}. {name}",
            fontsize=9,
            color="white" if booked else INK,
            va="top",
            family="monospace",
            bbox=dict(boxstyle="square,pad=0.35", facecolor=ACCENT, edgecolor="none") if booked else None,
        )
        y -= 0.07
    fig.text(0.04, 0.04, "Numeric scores for each account were not stored. The teal row is the booked account.", fontsize=8, color=GREY_DARK)
    save(fig, "flow.png")


def chart_errors() -> None:
    cards = examples()
    styles = [
        ("#166534", "Correct"),
        ("#a16207", "Wrong, but the booked account is in the top 3"),
        ("#b91c1c", "Wrong, and the booked account is outside the top 3"),
    ]
    fig = plt.figure(figsize=(12.2, 3.5), facecolor=PAPER)
    for i, (item, (color, verdict)) in enumerate(zip(cards, styles)):
        x = 0.03 + i * 0.32
        box = plt.Rectangle((x, 0.06), 0.30, 0.88, transform=fig.transFigure, facecolor="white", edgecolor=color, linewidth=2, zorder=2)
        fig.patches.append(box)
        body = (
            f"{verdict}\n\n"
            f"{item['merchant_raw']}\n"
            f"{item['department']}\n"
            f"{item['service_category']}\n"
            f"GBP {item['amount']}\n\n"
            f"Booked: {item['y']}\n"
            f"Model: {item['pred']}\n"
            f"Lookup: {item['lookup']}"
        )
        fig.text(x + 0.015, 0.84, body, fontsize=8, color=INK, va="top", zorder=3)
    save(fig, "error-cards.png")


def laptop_facts() -> dict:
    report = load_json(RESULTS / "eval_mlx_2018.json")
    facts = {
        "score_seconds": report["seconds"],
        "seconds_per_example": report["seconds_per_example"],
        "train_seconds": None,
        "peak_footprint_mb": None,
    }
    if TRAIN_LOG.exists():
        stat = TRAIN_LOG.stat()
        facts["train_seconds"] = round(stat.st_mtime - stat.st_birthtime, 1)
    peak_path = Path("/tmp/mlx-run/footprint-peak.txt")
    if peak_path.exists():
        for line in peak_path.read_text().splitlines():
            if line.startswith("peak_footprint_mb="):
                facts["peak_footprint_mb"] = line.split("=", 1)[1]
    return facts


def rounded_laptop(facts: dict) -> dict:
    from decimal import Decimal, ROUND_HALF_UP

    shown = {}
    if facts["train_seconds"] is not None:
        minutes = (Decimal(str(facts["train_seconds"])) / Decimal(60)).quantize(Decimal("1"), rounding=ROUND_HALF_UP)
        shown["train"] = f"~{minutes} min"
    if facts["peak_footprint_mb"] is not None:
        gb = (Decimal(str(facts["peak_footprint_mb"])) / Decimal(1000)).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        shown["peak"] = f"{gb} GB"
    if facts["seconds_per_example"] is not None:
        seconds = Decimal(str(facts["seconds_per_example"])).quantize(Decimal("0.1"), rounding=ROUND_HALF_UP)
        shown["line"] = f"{seconds} s"
    return shown


def chart_laptop(facts: dict) -> None:
    fig = plt.figure(figsize=(12.2, 3.2), facecolor=PAPER)
    fig.text(0.04, 0.82, "The 4-bit run on this Mac", fontsize=16, color=INK, fontweight="bold")
    shown = rounded_laptop(facts)
    tiles = []
    if "train" in shown:
        tiles.append((shown["train"], "Training"))
    if "peak" in shown:
        tiles.append((shown["peak"], "Peak footprint"))
    if "line" in shown:
        tiles.append((shown["line"], "Seconds per line"))
    for i, (value, caption) in enumerate(tiles):
        _tile(fig, 0.04 + i * 0.32, 0.16, 0.28, 0.48, value, caption)
    save(fig, "laptop.png")


def main() -> None:
    rows = fold_models()
    for row in rows:
        ci = row["top1_ci"]
        ci_text = "none stored" if not ci else f"{pct(ci[0])}–{pct(ci[1])}"
        new_ci = row["top1_new_ci"]
        new_text = "none stored" if not new_ci else f"{pct(new_ci[0])}–{pct(new_ci[1])}"
        print(f"PCT {row['label']} top1={pct(row['top1'])} ({ci_text}) new={pct(row['top1_new'])} ({new_text})")
    chart_top1(rows)
    chart_new_seen(rows)
    chart_scoreboard(rows)
    chart_progression(rows)
    chart_hero(rows)
    chart_flow()
    chart_errors()
    facts = laptop_facts()
    chart_laptop(facts)
    print("LAPTOP", facts)
    print("LAPTOP_ROUNDED", rounded_laptop(facts))
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
        "laptop": facts,
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
