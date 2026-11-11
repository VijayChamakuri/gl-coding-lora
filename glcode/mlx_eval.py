"""Closed-set scoring of an MLX LoRA adapter.

The ranking, bootstrap, and JSON keys match `glcode.evaluate`. MLX is imported
only inside `score_labels`, so the report builder can be tested on Linux.
No accuracy in this module was measured on the Linux development machine.
"""

from __future__ import annotations

import json
import platform
import time
from pathlib import Path

import numpy as np

from .baselines import lookup_on_rows
from .dataset import candidate_accounts, iter_folds, load_lines
from .hardware import describe
from .metrics import rank_labels, summarize
from .mlx_data import FALLBACK_MODEL, PRIMARY_MODEL, SEED
from .prompts import messages, user_message


def _as_ids(out) -> list[int]:
    if hasattr(out, "input_ids"):
        out = out.input_ids
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        if len(out) != 1:
            raise RuntimeError(f"expected one conversation, got batch {len(out)}")
        out = out[0]
    return [int(x) for x in out]


def encode_chat(tokenizer, convo: list[dict], add_generation_prompt: bool) -> list[int]:
    if not hasattr(tokenizer, "apply_chat_template"):
        raise RuntimeError("MLX tokenizer has no apply_chat_template; refusing to score a different prompt format")
    try:
        out = tokenizer.apply_chat_template(
            convo, add_generation_prompt=add_generation_prompt, tokenize=True
        )
    except TypeError:
        out = tokenizer.apply_chat_template(convo, tokenize=True, add_generation_prompt=add_generation_prompt)
    return _as_ids(out)


def _logits_array(model, token_ids: list[int]):
    import mlx.core as mx

    tokens = mx.array(token_ids)[None]
    raw = model(tokens)
    if isinstance(raw, tuple):
        raw = raw[0]
    mx.eval(raw)
    return np.array(raw[0].astype(mx.float32))


def score_labels(model, tokenizer, user: str, labels: list[str]) -> tuple[list[float], list[float]]:
    from .mlx_data import completion_token_logprobs

    prompt = encode_chat(tokenizer, messages(user), add_generation_prompt=True)
    sums: list[float] = []
    means: list[float] = []
    for label in labels:
        full = encode_chat(tokenizer, messages(user, label), add_generation_prompt=False)
        if full[: len(prompt)] != prompt:
            raise RuntimeError("chat template prompt is not a prefix of the full conversation")
        logits = _logits_array(model, full)
        total, mean = completion_token_logprobs(logits, full, len(prompt))
        sums.append(total)
        means.append(mean)
    return sums, means


def _metrics(rows: list[dict], pred_key: str, top_key: str) -> dict:
    correct = np.array([row[pred_key] == row["y"] for row in rows], dtype=float)
    top3 = np.array([row["y"] in row[top_key] for row in rows], dtype=float)
    seen = np.array([row["seen_merchant"] for row in rows], dtype=bool)
    return summarize(correct, top3, seen, y=[row["y"] for row in rows], pred=[row[pred_key] for row in rows])


def _lookup_metrics(rows: list[dict]) -> dict:
    correct = np.array([row["lookup"] == row["y"] for row in rows], dtype=float)
    seen = np.array([row["seen_merchant"] for row in rows], dtype=bool)
    summary = summarize(correct, correct, seen, y=[row["y"] for row in rows], pred=[row["lookup"] for row in rows])
    summary["top3"] = None
    summary["top3_ci"] = None
    summary["top3_seen_merchants"] = None
    summary["top3_new_merchants"] = None
    summary["top3_new_ci"] = None
    return summary


def _read_jsonl(path: Path) -> list[dict]:
    if not path.exists():
        return []
    rows = []
    with path.open() as handle:
        for line in handle:
            line = line.strip()
            if line:
                rows.append(json.loads(line))
    return rows


def assemble_report(
    rows: list[dict],
    fold_reports: list[dict],
    *,
    tag: str,
    model_name: str,
    adapter_dir: str,
    folds: list[int],
    seconds: float,
    n_outside: int,
    example_prompt: str,
    fallback_from: str | None,
    mlx_executed: bool,
) -> dict:
    complete = bool(fold_reports) and all(item["complete"] for item in fold_reports)
    hardware = describe()
    hardware["machine"] = platform.machine()
    hardware["system"] = platform.system()
    return {
        "measured": bool(mlx_executed and complete),
        "complete": complete,
        "mlx_executed": bool(mlx_executed),
        "untested_on_linux_vm": not mlx_executed,
        "tag": tag,
        "model_name": model_name,
        "fallback_from": fallback_from,
        "adapter_dir": adapter_dir,
        "use_adapter": True,
        "shots": 0,
        "folds": folds,
        "ranking": "mean_token_logprob",
        "secondary_ranking": "sum_token_logprob",
        "dtype": "4bit",
        "quantization": "mlx 4-bit base + LoRA (QLoRA as defined by mlx-lm)",
        "max_eval": 0,
        "seed": SEED,
        "n_scored": len(rows),
        "n_label_not_in_candidates": n_outside,
        "metrics": _metrics(rows, "pred", "top3") if rows else None,
        "metrics_sum_logprob": _metrics(rows, "pred_sum", "top3_sum") if rows else None,
        "paired_merchant_lookup": _lookup_metrics(rows) if rows else None,
        "per_fold": fold_reports,
        "format_check": {
            "n": 0,
            "exact_account": None,
            "in_candidate_list": None,
            "error": None,
            "rows": [],
            "note": "Free generation was not run. The headline is closed-set mean token log-probability.",
        },
        "seconds": round(seconds, 1),
        "seconds_per_example": round(seconds / len(rows), 3) if rows else None,
        "hardware": hardware,
        "cost_usd": None,
        "cost_note": "Not measured. This is a local Mac run; no invoice exists.",
        "protocol": {
            "features": ["merchant_raw", "department", "service_category", "amount"],
            "excluded": ["merchant_category"],
            "candidates": "accounts observed in training years of the fold, alphabetical",
            "split": "TrustGate rolling statement_year; only the folds listed above were scored",
            "train_rows": "every labelled line with statement_year < test year, including the valid.jsonl subset",
            "primary_model": PRIMARY_MODEL,
            "fallback_model": FALLBACK_MODEL,
        },
        "example_prompt": example_prompt,
    }


def run(cfg: dict) -> dict:
    import mlx.core as mx
    from mlx_lm import load

    df = load_lines()
    folds = [int(year) for year in cfg["folds"]]
    model_name = cfg["model_name"]
    adapter_root = Path(cfg["adapter_root"])
    tag = cfg["tag"]
    pred_dir = Path(cfg.get("prediction_dir", "results/predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pred_dir / f"{tag}.jsonl"
    meta_path = pred_dir / f"{tag}.meta.json"
    t0 = time.time()
    model = None
    tokenizer = None
    loaded_for = None
    fold_reports = []
    all_rows: list[dict] = []
    n_outside = 0
    example_prompt = ""
    for year, train, test in iter_folds(df):
        if year not in folds:
            continue
        accounts = candidate_accounts(train)
        adapter = adapter_root / f"fold_{year}"
        if not (adapter / "adapters.safetensors").exists():
            raise RuntimeError(f"missing adapter weights at {adapter / 'adapters.safetensors'}")
        planned = test.card_line_id.tolist()
        if jsonl_path.exists() and not meta_path.exists():
            raise RuntimeError(f"{jsonl_path} has no meta file; delete it to start over")
        if meta_path.exists() and jsonl_path.exists():
            meta = json.loads(meta_path.read_text())
            stored = meta.get("planned_ids_by_fold", {}).get(str(year))
            if stored is not None and stored != planned:
                raise RuntimeError(f"{meta_path} fold {year} does not match this data; delete the predictions")
        else:
            meta = {"planned_ids_by_fold": {}, "fold_account_lists": {}}
        meta.setdefault("planned_ids_by_fold", {})[str(year)] = planned
        meta.setdefault("fold_account_lists", {})[str(year)] = accounts
        meta["config_tag"] = tag
        meta_path.write_text(json.dumps(meta))
        if not example_prompt:
            example_prompt = user_message(test.sort_values("card_line_id").iloc[0], accounts)
        if loaded_for != str(adapter):
            del model, tokenizer
            model, tokenizer = load(model_name, adapter_path=str(adapter))
            loaded_for = str(adapter)
            if hasattr(mx, "clear_cache"):
                mx.clear_cache()
        done = {row["card_line_id"] for row in _read_jsonl(jsonl_path) if row.get("fold") == year}
        lookup = dict(zip(test.card_line_id, lookup_on_rows(train, test), strict=True))
        seen_keys = set(train.merchant_key)
        pending = test[~test.card_line_id.isin(done)]
        print(f"fold {year}: {len(pending)} to score, {len(done)} already saved, {len(accounts)} accounts", flush=True)
        with jsonl_path.open("a") as handle:
            for n, row in enumerate(pending.itertuples(index=False), start=1):
                user = user_message(row, accounts)
                sums, means = score_labels(model, tokenizer, user, accounts)
                ranked = rank_labels(accounts, means, sums)
                record = {
                    "card_line_id": row.card_line_id,
                    "fold": year,
                    "y": row.gl_account,
                    "seen_merchant": bool(row.merchant_key in seen_keys),
                    "label_in_candidates": bool(row.gl_account in accounts),
                    "pred": ranked["pred"],
                    "top3": ranked["top3"],
                    "pred_sum": ranked["pred_sum"],
                    "top3_sum": ranked["top3_sum"],
                    "lookup": lookup[row.card_line_id],
                }
                handle.write(json.dumps(record) + "\n")
                handle.flush()
                if n % 25 == 0:
                    print(f"  fold {year} {n}/{len(pending)} elapsed {time.time() - t0:.0f}s", flush=True)
        fold_rows = []
        seen_ids = set()
        for row in _read_jsonl(jsonl_path):
            if row.get("fold") == year and row["card_line_id"] not in seen_ids:
                seen_ids.add(row["card_line_id"])
                fold_rows.append(row)
        n_outside += sum(not row["label_in_candidates"] for row in fold_rows)
        fold_reports.append(
            {
                "year": year,
                "n_planned": len(planned),
                "n_scored": len(fold_rows),
                "complete": len(fold_rows) == len(planned),
                "n_train": int(len(train)),
                "n_candidates": len(accounts),
                "metrics": _metrics(fold_rows, "pred", "top3") if fold_rows else None,
                "paired_merchant_lookup": _lookup_metrics(fold_rows) if fold_rows else None,
            }
        )
        all_rows.extend(fold_rows)
    elapsed = time.time() - t0
    result = assemble_report(
        all_rows,
        fold_reports,
        tag=tag,
        model_name=model_name,
        adapter_dir=str(adapter_root),
        folds=folds,
        seconds=elapsed,
        n_outside=n_outside,
        example_prompt=example_prompt,
        fallback_from=cfg.get("fallback_from"),
        mlx_executed=True,
    )
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"wrote {out} n={result['n_scored']} complete={result['complete']} "
        f"top1={None if not result['metrics'] else round(result['metrics']['top1'], 4)}",
        flush=True,
    )
    return result
