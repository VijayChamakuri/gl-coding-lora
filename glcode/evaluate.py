"""Closed-set evaluation of a base model or a saved LoRA adapter.

Headline numbers use mean token log-probability over the training-fold
account list. Merchant lookup is fit on the full training fold and scored
on the same rows, so a subset still has a paired baseline. Free generation
on a handful of rows is reported separately and is not the headline.
"""

from __future__ import annotations

import json
import os
import time
from pathlib import Path

import numpy as np
import pandas as pd
import torch

from .baselines import lookup_on_rows
from .dataset import candidate_accounts, iter_folds, load_lines, select_exemplars, stratified_sample
from .hardware import describe
from .metrics import summarize
from .prompts import user_message
from .score import completion_logprobs, rank_labels

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def _load(model_name: str, adapter: Path | None):
    from transformers import AutoModelForCausalLM, AutoTokenizer

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=torch.float32)
    if adapter is not None:
        from peft import PeftModel

        model = PeftModel.from_pretrained(model, str(adapter))
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    model.eval()
    model.config.use_cache = True
    return tokenizer, model


def _dedupe(rows: list[dict]) -> list[dict]:
    by_id: dict[str, dict] = {}
    for row in rows:
        by_id[row["card_line_id"]] = row
    return list(by_id.values())


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


def _greedy(model, tokenizer, user: str, max_new: int = 16) -> str:
    from .chat import token_ids
    from .prompts import messages

    prompt = token_ids(tokenizer, messages(user), add_generation_prompt=True)
    device = next(model.parameters()).device
    ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model.generate(ids, max_new_tokens=max_new, do_sample=False, pad_token_id=tokenizer.pad_token_id)
    return tokenizer.decode(out[0, ids.shape[1] :], skip_special_tokens=True).strip()


def _metrics_from_rows(rows: list[dict], pred_key: str, top_key: str) -> dict:
    correct = np.array([row[pred_key] == row["y"] for row in rows], dtype=float)
    top3 = np.array([row["y"] in row[top_key] for row in rows], dtype=float)
    seen = np.array([row["seen_merchant"] for row in rows], dtype=bool)
    y = [row["y"] for row in rows]
    pred = [row[pred_key] for row in rows]
    return summarize(correct, top3, seen, y=y, pred=pred)


def run(cfg: dict) -> dict:
    if cfg.get("use_adapter") and int(cfg.get("shots") or 0) > 0:
        raise RuntimeError("the adapter was trained without exemplars; evaluate it with shots=0")
    df = load_lines()
    tag = cfg["tag"]
    pred_dir = Path(cfg.get("prediction_dir", "results/predictions"))
    pred_dir.mkdir(parents=True, exist_ok=True)
    jsonl_path = pred_dir / f"{tag}.jsonl"
    meta_path = pred_dir / f"{tag}.meta.json"
    torch.set_num_threads(os.cpu_count() or 1)
    t0 = time.time()
    tokenizer = None
    model = None
    fold_reports = []
    all_rows: list[dict] = []
    format_rows: list[dict] = []
    format_error = None
    n_outside = 0
    for year, train, test in iter_folds(df):
        if year not in set(int(y) for y in cfg["folds"]):
            continue
        if int(cfg.get("max_eval") or 0) > 0:
            test = stratified_sample(test, int(cfg["max_eval"]), seed=int(cfg["seed"]))
        else:
            rng = np.random.default_rng(int(cfg["seed"]))
            test = test.iloc[rng.permutation(len(test))].reset_index(drop=True)
        accounts = candidate_accounts(train)
        exemplars = select_exemplars(train, int(cfg.get("shots") or 0))
        overlap = set(exemplars.card_line_id) & set(test.card_line_id) if len(exemplars) else set()
        if overlap:
            raise RuntimeError(f"few-shot exemplars overlap the test fold: {sorted(overlap)[:3]}")
        adapter = None
        if cfg.get("use_adapter"):
            adapter = Path(cfg.get("adapter_dir", "results/adapters")) / f"fold_{year}"
            manifest_path = adapter / "train_manifest.json"
            if not manifest_path.exists():
                raise RuntimeError(f"missing adapter manifest at {manifest_path}")
            manifest = json.loads(manifest_path.read_text())
            if manifest.get("stopped_early"):
                raise RuntimeError(f"adapter for fold {year} was stopped early; refusing to score it as a full run")
            if manifest.get("candidate_accounts") != accounts:
                raise RuntimeError(f"adapter account list does not match the training fold for {year}")
        if model is None:
            tokenizer, model = _load(cfg["model_name"], adapter)
        elif adapter is not None:
            del model
            tokenizer, model = _load(cfg["model_name"], adapter)
        planned = test.card_line_id.tolist()
        if jsonl_path.exists() and not meta_path.exists():
            raise RuntimeError(f"{jsonl_path} has no meta file; delete it to start over")
        if meta_path.exists() and jsonl_path.exists():
            meta = json.loads(meta_path.read_text())
            stored_ids = meta.get("planned_ids_by_fold", {}).get(str(year))
            stored_accounts = meta.get("fold_account_lists", {}).get(str(year))
            if stored_ids is not None and stored_ids != planned:
                raise RuntimeError(f"{meta_path} fold {year} ids do not match this config; delete the predictions to start over")
            if stored_accounts is not None and stored_accounts != accounts:
                raise RuntimeError(f"{meta_path} fold {year} account list does not match; delete the predictions to start over")
        else:
            meta = {"planned_ids_by_fold": {}, "fold_account_lists": {}}
        meta.setdefault("planned_ids_by_fold", {})[str(year)] = planned
        meta.setdefault("fold_account_lists", {})[str(year)] = accounts
        meta["config_tag"] = tag
        meta_path.write_text(json.dumps(meta))
        done = {row["card_line_id"] for row in _read_jsonl(jsonl_path) if row.get("fold") == year}
        lookup = dict(zip(test.card_line_id, lookup_on_rows(train, test), strict=True))
        seen_keys = set(train.merchant_key)
        pending = test[~test.card_line_id.isin(done)]
        print(f"fold {year}: {len(pending)} to score, {len(done)} already saved, {len(accounts)} accounts", flush=True)
        with jsonl_path.open("a") as handle:
            for n, row in enumerate(pending.itertuples(index=False), start=1):
                user = user_message(row, accounts, exemplars if len(exemplars) else None)
                sums, means = completion_logprobs(model, tokenizer, user, accounts)
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
                    elapsed = time.time() - t0
                    print(f"  fold {year} {n}/{len(pending)} elapsed {elapsed:.0f}s", flush=True)
        fold_rows = _dedupe([row for row in _read_jsonl(jsonl_path) if row.get("fold") == year])
        if len(fold_rows) != len(planned):
            # A resumed file can hold this fold plus others; count this fold only.
            pass
        n_outside += sum(not row["label_in_candidates"] for row in fold_rows)
        fold_reports.append(
            {
                "year": year,
                "n_planned": len(planned),
                "n_scored": len(fold_rows),
                "complete": len(fold_rows) == len(planned),
                "n_train": int(len(train)),
                "n_candidates": len(accounts),
                "metrics": _metrics_from_rows(fold_rows, "pred", "top3") if fold_rows else None,
                "paired_merchant_lookup": _lookup_metrics(fold_rows) if fold_rows else None,
            }
        )
        all_rows.extend(fold_rows)
        if format_error is None and len(format_rows) < int(cfg.get("format_check") or 0):
            need = int(cfg["format_check"]) - len(format_rows)
            for row in test.itertuples(index=False):
                if need <= 0:
                    break
                user = user_message(row, accounts, exemplars if len(exemplars) else None)
                try:
                    text = _greedy(model, tokenizer, user)
                except Exception as exc:  # noqa: BLE001 - generation is auxiliary
                    format_error = f"{type(exc).__name__}: {exc}"
                    break
                format_rows.append(
                    {
                        "card_line_id": row.card_line_id,
                        "fold": year,
                        "y": row.gl_account,
                        "generation": text,
                        "equals_gold": text == row.gl_account,
                        "in_candidate_list": text in accounts,
                    }
                )
                need -= 1
    elapsed = time.time() - t0
    complete = all(item["complete"] for item in fold_reports) and bool(fold_reports)
    result = {
        "measured": True,
        "complete": complete,
        "tag": tag,
        "model_name": cfg["model_name"],
        "adapter_dir": cfg.get("adapter_dir") if cfg.get("use_adapter") else None,
        "use_adapter": bool(cfg.get("use_adapter")),
        "shots": int(cfg.get("shots") or 0),
        "folds": [int(y) for y in cfg["folds"]],
        "ranking": "mean_token_logprob",
        "secondary_ranking": "sum_token_logprob",
        "dtype": "float32",
        "max_eval": int(cfg.get("max_eval") or 0),
        "seed": int(cfg["seed"]),
        "n_scored": len(all_rows),
        "n_label_not_in_candidates": n_outside,
        "metrics": _metrics_from_rows(all_rows, "pred", "top3") if all_rows else None,
        "metrics_sum_logprob": _metrics_from_rows(all_rows, "pred_sum", "top3_sum") if all_rows else None,
        "paired_merchant_lookup": _lookup_metrics(all_rows) if all_rows else None,
        "per_fold": fold_reports,
        "format_check": {
            "n": len(format_rows),
            "exact_account": int(sum(row["equals_gold"] for row in format_rows)),
            "in_candidate_list": int(sum(row["in_candidate_list"] for row in format_rows)),
            "error": format_error,
            "rows": format_rows,
            "note": "Free generation is not the headline metric. Headline predictions are the closed-set ranking.",
        },
        "seconds": round(elapsed, 1),
        "seconds_per_example": round(elapsed / len(all_rows), 3) if all_rows else None,
        "hardware": describe(),
        "cost_usd": None,
        "cost_note": "Not measured. No GPU invoice was recorded for this run.",
        "protocol": {
            "features": ["merchant_raw", "department", "service_category", "amount"],
            "excluded": ["merchant_category"],
            "candidates": "accounts observed in training years of the fold, alphabetical",
            "split": "TrustGate rolling statement_year; only the folds listed above were scored",
        },
        "example_prompt": _example_prompt(df, cfg),
    }
    out = Path(cfg["output"])
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(result, indent=2))
    print(
        f"wrote {out} n={result['n_scored']} complete={complete} "
        f"top1={None if not result['metrics'] else round(result['metrics']['top1'], 4)} "
        f"in {elapsed:.1f}s",
        flush=True,
    )
    return result


def _lookup_metrics(rows: list[dict]) -> dict:
    correct = np.array([row["lookup"] == row["y"] for row in rows], dtype=float)
    seen = np.array([row["seen_merchant"] for row in rows], dtype=bool)
    # Lookup has no top-3 in the TrustGate definition. top3 mirrors top1 so the
    # shared summarizer still runs; callers must not report it as lookup top-3.
    summary = summarize(correct, correct, seen, y=[r["y"] for r in rows], pred=[r["lookup"] for r in rows])
    summary["top3"] = None
    summary["top3_ci"] = None
    summary["top3_seen_merchants"] = None
    summary["top3_new_merchants"] = None
    summary["top3_new_ci"] = None
    return summary


def _example_prompt(df, cfg) -> str:
    for year, train, test in iter_folds(df):
        if year == int(cfg["folds"][0]):
            accounts = candidate_accounts(train)
            exemplars = select_exemplars(train, int(cfg.get("shots") or 0))
            row = test.sort_values("card_line_id").iloc[0]
            return user_message(row, accounts, exemplars if len(exemplars) else None)
    return ""
