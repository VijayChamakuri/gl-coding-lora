"""LoRA / QLoRA fine-tune on one or more folds of the rolling split.

The measured CPU run uses float32 LoRA. QLoRA (4-bit NF4 via bitsandbytes)
is selected only when CUDA is available and bitsandbytes imports. That path
was not executed in this environment.
"""

from __future__ import annotations

import json
import math
import os
import random
import time
from contextlib import nullcontext
from pathlib import Path

import numpy as np
import torch

from .chat import split_prompt_completion
from .dataset import candidate_accounts, iter_folds, load_lines, stratified_sample
from .hardware import describe
from .paths import RESULTS
from .prompts import user_message

os.environ.setdefault("HF_HOME", str(Path(__file__).resolve().parents[1] / ".cache" / "huggingface"))
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")

ATTN = ["q_proj", "k_proj", "v_proj", "o_proj"]
ALL_LINEAR = ATTN + ["gate_proj", "up_proj", "down_proj"]


def resolve_targets(spec: str) -> list[str]:
    if spec == "attn":
        return list(ATTN)
    if spec == "all-linear":
        return list(ALL_LINEAR)
    raise ValueError(f"unknown target_modules {spec!r}")


def resolve_backend(qlora: str, cuda: bool) -> str:
    """Return 'qlora' or 'lora'. qlora='yes' fails closed when CUDA is absent."""
    if qlora not in ("auto", "yes", "no"):
        raise ValueError(qlora)
    if qlora == "no" or not cuda:
        if qlora == "yes":
            raise RuntimeError("QLoRA was requested but CUDA is not available")
        return "lora"
    try:
        import bitsandbytes  # noqa: F401
    except Exception as exc:
        if qlora == "yes":
            raise RuntimeError(f"QLoRA was requested but bitsandbytes did not import: {exc}") from exc
        return "lora"
    return "qlora"


def set_seed(seed: int) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def collate(pairs: list[tuple[list[int], list[int]]], pad_id: int) -> dict[str, torch.Tensor]:
    width = max(len(ids) for ids, _ in pairs)
    input_ids = torch.full((len(pairs), width), pad_id, dtype=torch.long)
    labels = torch.full((len(pairs), width), -100, dtype=torch.long)
    attention = torch.zeros((len(pairs), width), dtype=torch.long)
    for i, (ids, target) in enumerate(pairs):
        input_ids[i, : len(ids)] = torch.tensor(ids, dtype=torch.long)
        labels[i, : len(target)] = torch.tensor(target, dtype=torch.long)
        attention[i, : len(ids)] = 1
    return {"input_ids": input_ids, "labels": labels, "attention_mask": attention}


def _lr(step: int, total: int, base: float, warmup_ratio: float) -> float:
    warm = max(1, int(round(total * warmup_ratio)))
    if step <= warm:
        return base * step / warm
    if total <= warm:
        return base
    return base * max(0.0, 1 - (step - warm) / (total - warm))


def _load_model(model_name: str, backend: str, dtype_name: str):
    from transformers import AutoModelForCausalLM, AutoTokenizer, BitsAndBytesConfig

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    if backend == "qlora":
        compute = torch.bfloat16 if dtype_name in ("auto", "bfloat16") else torch.float16
        quant = BitsAndBytesConfig(
            load_in_4bit=True,
            bnb_4bit_quant_type="nf4",
            bnb_4bit_use_double_quant=True,
            bnb_4bit_compute_dtype=compute,
        )
        model = AutoModelForCausalLM.from_pretrained(model_name, quantization_config=quant, device_map="auto")
        from peft import prepare_model_for_kbit_training

        model = prepare_model_for_kbit_training(model, use_gradient_checkpointing=False)
        return tokenizer, model
    dtype = {"float32": torch.float32, "bfloat16": torch.bfloat16, "float16": torch.float16}[dtype_name]
    model = AutoModelForCausalLM.from_pretrained(model_name, dtype=dtype)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    model.to(device)
    return tokenizer, model


def _examples(train, tokenizer, accounts: list[str], max_length: int):
    built = []
    longest = 0
    for row in train.itertuples(index=False):
        user = user_message(row, accounts)
        prompt, completion = split_prompt_completion(tokenizer, user, row.gl_account)
        ids = prompt + completion
        longest = max(longest, len(ids))
        if len(ids) > max_length:
            raise RuntimeError(
                f"example {row.card_line_id} is {len(ids)} tokens, above max_length {max_length}; refusing to truncate"
            )
        built.append((ids, [-100] * len(prompt) + completion, row.card_line_id))
    return built, longest


def train_fold(cfg: dict, year: int, train_df, out_dir: Path) -> dict:
    from peft import LoraConfig, TaskType, get_peft_model

    set_seed(int(cfg["seed"]))
    torch.set_num_threads(os.cpu_count() or 1)
    cuda = torch.cuda.is_available()
    dtype_name = cfg.get("dtype", "float32")
    if dtype_name == "auto":
        dtype_name = "bfloat16" if cuda else "float32"
    backend = resolve_backend(cfg.get("qlora", "auto"), cuda)
    tokenizer, model = _load_model(cfg["model_name"], backend, dtype_name)
    accounts = candidate_accounts(train_df)
    print(f"fold {year}: tokenizing {len(train_df)} training lines, {len(accounts)} accounts", flush=True)
    if int(cfg.get("max_train") or 0) > 0:
        train_df = stratified_sample(train_df, int(cfg["max_train"]), seed=int(cfg["seed"]))
    examples, longest = _examples(train_df, tokenizer, accounts, int(cfg["max_length"]))
    lora = LoraConfig(
        r=int(cfg["lora_r"]),
        lora_alpha=int(cfg["lora_alpha"]),
        lora_dropout=float(cfg["lora_dropout"]),
        bias="none",
        task_type=TaskType.CAUSAL_LM,
        target_modules=resolve_targets(cfg.get("target_modules", "all-linear")),
    )
    model = get_peft_model(model, lora)
    if cfg.get("gradient_checkpointing"):
        model.gradient_checkpointing_enable()
    model.config.use_cache = False
    device = next(model.parameters()).device
    trainable = [p for p in model.parameters() if p.requires_grad]
    opt = torch.optim.AdamW(trainable, lr=float(cfg["learning_rate"]), weight_decay=float(cfg.get("weight_decay", 0.0)))
    batch_size = int(cfg["batch_size"])
    accum = int(cfg["grad_accum"])
    epochs = int(cfg["epochs"])
    micro_per_epoch = math.ceil(len(examples) / batch_size)
    total_steps = math.ceil(micro_per_epoch / accum) * epochs
    max_steps = int(cfg["max_steps"]) if cfg.get("max_steps") else None
    pad_id = tokenizer.pad_token_id
    rng = np.random.default_rng(int(cfg["seed"]))
    use_autocast = backend == "qlora" or (device.type == "cuda" and dtype_name in ("bfloat16", "float16"))
    autocast_dtype = torch.bfloat16 if dtype_name != "float16" else torch.float16
    step = 0
    micro = 0
    seen = 0
    running = 0.0
    t0 = time.time()
    stopped_early = False
    opt.zero_grad(set_to_none=True)
    model.train()
    for epoch in range(epochs):
        order = rng.permutation(len(examples))
        for start in range(0, len(order), batch_size):
            idx = order[start : start + batch_size]
            pairs = [(examples[i][0], examples[i][1]) for i in idx]
            batch = collate(pairs, pad_id)
            batch = {k: v.to(device) for k, v in batch.items()}
            ctx = torch.autocast(device_type=device.type, dtype=autocast_dtype) if use_autocast else nullcontext()
            with ctx:
                loss = model(**batch, use_cache=False).loss
            if not torch.isfinite(loss):
                raise RuntimeError(f"non-finite loss at epoch {epoch} micro-batch {start}")
            (loss / accum).backward()
            running += float(loss.detach().float()) * len(idx)
            seen += len(idx)
            micro += 1
            if micro == accum:
                step += 1
                torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("max_grad_norm", 1.0)))
                for group in opt.param_groups:
                    group["lr"] = _lr(step, total_steps, float(cfg["learning_rate"]), float(cfg.get("warmup_ratio", 0.03)))
                opt.step()
                opt.zero_grad(set_to_none=True)
                micro = 0
                if step % 10 == 0 or step == 1:
                    elapsed = time.time() - t0
                    print(
                        f"fold {year} step {step}/{total_steps} loss {running / seen:.4f} "
                        f"ex/s {seen / elapsed:.2f} lr {opt.param_groups[0]['lr']:.2e}",
                        flush=True,
                    )
                if max_steps is not None and step >= max_steps:
                    stopped_early = True
                    break
        if stopped_early:
            break
        if micro:
            step += 1
            torch.nn.utils.clip_grad_norm_(trainable, float(cfg.get("max_grad_norm", 1.0)))
            for group in opt.param_groups:
                group["lr"] = _lr(step, total_steps, float(cfg["learning_rate"]), float(cfg.get("warmup_ratio", 0.03)))
            opt.step()
            opt.zero_grad(set_to_none=True)
            micro = 0
    elapsed = time.time() - t0
    out_dir.mkdir(parents=True, exist_ok=True)
    model.save_pretrained(out_dir)
    n_trainable = sum(p.numel() for p in trainable)
    n_total = sum(p.numel() for p in model.parameters())
    manifest = {
        "measured": True,
        "fold": year,
        "backend": backend,
        "qlora_executed": backend == "qlora",
        "dtype": dtype_name,
        "model_name": cfg["model_name"],
        "epochs_requested": epochs,
        "stopped_early": stopped_early,
        "max_steps": max_steps,
        "optimizer_steps": step,
        "train_rows": len(examples),
        "train_card_line_ids": [examples[i][2] for i in range(len(examples))],
        "n_candidate_accounts": len(accounts),
        "candidate_accounts": accounts,
        "longest_tokens": longest,
        "lora_r": int(cfg["lora_r"]),
        "lora_alpha": int(cfg["lora_alpha"]),
        "lora_dropout": float(cfg["lora_dropout"]),
        "target_modules": resolve_targets(cfg.get("target_modules", "all-linear")),
        "learning_rate": float(cfg["learning_rate"]),
        "batch_size": batch_size,
        "grad_accum": accum,
        "trainable_params": n_trainable,
        "total_params": n_total,
        "seconds": round(elapsed, 1),
        "examples_per_second": round(seen / elapsed, 3) if elapsed else None,
        "mean_token_loss": round(running / seen, 6) if seen else None,
        "hardware": describe(),
        "seed": int(cfg["seed"]),
        "cost_usd": None,
        "cost_note": "Not measured. No GPU invoice was recorded for this run.",
    }
    (out_dir / "train_manifest.json").write_text(json.dumps(manifest, indent=2))
    print(f"fold {year} saved {out_dir} in {elapsed:.1f}s backend={backend} loss={manifest['mean_token_loss']}", flush=True)
    del model, opt
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
    return manifest


def run(cfg: dict) -> list[dict]:
    df = load_lines()
    out_root = Path(cfg.get("output_dir", RESULTS / "adapters"))
    manifests = []
    wanted = set(int(y) for y in cfg["folds"])
    for year, train, _test in iter_folds(df):
        if year not in wanted:
            continue
        manifests.append(train_fold(cfg, year, train, out_root / f"fold_{year}"))
    (out_root / "train_index.json").write_text(json.dumps(manifests, indent=2))
    return manifests
