"""MLX-LM data export and the 8GB QLoRA config.

Training rows are the same TrustGate fold as the PyTorch run: every labelled
line with statement year before the test year. `valid.jsonl` is a 128-row
subset of that training file, because mlx-lm reports a validation loss only
when the file exists. Those rows are not held out of training, and the test
year is in neither file. Validation loss is not the result.

A 4-bit base plus LoRA is MLX-LM's QLoRA path (`mlx_lm.lora` on an
`mlx-community` 4-bit repo). This module does not import MLX.
"""

from __future__ import annotations

import json
from pathlib import Path

import numpy as np

from .dataset import TEST_YEARS, candidate_accounts, iter_folds, load_lines, stratified_sample
from .paths import ROOT
from .prompts import messages, user_message

PRIMARY_MODEL = "mlx-community/Qwen2.5-1.5B-Instruct-4bit"
FALLBACK_MODEL = "mlx-community/Qwen2.5-0.5B-Instruct-4bit"
MAX_SEQ_LENGTH = 256
BATCH_SIZE = 1
NUM_LAYERS = 8
LORA_RANK = 8
LORA_SCALE = 20.0
LORA_DROPOUT = 0.0
LEARNING_RATE = 1e-4
VALID_ROWS = 128
SEED = 0
LORA_KEYS = [
    "self_attn.q_proj",
    "self_attn.k_proj",
    "self_attn.v_proj",
    "self_attn.o_proj",
]
MIN_BYTES_FOR_PRIMARY = 7 * 1024**3

OOM_MARKERS = (
    "out of memory",
    "outofmemory",
    "insufficient memory",
    "kiogpucommandbuffercallbackerroroutofmemory",
    "failed to allocate",
    "cannot allocate memory",
    "mmap failed",
    "metal command buffer",
)


def models_for_memory(mem_bytes: int) -> list[str]:
    """8GB tries 1.5B then 0.5B. Under 7GB starts on 0.5B."""
    if mem_bytes and mem_bytes < MIN_BYTES_FOR_PRIMARY:
        return [FALLBACK_MODEL]
    return [PRIMARY_MODEL, FALLBACK_MODEL]


def model_slug(model_name: str) -> str:
    leaf = model_name.rstrip("/").split("/")[-1].lower()
    return leaf.replace(".", "")


def is_oom(exit_code: int, log_text: str) -> bool:
    if exit_code == 0:
        return False
    if exit_code == 137:
        return True
    lowered = log_text.lower()
    return any(marker in lowered for marker in OOM_MARKERS)


def _record(row, accounts: list[str], year: int, split: str) -> dict:
    user = user_message(row, accounts)
    return {
        "messages": messages(user, row.gl_account),
        "card_line_id": row.card_line_id,
        "fold": year,
        "split": split,
    }


def _write_jsonl(path: Path, records: list[dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w") as handle:
        for record in records:
            handle.write(json.dumps(record, ensure_ascii=False) + "\n")


def export_fold(df, year: int, dest: Path, valid_rows: int = VALID_ROWS, seed: int = SEED) -> dict:
    train = None
    for fold_year, fold_train, _test in iter_folds(df):
        if fold_year == year:
            train = fold_train
            break
    if train is None:
        raise ValueError(f"fold {year} is not in {TEST_YEARS}")
    accounts = candidate_accounts(train)
    valid_frame = stratified_sample(train, valid_rows, seed=seed)
    train_records = [_record(row, accounts, year, "train") for row in train.itertuples(index=False)]
    valid_ids = set(valid_frame.card_line_id)
    valid_records = [record for record in train_records if record["card_line_id"] in valid_ids]
    fold_dir = dest / f"fold_{year}"
    _write_jsonl(fold_dir / "train.jsonl", train_records)
    _write_jsonl(fold_dir / "valid.jsonl", valid_records)
    meta = {
        "fold": year,
        "train_rows": len(train_records),
        "valid_rows": len(valid_records),
        "valid_is_subset_of_train": True,
        "valid_note": (
            "valid.jsonl is also inside train.jsonl. It exists so mlx-lm can print a "
            "validation loss. The test year is not in either file. Do not quote val loss as the result."
        ),
        "n_candidate_accounts": len(accounts),
        "candidate_accounts": accounts,
        "iters": len(train_records),
        "max_seq_length": MAX_SEQ_LENGTH,
        "seed": seed,
    }
    (fold_dir / "meta.json").write_text(json.dumps(meta, indent=2))
    return meta


def export_folds(years: list[int], dest: Path | None = None) -> list[dict]:
    dest = dest or (ROOT / "data" / "mlx")
    df = load_lines()
    return [export_fold(df, year, dest) for year in years]


def render_lora_config(
    model: str,
    data_dir: Path,
    adapter_path: Path,
    iters: int,
) -> str:
    keys = ", ".join(json.dumps(key) for key in LORA_KEYS)
    return "\n".join(
        [
            f"model: {json.dumps(model)}",
            "train: true",
            "fine_tune_type: lora",
            "optimizer: adamw",
            f"data: {json.dumps(str(data_dir))}",
            f"adapter_path: {json.dumps(str(adapter_path))}",
            f"seed: {SEED}",
            f"num_layers: {NUM_LAYERS}",
            f"batch_size: {BATCH_SIZE}",
            f"iters: {iters}",
            "val_batches: 8",
            f"learning_rate: {LEARNING_RATE}",
            "steps_per_report: 10",
            "steps_per_eval: 500",
            "save_every: 1000",
            "mask_prompt: true",
            "grad_checkpoint: true",
            "grad_accumulation_steps: 1",
            f"max_seq_length: {MAX_SEQ_LENGTH}",
            "test: false",
            "lora_parameters:",
            f"  keys: [{keys}]",
            f"  rank: {LORA_RANK}",
            f"  scale: {LORA_SCALE}",
            f"  dropout: {LORA_DROPOUT}",
            "",
        ]
    )


def write_lora_config(path: Path, model: str, data_dir: Path, adapter_path: Path, iters: int) -> Path:
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(render_lora_config(model, data_dir, adapter_path, iters))
    return path


def completion_token_logprobs(logits: np.ndarray, token_ids: list[int], prompt_len: int) -> tuple[float, float]:
    """Mean and sum log-prob of the completion, from next-token logits.

    `logits[i]` predicts `token_ids[i + 1]`. The completion starts at
    `prompt_len`. This is the same ranking quantity as the PyTorch scorer.
    """
    completion = token_ids[prompt_len:]
    if not completion:
        raise ValueError("empty completion")
    if logits.shape[0] < prompt_len - 1 + len(completion):
        raise ValueError(
            f"logits length {logits.shape[0]} cannot score a completion of {len(completion)} "
            f"tokens starting at {prompt_len}"
        )
    total = 0.0
    for offset, token_id in enumerate(completion):
        row = logits[prompt_len - 1 + offset].astype(np.float64)
        row = row - row.max()
        log_z = np.log(np.exp(row).sum())
        total += float(row[token_id] - log_z)
    return total, total / len(completion)
