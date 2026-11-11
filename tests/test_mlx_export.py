import json
from pathlib import Path

import numpy as np
import pytest

from glcode.dataset import load_lines
from glcode.mlx_data import (
    FALLBACK_MODEL,
    MAX_SEQ_LENGTH,
    PRIMARY_MODEL,
    completion_token_logprobs,
    export_fold,
    is_oom,
    models_for_memory,
    render_lora_config,
)
from glcode.mlx_eval import assemble_report
from glcode.prompts import SYSTEM


def test_memory_policy_tries_1_5b_then_0_5b_on_8gb():
    eight_gb = 8 * 1024**3
    assert models_for_memory(eight_gb) == [PRIMARY_MODEL, FALLBACK_MODEL]
    assert models_for_memory(4 * 1024**3) == [FALLBACK_MODEL]
    assert models_for_memory(0) == [PRIMARY_MODEL, FALLBACK_MODEL]


def test_oom_detector_matches_metal_and_sigkill():
    assert is_oom(137, "")
    assert is_oom(1, "kIOGPUCommandBufferCallbackErrorOutOfMemory")
    assert is_oom(1, "RuntimeError: failed to allocate 9000000000 bytes")
    assert not is_oom(1, "KeyError: fold")
    assert not is_oom(0, "out of memory in a comment that is not the exit")


def test_fold_2018_export_matches_the_pytorch_train_set(tmp_path):
    df = load_lines()
    meta = export_fold(df, 2018, tmp_path)
    train_path = tmp_path / "fold_2018" / "train.jsonl"
    valid_path = tmp_path / "fold_2018" / "valid.jsonl"
    train = [json.loads(line) for line in train_path.read_text().splitlines()]
    valid = [json.loads(line) for line in valid_path.read_text().splitlines()]
    assert meta["train_rows"] == 3095
    assert len(train) == 3095
    assert meta["iters"] == 3095
    assert len(valid) == 128
    train_ids = [row["card_line_id"] for row in train]
    assert len(set(train_ids)) == 3095
    assert set(row["card_line_id"] for row in valid).issubset(train_ids)
    test_ids = set(df.loc[df.statement_year == 2018, "card_line_id"])
    assert test_ids.isdisjoint(train_ids)
    sample = train[0]
    assert sample["messages"][0] == {"role": "system", "content": SYSTEM}
    assert sample["messages"][2]["role"] == "assistant"
    assert sample["messages"][2]["content"] in meta["candidate_accounts"]
    user = sample["messages"][1]["content"]
    assert "Merchant:" in user and "Amount GBP:" in user
    assert "Procurement" not in user
    assert "SENTINEL_CATEGORY_XYZ" not in user
    assert meta["n_candidate_accounts"] == 19


def test_lora_yaml_is_the_8gb_recipe():
    text = render_lora_config(PRIMARY_MODEL, Path("data/mlx/fold_2018"), Path("results/mlx_adapters/fold_2018"), 3095)
    assert "mlx-community/Qwen2.5-1.5B-Instruct-4bit" in text
    assert "fine_tune_type: lora" in text
    assert "mask_prompt: true" in text
    assert "grad_checkpoint: true" in text
    assert "batch_size: 1" in text
    assert "max_seq_length: 256" in text
    assert "num_layers: 8" in text
    assert "iters: 3095" in text
    assert "self_attn.q_proj" in text and "self_attn.o_proj" in text
    assert MAX_SEQ_LENGTH == 256


def test_completion_logprob_matches_a_hand_computed_softmax():
    # vocab 3. Sequence [0, 1, 2], prompt length 1, so the completion is [1, 2].
    # logits[0] predicts token 1, logits[1] predicts token 2.
    logits = np.array(
        [
            [0.0, 10.0, 0.0],
            [0.0, 0.0, 10.0],
            [5.0, 0.0, 0.0],
        ],
        dtype=np.float64,
    )
    total, mean = completion_token_logprobs(logits, [0, 1, 2], prompt_len=1)

    def one(row, token):
        shifted = row - row.max()
        return float(shifted[token] - np.log(np.exp(shifted).sum()))

    expected = one(logits[0], 1) + one(logits[1], 2)
    assert total == pytest.approx(expected)
    assert mean == pytest.approx(expected / 2)
    assert total < 0


def test_report_marks_unrun_mlx_as_unmeasured():
    report = assemble_report(
        [],
        [],
        tag="mlx_2018",
        model_name=PRIMARY_MODEL,
        adapter_dir="results/mlx_adapters/active",
        folds=[2018],
        seconds=0.0,
        n_outside=0,
        example_prompt="",
        fallback_from=None,
        mlx_executed=False,
    )
    assert report["measured"] is False
    assert report["mlx_executed"] is False
    assert report["ranking"] == "mean_token_logprob"
    assert report["dtype"] == "4bit"
    assert report["metrics"] is None
    assert "top1" in json.dumps(report["format_check"]) or report["format_check"]["n"] == 0
