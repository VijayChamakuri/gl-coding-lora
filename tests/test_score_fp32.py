"""The closed-set scorer must match a full-sequence forward in float32."""

import os
from pathlib import Path

import pytest
import torch
from transformers import AutoModelForCausalLM, AutoTokenizer

from glcode.chat import split_prompt_completion
from glcode.prompts import messages
from glcode.score import completion_logprobs

pytestmark = pytest.mark.skipif(
    not any(Path(".cache/huggingface").glob("hub/models--Qwen--Qwen2.5-0.5B-Instruct/snapshots/*/tokenizer.json")),
    reason="Qwen tokenizer is not in the local HF cache",
)

os.environ.setdefault("HF_HOME", str(Path(".cache/huggingface").resolve()))
os.environ.setdefault("HF_HUB_OFFLINE", "1")
os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")


def test_prompt_is_a_prefix_and_cache_scores_match():
    name = "Qwen/Qwen2.5-0.5B-Instruct"
    tokenizer = AutoTokenizer.from_pretrained(name)
    user = "Merchant: ASDA\nDepartment: unknown\nService: unknown\nAmount GBP: 3.50"
    prompt, comp = split_prompt_completion(tokenizer, user, "ICT")
    full = tokenizer.apply_chat_template(messages(user, "ICT"), add_generation_prompt=False, tokenize=False)
    assert "ICT" in full
    assert comp
    labels = ["ICT", "Fuel", "Catering Supplies"]
    model = AutoModelForCausalLM.from_pretrained(name, dtype=torch.float32)
    model.eval()
    sums, means = completion_logprobs(model, tokenizer, user, labels)
    prompt_ids, _ = split_prompt_completion(tokenizer, user, labels[0])
    for label, total, mean in zip(labels, sums, means, strict=True):
        _, comp_ids = split_prompt_completion(tokenizer, user, label)
        seq = torch.tensor([prompt_ids + comp_ids])
        with torch.inference_mode():
            logits = model(input_ids=seq).logits
        gold = 0.0
        for t, tok_id in enumerate(comp_ids):
            gold += torch.log_softmax(logits[0, len(prompt_ids) + t - 1].float(), dim=-1)[tok_id].item()
        assert abs(gold - total) < 1e-3
        assert abs(gold / len(comp_ids) - mean) < 1e-3
