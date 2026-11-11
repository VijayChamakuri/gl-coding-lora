"""Chat-template token ids shared by training and evaluation."""

from __future__ import annotations

from .prompts import messages


def token_ids(tokenizer, convo: list[dict], add_generation_prompt: bool) -> list[int]:
    out = tokenizer.apply_chat_template(convo, add_generation_prompt=add_generation_prompt, tokenize=True)
    if hasattr(out, "input_ids"):
        out = out.input_ids
    if hasattr(out, "tolist"):
        out = out.tolist()
    if out and isinstance(out[0], list):
        if len(out) != 1:
            raise RuntimeError(f"expected one conversation, got batch {len(out)}")
        out = out[0]
    return [int(x) for x in out]


def split_prompt_completion(tokenizer, user: str, assistant: str) -> tuple[list[int], list[int]]:
    prompt = token_ids(tokenizer, messages(user), add_generation_prompt=True)
    full = token_ids(tokenizer, messages(user, assistant), add_generation_prompt=False)
    if full[: len(prompt)] != prompt:
        raise RuntimeError("chat template prompt is not a prefix of the full conversation; refusing to train or score")
    completion = full[len(prompt) :]
    if not completion:
        raise RuntimeError("assistant completion tokenized to nothing")
    return prompt, completion
