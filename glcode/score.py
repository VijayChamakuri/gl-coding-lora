"""Closed-set scoring by completion log-probability.

Headline rank is the mean token log-probability of the assistant completion
(the account name plus the chat template's end-of-turn tokens). The sum of
those log-probabilities is also returned. Mean-token ranking was chosen
before any language-model result was measured, because account names differ
in length ("ICT" versus "Legal and Professional Fees") and the sum favours
short strings.

Scores use one prompt forward plus a batched continuation through the KV
cache. In float32 that matches a full-sequence forward to about 1e-4. bf16
did not, so evaluation loads the model in float32.
"""

from __future__ import annotations

import torch

from .chat import split_prompt_completion


def completion_logprobs(model, tokenizer, user: str, labels: list[str]) -> tuple[list[float], list[float]]:
    """Return (sum_logprob, mean_logprob) aligned with `labels`."""
    if not labels:
        return [], []
    prompt, first_comp = split_prompt_completion(tokenizer, user, labels[0])
    comps = [first_comp]
    for label in labels[1:]:
        prompt_i, comp = split_prompt_completion(tokenizer, user, label)
        if prompt_i != prompt:
            raise RuntimeError("prompt tokens changed across labels; the account list must stay fixed")
        comps.append(comp)
    device = next(model.parameters()).device
    input_ids = torch.tensor([prompt], dtype=torch.long, device=device)
    with torch.inference_mode():
        out = model(input_ids=input_ids, use_cache=True)
    first = torch.log_softmax(out.logits[0, -1].float(), dim=-1)
    sums = torch.tensor([first[comp[0]].item() for comp in comps], dtype=torch.float64)
    counts = torch.ones(len(comps), dtype=torch.float64)
    feed_lens = [len(comp) - 1 for comp in comps]
    max_feed = max(feed_lens)
    if max_feed > 0:
        out.past_key_values.batch_repeat_interleave(len(comps))
        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = tokenizer.eos_token_id
        feed = torch.full((len(comps), max_feed), pad_id, dtype=torch.long, device=device)
        for i, comp in enumerate(comps):
            if len(comp) > 1:
                feed[i, : len(comp) - 1] = torch.tensor(comp[:-1], dtype=torch.long)
        with torch.inference_mode():
            cont = model(input_ids=feed, past_key_values=out.past_key_values, use_cache=False)
        logp = torch.log_softmax(cont.logits.float(), dim=-1)
        for i, comp in enumerate(comps):
            for t in range(1, len(comp)):
                sums[i] += logp[i, t - 1, comp[t]].item()
                counts[i] += 1
    means = sums / counts
    return sums.tolist(), means.tolist()


def rank_labels(labels: list[str], means: list[float], sums: list[float], k: int = 3) -> dict:
    """Stable ranking: equal scores keep the incoming label order."""
    order_mean = sorted(range(len(labels)), key=lambda i: (-means[i], i))
    order_sum = sorted(range(len(labels)), key=lambda i: (-sums[i], i))
    top = [labels[i] for i in order_mean[:k]]
    return {
        "pred": labels[order_mean[0]],
        "top3": top,
        "pred_sum": labels[order_sum[0]],
        "top3_sum": [labels[i] for i in order_sum[:k]],
        "mean_logprob": means[order_mean[0]],
        "sum_logprob": sums[order_sum[0]],
    }
