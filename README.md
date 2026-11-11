# GL-LoRA

Fine-tune a small open model with LoRA to predict the general-ledger account on real UK council purchase-card lines, and compare it with the baselines from [TrustGate](https://github.com/VijayChamakuri/trustgate).

The label is the account finance staff booked. The inputs are what a card feed shows: merchant string, department, service, and amount. The council's procurement category is not used. Staff assigned that field with the label, so it leaks the answer (TrustGate's upstream setting, where a category lookup alone is about 90.8%).

## Measured results

Classical baselines were recomputed on the full rolling split (11,756 test lines, statement years 2018–2022). Majority class and merchant lookup match the published TrustGate figures exactly, including the bootstrap intervals. The label ceiling matches exactly. Character TF-IDF logistic regression is one labelled line off the published accuracy (0.609306 here, 0.609221 published).

Language-model numbers below are only for **fold 2018** (train on 2017, score 2018). They are not the pooled 11,756-line number. QLoRA was not run. Dollar cost was not measured.

### Card-feed rolling split, pooled 2018–2022

| Model | n | Top-1 | 95% CI | Top-3 | New merchants | New-merchant 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Majority class | 11,756 | 17.0% | 16.3–17.6% | 52.5% micro | 11.4% | 10.3–12.4% |
| Merchant → account lookup | 11,756 | 54.5% | 53.6–55.4% | — | 15.8% | 14.5–17.0% |
| Merchant top-3 (defined below) | 11,756 | 54.5% | 53.6–55.4% | 75.2% micro | 15.8% | 14.5–17.0% |
| Char TF-IDF + logistic regression | 11,756 | 60.9% | 60.0–61.8% | 80.1% micro / 80.7% year-macro | 34.6% | 33.0–36.3% |

Label ceiling on identical card-feed inputs, recomputed: **86.1%**. Same inputs were not always booked to the same account, so no deterministic model can exceed that on this population.

Merchant top-3 is not a TrustGate metric. For a seen merchant it is the merchant's most frequent accounts, padded to three with the global ranking; unseen merchants use department + service, then the global ranking. Majority top-3 means the booked account is one of the three most frequent accounts in the training years. TrustGate's published top-3 is the unweighted mean of the five yearly rates (`top3_year_macro`). Line-weighted top-3 is `top3_micro`.

### Published by TrustGate, not re-run here

Source: `results/real/gl_benchmark.json` at commit `66cf7ab2fb866140a75430415ac94dfa1f1907cb`. Same card-feed rolling split.

| Model | Top-1 | 95% CI | Top-3 (year-macro) | New merchants |
|---|---:|---:|---:|---:|
| TF-IDF + `BAAI/bge-small-en-v1.5` + logistic regression | 62.4% | 61.5–63.3% | 82.9% | 40.7% |

TrustGate also reports that, on a **different** window (live months of the autonomy rollout, not the 11,756-line pool), the booked account is in the top 3 **86.4%** of the time. That figure is not comparable to the 82.9% rolling-benchmark top-3.

### Language model, fold 2018 only

`Qwen/Qwen2.5-0.5B-Instruct`, float32. Ranked by mean token log-probability of the assistant completion over the 19 accounts observed in 2017. Closed-set ranking is the headline. Free generation on 20 shuffled rows is reported beside it and is not the accuracy.

<!-- LM_TABLE -->
Language-model rows are written here after `results/eval_qwen05_zeroshot_2018.json`, `results/eval_qwen05_fewshot4_2018.json`, and `results/eval_qwen05_lora_2018.json` exist. Until then they are unmeasured.
<!-- /LM_TABLE -->

Fold 2018 paired baselines, same 3,387 lines (these match TrustGate's 2018 row exactly):

| Model | Top-1 | Top-3 | New merchants (n=824) |
|---|---:|---:|---:|
| Merchant lookup | 56.9% | — | 11.4% |
| Char TF-IDF + logistic regression | 61.2% | 80.1% | 30.6% |

## What was not measured

- QLoRA. This machine has no CUDA device. The QLoRA branch is implemented and used only when CUDA and bitsandbytes are both available.
- The five-fold rolling language-model run (11,756 lines, five adapters).
- `BAAI/bge-small-en-v1.5` and TrustGate's combined TF-IDF + embedding model. The 62.4% / 40.7% / 82.9% figures above are citations.
- Any model larger than Qwen2.5-0.5B.
- Dollar cost. No invoice was recorded. Cost = measured GPU hours × the provider's price at the time you rent it. Do not quote a dollar figure from this repository.
- A second training epoch, a different rank, or early stopping on the test fold. The run is the config in `configs/cpu_fold2018.json`, once.

## Protocol

The labelled file is the JudgeGate export of TrustGate's dbt model `int_stockport__labelled_lines`: 14,851 lines, 21 accounts, January 2017–June 2022. `scripts/prepare_data.py --download` fetches the 111 monthly files from data.gov.uk and rebuilds that table. The rebuild matches the export on content and on `card_line_id` (see `results/rebuild_report.json`). One source date is `28.02.20`; DuckDB's `%Y` parsed it as year 0020, and the parser does the same.

For each test year *t* in 2018–2022 the classical models train on statement years before *t* and score year *t*. The year is the file's statement month, not the purchase date. Bootstrap intervals are 2,000 line resamples, seed 0, percentiles 2.5 and 97.5, the same procedure as TrustGate.

The language model uses one fold of that split: train on all 3,095 labelled lines from 2017, score all 3,387 lines from 2018. The prompt lists the 19 accounts that occur in 2017 (`COVID 19` and `Activities and Events` do not). No 2018 label falls outside that list. Department or service, when missing, is the word "unknown". Merchant category is not in the prompt.

Few-shot examples are four training-year lines: the earliest line of each of the four most common training accounts. They are not used when training or when scoring the adapter. The adapter is trained on the account name as the assistant message, loss only on that completion.

Ranking uses the mean log-probability of the completion tokens (account name plus the chat template's end-of-turn tokens). The sum of those log-probabilities is stored as a secondary metric because it favours short names such as `ICT`. The choice of the mean was made before any language-model accuracy was computed.

## Train and evaluate

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --check
python scripts/run_baselines.py
pytest
```

The CPU run that produced the fold-2018 adapter, on a 4-core machine with no GPU:

```bash
python scripts/train_lora.py --config configs/cpu_fold2018.json
python scripts/evaluate.py --config configs/eval_zeroshot_2018.json
python scripts/evaluate.py --config configs/eval_fewshot_2018.json
python scripts/evaluate.py --config configs/eval_lora_2018.json
```

LoRA settings in that config: rank 16, alpha 32, dropout 0.05, all Qwen linear modules, AdamW, learning rate 2e-4, one epoch, batch size 1, gradient accumulation 8, seed 0, float32. Evaluation appends to `results/predictions/<tag>.jsonl` and can be resumed with the same config. A partial file is `complete: false` and is not a full-fold result.

### Full rolling run on a rented GPU

Not run here. Install a CUDA build of PyTorch, then `pip install -r requirements-gpu.txt` and `pip install bitsandbytes`. QLoRA (NF4, double quant) is selected automatically when both CUDA and bitsandbytes import.

```bash
python scripts/train_lora.py --config configs/gpu_rolling.json
python scripts/evaluate.py --config configs/gpu_eval_rolling.json
```

That trains two epochs on each of the five folds, into `results/adapters_rolling/`. Record `seconds` from each `train_manifest.json` and multiply by the host's hourly price. This repository does not estimate that product.

## Data

Stockport Metropolitan Borough Council purchase-card transactions, data.gov.uk dataset `purchase-card-transactions`, Open Government Licence v3.0. Contains public sector information licensed under the Open Government Licence v3.0.

The purpose map in `data/stockport_purpose_map.csv` is the reviewed seed from TrustGate (MIT). It collapses 61 raw purpose strings to 21 accounts. The reference export is from JudgeGate commit `ff0b8f6b8d3ec119c3f3d2131dc1e8a8b190bb4e`, file `data/stockport_labelled_lines.csv.gz`, SHA-256 `d202870296c57f95014b9f6ad09bc5bc2b3ca68180eaafb37aad3bbc51d3e37b`.

## Layout

| Path | Role |
|---|---|
| `scripts/prepare_data.py` | Check the export, or download and rebuild it |
| `scripts/run_baselines.py` | Majority, merchant lookup, merchant top-3, TF-IDF logistic regression |
| `scripts/train_lora.py` | LoRA, or QLoRA when CUDA and bitsandbytes are available |
| `scripts/evaluate.py` | Closed-set log-probability ranking, with a paired merchant lookup |
| `glcode/` | The library those scripts call |
| `results/baselines.json` | Recomputed classical metrics |

## License

Code: MIT. Data: Open Government Licence v3.0, with the attribution above.
