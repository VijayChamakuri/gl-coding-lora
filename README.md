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
Closed-set ranking on all 3,387 lines of 2018, from `results/eval_qwen05_zeroshot_2018.json`, `results/eval_qwen05_fewshot4_2018.json`, and `results/eval_qwen05_lora_2018.json`. Each file has `complete: true` and `measured: true`. The headline is mean token log-probability. Every 2018 label is inside the 19 accounts from 2017 (`n_label_not_in_candidates` = 0). New-merchant rates use the 824 merchants absent from 2017.

| Model | n | Top-1 | 95% CI | Top-3 | Top-3 95% CI | New merchants | New-merchant 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B zero-shot | 3,387 | 16.5% | 15.4–17.8% | 29.1% | 27.6–30.7% | 5.3% | 3.9–6.9% |
| Qwen2.5-0.5B 4-shot | 3,387 | 29.1% | 27.5–30.6% | 54.9% | 53.1–56.5% | 14.3% | 12.0–16.7% |
| Qwen2.5-0.5B LoRA, 1 epoch | 3,387 | 58.9% | 57.2–60.5% | 80.2% | 78.9–81.6% | 33.0% | 30.0–36.4% |
| Paired merchant lookup (same rows in all three JSON files) | 3,387 | 56.9% | 55.1–58.5% | — | — | 11.4% | 9.3–13.6% |

Full precision from the eval JSON, mean token log-probability:

| Model | Top-1 | Top-1 95% CI | Top-3 | Top-3 95% CI | New-merchant top-1 | New-merchant 95% CI |
|---|---:|---:|---:|---:|---:|---:|
| Zero-shot | 0.165338057277827 | 0.1535281960436965–0.17832890463537054 | 0.2911130794213168 | 0.2760555063478004–0.3070637732506643 | 0.05339805825242718 | 0.038834951456310676–0.0691747572815534 |
| 4-shot | 0.2911130794213168 | 0.27516976675524063–0.30617065249483316 | 0.5485680543253617 | 0.5311485090050192–0.5653971065839977 | 0.14320388349514562 | 0.12014563106796117–0.16747572815533981 |
| LoRA, 1 epoch | 0.5893120755831119 | 0.5718925302627694–0.6046648951874816 | 0.8021848243283142 | 0.7886034839090641–0.8160614112784175 | 0.3300970873786408 | 0.29975728155339804–0.3640776699029126 |
| Paired merchant lookup | 0.5686448184233835 | 0.551225273103041–0.5845881310894597 | — | — | 0.11407766990291263 | 0.09344660194174757–0.13592233009708737 |

The paired lookup is identical in the three eval files and equals the fold-2018 merchant lookup in `results/baselines.json` (top-1 0.5686448184233835, new merchants 0.11407766990291263, n=3387). Lookup has no top-3 in that definition, so those cells are empty. LoRA's mean-logprob top-1 (0.5893120755831119) is above that paired lookup. The two marginal 95% intervals overlap. The same LoRA top-1 is below the fold-2018 character TF-IDF logistic regression (0.6120460584588131). LoRA top-3 is 0.8021848243283142 against that TF-IDF top-3 of 0.8012990847357544. LoRA new-merchant top-1 is 0.3300970873786408 against TF-IDF 0.3058252427184466.

On the completed prediction files, mean-logprob ranking used 2 account names in zero-shot (Catering Supplies 1,755; Care Payments 1,632) and 8 names in 4-shot (Travelling Expenses 1,497; Furniture and Equipment 1,319; Catering Supplies 540; Expenses and Allowances 12; Office Expenses 10; Publicity and Promotion 4; Transport and Vehicles 3; Repairs and Maintenance 2). After LoRA it used 12 of the 19 names (Travelling Expenses 828; Furniture and Equipment 692; Catering Supplies 579; Expenses and Allowances 323; Care Payments 299; Publicity and Promotion 192; Fees and Charges 146; ICT 141; Legal and Professional Fees 65; Office Expenses 46; Staff Training 38; Library Equipment 38).

Secondary sum-logprob top-1, stored beside the mean and kept out of the headline: zero-shot 0.07646885149099499, 4-shot 0.2273398287570121, LoRA 0.5934455270150576. Sum-logprob and mean-logprob disagree on zero-shot (0.07646885149099499 against 0.165338057277827) and on 4-shot (0.2273398287570121 against 0.2911130794213168). On LoRA the two top-1 figures are close (0.5934455270150576 against 0.5893120755831119).

Format check is free generation on 20 shuffled rows. It is separate from the closed-set accuracy. Zero-shot: 0/20 exact account strings, 1/20 inside the account list. 4-shot: 0/20 exact, 0/20 inside the list. LoRA: 15/20 exact, 20/20 inside the list. `format_check.error` is null in all three files.

The adapter at `results/adapters/fold_2018` trained for 4060.1 seconds, 0.762 examples/s, mean example loss 0.275091, on 3,095 rows and 19 accounts, for 1 epoch (387 optimizer steps). Trainable parameters 8,798,208 of 502,830,976. Float32, all-linear LoRA, rank 16, alpha 32, dropout 0.05, seed 0, `stopped_early` false, backend `lora`, `qlora_executed` false. Hardware: Linux x86_64, 4 CPU cores, 15.63 GB RAM, no GPU, torch 2.15.0+cpu. `cost_usd` is null.

Scoring the same fold on that machine took 2020.1 s for zero-shot (0.596 s/example), 2814.7 s for 4-shot (0.831 s/example), and 2130.2 s for the adapter (0.629 s/example). Eval `cost_usd` is null.
<!-- /LM_TABLE -->

Fold 2018 paired baselines, same 3,387 lines (these match TrustGate's 2018 row exactly):

| Model | Top-1 | Top-3 | New merchants (n=824) |
|---|---:|---:|---:|
| Merchant lookup | 56.9% | — | 11.4% |
| Char TF-IDF + logistic regression | 61.2% | 80.1% | 30.6% |

## What was not measured

- QLoRA on CUDA. This machine has no CUDA device. The bitsandbytes path is implemented and used only when CUDA and bitsandbytes are both available.
- The MLX run below. No Mac accuracy, loss, or runtime from that path is in this repository.
- The five-fold rolling language-model run on the PyTorch path (11,756 lines, five adapters).
- `BAAI/bge-small-en-v1.5` and TrustGate's combined TF-IDF + embedding model. The 62.4% / 40.7% / 82.9% figures above are citations.
- Any model larger than Qwen2.5-0.5B on this machine. The MLX path is aimed at a 4-bit 1.5B and has not been executed.
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

### MLX on an M2 MacBook Air (8GB, fanless)

Not run. This development machine is Linux x86_64, and MLX does not install there. There is no Mac loss, accuracy, tokens/second, or wall time in this repository. Do not fill one in.

The path is QLoRA in the sense MLX-LM uses: a 4-bit `mlx-community` base plus a LoRA adapter (`mlx_lm.lora`). The training rows are the same TrustGate fold as the PyTorch run. Fold 2018 trains on all 3,095 lines from 2017 and scores all 3,387 lines from 2018. Ranking is mean token log-probability over the training-year account list, with the same top-1, top-3, new-merchant, and bootstrap fields as `results/eval_qwen05_lora_2018.json`. `valid.jsonl` is 128 of those training rows copied out so mlx-lm can print a validation loss. They are not removed from `train.jsonl`, and the test year is in neither file. Do not quote that validation loss as the result.

Checked here, on Linux: the JSONL export (3,095 train rows, no 2018 ids), the YAML (batch 1, 8 LoRA layers, rank 8, sequence length 256, gradient checkpointing, prompt masking), the out-of-memory detector, and the log-probability arithmetic. Not checked: loading MLX, the Metal training loop, adapter scoring, and the 1.5B-to-0.5B fallback actually catching an allocation failure.

Use Python 3.11 or 3.12. The system Python on macOS is often 3.9, which is too old. This command is the one to run on the Air:

```bash
python3.12 --version
PYTHON=python3.12 make mlx
```

`make mlx` creates `.venv-mlx`, installs `requirements-mlx.txt`, exports `data/mlx/fold_2018`, trains, and writes `results/eval_mlx_2018.json`. If the 1.5B model dies with an out-of-memory error, the same command retries `mlx-community/Qwen2.5-0.5B-Instruct-4bit` and records `fallback_from` in that JSON. All five test years:

```bash
PYTHON=python3.12 make mlx-rolling
```

That trains five adapters (3,095 + 6,482 + 9,864 + 12,007 + 13,917 steps at batch size 1) and writes `results/eval_mlx_2018_2019_2020_2021_2022.json`.

Defaults, in `configs/mlx_m2_8gb.yaml`: `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, batch size 1, gradient checkpointing, max sequence length 256, LoRA on the last 8 attention projections (`q`, `k`, `v`, `o`) at rank 8, AdamW, learning rate 1e-4, one pass over the training rows, seed 0. With the Qwen2.5 tokenizer on this machine, the longest training chat across all five folds is 199 tokens (186 tokens for 2017 alone), so 256 does not cut off the account name. The export script checks that again when `transformers` is installed and refuses to train if a chat would be truncated. `requirements-mlx.txt` pins `mlx-lm[train]==0.32.2`. If that pin is rejected, the script installs `mlx-lm[train]` unpinned and prints the version. Record that version next to the result. It was not installed here.

The Air is fanless. After a few minutes the chassis warms up and Metal slows down, then slows further if the machine is on battery or other apps are open. Plug it in, close browsers, and let the run finish; a partial `results/predictions/mlx_*.jsonl` resumes. No duration is stated here because none was measured. The result JSON's `seconds` field is the measured scoring time once the Mac run exists, and the training log under `results/mlx_fold_*.log` is the measured training time.

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
| `scripts/run_mlx.sh` | Mac-only MLX setup, QLoRA train, and the same scoring protocol |
| `make mlx` | Fold 2018 on an M2 Air. `make mlx-rolling` is all five years |
| `glcode/` | The library those scripts call |
| `results/baselines.json` | Recomputed classical metrics |

## License

Code: MIT. Data: Open Government Licence v3.0, with the attribution above.
