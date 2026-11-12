# GL-LoRA

A small Qwen model, trained with LoRA, predicts the general-ledger account on real Stockport council purchase-card lines. The comparison is merchant lookup and character TF-IDF on the same 2018 fold.

The label is the account finance staff booked. The inputs are the merchant string, department, service, and amount. The council's procurement category is not an input: staff assigned it with the label, and a category lookup alone is about 90.8% in TrustGate's upstream setting.

On all 3,387 lines of 2018, float32 Qwen2.5-0.5B LoRA is the strongest language-model top-1 in this repo. The 4-bit 1.5B QLoRA run on an 8GB M2 Air did not beat it, and it did not beat character TF-IDF.

## TL;DR

Fold 2018 only: train on the 3,095 labelled lines from 2017, score the 3,387 lines from 2018. Language-model numbers are closed-set mean token log-probability. Cells are the stored floats. A dash means that file did not store the interval.

| Model | n | Top-1 | Top-1 95% CI | Top-3 | Top-3 95% CI | New-merchant top-1 | New-merchant 95% CI |
|---|---:|---:|---:|---:|---:|---:|---:|
| Qwen2.5-0.5B zero-shot | 3,387 | 0.165338057277827 | 0.1535281960436965–0.17832890463537054 | 0.2911130794213168 | 0.2760555063478004–0.3070637732506643 | 0.05339805825242718 | 0.038834951456310676–0.0691747572815534 |
| Qwen2.5-0.5B 4-shot | 3,387 | 0.2911130794213168 | 0.27516976675524063–0.30617065249483316 | 0.5485680543253617 | 0.5311485090050192–0.5653971065839977 | 0.14320388349514562 | 0.12014563106796117–0.16747572815533981 |
| Qwen2.5-0.5B LoRA, CPU | 3,387 | 0.5893120755831119 | 0.5718925302627694–0.6046648951874816 | 0.8021848243283142 | 0.7886034839090641–0.8160614112784175 | 0.3300970873786408 | 0.29975728155339804–0.3640776699029126 |
| Qwen2.5-1.5B-Instruct-4bit MLX QLoRA | 3,387 | 0.5674638322999704 | 0.5497416592855033–0.5851786241511662 | 0.7753173900206672 | 0.7608503100088574–0.7891939769707705 | 0.28398058252427183 | 0.25485436893203883–0.3143203883495146 |
| Merchant lookup | 3,387 | 0.5686448184233835 | 0.5515205196338943–0.584883377620313 | — | — | 0.11407766990291263 | 0.09223300970873786–0.13592233009708737 |
| Char TF-IDF + logistic regression | 3,387 | 0.6120460584588131 | — | 0.8012990847357544 | — | 0.3058252427184466 | — |

Sources: `results/eval_qwen05_zeroshot_2018.json`, `results/eval_qwen05_fewshot4_2018.json`, `results/eval_qwen05_lora_2018.json`, `results/eval_mlx_2018.json`, and the 2018 `tfidf_logreg` row of `results/baselines.json`. Lookup in the table is the paired lookup stored in the MLX eval file. It matches the 2018 `merchant_lookup` row in `results/baselines.json` (top-1 0.5686448184233835, new merchants 0.11407766990291263). That baseline file's per-year TF-IDF row has no bootstrap interval, so those cells are empty. Lookup has no top-3 in this definition.

The 1.5B top-1 (0.5674638322999704) is below the 0.5B LoRA top-1 (0.5893120755831119), below fold-2018 TF-IDF (0.6120460584588131), and just below merchant lookup (0.5686448184233835). The 1.5B and lookup top-1 intervals overlap. New-merchant top-1 for the 1.5B model is 0.28398058252427183, against 0.3300970873786408 for 0.5B LoRA and 0.11407766990291263 for lookup. `fallback_from` in the MLX file is empty: the 0.5B fallback did not train.

## One transaction

`2018-01_f41af6e3.csv:10`, the first correct MLX row in `card_line_id` order, and the same merchant string as the example prompt stored in `results/eval_mlx_2018.json`.

| Field | Value |
|---|---|
| Merchant | CARREFOUR MARKET ST H |
| Department | Services to People |
| Service | Adult Social Care |
| Amount | 14.49 GBP |
| Booked account | Expenses and Allowances |
| MLX top-1 | Expenses and Allowances |
| MLX top-3 | Expenses and Allowances, Travelling Expenses, Publicity and Promotion |
| Seen in 2017 | no |
| Merchant lookup | Travelling Expenses |

The model ranked the booked account first. Lookup, which has no memory of this merchant, assigned Travelling Expenses.

## Pipeline

```mermaid
flowchart LR
  lines[Stockport labelled lines]
  split[2017 train, 2018 score]
  prompt[Chat prompt listing 19 accounts]
  adapt[LoRA or 4-bit QLoRA]
  rank[Mean token log-probability]
  metrics[Top-1, top-3, new merchants]
  lines --> split --> prompt --> adapt --> rank --> metrics
```

`valid.jsonl` is 128 training rows copied out so mlx-lm can print a validation loss. Those rows stay in `train.jsonl`. The 2018 lines are in neither file. Do not quote that validation loss as the test result.

## Charts

`make figures` reads the eval JSON files, the prediction JSONL, `results/baselines.json`, and `results/mlx_train_curve_2018.json`.

![Fold 2018 top-1](docs/img/top1-ci.png)

Caps are the stored 95% intervals. The TF-IDF bar has no cap because `results/baselines.json` does not store one for the 2018 year row.

![Seen and new merchants](docs/img/new-vs-seen.png)

TF-IDF is absent here. Its 2018 year row stores `accuracy_new_merchants` (0.3058252427184466) and does not store a seen-merchant top-1.

![Top-k from stored lists](docs/img/topk.png)

k = 2 is computed from the stored top-3 list. Ranks past 3 were not saved, so the curve stops at 3.

| Model | k = 1 | k = 2 | k = 3 | n |
|---|---:|---:|---:|---:|
| Zero-shot 0.5B | 0.165338057277827 | 0.26956008266902864 | 0.2911130794213168 | 3387 |
| 4-shot 0.5B | 0.2911130794213168 | 0.42781222320637735 | 0.5485680543253617 | 3387 |
| LoRA 0.5B CPU | 0.5893120755831119 | 0.7260112193681725 | 0.8021848243283142 | 3387 |
| LoRA 1.5B MLX | 0.5674638322999704 | 0.70859167404783 | 0.7753173900206672 | 3387 |

k = 1 and k = 3 match the top-1 and top-3 fields in the eval JSON.

![Printed training loss](docs/img/train-loss.png)

mlx-lm printed 310 training losses and 8 validation losses. The first printed training loss is 1.596 at iteration 10. The last is 0.192 at iteration 3095. Printed validation loss starts at 5.077 on iteration 1 and ends at 0.412 on iteration 3095. Those strings are in `results/mlx_train_curve_2018.json`. They are the logger's rounded text, not a higher-precision loss.

![Confusion among the six largest 2018 accounts](docs/img/confusion.png)

The six accounts with the most 2018 labels, and how often the 1.5B model predicted one of those six:

| Booked \ predicted | Travelling Expenses | Catering Supplies | Furniture and Equipment | Publicity and Promotion | Care Payments | Expenses and Allowances | n |
|---|---:|---:|---:|---:|---:|---:|---:|
| Travelling Expenses | 667 | 0 | 3 | 5 | 46 | 5 | 727 |
| Catering Supplies | 33 | 524 | 21 | 11 | 10 | 70 | 676 |
| Furniture and Equipment | 9 | 51 | 246 | 25 | 42 | 18 | 479 |
| Publicity and Promotion | 4 | 0 | 95 | 105 | 16 | 4 | 251 |
| Care Payments | 67 | 0 | 39 | 10 | 133 | 0 | 249 |
| Expenses and Allowances | 75 | 0 | 16 | 16 | 18 | 30 | 176 |

Row totals in the matrix are short of `n` when the prediction was some other account. That happened 144 times. Catering Supplies is mostly kept (524 of 676). Expenses and Allowances is mostly sent to Travelling Expenses (75 of 176). Publicity and Promotion is split with Furniture and Equipment (95 predicted as furniture, 105 kept).

## Where it goes wrong

These three rows are the first match in `card_line_id` order, not a hand-picked sample.

**Correct, new merchant.** `2018-01_f41af6e3.csv:10`. CARREFOUR MARKET ST H, Services to People, Adult Social Care, 14.49 GBP. Booked Expenses and Allowances. MLX predicted Expenses and Allowances. Lookup predicted Travelling Expenses.

**Seen merchant, wrong.** `2018-01_f41af6e3.csv:106`. W M MORRISON PLC, Services to People, Childrens & Educational Services, 10.0 GBP. Booked Furniture and Equipment. MLX predicted Catering Supplies. The booked account was still inside the top 3: Catering Supplies, Office Expenses, Furniture and Equipment. Lookup also predicted Catering Supplies.

**New merchant, wrong.** `2018-01_f41af6e3.csv:109`. PAYPAL  VANESSATREA, Corporate and Support Services, Corporate & Democratic Core, 189.59 GBP. Booked Furniture and Equipment. MLX predicted Publicity and Promotion. Top 3 was Publicity and Promotion, Expenses and Allowances, Staff Training. The booked account was not in that list. Lookup predicted Expenses and Allowances.

## Reproduce

CPU fold, measured on Linux-6.12.113+-x86_64, 4 cores, 15.63 GB RAM, no GPU, torch 2.15.0+cpu. Training is `results/adapters/fold_2018/train_manifest.json`: 4060.1 seconds, 0.762 examples/s, mean token loss 0.275091, 8,798,208 trainable parameters of 502,830,976, rank 16, alpha 32, dropout 0.05, one epoch, 387 optimizer steps. Scoring from the eval JSON `seconds` fields: zero-shot 2020.1 s (0.596 s/example), 4-shot 2814.7 s (0.831 s/example), LoRA 2130.2 s (0.629 s/example).

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt
python scripts/prepare_data.py --check
python scripts/run_baselines.py
python scripts/train_lora.py --config configs/cpu_fold2018.json
python scripts/evaluate.py --config configs/eval_zeroshot_2018.json
python scripts/evaluate.py --config configs/eval_fewshot_2018.json
python scripts/evaluate.py --config configs/eval_lora_2018.json
```

M2 MacBook Air, 8GB unified memory, fanless, macOS-27.2-arm64, 8 CPU cores. The eval JSON records `ram_gb_memtotal` as null and `gpu` as null because the hardware helper reads `/proc/meminfo` and torch. This run was on AC power. `mlx-lm` Version: 0.32.2. Model `mlx-community/Qwen2.5-1.5B-Instruct-4bit`, batch size 1, rank 8, last 8 attention projections, max sequence length 256, 3,095 steps.

Train wall clock is the training log's lifetime, 2026-11-11T19:00:36-0500 through 2026-11-11T19:36:29-0500, which is 2153 seconds. Score time is `seconds` in `results/eval_mlx_2018.json`: 38911.2 (11.488 seconds/example). Peak memory is the maximum `footprint` sample of the train and score processes, taken every 20 seconds: 6181 MB at 2026-11-12T03:43:27Z, during scoring.

```bash
python3.12 --version
PYTHON=python3.12 make mlx
```

Charts need matplotlib on the interpreter you point at:

```bash
python3.12 -m pip install matplotlib
PYTHON=python3.12 make figures
```

On this Mac, matplotlib 3.11.2 was installed in `.venv-mlx`, and `PYTHON=.venv-mlx/bin/python make figures` wrote `docs/img/`.

## Limitations

- One language-model fold. The five-year rolling language-model run was not executed. Pooled classical numbers in `results/baselines.json` cover 11,756 lines and are a different population from this table.
- The 1.5B QLoRA top-1 does not clear merchant lookup or the 0.5B LoRA. The top-1 intervals for 1.5B and lookup overlap.
- Fold-2018 TF-IDF has no bootstrap interval in `results/baselines.json`.
- Prediction files store three names. There is no measured top-k for k greater than 3.
- The MLX validation loss is not a test score. `format_check.n` in the MLX JSON is 0. Free generation was not run for that model.
- Peak memory is a 20-second sample of process footprint, not a device-wide counter and not a torch peak-allocated figure. The JSON does not contain it.
- Training loss in the chart is the text mlx-lm printed.
- Dollar cost was not measured.
- The label ceiling on card-feed inputs, recomputed in `results/baselines.json`, is 0.8608174533701434. The same inputs were not always booked to the same account.
- QLoRA on CUDA was not run. This Air has no CUDA device.

## Repo map

| Path | Role |
|---|---|
| `glcode/` | Split, prompts, metrics, MLX export and scoring |
| `scripts/train_lora.py` | CPU LoRA |
| `scripts/evaluate.py` | Closed-set log-probability ranking |
| `scripts/run_mlx.sh` | Mac QLoRA train and score |
| `scripts/make_figures.py` | Charts in `docs/img/` |
| `configs/cpu_fold2018.json` | The measured 0.5B CPU run |
| `configs/mlx_m2_8gb.yaml` | The 8GB QLoRA defaults |
| `results/eval_*_2018.json` | Measured fold-2018 scores |
| `results/baselines.json` | Classical baselines |
| `results/mlx_train_curve_2018.json` | Losses mlx-lm printed |
| `data/reference/stockport_labelled_lines.csv.gz` | The labelled export |

## Citation

Stockport Metropolitan Borough Council purchase-card transactions, data.gov.uk dataset `purchase-card-transactions`, Open Government Licence v3.0. Contains public sector information licensed under the Open Government Licence v3.0.

The purpose map is the reviewed TrustGate seed. TrustGate: [https://github.com/VijayChamakuri/trustgate](https://github.com/VijayChamakuri/trustgate). The labelled export matches JudgeGate commit `ff0b8f6b8d3ec119c3f3d2131dc1e8a8b190bb4e`, file `data/stockport_labelled_lines.csv.gz`, SHA-256 `d202870296c57f95014b9f6ad09bc5bc2b3ca68180eaafb37aad3bbc51d3e37b`.

Language-model bases: `Qwen/Qwen2.5-0.5B-Instruct` and `mlx-community/Qwen2.5-1.5B-Instruct-4bit`. This repository is MIT, copyright 2026 Vijay Chamakuri.
