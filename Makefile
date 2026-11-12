.PHONY: check baselines test train-cpu eval-cpu mlx mlx-rolling mlx-export figures demo

PYTHON ?= python3

check:
	python scripts/prepare_data.py --check

baselines:
	python scripts/run_baselines.py

test:
	pytest

train-cpu:
	python scripts/train_lora.py --config configs/cpu_fold2018.json

eval-cpu:
	python scripts/evaluate.py --config configs/eval_zeroshot_2018.json
	python scripts/evaluate.py --config configs/eval_fewshot_2018.json
	python scripts/evaluate.py --config configs/eval_lora_2018.json

# Apple Silicon only. Exports JSONL, trains Qwen2.5-1.5B-Instruct 4-bit LoRA
# on 2017, scores 2018. Falls back to the 0.5B 4-bit model if 1.5B runs out of memory.
mlx:
	PYTHON="$(PYTHON)" bash scripts/run_mlx.sh

# Same path for test years 2018, 2019, 2020, 2021, and 2022.
mlx-rolling:
	PYTHON="$(PYTHON)" bash scripts/run_mlx.sh --folds 2018,2019,2020,2021,2022

# Data export and YAML only. This is the part that runs on Linux.
mlx-export:
	PYTHON="$(PYTHON)" bash scripts/run_mlx.sh --export-only

# Charts in docs/img. Reads result JSON, prediction JSONL, and the training log when present.
figures:
	$(PYTHON) scripts/make_figures.py

# One 2018 purchase through the local 4-bit adapter. The weights are not in git.
demo:
	.venv-mlx/bin/python scripts/demo_one.py
