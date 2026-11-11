.PHONY: check baselines test train-cpu eval-cpu

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
