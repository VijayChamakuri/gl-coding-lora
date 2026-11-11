#!/usr/bin/env bash
# Setup, QLoRA-train, and score on an Apple Silicon Mac.
#
# Fold 2018 (train on 2017, score 2018), the run that matches the published
# PyTorch fold:
#   make mlx
#
# All five test years, one adapter each, then one pooled score file:
#   make mlx-rolling
#
# Export the JSONL on any machine, including this Linux VM:
#   make mlx-export
set -euo pipefail

ROOT="$(cd "$(dirname "$0")/.." && pwd)"
cd "$ROOT"

FOLDS="2018"
EXPORT_ONLY=0
while [[ $# -gt 0 ]]; do
  case "$1" in
    --folds)
      FOLDS="$2"
      shift 2
      ;;
    --export-only)
      EXPORT_ONLY=1
      shift
      ;;
    *)
      echo "unknown argument: $1" >&2
      exit 2
      ;;
  esac
done

PYTHON="${PYTHON:-python3}"

export_data() {
  local py="$1"
  local model="${2:-}"
  if [[ -n "$model" ]]; then
    "$py" scripts/export_mlx_data.py --folds "$FOLDS" --model "$model"
  else
    "$py" scripts/export_mlx_data.py --folds "$FOLDS"
  fi
}

if [[ "$EXPORT_ONLY" -eq 1 ]]; then
  export_data "$PYTHON"
  echo "Export only. MLX training and scoring were not run."
  exit 0
fi

if [[ "$(uname -s)" != "Darwin" || "$(uname -m)" != "arm64" ]]; then
  if "$PYTHON" -c 'import pandas' >/dev/null 2>&1; then
    export_data "$PYTHON" || true
  fi
  echo "MLX training was not run. This machine is $(uname -s) $(uname -m)." >&2
  echo "On the M2 Air, from this repo: make mlx" >&2
  exit 2
fi

PY_MM="$("$PYTHON" -c 'import sys; print("%d.%d" % sys.version_info[:2])')"
PY_MAJOR="${PY_MM%%.*}"
PY_MINOR="${PY_MM#*.}"
if [[ "$PY_MAJOR" -lt 3 || ( "$PY_MAJOR" -eq 3 && "$PY_MINOR" -lt 11 ) ]]; then
  echo "Python ${PY_MM} is too old. Install Python 3.11 or 3.12 and rerun with PYTHON=python3.12." >&2
  exit 2
fi

if [[ ! -d .venv-mlx ]]; then
  "$PYTHON" -m venv .venv-mlx
fi
# shellcheck disable=SC1091
source .venv-mlx/bin/activate
python -m pip install -U pip
if ! python -m pip install -r requirements-mlx.txt; then
  echo "Pinned mlx-lm install failed. Installing mlx-lm[train] without that pin." >&2
  python -m pip install "mlx-lm[train]" "numpy==2.5.4" "pandas==3.1.1" "scikit-learn==1.9.1" "scipy==1.18.1"
fi
python -m pip show mlx-lm | awk '/^Version:/{print "mlx-lm", $0}'

export MEM_BYTES
MEM_BYTES="$(sysctl -n hw.memsize)"
if [[ "${MLX_FORCE_FALLBACK:-0}" == 1 ]]; then
  MEM_BYTES=1
fi
MODEL_LIST=()
while IFS= read -r line; do
  [[ -n "$line" ]] && MODEL_LIST+=("$line")
done < <(python -c 'import os; from glcode.mlx_data import models_for_memory; print("\n".join(models_for_memory(int(os.environ["MEM_BYTES"]))))')

IFS=',' read -r -a YEAR_LIST <<< "$FOLDS"
CHOSEN=""
FALLBACK_FROM=""
for year in "${YEAR_LIST[@]}"; do
  candidates=()
  if [[ -n "$CHOSEN" ]]; then
    candidates=("$CHOSEN")
  else
    candidates=("${MODEL_LIST[@]}")
  fi
  trained=""
  for model in "${candidates[@]}"; do
    slug="$(python -c 'from glcode.mlx_data import model_slug; import sys; print(model_slug(sys.argv[1]))' "$model")"
    python scripts/export_mlx_data.py --folds "$year" --model "$model" >/dev/null
    cfg="results/mlx_configs/fold_${year}_${slug}.yaml"
    log="results/mlx_fold_${year}_${slug}.log"
    iters="$(python -c 'import json,sys; print(json.load(open(sys.argv[1]))["iters"])' "data/mlx/fold_${year}/meta.json")"
    echo "Training fold ${year} on ${model} for ${iters} steps. Log: ${log}"
    set +e
    python -m mlx_lm lora \
      --config "$cfg" \
      --train \
      --mask-prompt \
      --grad-checkpoint \
      --batch-size 1 \
      --num-layers 8 \
      --max-seq-length 256 \
      --learning-rate 1e-4 \
      --seed 0 \
      --iters "$iters" \
      --clear-cache-threshold 2GB \
      >"$log" 2>&1
    code=$?
    set -e
    if [[ "$code" -eq 0 ]]; then
      trained="$model"
      if [[ -z "$CHOSEN" && "$model" != "${MODEL_LIST[0]}" ]]; then
        FALLBACK_FROM="${MODEL_LIST[0]}"
      fi
      CHOSEN="$model"
      break
    fi
    oom=0
    if python -c 'import sys; from glcode.mlx_data import is_oom; raise SystemExit(0 if is_oom(int(sys.argv[1]), open(sys.argv[2], encoding="utf-8", errors="replace").read()) else 1)' "$code" "$log"; then
      oom=1
    fi
    last_candidate="${candidates[${#candidates[@]}-1]}"
    if [[ "$oom" -eq 1 && -z "$CHOSEN" && "$model" != "$last_candidate" ]]; then
      echo "Out of memory on ${model}. Retrying the smaller model. Log: ${log}" >&2
      continue
    fi
    echo "Training failed (exit ${code}). Last lines of ${log}:" >&2
    tail -n 40 "$log" >&2
    exit "$code"
  done
  if [[ -z "$trained" ]]; then
    echo "Fold ${year} did not train." >&2
    exit 1
  fi
  slug="$(python -c 'from glcode.mlx_data import model_slug; import sys; print(model_slug(sys.argv[1]))' "$trained")"
  python - "$year" "$trained" "$FALLBACK_FROM" "$slug" << 'PY'
import json, sys
from pathlib import Path
year, model, fallback, slug = sys.argv[1:]
path = Path(f"results/mlx_adapters/fold_{year}/{slug}/launch.json")
path.parent.mkdir(parents=True, exist_ok=True)
path.write_text(json.dumps({
    "model": model,
    "fallback_from": fallback or None,
    "fold": int(year),
    "mlx_executed": True,
}, indent=2))
PY
done

TAG="mlx_$(echo "$FOLDS" | tr ',' '_')"
SLUG="$(python -c 'from glcode.mlx_data import model_slug; import sys; print(model_slug(sys.argv[1]))' "$CHOSEN")"
mkdir -p results/mlx_adapters/active
for year in "${YEAR_LIST[@]}"; do
  src="$(cd "results/mlx_adapters/fold_${year}/${SLUG}" && pwd)"
  dst="results/mlx_adapters/active/fold_${year}"
  rm -rf "$dst"
  ln -s "$src" "$dst"
done
python scripts/eval_mlx.py \
  --folds "$FOLDS" \
  --model "$CHOSEN" \
  --adapter-root "results/mlx_adapters/active" \
  --output "results/eval_${TAG}.json" \
  --tag "$TAG" \
  --fallback-from "$FALLBACK_FROM"
echo "Wrote results/eval_${TAG}.json"
