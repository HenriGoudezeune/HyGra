#!/usr/bin/env bash
# Full v2 pipeline: tokenizer -> generator -> evaluation. Logs go to logs/.
set -euo pipefail
cd "$(dirname "$0")"
mkdir -p logs
export PYTHONUNBUFFERED=1

if [ ! -f ckpt/tokenizer.pt ] || [ "${RETRAIN_TOKENIZER:-0}" = 1 ]; then
  uv run python train_tokenizer.py "$@" 2>&1 | tee logs/tokenizer.log
fi
uv run python demo_tokenizer.py 2>&1 | tee logs/demo_tokenizer.log
uv run python train_generator.py 2>&1 | tee logs/generator.log
uv run python eval_code.py 2>&1 | tee logs/eval.log
