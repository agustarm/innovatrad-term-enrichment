#!/bin/bash
set -e

cd "$(dirname "$0")"

echo "📌 [1/3] Activando venv..."
source venv/bin/activate

echo "📌 [2/3] Preparando dataset econstor_test para MDERank..."
python scripts/prepare_econstor_for_mderank.py

echo "📌 [3/3] Ejecutando MDERank + RoBERTa..."
mkdir -p logs results/mderank

python mdeRank/MDERank/mderank_main.py \
  --dataset_dir mdeRank/data/econstor_test \
  --dataset_name econstor_test \
  --doc_embed_mode mean \
  --batch_size 16 \
  --log_dir logs/ \
  --model_type roberta \
  --model_name_or_path roberta-base \
  --lang en \
  --type_execution eval

echo "🎉 MDERank + RoBERTa finalizado"