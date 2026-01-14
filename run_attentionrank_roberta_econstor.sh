#!/bin/bash
set -e

# Ir a la raíz del repo (la carpeta donde está este .sh)
cd "$(dirname "$0")"

echo "📌 [1/5] Activando entorno virtual..."
source venv/bin/activate

echo "📌 [2/5] Limpiando salidas anteriores de Econstor..."
rm -rf econstor/processed_econstor
rm -rf econstor/res*
# OJO: NO borramos econstor_finance_en.jsonl ni econstor/docsutf8 si ya está bien
# (si quieres reconstruir siempre docsutf8, deja el paso 3)

echo "📌 [3/5] Reconstruyendo econstor/docsutf8 desde econstor_finance_en.jsonl..."
python scripts/rebuild_econstor_docs_for_attentionrank.py

echo "📌 [4/5] Ejecutando AttentionRank (pipeline completa) con RoBERTa..."
python attentionrank/main.py \
  --dataset_name econstor \
  --model_name_or_path roberta-base \
  --model_type roberta \
  --lang en \
  --type_execution exec \
  --k_value 15

echo "✅ Pipeline AttentionRank completada."

echo "📌 [5/5] Exportando top-15 keywords..."
python attentionrank/export_attentionrank_predictions.py \
  --dataset_name econstor \
  --top_k 15

# Renombramos el fichero genérico a uno específico del modelo
if [ -f "econstor/predictions_top15.jsonl" ]; then
  mv econstor/predictions_top15.jsonl econstor/predictions_roberta_top15.jsonl
  echo "✅ Predicciones guardadas en econstor/predictions_roberta_top15.jsonl"
else
  echo "⚠️ No se ha encontrado econstor/predictions_top15.jsonl. Algo ha fallado en la exportación."
fi

echo "🎉 Pipeline completa para RoBERTa finalizada."