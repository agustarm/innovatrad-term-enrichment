## 🔬 Reproducción de los experimentos

Este repositorio contiene el código necesario para reproducir los experimentos de **extracción automática de términos financieros** realizados en el Trabajo de Fin de Grado, utilizando distintos enfoques y modelos de lenguaje.

Se incluyen:
- Métodos **no supervisados**: AttentionRank y MDERank  
- Distintos **modelos base**: BERT, FinBERT y RoBERTa  
- Un enfoque **supervisado** mediante *fine-tuning* de modelos Transformer  

---

## 1️⃣ Preparación del entorno

### Crear entorno virtual
```bash
python3 -m venv venv
source venv/bin/activate


pip install -r requirements.txt
```
## Dataset

data/
 ├── train.jsonl
 ├── dev.jsonl
 └── test.jsonl

Cada documento sigue la estrcutura:

{
  "doc_id": "...",
  "text": "...",
  "keywords": ["...", "..."]
}

## 3 AttentionRank
### 3.1 Preparación de documentos

```bash
python scripts/rebuild_econstor_docs_for_attentionrank.py

```

econstor/docsutf8/

### 3.2 Ejecución de AttentionRank

Ejemplo con BERT
```bash
python attentionrank/main.py \
  --dataset_name econstor \
  --model_name_or_path bert-base-uncased \
  --model_type bert \
  --lang en \
  --type_execution exec \
  --k_value 15
```
Ejemplo con RoBERTa:
```bash
python attentionrank/main.py \
  --dataset_name econstor \
  --model_name_or_path roberta-base \
  --model_type roberta \
  --lang en \
  --type_execution exec \
  --k_value 15
```
### 3.3 Exportación de prediciones
```bash
python attentionrank/export_attentionrank_predictions.py \
  --dataset_name econstor \
  --top_k 15
```
Se genera el fichero

econstor/predictions_<modelo>_top15.jsonl

## 4.Evaluación (Precisi@K,Recall@K,F1@K)
```bash
python scripts/eval_keywords.py \
  --gold_path data/test.jsonl \
  --pred_path econstor/predictions_bert_top15.jsonl \
  --k_values 5,10,15
```
## MDERank (enfoque no supervisado)

### 5.1 Preparación del dataset para MDERank
```bash
python scripts/prepare_econstor_for_mderank.py
```
Se genera la estructura:

mdeRank/data/econstor_test/
 ├── docs/
 └── gold/

 ### 5.2 Ejecución de MDERank

 Ejemplo con RoBerta:
```bash
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
```
  MDERank evalúa directamente para K = 5, 10 y 15, produciendo métricas comparables con AttentionRank.

  # 6.Enfoque supervisado mediante fine-tuning 

  El enfoque supervisado se plantea como un experimento complementario, basado en la clasificación binaria de candidatos terminológicos.


## 6.1 Generación del dataset supervisado 
```bash
python scripts/build_candidate_dataset_for_finetuning.py
```
se generan los ficheros:

data/train_pairs.jsonl
data/dev_pairs.jsonl
data/test_pairs.jsonl

## 6.2 Entrenamiento (ejemplo FinBERT)
```bash
python scripts/train_finetune_finbert.py
```
los modelos entrenados se almacenan en:

models/<modelo>_finetuned_candidates/