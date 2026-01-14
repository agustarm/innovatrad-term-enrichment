#!/usr/bin/env python
import json
import random
from pathlib import Path
from typing import List, Set

ROOT = Path(__file__).resolve().parent.parent

TRAIN_IN = ROOT / "data" / "train.jsonl"
DEV_IN   = ROOT / "data" / "dev.jsonl"
TEST_IN  = ROOT / "data" / "test.jsonl"

TRAIN_OUT = ROOT / "data" / "train_pairs.jsonl"
DEV_OUT   = ROOT / "data" / "dev_pairs.jsonl"
TEST_OUT  = ROOT / "data" / "test_pairs.jsonl"

random.seed(42)

def load_jsonl(path: Path):
    data = []
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            data.append(json.loads(line))
    return data

def write_jsonl(path: Path, rows):
    with path.open("w", encoding="utf-8") as f:
        for obj in rows:
            f.write(json.dumps(obj, ensure_ascii=False) + "\n")

def extract_ngrams(tokens: List[str], max_n: int = 3) -> Set[str]:
    """Devuelve todos los n-gramas (1..max_n) como strings en minúscula."""
    ngrams = set()
    n_tokens = len(tokens)
    for n in range(1, max_n + 1):
        for i in range(0, n_tokens - n + 1):
            span = tokens[i:i+n]
            s = " ".join(span).strip()
            if len(s) < 3:
                continue
            ngrams.add(s.lower())
    return ngrams

def build_pairs(split_name: str, input_path: Path, output_path: Path,
                max_neg_per_doc: int = 20):
    print(f"📄 Procesando {split_name} desde {input_path}...")
    docs = load_jsonl(input_path)
    pairs = []

    for doc in docs:
        doc_id = doc.get("doc_id")
        text = doc.get("text") or doc.get("text_en") or ""
        gold = doc.get("keywords", [])
        if not text.strip():
            continue

        gold_norm = [g.strip().lower() for g in gold if g.strip()]
        gold_set = set(gold_norm)

        # Positivos: cada keyword gold
        for kw in gold_norm:
            pairs.append({
                "doc_id": doc_id,
                "text": text,
                "candidate": kw,
                "label": 1
            })

        # Negativos: n-gramas que NO estén en las gold
        tokens = text.split()
        all_ngrams = extract_ngrams(tokens, max_n=3)
        neg_candidates = list(all_ngrams - gold_set)

        random.shuffle(neg_candidates)
        neg_candidates = neg_candidates[:max_neg_per_doc]

        for kw in neg_candidates:
            pairs.append({
                "doc_id": doc_id,
                "text": text,
                "candidate": kw,
                "label": 0
            })

    print(f"✅ {split_name}: generados {len(pairs)} pares (pos+neg).")
    write_jsonl(output_path, pairs)
    print(f"💾 Guardado en {output_path}")

def main():
    build_pairs("train", TRAIN_IN, TRAIN_OUT, max_neg_per_doc=20)
    build_pairs("dev",   DEV_IN,   DEV_OUT,   max_neg_per_doc=20)
    build_pairs("test",  TEST_IN,  TEST_OUT,  max_neg_per_doc=20)
    print("🎉 Dataset supervisado preparado.")

if __name__ == "__main__":
    main()