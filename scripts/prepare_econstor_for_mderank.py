#!/usr/bin/env python
import json
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent

TEST_JSONL = ROOT / "data" / "test.jsonl"

OUT_DIR = ROOT / "mdeRank" / "data" / "econstor_test"
DOCS_DIR = OUT_DIR / "docsutf8"
KEYS_DIR = OUT_DIR / "keys"

def main():
    print(f"📄 Leyendo {TEST_JSONL}...")
    docs = []
    with TEST_JSONL.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            obj = json.loads(line)
            doc_id = obj.get("doc_id")
            text = obj.get("text") or obj.get("text_en") or ""
            keywords = obj.get("keywords", [])
            if not text.strip():
                continue
            docs.append((doc_id, text, keywords))

    print(f"✅ Cargados {len(docs)} documentos")

    DOCS_DIR.mkdir(parents=True, exist_ok=True)
    KEYS_DIR.mkdir(parents=True, exist_ok=True)

    for idx, (_doc_id, text, keywords) in enumerate(docs):
        (DOCS_DIR / f"{idx}.txt").write_text(text, encoding="utf-8")

        with (KEYS_DIR / f"{idx}.key").open("w", encoding="utf-8") as gf:
            for kw in keywords:
                kw = str(kw).strip()
                if kw:
                    gf.write(kw.lower() + "\n")

    print(f"📂 docsutf8: {DOCS_DIR}")
    print(f"📂 keys:    {KEYS_DIR}")
    print("🎉 Dataset econstor_test preparado para MDERank (formato SemEval2017).")

if __name__ == "__main__":
    main()