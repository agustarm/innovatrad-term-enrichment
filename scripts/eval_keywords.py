import json
from pathlib import Path
import argparse
from typing import Iterable, List


def load_jsonl(path: Path):
    with path.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            yield json.loads(line)


def normalize_kw_list(kws):
    if isinstance(kws, str):
        # Por si en algún sitio vinieran como cadena separada por |
        parts = [p.strip() for p in kws.split("|")]
    else:
        parts = kws or []
    return {p.lower().strip() for p in parts if p.strip()}


def eval_model(gold_path: Path, pred_path: Path, k_values: List[int]):
    gold_data = list(load_jsonl(gold_path))
    pred_data = list(load_jsonl(pred_path))

    # Mapeamos predicciones por doc_id = índice (string)
    pred_by_id = {str(obj["doc_id"]): obj for obj in pred_data}

    # Acumuladores por K
    totals = {k: {"p": 0.0, "r": 0.0, "f1": 0.0, "count": 0} for k in k_values}

    for idx, gold_obj in enumerate(gold_data):
        gold_kws = gold_obj.get("keywords") or gold_obj.get("gold_keywords") or []
        gold_set = normalize_kw_list(gold_kws)

        pred_obj = pred_by_id.get(str(idx))
        pred_list = []
        if pred_obj is not None:
            pred_list = pred_obj.get("predicted_keywords", []) or []

        # Para cada K, cogemos el top-K y evaluamos
        for k in k_values:
            pred_topk = pred_list[:k]
            pred_set = normalize_kw_list(pred_topk)

            # Caso: no hay predicciones y/o gold
            if not pred_set and not gold_set:
                totals[k]["count"] += 1
                continue

            inter = gold_set & pred_set
            tp = len(inter)
            p = tp / max(len(pred_set), 1)
            r = tp / max(len(gold_set), 1)
            f1 = 2 * p * r / (p + r) if (p + r) > 0 else 0.0

            totals[k]["p"] += p
            totals[k]["r"] += r
            totals[k]["f1"] += f1
            totals[k]["count"] += 1

    # Medias por K
    means = {}
    for k in k_values:
        c = totals[k]["count"]
        if c == 0:
            means[k] = (0.0, 0.0, 0.0)
        else:
            means[k] = (totals[k]["p"] / c, totals[k]["r"] / c, totals[k]["f1"] / c)

    return means


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "--gold_path",
        type=str,
        default="data/test.jsonl",
        help="Ruta al JSONL con el gold estándar (text + keywords).",
    )
    parser.add_argument(
        "--pred_path",
        type=str,
        required=True,
        help="Ruta al JSONL con las predicciones de un modelo.",
    )
    parser.add_argument(
        "--k_values",
        type=str,
        default="15",
        help="Valores de K separados por comas (ej: 5,10,15). Por defecto: 15",
    )
    args = parser.parse_args()

    gold = Path(args.gold_path)
    pred = Path(args.pred_path)

    # Parse K values
    try:
        k_values = [int(x.strip()) for x in args.k_values.split(",") if x.strip()]
    except ValueError:
        raise SystemExit("Error: --k_values debe ser una lista de enteros separada por comas, p.ej. 5,10,15")

    if not k_values:
        k_values = [15]

    # Aseguramos orden ascendente y sin duplicados
    k_values = sorted(set(k_values))

    means = eval_model(gold, pred, k_values)

    print(f"Evaluación sobre {gold}:")
    print(f"  Predicciones: {pred}")
    for k in k_values:
        p, r, f1 = means[k]
        print(f"\nK={k}")
        print(f"  Precision@K media: {p:.4f}")
        print(f"  Recall@K medio:    {r:.4f}")
        print(f"  F1@K medio:        {f1:.4f}")


if __name__ == "__main__":
    main()