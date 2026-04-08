import json
import re
import sys
from pathlib import Path
from typing import Set, Tuple, Dict, Any, List, Optional

from semantics_analysis.entities import read_sentences, Sentence


# =====================================================
# Triple definition
# =====================================================

Triple = Tuple[str, str, str]   # (value1, predicate, value2)

TRIPLE_RE = re.compile(r"\((.*?)\)\s+(\S+)\s+\((.*?)\)")




# =====================================================
# GOLD GRAPH → triples
# =====================================================

def extract_gold_triples(sentences_path: str) -> Set[Triple]:
    sentences: List[Sentence] = read_sentences(sentences_path)

    gold_triples: Set[Triple] = set()

    for sent in sentences:
        for rel in sent.relations:
            gold_triples.add(
                (rel.term1.value, rel.predicate, rel.term2.value)
            )

    return gold_triples


# =====================================================
# PREDICTED GRAPH → triples
# =====================================================

def parse_triple_from_string(s: str) -> Triple | None:
    """
    '(A) relation (B)' → ('A', 'relation', 'B')
    """
    m = TRIPLE_RE.match(s)
    if not m:
        return None
    return m.group(1), m.group(2), m.group(3)


def extract_predicted_triples(results_path: str) -> Set[Triple]:
    """Из файла результатов извлечения отношений (shot_rel_*.json)."""
    with open(results_path, "r", encoding="utf-8") as f:
        data = json.load(f)

    predicted_triples: Set[Triple] = set()
    for block in data.values():
        if isinstance(block, dict) and "predicted" in block:
            for section in ("correct", "incorrect"):
                for ex in block["predicted"][section]["examples"]:
                    triple = parse_triple_from_string(ex["relation"])
                    if triple:
                        predicted_triples.add(triple)
    return predicted_triples


def extract_triples_from_kg(kg_path: str) -> Set[Triple]:
    """
    Из построенного графа знаний (ont_entities.json: objects + relations).
    Имя связи имеет вид Class1_predicate_Class2 — в тройку берём predicate.
    """
    with open(kg_path, "r", encoding="utf-8") as f:
        data = json.load(f)
    objects = {obj["id"]: obj for obj in data.get("objects", [])}
    relations = data.get("relations", [])
    triples: Set[Triple] = set()
    for r in relations:
        o1_id = r.get("object1_id")
        o2_id = r.get("object2_id")
        name = r.get("name", "")
        if o1_id not in objects or o2_id not in objects:
            continue
        # name = "Method_solves_Task" -> predicate = "solves"
        parts = name.split("_")
        if len(parts) >= 3:
            predicate = "_".join(parts[1:-1])  # на случай предиката из нескольких частей
        else:
            predicate = name
        v1 = objects[o1_id]["value"]
        v2 = objects[o2_id]["value"]
        triples.add((v1, predicate, v2))
    return triples


# =====================================================
# GRAPH-LEVEL METRICS
# =====================================================

def compute_graph_metrics(
        gold: Set[Triple],
        predicted: Set[Triple]
) -> Dict[str, Any]:

    tp = len(gold & predicted)
    fp = len(predicted - gold)
    fn = len(gold - predicted)

    precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
    f1 = (
        2 * precision * recall / (precision + recall)
        if (precision + recall) > 0 else 0.0
    )

    return {
        "TP": tp,
        "FP": fp,
        "FN": fn,
        "precision": round(precision, 4),
        "recall": round(recall, 4),
        "f1": round(f1, 4),
        "gold_triples": len(gold),
        "predicted_triples": len(predicted),
    }


def structural_stats(triples: Set[Triple]) -> Dict[str, int]:
    """Число узлов (уникальных сущностей) и рёбер (троек)."""
    nodes = set()
    for v1, _, v2 in triples:
        nodes.add(v1)
        nodes.add(v2)
    return {"nodes": len(nodes), "edges": len(triples)}


def load_predicted_triples(path: str) -> Set[Triple]:
    """По содержимому выбирает способ загрузки предсказаний."""
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    # Список троек [[v1, pred, v2], ...] (например из evaluate_on_dataset_kristina --save-triples)
    if isinstance(data, list) and data and isinstance(data[0], (list, tuple)) and len(data[0]) == 3:
        return set((t[0], t[1], t[2]) for t in data)
    if "objects" in data and "relations" in data:
        # Построенный ГЗ (ont_entities.json)
        objects = {obj["id"]: obj for obj in data["objects"]}
        triples: Set[Triple] = set()
        for r in data["relations"]:
            o1_id, o2_id = r.get("object1_id"), r.get("object2_id")
            if o1_id not in objects or o2_id not in objects:
                continue
            name = r.get("name", "")
            parts = name.split("_")
            predicate = "_".join(parts[1:-1]) if len(parts) >= 3 else name
            v1, v2 = objects[o1_id]["value"], objects[o2_id]["value"]
            triples.add((v1, predicate, v2))
        return triples
    # shot_rel_*.json
    return extract_predicted_triples(path)


# =====================================================
# MAIN
# =====================================================

if __name__ == "__main__":
    GOLD_PATH = "tests/short_sent.json"
    PREDICTED_PATH = "results/shot_rel_new_prompts.json"
    if len(sys.argv) >= 2:
        GOLD_PATH = sys.argv[1]
    if len(sys.argv) >= 3:
        PREDICTED_PATH = sys.argv[2]

    gold_triples = extract_gold_triples(GOLD_PATH)
    predicted_triples = load_predicted_triples(PREDICTED_PATH)

    metrics = compute_graph_metrics(gold_triples, predicted_triples)
    metrics["gold_structure"] = structural_stats(gold_triples)
    metrics["predicted_structure"] = structural_stats(predicted_triples)

    print("GRAPH-LEVEL METRICS (triple-based)")
    for k, v in metrics.items():
        if k not in ("gold_structure", "predicted_structure"):
            print(f"  {k}: {v}")
    print("  gold_structure:", metrics["gold_structure"])
    print("  predicted_structure:", metrics["predicted_structure"])

    out_dir = Path("results")
    out_dir.mkdir(parents=True, exist_ok=True)
    with open(out_dir / "graph_metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics, f, ensure_ascii=False, indent=2)
    with open(out_dir / "graph_metrics.md", "w", encoding="utf-8") as f:
        f.write("# Метрики построения графа знаний\n\n")
        f.write("| Метрика | Значение |\n|---------|----------|\n")
        f.write(f"| Precision (тройки) | {metrics['precision']} |\n")
        f.write(f"| Recall (тройки) | {metrics['recall']} |\n")
        f.write(f"| F1 (тройки) | {metrics['f1']} |\n")
        f.write(f"| Gold: узлов / рёбер | {metrics['gold_structure']['nodes']} / {metrics['gold_structure']['edges']} |\n")
        f.write(f"| Predicted: узлов / рёбер | {metrics['predicted_structure']['nodes']} / {metrics['predicted_structure']['edges']} |\n")
    print(f"\nСохранено: results/graph_metrics.json, results/graph_metrics.md")
