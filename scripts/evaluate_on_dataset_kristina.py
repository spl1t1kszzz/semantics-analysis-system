#!/usr/bin/env python3
"""
Оценка качества извлечения отношений на датасете Кристины.

Использование:
  1. Сначала конвертируйте датасет в JSON:
     poetry run python scripts/convert_dataset_kristina_to_sentences.py

  2. Запустите оценку (из корня проекта):
     poetry run python scripts/evaluate_on_dataset_kristina.py [sentences.json] [--multi-agent] [--limit N]

  --multi-agent: включить разрешение конфликтов перед подсчётом метрик
  --limit N: обработать только первые N предложений (для отладки)

Метрики сохраняются в results/dataset_kristina_scores.json и results/dataset_kristina_metrics.md
"""

import json
import os
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
os.chdir(PROJECT_ROOT)
sys.path.insert(0, str(PROJECT_ROOT))

from rich.progress import Progress

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence, Relation, Term
from semantics_analysis.ontology_utils import loaded_relation_ids
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from calculate_scores import update_scores


def run_evaluation(
    sentences: List[Sentence],
    relation_extractor,
    reference_resolver,
    conflict_resolver=None,
    progress=None,
    limit: Optional[int] = None,
    collect_triples: bool = False,
) -> tuple:
    """Считает метрики: для каждого типа отношений — precision, recall; общие — micro/macro."""
    scores = {}
    for rel_id in loaded_relation_ids:
        scores[rel_id] = {
            "predicted": {"incorrect": {"count": 0, "examples": []}, "correct": {"count": 0, "examples": []}},
            "expected": {"not_found": {"count": 0, "examples": []}, "found": {"count": 0, "examples": []}},
        }
    ignored = set()
    sentences_to_check = sentences[:limit] if limit else sentences
    total = len(sentences_to_check)
    task = progress.add_task(description="Sentences", total=total) if progress else None
    all_predicted_triples = set() if collect_triples else None  # (value1, predicate, value2)

    for idx, sent in enumerate(sentences_to_check):
        expected_relations = {rel for rel in sent.relations if rel.id in loaded_relation_ids}
        if not expected_relations:
            if progress:
                progress.update(task, advance=1)
            continue

        term_mentions = []
        for term in sent.terms:
            term_mentions.extend(term.mentions)
        try:
            grouped_terms = reference_resolver(term_mentions, sent.text)
        except Exception as e:
            if progress:
                progress.update(task, advance=1)
            continue

        term_variants = {}
        predicted_relations = set()
        for term in grouped_terms:
            variants = [Term(term.class_, m.value, [m]) for m in term.mentions]
            term_variants[term] = variants
            if len(variants) > 1:
                for i in range(len(variants)):
                    for j in range(i + 1, len(variants)):
                        if variants[i].value != variants[j].value:
                            predicted_relations.add(Relation(variants[i], "isAlternativeNameFor", variants[j]))

        try:
            relations_iter = relation_extractor(sent.text, grouped_terms)
            for rel in relations_iter.items:
                if rel is not None:
                    predicted_relations.add(rel)
                    for t1 in term_variants.get(rel.term1, [rel.term1]):
                        for t2 in term_variants.get(rel.term2, [rel.term2]):
                            r2 = Relation(t1, rel.predicate, t2)
                            if r2 in expected_relations:
                                predicted_relations.add(r2)
        except Exception as e:
            if progress:
                progress.update(task, advance=1)
            continue

        if conflict_resolver and len(predicted_relations) > 0:
            resolved, _, _ = conflict_resolver.resolve_all(sent.text, list(predicted_relations))
            predicted_relations = set(resolved)

        if collect_triples:
            for rel in predicted_relations:
                all_predicted_triples.add((rel.term1.value, rel.predicate, rel.term2.value))

        update_scores(sent, predicted_relations, expected_relations, ignored, scores)
        if progress:
            progress.update(task, advance=1, description=f"Sentence {idx+1}/{total}")

    # Агрегируем метрики
    result = {"by_relation": {}, "micro": {"tp": 0, "fp": 0, "fn": 0}, "macro": {"precision": [], "recall": []}}
    for rel_id, s in scores.items():
        correct = s["predicted"]["correct"]["count"]
        incorrect = s["predicted"]["incorrect"]["count"]
        found = s["expected"]["found"]["count"]
        not_found = s["expected"]["not_found"]["count"]
        if found + not_found == 0:
            continue
        prec = correct / (correct + incorrect) if (correct + incorrect) > 0 else 0.0
        rec = found / (found + not_found)
        result["by_relation"][rel_id] = {
            "precision": round(prec, 4),
            "recall": round(rec, 4),
            "f1": round(2 * prec * rec / (prec + rec), 4) if (prec + rec) > 0 else 0.0,
            "support": found + not_found,
        }
        result["micro"]["tp"] += correct
        result["micro"]["fp"] += incorrect
        result["micro"]["fn"] += not_found
        result["macro"]["precision"].append(prec)
        result["macro"]["recall"].append(rec)

    n = len(result["macro"]["precision"])
    result["macro"]["precision"] = round(sum(result["macro"]["precision"]) / n, 4) if n else 0.0
    result["macro"]["recall"] = round(sum(result["macro"]["recall"]) / n, 4) if n else 0.0
    tp, fp, fn = result["micro"]["tp"], result["micro"]["fp"], result["micro"]["fn"]
    result["micro"]["precision"] = round(tp / (tp + fp), 4) if (tp + fp) > 0 else 0.0
    result["micro"]["recall"] = round(tp / (tp + fn), 4) if (tp + fn) > 0 else 0.0
    p, r = result["micro"]["precision"], result["micro"]["recall"]
    result["micro"]["f1"] = round(2 * p * r / (p + r), 4) if (p + r) > 0 else 0.0
    return result, all_predicted_triples


def main():
    sentences_path = PROJECT_ROOT / "tests" / "dataset_kristina_sentences.json"
    use_multi_agent = "--multi-agent" in sys.argv
    save_triples = "--save-triples" in sys.argv
    limit = None
    args = [a for a in sys.argv[1:] if a not in ("--multi-agent", "--save-triples")]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if args[i].endswith(".json"):
            sentences_path = Path(args[i])
        i += 1

    if not sentences_path.is_file():
        print(f"Файл не найден: {sentences_path}")
        print("Сначала выполните: poetry run python scripts/convert_dataset_kristina_to_sentences.py")
        sys.exit(1)

    config = load_config(PROJECT_ROOT / "config.yml")
    sentences = read_sentences(str(sentences_path))
    print(f"Загружено предложений: {len(sentences)}")
    if limit:
        print(f"Ограничение: первые {limit}")
    print(f"Мультиагентный режим (разрешение конфликтов): {use_multi_agent}")
    print(f"Сохранять предсказанные тройки для метрик графа: {save_triples}")

    with Progress() as progress:
        ref_resolver = LLMReferenceResolver(progress=progress, model=config.llm, use_all_tokens=True)
        relation_extractor = LLMRelationExtractor(model=config.llm, use_all_tokens=True)
        conflict_resolver = None
        if use_multi_agent:
            from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver
            conflict_resolver = RelationConflictResolver()

        result, predicted_triples = run_evaluation(
            sentences,
            relation_extractor,
            ref_resolver,
            conflict_resolver=conflict_resolver,
            progress=progress,
            limit=limit,
            collect_triples=save_triples,
        )

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    scores_path = out_dir / "dataset_kristina_scores.json"
    md_path = out_dir / "dataset_kristina_metrics.md"

    with open(scores_path, "w", encoding="utf-8") as f:
        json.dump(result, f, ensure_ascii=False, indent=2)
    print(f"Метрики (JSON): {scores_path}")

    lines = [
        "# Метрики на датасете Кристины",
        "",
        "## Micro (по всем парам)",
        f"- Precision: {result['micro']['precision']:.4f}",
        f"- Recall: {result['micro']['recall']:.4f}",
        f"- F1: {result['micro']['f1']:.4f}",
        "",
        "## Macro (среднее по типам отношений)",
        f"- Precision: {result['macro']['precision']:.4f}",
        f"- Recall: {result['macro']['recall']:.4f}",
        "",
        "## По типам отношений",
        "| Relation | Precision | Recall | F1 | Support |",
        "|----------|-----------|--------|-----|---------|",
    ]
    for rel_id, m in sorted(result["by_relation"].items(), key=lambda x: -x[1]["support"]):
        lines.append(f"| {rel_id} | {m['precision']:.4f} | {m['recall']:.4f} | {m['f1']:.4f} | {m['support']} |")
    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Отчёт (Markdown): {md_path}")

    if save_triples and predicted_triples is not None:
        triples_path = out_dir / "dataset_kristina_predicted_triples.json"
        with open(triples_path, "w", encoding="utf-8") as f:
            json.dump([list(t) for t in predicted_triples], f, ensure_ascii=False, indent=0)
        print(f"Предсказанные тройки для метрик графа: {triples_path}")
        print("  Далее: python calculate_graph_metrics.py tests/dataset_kristina_sentences.json results/dataset_kristina_predicted_triples.json")


if __name__ == "__main__":
    main()
