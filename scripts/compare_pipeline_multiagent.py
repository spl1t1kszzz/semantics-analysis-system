#!/usr/bin/env python3
"""
Сравнение пайплайна без мультиагентного шага и с ним (с разрешением конфликтов).

Запускает оценку дважды на одном датасете:
  1) Базовый пайплайн: извлечение отношений без разрешения конфликтов.
  2) С мультиагентным шагом: после извлечения отношений вызывается агент разрешения конфликтов.

Использование (из корня проекта):
  poetry run python scripts/compare_pipeline_multiagent.py [sentences.json] [--limit N]

Результаты: results/compare_baseline_vs_multiagent.json и results/compare_baseline_vs_multiagent.md
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

from rich.progress import Progress

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor

# Импорт run_evaluation из того же каталога scripts
import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval_kristina",
    PROJECT_ROOT / "scripts" / "evaluate_on_dataset_kristina.py",
)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
run_evaluation = _eval_mod.run_evaluation


def run_multiagent_only(
    sentences: List[Sentence],
    relation_extractor,
    reference_resolver,
    progress: Progress,
    limit: Optional[int] = None,
    conflict_resolver=None,
):
    """Запуск мультиагентной оценки (multi-probe + опциональное разрешение конфликтов)."""
    result, triples, pre_conflict_result = run_evaluation(
        sentences,
        relation_extractor,
        reference_resolver,
        conflict_resolver=conflict_resolver,
        progress=progress,
        limit=limit,
        collect_triples=True,
    )
    return result, triples, pre_conflict_result


def run_both(
    sentences: List[Sentence],
    relation_extractor,
    reference_resolver,
    progress: Progress,
    limit: Optional[int] = None,
    multiagent_relation_extractor=None,
    conflict_resolver=None,
):
    """Запуск оценки без и с разрешением конфликтов.

    Возвращает (result_baseline, result_multiagent, triples_baseline, triples_multiagent, pre_conflict_result).
    pre_conflict_result — метрики мультиагентного пайплайна до разрешения конфликтов.
    """
    # 1) Без мультиагента (без разрешения конфликтов)
    result_baseline, triples_baseline, _ = run_evaluation(
        sentences,
        relation_extractor,
        reference_resolver,
        conflict_resolver=None,
        progress=progress,
        limit=limit,
        collect_triples=True,
    )
    # 2) С мультиагентом (multi-probe + разрешение конфликтов)
    ma_extractor = multiagent_relation_extractor or relation_extractor
    result_multiagent, triples_multiagent, pre_conflict_result = run_multiagent_only(
        sentences, ma_extractor, reference_resolver, progress, limit=limit,
        conflict_resolver=conflict_resolver,
    )
    return result_baseline, result_multiagent, triples_baseline, triples_multiagent, pre_conflict_result


def _macro_f1(precision: float, recall: float) -> float:
    if precision + recall <= 0:
        return 0.0
    return round(2 * precision * recall / (precision + recall), 4)


def build_comparison(
    result_baseline: Dict[str, Any],
    result_multiagent: Dict[str, Any],
    gold_triples: Optional[set] = None,
    triples_baseline: Optional[set] = None,
    triples_multiagent: Optional[set] = None,
) -> Dict[str, Any]:
    """Строит сравнение: micro/macro (в т.ч. macro F1), по типам отношений, опционально P/R/F1 по тройкам графа."""
    b_macro = result_baseline["macro"]
    m_macro = result_multiagent["macro"]
    b_macro_f1 = _macro_f1(b_macro["precision"], b_macro["recall"])
    m_macro_f1 = _macro_f1(m_macro["precision"], m_macro["recall"])

    out = {
        "baseline": {
            "micro": result_baseline["micro"],
            "macro": {
                "precision": b_macro["precision"],
                "recall": b_macro["recall"],
                "f1": b_macro_f1,
            },
            "by_relation": result_baseline["by_relation"],
        },
        "multiagent": {
            "micro": result_multiagent["micro"],
            "macro": {
                "precision": m_macro["precision"],
                "recall": m_macro["recall"],
                "f1": m_macro_f1,
            },
            "by_relation": result_multiagent["by_relation"],
        },
        "diff": {
            "micro_precision": round(result_multiagent["micro"]["precision"] - result_baseline["micro"]["precision"], 4),
            "micro_recall": round(result_multiagent["micro"]["recall"] - result_baseline["micro"]["recall"], 4),
            "micro_f1": round(result_multiagent["micro"]["f1"] - result_baseline["micro"]["f1"], 4),
            "macro_precision": round(m_macro["precision"] - b_macro["precision"], 4),
            "macro_recall": round(m_macro["recall"] - b_macro["recall"], 4),
            "macro_f1": round(m_macro_f1 - b_macro_f1, 4),
        },
    }
    if triples_baseline is not None and triples_multiagent is not None and gold_triples is not None:
        from calculate_graph_metrics import compute_graph_metrics
        metrics_b = compute_graph_metrics(gold_triples, triples_baseline)
        metrics_m = compute_graph_metrics(gold_triples, triples_multiagent)
        out["graph_triples"] = {
            "baseline_count": len(triples_baseline),
            "multiagent_count": len(triples_multiagent),
            "gold_count": len(gold_triples),
            "baseline": {"precision": metrics_b["precision"], "recall": metrics_b["recall"], "f1": metrics_b["f1"]},
            "multiagent": {"precision": metrics_m["precision"], "recall": metrics_m["recall"], "f1": metrics_m["f1"]},
            "diff": {
                "precision": round(metrics_m["precision"] - metrics_b["precision"], 4),
                "recall": round(metrics_m["recall"] - metrics_b["recall"], 4),
                "f1": round(metrics_m["f1"] - metrics_b["f1"], 4),
            },
        }
    return out


def main():
    sentences_path = PROJECT_ROOT / "tests" / "dataset_kristina_sentences.json"
    limit = None
    multiagent_only = "--multiagent-only" in sys.argv
    args = [a for a in sys.argv[1:] if a != "--multiagent-only"]
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
        print("Укажите JSON с предложениями (например tests/dataset_kristina_sentences.json)")
        sys.exit(1)

    config = load_config(PROJECT_ROOT / "config.yml")
    sentences = read_sentences(str(sentences_path))
    print(f"Загружено предложений: {len(sentences)}")
    if limit:
        print(f"Ограничение: первые {limit} предложений")

    if multiagent_only:
        print("Режим: только мультиагент (baseline F1=0.7640 из кэша).")
        with Progress() as progress:
            ref_resolver = LLMReferenceResolver(progress=progress, model=config.llm)
            multiagent_extractor = LLMRelationExtractor(model=config.llm, use_multi_probe=True)

            from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver
            from semantics_analysis.llm_agent import LLMAgent
            llm_agent = LLMAgent(model=config.llm)
            cr = RelationConflictResolver(llm_agent=llm_agent)
            cr.reverify_callback = multiagent_extractor.verify_relation

            result_multiagent, triples_multiagent, pre_conflict_result = run_multiagent_only(
                sentences, multiagent_extractor, ref_resolver, progress, limit=limit,
                conflict_resolver=cr,
            )
        m = result_multiagent["micro"]
        print(f"\nМультиагент micro: P={m['precision']:.4f} R={m['recall']:.4f} F1={m['f1']:.4f} (TP={m['tp']} FP={m['fp']} FN={m['fn']})")
        print(f"Разница с baseline: F1 {m['f1'] - 0.7640:+.4f}")
        if pre_conflict_result:
            pc = pre_conflict_result["micro"]
            print(f"До конфликтов:      P={pc['precision']:.4f} R={pc['recall']:.4f} F1={pc['f1']:.4f} (TP={pc['tp']} FP={pc['fp']} FN={pc['fn']})")
        return

    print("Сравнение: базовый пайплайн vs с разрешением конфликтов (два прохода по датасету).")
    with Progress() as progress:
        ref_resolver = LLMReferenceResolver(progress=progress, model=config.llm)
        relation_extractor = LLMRelationExtractor(model=config.llm)
        multiagent_extractor = LLMRelationExtractor(model=config.llm, use_multi_probe=True)

        # Conflict resolver: re-verify конфликтные группы через verify_relation
        from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver
        from semantics_analysis.llm_agent import LLMAgent
        llm_agent = LLMAgent(model=config.llm)
        conflict_resolver = RelationConflictResolver(
            llm_agent=llm_agent,
            log_prompts=config.log_prompts,
            log_responses=config.log_llm_responses,
        )
        conflict_resolver.reverify_callback = multiagent_extractor.verify_relation

        result_baseline, result_multiagent, triples_baseline, triples_multiagent, pre_conflict_result = run_both(
            sentences, relation_extractor, ref_resolver, progress, limit=limit,
            multiagent_relation_extractor=multiagent_extractor,
            conflict_resolver=conflict_resolver,
        )
    sentences_to_check = sentences[:limit] if limit else sentences
    gold_triples = set(
        (r.term1.value, r.predicate, r.term2.value)
        for s in sentences_to_check
        for r in s.relations
    )

    # --- Статистика по классам терминов ---
    from collections import Counter
    class_counts = Counter()
    for s in sentences_to_check:
        for term in s.terms:
            class_counts[term.class_] += 1

    # --- Статистика по этапам: reference resolution ---
    ref_stats = {"total_mentions": 0, "total_terms_after": 0}
    for s in sentences_to_check:
        for term in s.terms:
            ref_stats["total_mentions"] += len(term.mentions)
        ref_stats["total_terms_after"] += len(s.terms)

    comparison = build_comparison(
        result_baseline,
        result_multiagent,
        gold_triples=gold_triples,
        triples_baseline=triples_baseline,
        triples_multiagent=triples_multiagent,
    )

    # Добавляем промежуточные метрики (до разрешения конфликтов)
    if pre_conflict_result:
        comparison["multiagent_pre_conflict"] = {
            "micro": pre_conflict_result["micro"],
            "macro": {
                "precision": pre_conflict_result["macro"]["precision"],
                "recall": pre_conflict_result["macro"]["recall"],
                "f1": _macro_f1(pre_conflict_result["macro"]["precision"], pre_conflict_result["macro"]["recall"]),
            },
        }

    # Добавляем метаданные
    comparison["meta"] = {
        "model": config.llm,
        "num_sentences": len(sentences_to_check),
        "class_distribution": dict(class_counts.most_common()),
        "total_gold_relations": sum(len(s.relations) for s in sentences_to_check),
        "ref_resolution": ref_stats,
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.llm.replace(":", "_").replace("/", "_")
    n_sent = limit or len(sentences)
    suffix = f"_{model_name}_{n_sent}sent"
    json_path = out_dir / f"compare_baseline_vs_multiagent{suffix}.json"
    md_path = out_dir / f"compare_baseline_vs_multiagent{suffix}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Сравнение (JSON): {json_path}")

    # Markdown-отчёт
    b, m, d = comparison["baseline"], comparison["multiagent"], comparison["diff"]
    lines = [
        f"# Сравнение: базовый vs мультиагентный",
        f"",
        f"**Модель:** `{config.llm}` | **Предложений:** {len(sentences_to_check)} | "
        f"**Эталонных отношений:** {comparison['meta']['total_gold_relations']}",
        "",
        "## 1. Сводка (micro)",
        "| Метрика | Базовый | Мультиагентный | Разница |",
        "|---------|---------|----------------|---------|",
        f"| Precision | {b['micro']['precision']:.4f} | {m['micro']['precision']:.4f} | {d['micro_precision']:+.4f} |",
        f"| Recall | {b['micro']['recall']:.4f} | {m['micro']['recall']:.4f} | {d['micro_recall']:+.4f} |",
        f"| F1 | {b['micro']['f1']:.4f} | {m['micro']['f1']:.4f} | {d['micro_f1']:+.4f} |",
        f"",
        f"TP={b['micro']['tp']}/{m['micro']['tp']}, "
        f"FP={b['micro']['fp']}/{m['micro']['fp']}, "
        f"FN={b['micro']['fn']}/{m['micro']['fn']} (baseline/multiagent)",
        "",
        "## 2. Macro",
        "| Метрика | Базовый | Мультиагентный | Разница |",
        "|---------|---------|----------------|---------|",
        f"| Precision | {b['macro']['precision']:.4f} | {m['macro']['precision']:.4f} | {d['macro_precision']:+.4f} |",
        f"| Recall | {b['macro']['recall']:.4f} | {m['macro']['recall']:.4f} | {d['macro_recall']:+.4f} |",
        f"| F1 | {b['macro']['f1']:.4f} | {m['macro']['f1']:.4f} | {d['macro_f1']:+.4f} |",
        "",
        "## 3. По типам отношений",
        "| Relation | Support | Baseline P/R/F1 | Multiagent P/R/F1 | Δ F1 |",
        "|----------|---------|-----------------|-------------------|------|",
    ]
    all_rel = sorted(set(b["by_relation"]) | set(m["by_relation"]), key=lambda r: -(b["by_relation"].get(r, {}).get("support", 0) or m["by_relation"].get(r, {}).get("support", 0)))
    for rel_id in all_rel:
        br = b["by_relation"].get(rel_id, {})
        mr = m["by_relation"].get(rel_id, {})
        bp, br_r, bf = br.get("precision", 0), br.get("recall", 0), br.get("f1", 0)
        mp, mr_r, mf = mr.get("precision", 0), mr.get("recall", 0), mr.get("f1", 0)
        support = br.get("support", mr.get("support", 0))
        delta = mf - bf
        lines.append(f"| {rel_id} | {support} | {bp:.2f}/{br_r:.2f}/{bf:.2f} | {mp:.2f}/{mr_r:.2f}/{mf:.2f} | {delta:+.2f} |")

    # --- Распределение классов терминов ---
    lines.extend([
        "",
        "## 4. Распределение классов терминов (gold)",
        "| Класс | Количество |",
        "|-------|------------|",
    ])
    for cls, cnt in class_counts.most_common():
        lines.append(f"| {cls} | {cnt} |")
    lines.append(f"| **Всего** | **{sum(class_counts.values())}** |")

    # --- Граф ---
    if "graph_triples" in comparison:
        gt = comparison["graph_triples"]
        gb, gm, gd = gt.get("baseline", {}), gt.get("multiagent", {}), gt.get("diff", {})
        lines.extend([
            "",
            "## 5. Предсказанные тройки (граф)",
            "| Метрика | Базовый | Мультиагентный | Разница |",
            "|---------|---------|----------------|---------|",
        ])
        if gb and gm and gd:
            lines.append(f"| Precision | {gb.get('precision', 0):.4f} | {gm.get('precision', 0):.4f} | {gd.get('precision', 0):+.4f} |")
            lines.append(f"| Recall | {gb.get('recall', 0):.4f} | {gm.get('recall', 0):.4f} | {gd.get('recall', 0):+.4f} |")
            lines.append(f"| F1 | {gb.get('f1', 0):.4f} | {gm.get('f1', 0):.4f} | {gd.get('f1', 0):+.4f} |")
        lines.extend([
            "",
            f"- Базовый: {gt['baseline_count']} троек",
            f"- Мультиагентный: {gt['multiagent_count']} троек",
            f"- Эталон: {gt['gold_count']} троек",
        ])

    # --- Поэтапное сравнение ---
    lines.extend([
        "",
        "## 6. Поэтапное сравнение (метрики по этапам)",
        "",
        "| Этап | Precision | Recall | F1 | TP | FP | FN |",
        "|------|-----------|--------|-----|----|----|-----|",
        f"| Базовый: извлечение | {b['micro']['precision']:.4f} | {b['micro']['recall']:.4f} | {b['micro']['f1']:.4f} | {b['micro']['tp']} | {b['micro']['fp']} | {b['micro']['fn']} |",
    ])
    if "multiagent_pre_conflict" in comparison:
        pc = comparison["multiagent_pre_conflict"]["micro"]
        lines.append(
            f"| Мультиагент: multi-probe (до конфликтов) | {pc['precision']:.4f} | {pc['recall']:.4f} | {pc['f1']:.4f} | {pc['tp']} | {pc['fp']} | {pc['fn']} |"
        )
    lines.append(
        f"| Мультиагент: после разрешения конфликтов | {m['micro']['precision']:.4f} | {m['micro']['recall']:.4f} | {m['micro']['f1']:.4f} | {m['micro']['tp']} | {m['micro']['fp']} | {m['micro']['fn']} |"
    )
    if "graph_triples" in comparison:
        lines.extend([
            "",
            f"Троек (граф): базовый {comparison['graph_triples']['baseline_count']}, "
            f"мультиагентный {comparison['graph_triples']['multiagent_count']}, "
            f"эталон {comparison['graph_triples']['gold_count']}",
        ])

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))
    print(f"Отчёт (Markdown): {md_path}")
    print()
    print("Разница (multiagent - baseline):")
    print(f"  Micro F1: {d['micro_f1']:+.4f}")
    print(f"  Macro Precision: {d['macro_precision']:+.4f}, Recall: {d['macro_recall']:+.4f}, F1: {d['macro_f1']:+.4f}")
    if "graph_triples" in comparison and "diff" in comparison["graph_triples"]:
        gd = comparison["graph_triples"]["diff"]
        print(f"  Граф (тройки) P/R/F1: {gd['precision']:+.4f} / {gd['recall']:+.4f} / {gd['f1']:+.4f}")


if __name__ == "__main__":
    main()
