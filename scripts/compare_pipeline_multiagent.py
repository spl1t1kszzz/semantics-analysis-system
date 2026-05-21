#!/usr/bin/env python3
"""
Оценка мультиагентного пайплайна и сравнение с зафиксированным baseline.

По умолчанию: один прогон multi-agent, baseline берётся из
benchmarks/frozen_baseline_qwen2.5_7b_100sent.json (не пересчитывается).

Использование (из корня проекта):
  poetry run python scripts/compare_pipeline_multiagent.py [sentences.json] [--limit N]

  --run-both     пересчитать baseline и multi-agent (два прогона, как раньше)
  --frozen PATH  другой frozen baseline JSON (должен содержать ключ "baseline")

Результаты: results/compare_vs_frozen_baseline_{model}_{N}sent.json|.md
"""

import json
import sys
from pathlib import Path
from typing import List, Optional, Dict, Any, Tuple

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))
import os
os.chdir(PROJECT_ROOT)

from rich.progress import Progress

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor

import importlib.util
_spec = importlib.util.spec_from_file_location(
    "eval_kristina",
    PROJECT_ROOT / "scripts" / "evaluate_on_dataset_kristina.py",
)
_eval_mod = importlib.util.module_from_spec(_spec)
_spec.loader.exec_module(_eval_mod)
run_evaluation = _eval_mod.run_evaluation

DEFAULT_FROZEN_BASELINE = (
    PROJECT_ROOT / "benchmarks" / "frozen_baseline_qwen2.5_7b_100sent.json"
)


def load_frozen_baseline(path: Path) -> Tuple[Dict[str, Any], Dict[str, Any]]:
    """Возвращает (result_baseline, frozen_extras) — extras: graph_triples baseline, meta."""
    if not path.is_file():
        raise FileNotFoundError(
            f"Frozen baseline не найден: {path}\n"
            "Ожидается benchmarks/frozen_baseline_qwen2.5_7b_100sent.json"
        )
    with open(path, "r", encoding="utf-8") as f:
        data = json.load(f)
    if "baseline" not in data:
        raise ValueError(f'В {path} нет ключа "baseline"')
    extras = {
        "graph_triples": data.get("graph_triples"),
        "meta": data.get("meta", {}),
        "frozen_source": str(path),
    }
    return data["baseline"], extras


def run_multiagent(
    sentences: List[Sentence],
    relation_extractor,
    reference_resolver,
    progress: Progress,
    limit: Optional[int] = None,
    conflict_resolver=None,
):
    return run_evaluation(
        sentences,
        relation_extractor,
        reference_resolver,
        conflict_resolver=conflict_resolver,
        progress=progress,
        limit=limit,
        collect_triples=True,
    )


def run_both_live(
    sentences: List[Sentence],
    relation_extractor,
    reference_resolver,
    progress: Progress,
    limit: Optional[int] = None,
    multiagent_relation_extractor=None,
    conflict_resolver=None,
):
    result_baseline, triples_baseline, _ = run_evaluation(
        sentences,
        relation_extractor,
        reference_resolver,
        conflict_resolver=None,
        progress=progress,
        limit=limit,
        collect_triples=True,
    )
    ma_extractor = multiagent_relation_extractor or relation_extractor
    result_multiagent, triples_multiagent, pre_conflict_result = run_multiagent(
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
    frozen_graph_baseline: Optional[Dict[str, Any]] = None,
) -> Dict[str, Any]:
    b_macro = result_baseline["macro"]
    m_macro = result_multiagent["macro"]
    b_macro_f1 = b_macro.get("f1") or _macro_f1(b_macro["precision"], b_macro["recall"])
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
            "micro_precision": round(
                result_multiagent["micro"]["precision"] - result_baseline["micro"]["precision"], 4
            ),
            "micro_recall": round(
                result_multiagent["micro"]["recall"] - result_baseline["micro"]["recall"], 4
            ),
            "micro_f1": round(
                result_multiagent["micro"]["f1"] - result_baseline["micro"]["f1"], 4
            ),
            "macro_precision": round(m_macro["precision"] - b_macro["precision"], 4),
            "macro_recall": round(m_macro["recall"] - b_macro["recall"], 4),
            "macro_f1": round(m_macro_f1 - b_macro_f1, 4),
        },
    }

    if triples_multiagent is not None and gold_triples is not None:
        from calculate_graph_metrics import compute_graph_metrics
        metrics_m = compute_graph_metrics(gold_triples, triples_multiagent)
        graph_block = {
            "multiagent_count": len(triples_multiagent),
            "gold_count": len(gold_triples),
            "multiagent": {
                "precision": metrics_m["precision"],
                "recall": metrics_m["recall"],
                "f1": metrics_m["f1"],
            },
        }
        if triples_baseline is not None:
            metrics_b = compute_graph_metrics(gold_triples, triples_baseline)
            graph_block["baseline_count"] = len(triples_baseline)
            graph_block["baseline"] = {
                "precision": metrics_b["precision"],
                "recall": metrics_b["recall"],
                "f1": metrics_b["f1"],
            }
            graph_block["diff"] = {
                "precision": round(metrics_m["precision"] - metrics_b["precision"], 4),
                "recall": round(metrics_m["recall"] - metrics_b["recall"], 4),
                "f1": round(metrics_m["f1"] - metrics_b["f1"], 4),
            }
        elif frozen_graph_baseline:
            gb = frozen_graph_baseline.get("baseline", {})
            graph_block["baseline_count"] = frozen_graph_baseline.get("baseline_count")
            graph_block["baseline"] = gb
            graph_block["diff"] = {
                "precision": round(metrics_m["precision"] - gb.get("precision", 0), 4),
                "recall": round(metrics_m["recall"] - gb.get("recall", 0), 4),
                "f1": round(metrics_m["f1"] - gb.get("f1", 0), 4),
            }
        out["graph_triples"] = graph_block

    return out


def write_comparison_report(
    comparison: Dict[str, Any],
    md_path: Path,
    config_llm: str,
    num_sentences: int,
    baseline_label: str = "Базовый (frozen)",
):
    b, m, d = comparison["baseline"], comparison["multiagent"], comparison["diff"]
    meta = comparison.get("meta", {})
    total_gold = meta.get("total_gold_relations", "—")

    lines = [
        "# Сравнение: frozen baseline vs мультиагентный (текущий прогон)",
        "",
        f"**Baseline:** {baseline_label}  ",
        f"**Модель прогона:** `{config_llm}` | **Предложений:** {num_sentences} | "
        f"**Эталонных отношений:** {total_gold}",
        "",
        "## 1. Сводка (micro)",
        "| Метрика | Baseline | Мультиагентный | Δ |",
        "|---------|----------|----------------|---|",
        f"| Precision | {b['micro']['precision']:.4f} | {m['micro']['precision']:.4f} | {d['micro_precision']:+.4f} |",
        f"| Recall | {b['micro']['recall']:.4f} | {m['micro']['recall']:.4f} | {d['micro_recall']:+.4f} |",
        f"| F1 | {b['micro']['f1']:.4f} | {m['micro']['f1']:.4f} | {d['micro_f1']:+.4f} |",
        "",
        f"TP={b['micro']['tp']}/{m['micro']['tp']}, "
        f"FP={b['micro']['fp']}/{m['micro']['fp']}, "
        f"FN={b['micro']['fn']}/{m['micro']['fn']} (baseline/multiagent)",
        "",
        "## 2. Macro",
        "| Метрика | Baseline | Мультиагентный | Δ |",
        "|---------|----------|----------------|---|",
        f"| Precision | {b['macro']['precision']:.4f} | {m['macro']['precision']:.4f} | {d['macro_precision']:+.4f} |",
        f"| Recall | {b['macro']['recall']:.4f} | {m['macro']['recall']:.4f} | {d['macro_recall']:+.4f} |",
        f"| F1 | {b['macro']['f1']:.4f} | {m['macro']['f1']:.4f} | {d['macro_f1']:+.4f} |",
        "",
        "## 3. По типам отношений",
        "| Relation | Support | Baseline P/R/F1 | Multiagent P/R/F1 | Δ F1 |",
        "|----------|---------|-----------------|-------------------|------|",
    ]
    all_rel = sorted(
        set(b["by_relation"]) | set(m["by_relation"]),
        key=lambda r: -(
            b["by_relation"].get(r, {}).get("support", 0)
            or m["by_relation"].get(r, {}).get("support", 0)
        ),
    )
    for rel_id in all_rel:
        br = b["by_relation"].get(rel_id, {})
        mr = m["by_relation"].get(rel_id, {})
        bp, br_r, bf = br.get("precision", 0), br.get("recall", 0), br.get("f1", 0)
        mp, mr_r, mf = mr.get("precision", 0), mr.get("recall", 0), mr.get("f1", 0)
        support = br.get("support", mr.get("support", 0))
        lines.append(
            f"| {rel_id} | {support} | {bp:.2f}/{br_r:.2f}/{bf:.2f} | "
            f"{mp:.2f}/{mr_r:.2f}/{mf:.2f} | {mf - bf:+.2f} |"
        )

    if "graph_triples" in comparison:
        gt = comparison["graph_triples"]
        gb, gm, gd = gt.get("baseline", {}), gt.get("multiagent", {}), gt.get("diff", {})
        lines.extend([
            "",
            "## 4. Предсказанные тройки (граф)",
            "| Метрика | Baseline | Мультиагентный | Δ |",
            "|---------|----------|----------------|---|",
        ])
        if gb and gm and gd:
            lines.append(
                f"| Precision | {gb.get('precision', 0):.4f} | {gm.get('precision', 0):.4f} | "
                f"{gd.get('precision', 0):+.4f} |"
            )
            lines.append(
                f"| Recall | {gb.get('recall', 0):.4f} | {gm.get('recall', 0):.4f} | "
                f"{gd.get('recall', 0):+.4f} |"
            )
            lines.append(
                f"| F1 | {gb.get('f1', 0):.4f} | {gm.get('f1', 0):.4f} | {gd.get('f1', 0):+.4f} |"
            )
        lines.extend([
            "",
            f"- Baseline (frozen): {gt.get('baseline_count', '—')} троек",
            f"- Мультиагентный (прогон): {gt.get('multiagent_count', '—')} троек",
            f"- Эталон: {gt.get('gold_count', '—')} троек",
        ])

    lines.extend([
        "",
        "## 5. Поэтапное сравнение",
        "",
        "| Этап | Precision | Recall | F1 | TP | FP | FN |",
        "|------|-----------|--------|-----|----|----|-----|",
        f"| Baseline (frozen) | {b['micro']['precision']:.4f} | {b['micro']['recall']:.4f} | "
        f"{b['micro']['f1']:.4f} | {b['micro']['tp']} | {b['micro']['fp']} | {b['micro']['fn']} |",
    ])
    if "multiagent_pre_conflict" in comparison:
        pc = comparison["multiagent_pre_conflict"]["micro"]
        lines.append(
            f"| Multi-agent: до конфликтов | {pc['precision']:.4f} | {pc['recall']:.4f} | "
            f"{pc['f1']:.4f} | {pc['tp']} | {pc['fp']} | {pc['fn']} |"
        )
    lines.append(
        f"| Multi-agent: после конфликтов | {m['micro']['precision']:.4f} | {m['micro']['recall']:.4f} | "
        f"{m['micro']['f1']:.4f} | {m['micro']['tp']} | {m['micro']['fp']} | {m['micro']['fn']} |"
    )

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))


def _make_conflict_resolver(config, multiagent_extractor):
    from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver
    from semantics_analysis.llm_agent import LLMAgent

    llm_agent = LLMAgent(model=config.llm)
    cr = RelationConflictResolver(
        llm_agent=llm_agent,
        log_prompts=config.log_prompts,
        log_responses=config.log_llm_responses,
        use_dialogue=config.use_conflict_dialogue,
    )
    if config.use_reverify_after_resolve:
        cr.reverify_callback = multiagent_extractor.verify_relation
    return cr


def parse_args(argv: List[str]):
    run_both = "--run-both" in argv
    frozen_path = DEFAULT_FROZEN_BASELINE
    limit = None
    sentences_path = PROJECT_ROOT / "tests" / "dataset_kristina_sentences.json"

    args = [a for a in argv if a not in ("--run-both", "--multiagent-only")]
    i = 0
    while i < len(args):
        if args[i] == "--limit" and i + 1 < len(args):
            try:
                limit = int(args[i + 1])
            except ValueError:
                pass
            i += 2
            continue
        if args[i] == "--frozen" and i + 1 < len(args):
            frozen_path = Path(args[i + 1])
            i += 2
            continue
        if args[i].endswith(".json") and "frozen" not in args[i].lower():
            sentences_path = Path(args[i])
        i += 1

    return run_both, frozen_path, limit, sentences_path


def main():
    run_both, frozen_path, limit, sentences_path = parse_args(sys.argv[1:])

    if not sentences_path.is_file():
        print(f"Файл не найден: {sentences_path}")
        sys.exit(1)

    multiagent_config_path = PROJECT_ROOT / "config.multiagent.yml"
    config = load_config(
        multiagent_config_path if multiagent_config_path.is_file() else PROJECT_ROOT / "config.yml"
    )

    sentences = read_sentences(str(sentences_path))
    print(f"Загружено предложений: {len(sentences)}")
    if limit:
        print(f"Ограничение: первые {limit} предложений")

    sentences_to_check = sentences[:limit] if limit else sentences
    gold_triples = set(
        (r.term1.value, r.predicate, r.term2.value)
        for s in sentences_to_check
        for r in s.relations
    )

    if run_both:
        print("Режим --run-both: пересчёт baseline и multi-agent (два прогона).")
        baseline_config = load_config(PROJECT_ROOT / "config.baseline.yml")
        with Progress() as progress:
            ref_resolver = LLMReferenceResolver(progress=progress, model=baseline_config.llm)
            baseline_extractor = LLMRelationExtractor(model=baseline_config.llm, use_multi_probe=False)
            multiagent_extractor = LLMRelationExtractor(model=config.llm, use_multi_probe=True)
            conflict_resolver = _make_conflict_resolver(config, multiagent_extractor)
            result_baseline, result_multiagent, triples_baseline, triples_multiagent, pre_conflict_result = (
                run_both_live(
                    sentences, baseline_extractor, ref_resolver, progress, limit=limit,
                    multiagent_relation_extractor=multiagent_extractor,
                    conflict_resolver=conflict_resolver,
                )
            )
        frozen_label = "Базовый (live)"
        frozen_graph = None
    else:
        print(f"Режим: только multi-agent; baseline из {frozen_path}")
        if limit and limit != 100 and "100sent" in frozen_path.name:
            print(
                f"  Предупреждение: frozen baseline для 100 предложений, а --limit {limit}. "
                "Сравнение relation-level может быть некорректным; для графа используется текущий gold."
            )
        result_baseline, frozen_extras = load_frozen_baseline(frozen_path)
        frozen_label = f"frozen: {frozen_path.name}"
        frozen_graph = frozen_extras.get("graph_triples")

        with Progress() as progress:
            ref_resolver = LLMReferenceResolver(progress=progress, model=config.llm)
            multiagent_extractor = LLMRelationExtractor(model=config.llm, use_multi_probe=True)
            conflict_resolver = _make_conflict_resolver(config, multiagent_extractor)
            result_multiagent, triples_multiagent, pre_conflict_result = run_multiagent(
                sentences, multiagent_extractor, ref_resolver, progress, limit=limit,
                conflict_resolver=conflict_resolver,
            )
        triples_baseline = None

    comparison = build_comparison(
        result_baseline,
        result_multiagent,
        gold_triples=gold_triples,
        triples_baseline=triples_baseline,
        triples_multiagent=triples_multiagent,
        frozen_graph_baseline=frozen_graph,
    )

    if pre_conflict_result:
        comparison["multiagent_pre_conflict"] = {
            "micro": pre_conflict_result["micro"],
            "macro": {
                "precision": pre_conflict_result["macro"]["precision"],
                "recall": pre_conflict_result["macro"]["recall"],
                "f1": _macro_f1(
                    pre_conflict_result["macro"]["precision"],
                    pre_conflict_result["macro"]["recall"],
                ),
            },
        }

    from collections import Counter
    class_counts = Counter()
    for s in sentences_to_check:
        for term in s.terms:
            class_counts[term.class_] += 1

    comparison["meta"] = {
        "model": config.llm,
        "num_sentences": len(sentences_to_check),
        "class_distribution": dict(class_counts.most_common()),
        "total_gold_relations": sum(len(s.relations) for s in sentences_to_check),
        "frozen_baseline": str(frozen_path) if not run_both else None,
        "run_mode": "run_both" if run_both else "multiagent_vs_frozen",
    }

    out_dir = PROJECT_ROOT / "results"
    out_dir.mkdir(parents=True, exist_ok=True)
    model_name = config.llm.replace(":", "_").replace("/", "_")
    n_sent = limit or len(sentences)
    suffix = f"_{model_name}_{n_sent}sent"
    json_path = out_dir / f"compare_vs_frozen_baseline{suffix}.json"
    md_path = out_dir / f"compare_vs_frozen_baseline{suffix}.md"

    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(comparison, f, ensure_ascii=False, indent=2)
    print(f"Сравнение (JSON): {json_path}")

    write_comparison_report(
        comparison, md_path, config.llm, len(sentences_to_check), baseline_label=frozen_label
    )
    print(f"Отчёт (Markdown): {md_path}")

    d = comparison["diff"]
    m = comparison["multiagent"]["micro"]
    b = comparison["baseline"]["micro"]
    print()
    print(f"Baseline micro F1: {b['f1']:.4f}  →  Multi-agent: {m['f1']:.4f}  (Δ {d['micro_f1']:+.4f})")
    if "graph_triples" in comparison and "diff" in comparison["graph_triples"]:
        gd = comparison["graph_triples"]["diff"]
        print(f"Graph triples F1 Δ: {gd.get('f1', 0):+.4f}")


if __name__ == "__main__":
    main()
