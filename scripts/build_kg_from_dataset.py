#!/usr/bin/env python3
"""
Построение графа знаний по датасету в мультиагентном режиме.

Полный пайплайн: RoBERTa + LLM (verify, normalize, reference resolution),
multi-probe извлечение отношений, разрешение конфликтов.

Результаты в каталоге output (по умолчанию results/kg_<N>sent/):
  - ont_entities.json — объекты и связи онтологии
  - knowledge_graph.html — интерактивный граф (pyvis)
  - knowledge_graph.png — статичная картинка (matplotlib)

Использование (из корня проекта):
  poetry run python scripts/build_kg_from_dataset.py
  poetry run python scripts/build_kg_from_dataset.py tests/dataset_kristina_sentences.json --limit 100
  poetry run python scripts/build_kg_from_dataset.py --limit 10 --output-dir results/kg_demo
"""

from __future__ import annotations

import argparse
import json
import math
import sys
from pathlib import Path
from typing import List, Optional

PROJECT_ROOT = Path(__file__).resolve().parent.parent
sys.path.insert(0, str(PROJECT_ROOT))

import os

os.chdir(PROJECT_ROOT)

import matplotlib.pyplot as plt
from rich.progress import Progress

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence, Relation, Term
from semantics_analysis.factory import build_pipeline
from semantics_analysis.knowledge_graph import build_knowledge_graph
from semantics_analysis.pipelines import AnalysisResult
from semantics_analysis.utils import union_term_mentions
from semantics_analysis.visualization import display_relation_graph, get_color


DEFAULT_SENTENCES = PROJECT_ROOT / "tests" / "dataset_kristina_sentences.json"
DEFAULT_MA_CONFIG = PROJECT_ROOT / "config.multiagent.yml"
FALLBACK_CONFIG = PROJECT_ROOT / "config.yml"


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Мультиагентный пайплайн + построение графа знаний по датасету.",
    )
    parser.add_argument(
        "sentences",
        nargs="?",
        default=str(DEFAULT_SENTENCES),
        help="JSON с предложениями (по умолчанию tests/dataset_kristina_sentences.json)",
    )
    parser.add_argument(
        "--limit",
        type=int,
        default=None,
        help="Обработать только первые N предложений",
    )
    parser.add_argument(
        "--output-dir",
        type=str,
        default=None,
        help="Каталог для результатов (по умолчанию results/kg_<N>sent/)",
    )
    parser.add_argument(
        "--config",
        type=str,
        default=None,
        help="YAML-конфиг (по умолчанию config.multiagent.yml)",
    )
    parser.add_argument(
        "--no-png",
        action="store_true",
        help="Не сохранять PNG (только HTML и JSON)",
    )
    return parser.parse_args()


def resolve_config_path(config_arg: Optional[str]) -> Path:
    if config_arg:
        path = Path(config_arg)
        if path.is_file():
            return path
        print(f"Конфиг не найден: {path}, используется {DEFAULT_MA_CONFIG.name}")
    if DEFAULT_MA_CONFIG.is_file():
        return DEFAULT_MA_CONFIG
    return FALLBACK_CONFIG


def load_ma_config(config_path: Path):
    config = load_config(config_path)
    if not config.use_multi_agent:
        print(
            f"Предупреждение: в {config_path.name} use-multi-agent=false; "
            "multi-probe и conflict resolver могут быть отключены."
        )
    return config


def run_pipeline_on_dataset(
    sentences: List[Sentence],
    limit: Optional[int],
    progress: Progress,
    config,
) -> tuple[List[Term], List[Relation], int]:
    pipeline = build_pipeline(config, progress)

    to_process = sentences[:limit] if limit else sentences
    all_terms: List[Term] = []
    all_relations: List[Relation] = []
    processed = 0

    task = progress.add_task("Предложения", total=len(to_process))

    for sent in to_process:
        result = pipeline(AnalysisResult(text=sent.text))
        all_terms.extend(result.terms)
        all_relations.extend(result.relations)
        processed += 1
        progress.update(task, advance=1, description=f"Предложение {processed}/{len(to_process)}")

    progress.remove_task(task)
    return all_terms, all_relations, processed


def merge_corpus(terms: List[Term], relations: List[Relation]) -> tuple[List[Term], List[Relation]]:
    merged_terms = union_term_mentions(terms) if terms else []
    merged_relations = list(set(relations))
    return merged_terms, merged_relations


def export_kg_png(relations: List[Relation], output_path: Path, figsize=(16, 12), dpi=150) -> None:
    """Статичная картинка графа (круговая раскладка, matplotlib)."""
    if not relations:
        return

    node_keys: List[tuple[str, str]] = []
    node_labels: dict[tuple[str, str], str] = {}
    node_colors: dict[tuple[str, str], str] = {}

    for rel in relations:
        for term in (rel.term1, rel.term2):
            key = (term.class_, term.value)
            if key not in node_labels:
                node_keys.append(key)
                node_labels[key] = term.value
                node_colors[key] = get_color(term.class_)

    n = len(node_keys)
    pos = {}
    for i, key in enumerate(node_keys):
        angle = 2 * math.pi * i / n
        pos[key] = (math.cos(angle), math.sin(angle))

    fig, ax = plt.subplots(figsize=figsize)
    ax.set_aspect("equal")
    ax.axis("off")

    seen_edges: set[tuple[tuple[str, str], tuple[str, str], str]] = set()
    for rel in relations:
        k1 = (rel.term1.class_, rel.term1.value)
        k2 = (rel.term2.class_, rel.term2.value)
        edge_key = (k1, k2, rel.predicate)
        if edge_key in seen_edges:
            continue
        seen_edges.add(edge_key)

        x1, y1 = pos[k1]
        x2, y2 = pos[k2]
        ax.annotate(
            "",
            xy=(x2, y2),
            xytext=(x1, y1),
            arrowprops=dict(arrowstyle="-|>", color="#888888", lw=0.9, shrinkA=12, shrinkB=12),
        )
        mx, my = (x1 + x2) / 2, (y1 + y2) / 2
        ax.text(mx, my, rel.predicate, fontsize=5, ha="center", va="center", color="#555555")

    for key in node_keys:
        x, y = pos[key]
        color = node_colors[key]
        ax.plot(x, y, "o", markersize=14, color=color, markeredgecolor="#333333", markeredgewidth=0.6)
        label = node_labels[key]
        if len(label) > 28:
            label = label[:25] + "..."
        ax.text(x, y + 0.06, label, fontsize=7, ha="center", va="bottom", wrap=True)

    ax.set_title(f"Граф знаний ({len(node_keys)} узлов, {len(seen_edges)} рёбер)", fontsize=12)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight", facecolor="white")
    plt.close(fig)


def default_output_dir(limit: Optional[int], processed: int) -> Path:
    n = limit if limit else processed
    return PROJECT_ROOT / "results" / f"kg_{n}sent"


def main() -> None:
    args = parse_args()
    sentences_path = Path(args.sentences)
    if not sentences_path.is_file():
        print(f"Файл не найден: {sentences_path}")
        print("Сначала: poetry run python scripts/convert_dataset_kristina_to_sentences.py")
        sys.exit(1)

    config_path = resolve_config_path(args.config)
    config = load_ma_config(config_path)
    print(f"Конфиг: {config_path.name}")
    print(f"LLM: {config.llm}")
    print(f"multi-probe: {config.use_multi_agent}")
    print(f"conflict resolver: {config.use_multi_agent}")
    print(f"conflict dialogue: {config.use_conflict_dialogue}")
    print(f"re-verify: {config.use_reverify_after_resolve}")

    sentences = read_sentences(str(sentences_path))
    print(f"Загружено предложений: {len(sentences)}")
    if args.limit:
        print(f"Ограничение: первые {args.limit}")

    with Progress() as progress:
        terms, relations, processed = run_pipeline_on_dataset(
            sentences, args.limit, progress, config
        )

    terms, relations = merge_corpus(terms, relations)
    print(f"Обработано предложений: {processed}")
    print(f"Терминов (после объединения): {len(terms)}")
    print(f"Отношений (уникальных): {len(relations)}")

    out_dir = Path(args.output_dir) if args.output_dir else default_output_dir(args.limit, processed)
    out_dir.mkdir(parents=True, exist_ok=True)

    kg_json = build_knowledge_graph(
        terms, relations, deduplicate=True, llm_model=config.llm
    )
    kg_json["meta"] = {
        "sentences_file": str(sentences_path),
        "sentences_processed": processed,
        "limit": args.limit,
        "llm": config.llm,
        "multi_agent": config.use_multi_agent,
        "objects_count": len(kg_json.get("objects", [])),
        "relations_count": len(kg_json.get("relations", [])),
    }

    json_path = out_dir / "ont_entities.json"
    with open(json_path, "w", encoding="utf-8") as f:
        json.dump(kg_json, f, ensure_ascii=False, indent=2)
    print(f"Онтология (JSON): {json_path}")

    html_path = out_dir / "knowledge_graph.html"
    if relations:
        display_relation_graph(terms, relations, output_file=str(html_path))
        print(f"Интерактивный граф (HTML): {html_path}")
    else:
        print("Отношений нет — HTML-граф не построен.")

    if not args.no_png and relations:
        png_path = out_dir / "knowledge_graph.png"
        export_kg_png(relations, png_path)
        print(f"Картинка (PNG): {png_path}")

    summary_path = out_dir / "build_summary.json"
    with open(summary_path, "w", encoding="utf-8") as f:
        json.dump(
            {
                "terms": len(terms),
                "relations": len(relations),
                "objects": len(kg_json.get("objects", [])),
                "ont_relations": len(kg_json.get("relations", [])),
                "output_files": {
                    "json": str(json_path),
                    "html": str(html_path) if relations else None,
                    "png": str(out_dir / "knowledge_graph.png") if relations and not args.no_png else None,
                },
            },
            f,
            ensure_ascii=False,
            indent=2,
        )
    print(f"Сводка: {summary_path}")


if __name__ == "__main__":
    main()
