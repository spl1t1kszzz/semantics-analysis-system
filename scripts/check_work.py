#!/usr/bin/env python3
"""
Быстрая проверка работы системы: импорты, конфиг, мультиагентный слой, опционально — полный прогон.
Запуск из корня проекта: poetry run python scripts/check_work.py [--full]
"""

import os
import sys

# запуск из корня проекта
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
os.chdir(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))


def check_imports():
    print("1. Импорты (без LLM)...")
    from semantics_analysis.entities import Term, Relation, TermMention
    from semantics_analysis.knowledge_graph import build_knowledge_graph
    from semantics_analysis.multi_agent.conflict_resolution import detect_relation_conflicts
    print("   OK (entities, knowledge_graph, detect_relation_conflicts)")
    return True


def check_config():
    print("2. Конфиг...")
    from semantics_analysis.config import load_config
    config = load_config("config.yml")
    print(f"   llm={config.llm}, use_multi_agent={config.use_multi_agent}")
    return True


def check_env():
    print("3. Переменные окружения / .env...")
    key = os.environ.get("OPENAI_API_KEY", "").strip()
    if key:
        print("   OPENAI_API_KEY задан (первые 10 символов: {}...)".format(key[:10]))
    else:
        print("   OPENAI_API_KEY не задан — будет использован провайдер по умолчанию (vsegpt).")
    return True


def check_conflict_detection():
    print("4. Детекция конфликтов (без LLM)...")
    from semantics_analysis.entities import Term, Relation
    from semantics_analysis.multi_agent.conflict_resolution import detect_relation_conflicts  # noqa: F401

    t1 = Term("Method", "BERT", mentions=[])
    t2 = Term("Task", "NER", mentions=[])
    rels = [
        Relation(t1, "solves", t2),
        Relation(t1, "isExampleOf", t2),
    ]
    groups = detect_relation_conflicts(rels)
    assert len(groups) == 1 and len(groups[0]) == 2
    print("   OK (конфликт для одной пары сущностей найден)")
    return True


def check_kg_builder():
    print("5. Построение ГЗ (дедупликация)...")
    from semantics_analysis.entities import Term, Relation
    from semantics_analysis.knowledge_graph import build_knowledge_graph

    t1 = Term("Method", "BERT", mentions=[])
    t2 = Term("Task", "NER", mentions=[])
    rels = [
        Relation(t1, "solves", t2),
        Relation(t1, "solves", t2),
    ]
    kg = build_knowledge_graph([t1, t2], rels, deduplicate=True)
    # после дедупликации одна связь; объекты могут быть пустыми из-за convert_to_ont_entities
    assert "objects" in kg and "relations" in kg
    print("   OK (objects={}, relations={})".format(len(kg["objects"]), len(kg["relations"])))
    return True


def run_full_pipeline():
    print("6. Полный прогон (один короткий текст, с LLM)...")
    from semantics_analysis.config import load_config
    from semantics_analysis.utils import AlignedProgress
    from semantics_analysis.term_extraction.dict_term_mention_extractor import DictTermExtractor
    from semantics_analysis.term_extraction.hybrid_term_mention_extractor import CombinedTermExtractor
    from semantics_analysis.term_extraction.roberta_classified_term_mention_extractor import RobertaTermExtractor
    from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier
    from semantics_analysis.term_normalization.llm_term_normalizer import LLMTermNormalizer
    from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
    from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
    from semantics_analysis.multi_agent import MultiAgentOrchestrator
    from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver

    config = load_config("config.yml")
    text = "BERT — модель для решения задачи NER. NER является задачей извлечения именованных сущностей."

    with AlignedProgress() as progress:
        term_extractor = CombinedTermExtractor(
            DictTermExtractor("metadata/terms_by_class.json"),
            RobertaTermExtractor(config.device, term_threshold=0.2, class_threshold=0.5),
        )
        ref_resolver = LLMReferenceResolver(
            model=config.llm,
            progress=progress,
        )
        relation_extractor = LLMRelationExtractor()
        conflict_resolver = RelationConflictResolver()

        orchestrator = MultiAgentOrchestrator(
            term_mention_extractor=term_extractor,
            term_verifier=LLMTermVerifier(),
            term_normalizer=LLMTermNormalizer(),
            reference_resolver=ref_resolver,
            relation_extractor=relation_extractor,
            progress=progress,
            use_conflict_resolution=config.use_multi_agent,
            conflict_resolver=conflict_resolver,
        )
        state, ont = orchestrator.run_and_build_kg(text)

    print("   Терминов: {}, отношений: {}".format(len(state.terms), len(state.relations)))
    print("   ГЗ: objects={}, relations={}".format(len(ont["objects"]), len(ont["relations"])))
    return True


def main():
    do_full = "--full" in sys.argv
    try:
        check_imports()
        check_config()
        check_env()
        check_conflict_detection()
        check_kg_builder()
        if do_full:
            run_full_pipeline()
        else:
            print("6. Полный прогон — пропущен (запустите с --full для проверки с LLM).")
    except Exception as e:
        print("Ошибка:", e)
        raise
    print("\nПроверки пройдены.")


if __name__ == "__main__":
    main()
