"""
Построение графа знаний: дедупликация отношений и вызов конвертации в онтологические сущности.
"""

from typing import List, Dict, Any, Tuple

from semantics_analysis.entities import Term, Relation


def _relation_key(rel: Relation) -> Tuple[str, str, str, str, str]:
    """Ключ для дедупликации: (class1, value1, class2, value2, predicate)."""
    t1, t2 = rel.term1, rel.term2
    return (
        t1.class_, t1.value.strip().lower(),
        t2.class_, t2.value.strip().lower(),
        rel.predicate,
    )


def deduplicate_relations(relations: List[Relation]) -> List[Relation]:
    """Убирает дубликаты отношений (одинаковая пара сущностей и предикат)."""
    seen = set()
    result = []
    for rel in relations:
        key = _relation_key(rel)
        if key in seen:
            continue
        seen.add(key)
        result.append(rel)
    return result


def build_knowledge_graph(
    terms: List[Term],
    relations: List[Relation],
    deduplicate: bool = True,
) -> Dict[str, Any]:
    """
    Строит граф знаний (objects + ont_relations) из терминов и отношений.

    - Если deduplicate=True, перед конвертацией удаляются дубликаты отношений.
    - Ожидается, что отношения уже прошли этап разрешения конфликтов (мультиагентный пайплайн).

    Возвращает словарь с ключами 'objects' и 'relations' (JSON-сериализуемые).
    """
    if deduplicate:
        relations = deduplicate_relations(relations)

    from ontology_entities import convert_to_ont_entities

    objects, ont_relations = convert_to_ont_entities(terms, relations)
    return {
        "objects": [o.to_json() for o in objects],
        "relations": [r.to_json() for r in ont_relations],
    }
