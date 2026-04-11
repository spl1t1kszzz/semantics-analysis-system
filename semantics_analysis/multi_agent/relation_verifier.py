"""
Агент верификации отношений.

Проверяет каждое извлечённое отношение через независимый LLM-вызов.
В отличие от базового пайплайна (который верифицирует только same-class пары),
этот агент верифицирует все отношения, снижая FP.
"""

from typing import List, Optional, Callable, TYPE_CHECKING

from semantics_analysis.entities import Relation

if TYPE_CHECKING:
    from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor


class RelationVerifier:
    """
    Мультиагентный верификатор: проверяет каждое отношение
    через verify_relation callback.
    """

    def __init__(
        self,
        verify_callback: Callable[[str, "Relation"], bool],
        log_responses: bool = False,
    ):
        self.verify_callback = verify_callback
        self.log_responses = log_responses

    def verify_all(self, text: str, relations: List[Relation]) -> List[Relation]:
        """Проверяет каждое отношение, возвращает только подтверждённые."""
        verified = []
        for rel in relations:
            # same-class пары уже верифицированы на этапе извлечения — пропускаем
            if rel.term1.class_ == rel.term2.class_:
                verified.append(rel)
                continue

            if self.verify_callback(text, rel):
                verified.append(rel)
            elif self.log_responses:
                from semantics_analysis.utils import log
                log(f"[VERIFY REJECT] {rel.term1.value} --{rel.predicate}--> {rel.term2.value}\n")

        return verified
