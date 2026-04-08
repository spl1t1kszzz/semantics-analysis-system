"""
Базовые типы для мультиагентной системы: роли агентов и общее состояние.
"""

from dataclasses import dataclass, field
from enum import Enum
from typing import List, Optional

from semantics_analysis.entities import (
    Term,
    TermMention,
    Relation,
)


class AgentRole(str, Enum):
    """Роли агентов в пайплайне извлечения и построения ГЗ."""
    EXTRACTOR = "extractor"           # извлечение терминов и кандидатов отношений
    VERIFIER = "verifier"             # верификация терминов и отношений
    CONFLICT_RESOLVER = "conflict_resolver"  # поиск и разрешение конфликтов
    KG_INTEGRATOR = "kg_integrator"   # добавление в ГЗ, идентификация, референция


@dataclass
class SharedState:
    """
    Общее состояние, передаваемое между агентами.
    Соответствует AnalysisResult, расширено для мультиагентного сценария.
    """
    text: str
    term_mentions: List[TermMention] = field(default_factory=list)
    terms: List[Term] = field(default_factory=list)
    relations: List[Relation] = field(default_factory=list)
    # Метаданные для отладки и оценки
    conflicts_detected: int = 0
    conflicts_resolved: int = 0

    def to_analysis_result(self):
        """Приведение к формату AnalysisResult для совместимости с существующим пайплайном."""
        from semantics_analysis.pipelines import AnalysisResult
        return AnalysisResult(
            text=self.text,
            term_mentions=self.term_mentions,
            terms=self.terms,
            relations=self.relations,
        )

    @classmethod
    def from_analysis_result(cls, result):
        """Создание SharedState из AnalysisResult."""
        return cls(
            text=result.text,
            term_mentions=getattr(result, "term_mentions", []) or [],
            terms=getattr(result, "terms", []) or [],
            relations=getattr(result, "relations", []) or [],
        )
