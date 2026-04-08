"""
Мультиагентный подход к извлечению информации и построению графа знаний.

Агенты:
- Извлечение терминов и отношений (базовое извлечение).
- Верификация (самопроверка, уточнение).
- Разрешение конфликтов (диалог, выбор наилучшего варианта).
- Интеграция в граф знаний (идентификация сущностей, разрешение референции).
"""

from semantics_analysis.multi_agent.base import AgentRole, SharedState
from semantics_analysis.multi_agent.conflict_resolution import (
    detect_relation_conflicts,
    RelationConflictResolver,
)

# Ленивый импорт: оркестратор тянет rich, pipelines и др. — не грузим при импорте пакета.
def __getattr__(name: str):
    if name == "MultiAgentOrchestrator":
        from semantics_analysis.multi_agent.orchestrator import MultiAgentOrchestrator
        return MultiAgentOrchestrator
    raise AttributeError(f"module {__name__!r} has no attribute {name!r}")


__all__ = [
    "AgentRole",
    "SharedState",
    "detect_relation_conflicts",
    "RelationConflictResolver",
    "MultiAgentOrchestrator",
]
