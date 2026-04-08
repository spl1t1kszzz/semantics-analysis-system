"""
Оркестратор мультиагентного пайплайна.

Последовательность агентов:
1) Извлечение терминов (Extractor)
2) Верификация терминов (Verifier)
3) Нормализация, референция (в т.ч. идентификация для ГЗ)
4) Извлечение отношений (Extractor)
5) Разрешение конфликтов между отношениями (ConflictResolver) — опционально
6) Построение графа знаний (KG Integrator)
"""

from typing import Optional, List, Any, Dict

from rich.progress import Progress

from semantics_analysis.entities import TermMention, Term, Relation
from semantics_analysis.pipelines import (
    AnalysisResult,
    SequencePipeline,
    PredictTerms,
    PreprocessTerms,
    VerifyTerms,
    NormalizeTerms,
    DropEmptyTermMentions,
    NormalizeLanguages,
    ResolveReference,
    PredictSemanticRelations,
    ResolveRelationConflicts,
)
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor
from semantics_analysis.term_extraction.term_verifier import TermVerifier
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor
from semantics_analysis.term_post_processing.computer_science_term_post_processor import (
    ResolveLibraries,
)
from semantics_analysis.term_post_processing.merge_close_term_post_processor import (
    MergeCloseTerms,
)

from semantics_analysis.multi_agent.base import SharedState
from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver


class MultiAgentOrchestrator:
    """
    Запускает полный пайплайн извлечения с опциональным шагом
    разрешения конфликтов между отношениями (мультиагентный подход).
    """

    def __init__(
        self,
        term_mention_extractor: TermMentionExtractor,
        term_verifier: TermVerifier,
        term_normalizer: TermNormalizer,
        reference_resolver: ReferenceResolver,
        relation_extractor: RelationExtractor,
        progress: Progress,
        postprocessors: Optional[tuple] = None,
        use_conflict_resolution: bool = True,
        conflict_resolver: Optional[RelationConflictResolver] = None,
        use_conflict_dialogue: bool = False,
        use_reverify_after_resolve: bool = False,
    ):
        self.term_mention_extractor = term_mention_extractor
        self.term_verifier = term_verifier
        self.term_normalizer = term_normalizer
        self.reference_resolver = reference_resolver
        self.relation_extractor = relation_extractor
        self.progress = progress
        self._postprocessors = postprocessors if postprocessors is not None else (ResolveLibraries(), MergeCloseTerms())
        self.use_conflict_resolution = use_conflict_resolution
        reverify_callback = None
        if use_reverify_after_resolve and hasattr(relation_extractor, "verify_relation"):
            reverify_callback = relation_extractor.verify_relation
        self.conflict_resolver = conflict_resolver or RelationConflictResolver(
            use_dialogue=use_conflict_dialogue,
            reverify_callback=reverify_callback,
        )
        if conflict_resolver is not None:
            if use_conflict_dialogue:
                self.conflict_resolver.use_dialogue = True
            if use_reverify_after_resolve and hasattr(relation_extractor, "verify_relation"):
                self.conflict_resolver.reverify_callback = relation_extractor.verify_relation

    def run(self, text: str) -> SharedState:
        """Выполняет пайплайн над текстом и возвращает общее состояние (термины, отношения)."""
        state = AnalysisResult(text=text)

        steps = [
            PredictTerms(self.term_mention_extractor),
            PreprocessTerms(*self._postprocessors),
            VerifyTerms(self.term_verifier, self.progress),
            NormalizeTerms(self.term_normalizer, self.progress),
            DropEmptyTermMentions(),
            NormalizeLanguages(),
            ResolveReference(self.reference_resolver, self.progress),
            PredictSemanticRelations(self.relation_extractor, self.progress),
        ]
        if self.use_conflict_resolution:
            steps.append(ResolveRelationConflicts(self.conflict_resolver, self.progress))

        pipeline = SequencePipeline(*steps, progress=self.progress)
        result = pipeline(state)

        return SharedState.from_analysis_result(result)

    def run_and_build_kg(
        self,
        text: str,
        kg_builder=None,
    ) -> tuple[SharedState, Dict[str, Any]]:
        """
        Запускает пайплайн и строит граф знаний (objects + ont_relations).
        Возвращает (SharedState, ont_entities_dict с ключами 'objects', 'relations').
        """
        state = self.run(text)
        if kg_builder is not None:
            ont = kg_builder(state.terms, state.relations)
        else:
            from ontology_entities import convert_to_ont_entities
            objects, ont_relations = convert_to_ont_entities(state.terms, state.relations)
            ont = {
                "objects": [o.to_json() for o in objects],
                "relations": [r.to_json() for r in ont_relations],
            }
        return state, ont


