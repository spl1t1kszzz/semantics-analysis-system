from abc import ABC, abstractmethod
from typing import List, Optional

from colorama import Style
from rich.progress import Progress
from rich.spinner import Spinner

from semantics_analysis.entities import TermMention, Relation, Term, BoundedIterator
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor
from semantics_analysis.term_extraction.term_verifier import TermVerifier
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor
from semantics_analysis.utils import log_iterations, log_labeled_terms, log, LOG_STYLE, log_grouped_terms


class AnalysisResult:
    text: str
    term_mentions: Optional[List[TermMention]]
    terms: Optional[List[Term]]
    relations: Optional[List[Relation]]

    def __init__(
            self,
            text: str,
            term_mentions: Optional[List[TermMention]] = None,
            terms: Optional[List[Term]] = None,
            relations: Optional[List[Relation]] = None
    ):
        self.text = text
        self.term_mentions = term_mentions if term_mentions else []
        self.terms = terms if terms else []
        self.relations = relations if relations else []


class Pipeline(ABC):
    @abstractmethod
    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        raise NotImplemented('Abstract method.')


class SequencePipeline(Pipeline):

    def __init__(self, *pipelines: Pipeline, progress: Progress):
        self.pipelines = pipelines
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        for idx, pipeline in enumerate(self.pipelines):

            state = pipeline(state)

        return state


class PredictTerms(Pipeline):

    def __init__(self, term_mention_extractor: TermMentionExtractor):
        self.term_mention_extractor = term_mention_extractor

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.text:
            return state

        mentions = self.term_mention_extractor(state.text)

        state.term_mentions = mentions
        return state


class PreprocessTerms(Pipeline):

    def __init__(self, *term_postprocessors: TermPostProcessor):
        self.term_postprocessor = term_postprocessors

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.term_mentions:
            return state

        for postprocessor in self.term_postprocessor:
            state.term_mentions = postprocessor(state.term_mentions)

        return state


class VerifyTerms(Pipeline):

    def __init__(self, term_verifier: TermVerifier, progress: Progress):
        self.term_verifier = term_verifier
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.term_mentions:
            return state

        iterator = self.term_verifier.filter_terms(state.term_mentions)

        verified_terms = []

        log_iterations(
            description='Verifying terms',
            iterator=BoundedIterator(len(state.term_mentions), iterator),
            progress=self.progress,
            item_handler=lambda term: verified_terms.append(term) if term else None
        )

        state.term_mentions = verified_terms
        return state


class Log(Pipeline):

    def __init__(self, message: str):
        self.message = message

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        log(f'{LOG_STYLE}[      INFO      ]{Style.RESET_ALL}: {self.message}\n')

        return state


class LogLabeledTerms(Pipeline):

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        log_labeled_terms(state.text, state.term_mentions)

        return state


class NormalizeTerms(Pipeline):

    def __init__(self, term_normalizer: TermNormalizer, progress: Progress):
        self.term_normalizer = term_normalizer
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.term_mentions:
            return state

        normalized_terms = self.term_normalizer.normalize_all(state.term_mentions)

        log_iterations(
            description='Normalizing terms',
            iterator=BoundedIterator(len(state.term_mentions), normalized_terms),
            progress=self.progress,
            item_handler=lambda term: term
        )

        return state


class NormalizeLanguages(Pipeline):

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        for term in state.term_mentions:
            if term.class_ == 'Lang' and term.norm_value.endswith(' язык'):
                term.norm_value = term.norm_value[:-5].strip()

        return state


class DropEmptyTermMentions(Pipeline):

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        state.term_mentions = [
            m for m in state.term_mentions

            if m.norm_value and m.value
        ]

        return state


class LogNormalizedTerms(Pipeline):

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        log(f'{LOG_STYLE}[NORMALIZED TERMS]{Style.RESET_ALL}:')

        for term in state.term_mentions:
            log(f' - {term.value} -> {term.norm_value}')

        log()

        return state


class ResolveReference(Pipeline):
    def __init__(self, reference_resolver: ReferenceResolver, progress: Progress):
        self.reference_resolver = reference_resolver
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.term_mentions:
            return state

        state.terms = self.reference_resolver(state.term_mentions, state.text)
        return state


class LogGroupedTerms(Pipeline):

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        log_grouped_terms(state.terms)

        return state


class PredictSemanticRelations(Pipeline):
    def __init__(self, relation_extractor: RelationExtractor, progress: Progress):
        self.relation_extractor = relation_extractor
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.terms:
            return state

        predicted_relations_iterator = self.relation_extractor(state.text, state.terms)

        predicted_relations = []

        log_iterations(
            description='Predicting relations',
            iterator=predicted_relations_iterator,
            progress=self.progress,
            item_handler=lambda rel: predicted_relations.append(rel) if rel else None
        )

        state.relations = predicted_relations
        return state


class ResolveRelationConflicts(Pipeline):
    """
    Мультиагентный шаг: агент разрешения конфликтов.
    Для пар сущностей с несколькими разными предикатами оставляет один выбранный LLM.
    """
    def __init__(self, conflict_resolver, progress: Progress):
        self.conflict_resolver = conflict_resolver
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        from semantics_analysis.multi_agent.conflict_resolution import detect_relation_conflicts

        if not state.relations:
            return state

        conflicts = detect_relation_conflicts(state.relations)
        if not conflicts:
            return state

        task = self.progress.add_task(
            description=f'Resolving {len(conflicts)} relation conflicts',
            total=len(conflicts),
        )
        resolved, n_detected, n_resolved = self.conflict_resolver.resolve_all(
            state.text, state.relations
        )
        state.relations = resolved
        self.progress.update(task, advance=len(conflicts))
        self.progress.remove_task(task)
        return state
