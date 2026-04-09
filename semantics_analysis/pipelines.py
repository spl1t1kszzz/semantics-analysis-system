from abc import ABC, abstractmethod
from typing import List, Optional, Callable

from colorama import Style
from rich.progress import Progress

from semantics_analysis.entities import TermMention, Relation, Term, BoundedIterator
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor
from semantics_analysis.term_extraction.term_verifier import TermVerifier
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer
from semantics_analysis.term_post_processing.term_post_processor import TermPostProcessor
from semantics_analysis.utils import log_iterations, log_labeled_terms, log, LOG_STYLE, log_grouped_terms


class AnalysisResult:
    def __init__(
            self,
            text: str,
            term_mentions: Optional[List[TermMention]] = None,
            terms: Optional[List[Term]] = None,
            relations: Optional[List[Relation]] = None
    ):
        self.text = text
        self.term_mentions = term_mentions or []
        self.terms = terms or []
        self.relations = relations or []


# A pipeline step is any callable: AnalysisResult -> AnalysisResult
PipelineStep = Callable[[AnalysisResult], AnalysisResult]


class Pipeline(ABC):
    @abstractmethod
    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        ...


class SequencePipeline(Pipeline):
    def __init__(self, *steps: PipelineStep, progress: Progress = None):
        self.steps = steps
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        for step in self.steps:
            state = step(state)
        return state


# --- Lightweight pipeline steps as functions ---

def log_message(message: str) -> PipelineStep:
    def step(state: AnalysisResult) -> AnalysisResult:
        log(f'{LOG_STYLE}[      INFO      ]{Style.RESET_ALL}: {message}\n')
        return state
    return step


def log_labeled_terms_step(state: AnalysisResult) -> AnalysisResult:
    log_labeled_terms(state.text, state.term_mentions)
    return state


def log_normalized_terms_step(state: AnalysisResult) -> AnalysisResult:
    log(f'{LOG_STYLE}[NORMALIZED TERMS]{Style.RESET_ALL}:')
    for term in state.term_mentions:
        log(f' - {term.value} -> {term.norm_value}')
    log()
    return state


def log_grouped_terms_step(state: AnalysisResult) -> AnalysisResult:
    log_grouped_terms(state.terms)
    return state


def drop_empty_term_mentions(state: AnalysisResult) -> AnalysisResult:
    state.term_mentions = [m for m in state.term_mentions if m.norm_value and m.value]
    return state


def normalize_languages(state: AnalysisResult) -> AnalysisResult:
    for term in state.term_mentions:
        if term.class_ == 'Lang' and term.norm_value and term.norm_value.endswith(' язык'):
            term.norm_value = term.norm_value[:-5].strip()
    return state


# --- Heavy pipeline steps as classes ---

class PredictTerms(Pipeline):
    def __init__(self, extractor: TermMentionExtractor):
        self.extractor = extractor

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if state.text:
            state.term_mentions = self.extractor(state.text)
        return state


class PreprocessTerms(Pipeline):
    def __init__(self, *postprocessors: TermPostProcessor):
        self.postprocessors = postprocessors

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        for pp in self.postprocessors:
            state.term_mentions = pp(state.term_mentions)
        return state


class VerifyTerms(Pipeline):
    def __init__(self, verifier: TermVerifier, progress: Progress):
        self.verifier = verifier
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.term_mentions:
            return state
        iterator = self.verifier.filter_terms(state.term_mentions)
        verified = []
        log_iterations(
            description='Verifying terms',
            iterator=BoundedIterator(len(state.term_mentions), iterator),
            progress=self.progress,
            item_handler=lambda term: verified.append(term) if term else None
        )
        state.term_mentions = verified
        return state


class NormalizeTerms(Pipeline):
    def __init__(self, normalizer: TermNormalizer, progress: Progress):
        self.normalizer = normalizer
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.term_mentions:
            return state
        normalized = self.normalizer.normalize_all(state.term_mentions)
        log_iterations(
            description='Normalizing terms',
            iterator=BoundedIterator(len(state.term_mentions), normalized),
            progress=self.progress,
            item_handler=lambda term: term
        )
        return state


class ResolveReference(Pipeline):
    def __init__(self, resolver: ReferenceResolver, progress: Progress):
        self.resolver = resolver
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if state.term_mentions:
            state.terms = self.resolver(state.term_mentions, state.text)
        return state


class PredictSemanticRelations(Pipeline):
    def __init__(self, extractor: RelationExtractor, progress: Progress):
        self.extractor = extractor
        self.progress = progress

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.terms:
            return state
        predicted_iter = self.extractor(state.text, state.terms)
        predicted = []
        log_iterations(
            description='Predicting relations',
            iterator=predicted_iter,
            progress=self.progress,
            item_handler=lambda rel: predicted.append(rel) if rel else None
        )
        state.relations = predicted
        return state


class ResolveRelationConflicts(Pipeline):
    def __init__(self, conflict_resolver):
        self.conflict_resolver = conflict_resolver

    def __call__(self, state: AnalysisResult) -> AnalysisResult:
        if not state.relations:
            return state
        resolved, _, _ = self.conflict_resolver.resolve_all(state.text, state.relations)
        state.relations = resolved
        return state


# Backward-compatible aliases
Log = log_message
LogLabeledTerms = type('LogLabeledTerms', (Pipeline,), {'__call__': staticmethod(log_labeled_terms_step)})
LogNormalizedTerms = type('LogNormalizedTerms', (Pipeline,), {'__call__': staticmethod(log_normalized_terms_step)})
LogGroupedTerms = type('LogGroupedTerms', (Pipeline,), {'__call__': staticmethod(log_grouped_terms_step)})
DropEmptyTermMentions = type('DropEmptyTermMentions', (Pipeline,), {'__call__': staticmethod(drop_empty_term_mentions)})
NormalizeLanguages = type('NormalizeLanguages', (Pipeline,), {'__call__': staticmethod(normalize_languages)})
