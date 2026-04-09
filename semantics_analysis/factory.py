from rich.progress import Progress

from semantics_analysis.config import Config
from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.pipelines import (
    SequencePipeline,
    Log,
    PredictTerms,
    PreprocessTerms,
    LogLabeledTerms,
    VerifyTerms,
    NormalizeTerms,
    DropEmptyTermMentions,
    NormalizeLanguages,
    LogNormalizedTerms,
    ResolveReference,
    LogGroupedTerms,
    PredictSemanticRelations,
    ResolveRelationConflicts,
)
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.term_extraction.dict_term_mention_extractor import DictTermExtractor
from semantics_analysis.term_extraction.hybrid_term_mention_extractor import CombinedTermExtractor
from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier
from semantics_analysis.term_extraction.roberta_classified_term_mention_extractor import RobertaTermExtractor
from semantics_analysis.term_normalization.llm_term_normalizer import LLMTermNormalizer
from semantics_analysis.term_post_processing.computer_science_term_post_processor import ResolveLibraries
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTerms


def build_term_extractor(config: Config):
    extractors = []
    if config.use_dict:
        extractors.append(DictTermExtractor('metadata/terms_by_class.json'))
    extractors.append(
        RobertaTermExtractor(
            config.device,
            term_threshold=config.term_threshold,
            class_threshold=config.class_threshold,
        )
    )
    if len(extractors) == 1:
        return extractors[0]
    return CombinedTermExtractor(*extractors)


def build_pipeline(config: Config, progress: Progress) -> SequencePipeline:
    llm_agent = LLMAgent(model=config.llm)

    term_extractor = build_term_extractor(config)
    term_verifier = LLMTermVerifier(llm_agent=llm_agent)
    term_normalizer = LLMTermNormalizer(llm_agent=llm_agent)

    reference_resolver = LLMReferenceResolver(
        model=config.llm,
        show_explanation=config.show_explanation,
        log_prompts=config.log_prompts,
        log_llm_responses=config.log_llm_responses,
        progress=progress,
    )

    relation_extractor = LLMRelationExtractor(
        model=config.llm,
        show_explanation=config.show_explanation,
        log_prompts=config.log_prompts,
        log_llm_responses=config.log_llm_responses,
        max_term_distance=config.max_term_distance,
    )

    stages = [
        Log(message='Predicting terms...'),
        PredictTerms(term_extractor),
        PreprocessTerms(ResolveLibraries(), MergeCloseTerms()),
        LogLabeledTerms(),
        VerifyTerms(term_verifier, progress),
        Log(message='Verified terms'),
        LogLabeledTerms(),
        NormalizeTerms(term_normalizer, progress),
        DropEmptyTermMentions(),
        NormalizeLanguages(),
        LogNormalizedTerms(),
        ResolveReference(reference_resolver, progress),
        LogGroupedTerms(),
        PredictSemanticRelations(relation_extractor, progress),
    ]

    if config.use_multi_agent:
        from semantics_analysis.multi_agent.conflict_resolution import RelationConflictResolver
        conflict_resolver = RelationConflictResolver(
            llm_agent=llm_agent,
            log_prompts=config.log_prompts,
            log_responses=config.log_llm_responses,
            use_dialogue=config.use_conflict_dialogue,
        )
        if config.use_reverify_after_resolve:
            conflict_resolver.reverify_callback = relation_extractor.verify_relation
        stages.append(ResolveRelationConflicts(conflict_resolver))

    return SequencePipeline(*stages, progress=progress)
