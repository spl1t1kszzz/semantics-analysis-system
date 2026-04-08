import json
from typing import Optional

import inquirer
from colorama import init as colorama_init

from ontology_entities import convert_to_ont_entities
from plot_graph import display_relation_graph
from semantics_analysis.config import load_config, Config
from semantics_analysis.pipelines import (
    AnalysisResult,
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
)
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.term_extraction.dict_term_mention_extractor import DictTermExtractor
from semantics_analysis.term_extraction.hybrid_term_mention_extractor import CombinedTermExtractor
from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier
from semantics_analysis.term_extraction.roberta_classified_term_mention_extractor import (
    RobertaTermExtractor,
)
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor
from semantics_analysis.term_normalization.llm_term_normalizer import LLMTermNormalizer
from semantics_analysis.term_post_processing.computer_science_term_post_processor import (
    ResolveLibraries,
)
from semantics_analysis.term_post_processing.merge_close_term_post_processor import MergeCloseTerms
from semantics_analysis.utils import log_found_relations, AlignedProgress
from colorama import Style

LOG_STYLE = Style.DIM


def _build_kg_for_orchestrator(terms, relations):
    from semantics_analysis.knowledge_graph import build_knowledge_graph
    return build_knowledge_graph(terms, relations, deduplicate=True)


def analyze_text(app_config: Config, roberta_term_predictor: Optional[TermMentionExtractor] = None):
    if not roberta_term_predictor:
        roberta_term_predictor = CombinedTermExtractor(
            DictTermExtractor('metadata/terms_by_class.json'),
            RobertaTermExtractor(app_config.device, term_threshold=0.2, class_threshold=0.5)
        )

    llm_term_verifier = LLMTermVerifier()
    llm_term_normalizer = LLMTermNormalizer()

    llm_relation_predictor = LLMRelationExtractor(
        show_explanation=app_config.show_explanation,
        log_prompts=app_config.log_prompts,
        log_llm_responses=app_config.log_llm_responses,
    )

    text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')
    print()

    with AlignedProgress() as progress:
        llm_reference_resolver = LLMReferenceResolver(
            model=app_config.llm,
            show_explanation=app_config.show_explanation,
            log_prompts=app_config.log_prompts,
            log_llm_responses=app_config.log_llm_responses,
            progress=progress,
        )

        if app_config.use_multi_agent:
            from semantics_analysis.multi_agent import MultiAgentOrchestrator
            from semantics_analysis.multi_agent.conflict_resolution import (
                RelationConflictResolver,
            )
            conflict_resolver = RelationConflictResolver(
                log_prompts=app_config.log_prompts,
                log_responses=app_config.log_llm_responses,
            )
            orchestrator = MultiAgentOrchestrator(
                term_mention_extractor=roberta_term_predictor,
                term_verifier=llm_term_verifier,
                term_normalizer=llm_term_normalizer,
                reference_resolver=llm_reference_resolver,
                relation_extractor=llm_relation_predictor,
                progress=progress,
                use_conflict_resolution=True,
                conflict_resolver=conflict_resolver,
                use_conflict_dialogue=app_config.use_conflict_dialogue,
                use_reverify_after_resolve=app_config.use_reverify_after_resolve,
            )
            state, ont_entities_json = orchestrator.run_and_build_kg(
                text, kg_builder=_build_kg_for_orchestrator
            )
            result_terms, result_relations = state.terms, state.relations
        else:
            semantics_analysis = SequencePipeline(
                Log(message='Predicting terms...'),
                PredictTerms(roberta_term_predictor),
                PreprocessTerms(ResolveLibraries(), MergeCloseTerms()),
                LogLabeledTerms(),
                VerifyTerms(llm_term_verifier, progress),
                Log(message='Verified terms'),
                LogLabeledTerms(),
                NormalizeTerms(llm_term_normalizer, progress),
                DropEmptyTermMentions(),
                NormalizeLanguages(),
                LogNormalizedTerms(),
                ResolveReference(llm_reference_resolver, progress),
                LogGroupedTerms(),
                PredictSemanticRelations(llm_relation_predictor, progress),
                progress=progress
            )
            result = semantics_analysis(AnalysisResult(text))
            result_terms, result_relations = result.terms, result.relations
            objects, ont_relations = convert_to_ont_entities(
                result.terms, result.relations
            )
            ont_entities_json = {
                'objects': [o.to_json() for o in objects],
                'relations': [r.to_json() for r in ont_relations]
            }

    with open('ont_entities.json', 'w', encoding='utf-8') as wf:
        json.dump(ont_entities_json, wf, ensure_ascii=False, indent=2)

    if result_relations and app_config.display_graph:
        display_relation_graph(result_terms, result_relations)
    else:
        log_found_relations(result_relations)


def main():
    colorama_init()

    app_config = load_config('config.yml')

    term_predictor = CombinedTermExtractor(
        DictTermExtractor('metadata/terms_by_class.json'),
        RobertaTermExtractor(app_config.device, term_threshold=0.2, class_threshold=0.5)
    )

    while True:
        analyze_text(app_config, roberta_term_predictor=term_predictor)

        question = inquirer.questions.List(
            name='answer',
            message='Продолжить анализировать тексты дальше',
            choices=['Нет.', 'Да.']
        )

        answer = inquirer.prompt([question])['answer']

        if answer == 'Да.':
            continue
        else:
            break


if __name__ == '__main__':
    main()
