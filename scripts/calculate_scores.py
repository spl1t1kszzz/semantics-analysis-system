import json
import os
import sys
import traceback
from typing import List, Dict, Any, Set, Optional

from rich.progress import Progress

from semantics_analysis.config import load_config
from semantics_analysis.entities import read_sentences, Sentence, Relation, Term
from semantics_analysis.ontology_utils import loaded_relation_ids
from semantics_analysis.reference_resolution.llm_reference_resolver import LLMReferenceResolver
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.relation_extraction.llm_relation_extractor import LLMRelationExtractor
from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor


def update_scores(
        sent: Sentence,
        predicted_relations: Set[Relation],
        expected_relations: Set[Relation],
        ignored_relations: Set[Relation],
        scores: Dict[str, Any]
):
    temp = predicted_relations

    predicted_relations = set()

    alternative_name_pairs = []

    for rel in temp:
        if rel.id not in loaded_relation_ids:
            ignored_relations.add(rel.id)
            continue
        else:
            if rel.predicate == 'isAlternativeNameFor':
                term1, term2 = rel.term1.value.lower(), rel.term2.value.lower()

                if (term1, term2) in alternative_name_pairs:
                    continue
                else:
                    alternative_name_pairs.append((term1, term2))

                if rel.term2.value == rel.term1.value:
                    continue

                if rel not in expected_relations and rel.inverse() in expected_relations:
                    predicted_relations.add(rel.inverse())
                else:
                    predicted_relations.add(rel)
            else:
                predicted_relations.add(rel)

    for rel in expected_relations:
        if rel.id not in scores:
            ignored_relations.add(rel.id)
            continue

        if rel in predicted_relations:
            scores[rel.id]['expected']['found']['count'] += 1
            scores[rel.id]['expected']['found']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})
        else:
            scores[rel.id]['expected']['not_found']['count'] += 1
            scores[rel.id]['expected']['not_found']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})

    for rel in predicted_relations:
        if rel.id not in scores:
            ignored_relations.add(rel.id)
            continue

        if rel in expected_relations:
            scores[rel.id]['predicted']['correct']['count'] += 1
            scores[rel.id]['predicted']['correct']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})
        else:
            scores[rel.id]['predicted']['incorrect']['count'] += 1
            scores[rel.id]['predicted']['incorrect']['examples'].append(
                {'text': sent.text, 'relation': rel.as_str()})


def calculate_scores(
        scores: Dict[str, Any],
        relation_extractor: RelationExtractor,
        reference_resolver: ReferenceResolver,
        sentences_to_check: List[Sentence],
        last_sent_id: int,
        progress: Progress,
        relation_to_consider: Optional[str] = None
) -> (int, bool):

    relations_to_consider = {relation_to_consider} if relation_to_consider else loaded_relation_ids

    # инициализация счётчиков
    for rel_id in relations_to_consider:
        if rel_id not in scores:
            scores[rel_id] = {
                'predicted': {
                    'incorrect': {'count': 0, 'examples': []},
                    'correct': {'count': 0, 'examples': []}
                },
                'expected': {
                    'not_found': {'count': 0, 'examples': []},
                    'found': {'count': 0, 'examples': []}
                }
            }

    counter = 1
    total_sentences = len(sentences_to_check)

    sentence_task = progress.add_task(
        description=f'[green]Sentence {counter}/{total_sentences}',
        total=total_sentences
    )

    for sent in sentences_to_check:
        if sent.id <= last_sent_id:
            counter += 1
            progress.update(sentence_task, advance=1,
                            description=f'[green]Sentence {counter}/{total_sentences}')
            continue

        # --- эталонные отношения ---
        expected_relations = {
            rel for rel in sent.relations if rel.id in loaded_relation_ids
        }

        if not expected_relations:
            counter += 1
            progress.update(sentence_task, advance=1,
                            description=f'[green]Sentence {counter}/{total_sentences}')
            continue

        # --- собираем mentions для resolver ---
        term_mentions = []
        for term in sent.terms:
            term_mentions.extend(term.mentions)

        try:
            grouped_terms: List[Term] = reference_resolver(term_mentions, sent.text)
        except Exception as e:
            print(traceback.format_exc())
            progress.remove_task(sentence_task)
            return last_sent_id, False

        predicted_relations: Set[Relation] = set()

        # соответствие Term → все его варианты (через mentions)
        term_variants: Dict[Term, List[Term]] = {}

        for term in grouped_terms:
            variants = [
                Term(term.class_, m.value, [m]) for m in term.mentions
            ]
            term_variants[term] = variants

            # автоматически добавляем isAlternativeNameFor
            if len(variants) > 1:
                for i in range(len(variants)):
                    for j in range(i + 1, len(variants)):
                        if variants[i].value != variants[j].value:
                            predicted_relations.add(
                                Relation(variants[i], 'isAlternativeNameFor', variants[j])
                            )

        # --- извлечение отношений LLM ---
        try:
            if relation_to_consider:
                class1, _, class2 = relation_to_consider.split('_')
                relations = relation_extractor(sent.text, grouped_terms, progress, class1, class2)
            else:
                relations = relation_extractor(sent.text, grouped_terms)

                for rel in relations.items:
                    if rel is not None:
                        predicted_relations.add(rel)

                        # coreference-aware сопоставление
                        for t1 in term_variants.get(rel.term1, [rel.term1]):
                            for t2 in term_variants.get(rel.term2, [rel.term2]):
                                rel_option = Relation(t1, rel.predicate, t2)
                                if rel_option in expected_relations:
                                    predicted_relations.add(rel_option)

        except Exception as e:
            print(traceback.format_exc())
            progress.remove_task(sentence_task)
            return last_sent_id, False

        update_scores(
            sent,
            predicted_relations,
            expected_relations,
            ignored_relations=set(),
            scores=scores
        )

        last_sent_id = sent.id
        counter += 1
        progress.update(sentence_task, advance=1,
                        description=f'[green]Sentence {counter}/{total_sentences}')

    if counter < total_sentences:
        progress.remove_task(sentence_task)

    return last_sent_id, True


def main():
    relation_to_consider = None

    argv = sys.argv

    sentences_file = 'tests/short_sent.json'

    for arg in argv:
        if arg.endswith('.json'):
            sentences_file = arg
            argv.remove(arg)
            break

    if len(argv) > 1:
        relation_to_consider = argv[1]

        if relation_to_consider not in loaded_relation_ids:
            print(f'Invalid relation id: {relation_to_consider}')
            return 0

    config = load_config('config.yml')

    print(f'Sentences file: {sentences_file}')
    print()
    sentences = read_sentences(sentences_file)

    if not relation_to_consider:
        sentences_to_check = sentences
    else:
        sentences_to_check = []

        class1, _, class2 = relation_to_consider.split('_')

        for sent in sentences:
            if class1 == class2 and len(sent.find_terms_by_class(class1)) >= 2:
                sentences_to_check.append(sent)
                continue

            if class1 != class2 and sent.find_term_by_class(class1) and sent.find_term_by_class(class2):
                sentences_to_check.append(sent)
                continue

    # if relation_to_consider:
    #     if not os.path.exists(f'tests/{relation_to_consider}'):
    #         os.makedirs(f'tests/{relation_to_consider}')
    #
    #     lock_path = f'tests/{relation_to_consider}/last_stop.lock'
    #     scores_path = f'tests/{relation_to_consider}/new_scores.json'
    # else:
    lock_path = 'tests/last_stop.lock'
    scores_path = 'results/shot_rel_new_prompts.json'

    try:
        with open(lock_path, 'r', encoding='utf-8') as f:
            last_sent_id = int(f.readline().strip())

        with open(scores_path, 'r', encoding='utf-8') as f:
            scores = json.load(f)
    except Exception:
        scores = {}
        last_sent_id = 0

    with Progress() as progress:
        while True:
            prev_last_sent_id = last_sent_id

            relation_extractor = LLMRelationExtractor(
                model=config.llm,
            )

            reference_resolver = LLMReferenceResolver(
                progress,
                model=config.llm,
            )

            last_sent_id, finished = calculate_scores(
                scores,
                relation_extractor,
                reference_resolver,
                sentences_to_check,
                last_sent_id,
                progress=progress,
                relation_to_consider=relation_to_consider
            )

            if finished:
                with open(lock_path, 'w', encoding='utf-8') as f:
                    f.write(f'{last_sent_id}')

                with open(scores_path, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
                print('Scores were saved.')
                break

            if prev_last_sent_id != last_sent_id:
                with open(lock_path, 'w', encoding='utf-8') as f:
                    f.write(f'{last_sent_id}')

                with open(scores_path, 'w', encoding='utf-8') as f:
                    json.dump(scores, f, ensure_ascii=False, indent=2)
                print('Scores were saved.')


if __name__ == '__main__':
    main()
