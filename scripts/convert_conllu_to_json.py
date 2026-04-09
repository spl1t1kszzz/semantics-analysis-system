import json
import sys

import conllu
import os
from conllu import TokenList

from semantics_analysis.entities import Term, Relation, Sentence


def preprocess_label(label_: str) -> str:
    if label_ == 'Technology' or label_ == 'App_system':
        return 'Application'

    return label_


def parse_sentence(sent_bio: TokenList) -> Sentence:
    metadata = sent_bio.metadata
    text = metadata['text'].strip()

    words = [word['form'].split()[0] for word in sent_bio]
    labels = [word['form'].split()[1] for word in sent_bio]

    terms = []

    curr_text = ''
    curr_term = ''
    curr_label = ''
    is_under_term = False

    for label, token in zip(labels, words):
        remain_text = text[len(curr_text):]

        if label.startswith('B-'):
            is_under_term = True

            if curr_term:
                terms.append(Term(ontology_class=curr_label, value=curr_term.strip(), text=text, end_pos=len(curr_text)))
                curr_term = ''
            curr_label = preprocess_label(label[2:])
        elif label == 'O':
            is_under_term = False
            if curr_term:
                terms.append(Term(ontology_class=curr_label, value=curr_term.strip(), text=text, end_pos=len(curr_text)))
                curr_term = ''

        for i in range(len(remain_text)):
            curr_text += remain_text[i]

            if is_under_term:
                curr_term += remain_text[i]

            if curr_text.endswith(token):
                break

    if curr_term:
        terms.append(Term(ontology_class=curr_label, value=curr_term.strip(), text=text, end_pos=len(curr_text)))

    cleared_terms = []

    # clear from predicate terms
    for term in terms:
        if '_' in term.class_:
            continue
        cleared_terms.append(term)

    terms = cleared_terms

    terms_by_class = {}

    for term in terms:
        if term.class_ in terms_by_class:
            terms_by_class[term.class_].append(term)
        else:
            terms_by_class[term.class_] = [term]

    relations_str = metadata['relations'] if 'relations' in metadata else '""'

    relations_str = relations_str[1:-1]  # drop quotes

    relations = []

    if relations_str:
        relations_str_list = relations_str.split(', ')

        for relation_str in relations_str_list:
            relation_id, term1_idx, term2_idx = relation_str.split(' ')

            term1_idx, term2_idx = int(term1_idx), int(term2_idx)

            class1, predicate, class2 = relation_id.split('_')

            relations.append(Relation(
                term1=terms_by_class[class1][term1_idx],
                predicate=predicate,
                term2=terms_by_class[class2][term2_idx]
            ))

    return Sentence(text=text, terms=terms, relations=relations, sent_id=int(metadata['sent_id']))


args = sys.argv

if len(sys.argv) != 1 + 2:
    print('[USAGE]: python convert_conllu_to_json.py <input>.conllu <output>.json')
    exit(0)

input_file_name, output_file_name = sys.argv[-2], sys.argv[-1]

sentences_bio = conllu.parse_incr(open(input_file_name, 'r', encoding='utf-8'))

sentences = []

for sent in sentences_bio:
    try:
        sentences.append(parse_sentence(sent))
    except Exception as e:
        print(sent.metadata['sent_id'])

        raise e

sentences_json = {
    'sentences': [s.to_json() for s in sentences]
}

json.dump(sentences_json, open(output_file_name, 'w', encoding='utf-8'), ensure_ascii=False, indent=2)
