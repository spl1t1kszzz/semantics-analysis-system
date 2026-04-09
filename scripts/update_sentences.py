import json

from semantics_analysis.entities import Sentence, Relation

with open('tests/sentences.json', 'r', encoding='utf-8') as f:
    sentences_json = json.load(f)['sentences']

    sentences = [Sentence.from_json(s) for s in sentences_json]

new_sentences = []
for sent in sentences:
    new_relations = []

    for rel in sent.relations:
        if rel.predicate != 'includes':
            new_relations.append(rel)
            continue

        if rel.term1.class_ == 'Object':
            new_relations.append(Relation(rel.term2, 'isPartOf', rel.term1))
        elif rel.term1.class_ == 'Method':
            new_relations.append(Relation(rel.term2, 'isPartOf', rel.term1))
        elif rel.term1.class_ == 'Model':
            new_relations.append(Relation(rel.term2, 'isPartOf', rel.term1))
        else:
            new_relations.append(Relation(rel.term2, 'isExampleOf', rel.term1))

    new_sentences.append(
        Sentence(sent.id, sent.text, sent.terms, new_relations))

sentences_dict = {
    'sentences': [s.to_json() for s in new_sentences]
}

with open('tests/sentences.json', 'w', encoding='utf-8') as f:
    json.dump(sentences_dict, f, ensure_ascii=False, indent=2)
