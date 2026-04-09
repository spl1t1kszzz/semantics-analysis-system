import json
import re
from typing import Dict, List, Any, Set, Tuple

from semantics_analysis.entities import Relation, Term
from alphabet_detector import AlphabetDetector

ATTR_CLASSES = {'Date', 'Lang', 'Value'}

MAP_RELATIONS = {
    'isExampleOf': ('includes', True),
    'isPartOf': ('includes', True)
}

PERSON_PATTERN = re.compile(r'(?P<name>\w+)\nФамилия: (?P<surname>\w+)\.?')


class Attribute:
    name: str
    value: str
    lang: str

    def __init__(self, name: str, value: str, lang: str):
        self.name = name
        self.value = value
        self.lang = lang

    def __repr__(self):
        return f'Attribute(name={self.name}, value={self.value}, lang={self.lang})'

    def __eq__(self, other):
        if not isinstance(other, Attribute):
            return False

        return other.name == self.name and other.value == self.value and other.lang == self.lang

    def __hash__(self):
        return hash((self.name, self.value, self.lang))

    def to_json(self):
        return {
            'name': self.name,
            'value': self.value,
            'lang': self.lang
        }

    @staticmethod
    def from_json(attribute_json: Dict[str, str]):
        return Attribute(
            attribute_json['name'],
            attribute_json['value'],
            attribute_json['lang']
        )


class Object:
    id: int
    value: str
    class_: str
    lang: str
    attributes: List[Attribute]

    def __init__(self, id: int, value: str, class_: str, lang: str, attributes: List[Attribute]):
        self.id = id
        self.value = value
        self.class_ = class_
        self.lang = lang
        self.attributes = attributes

    def __repr__(self):
        return f'Object(id={self.id}, value={self.value}, class={self.class_}, lang={self.lang}, attributes={self.attributes})'

    def __eq__(self, other):
        if not isinstance(other, Object):
            return False

        return (other.id == self.id and other.value == self.value and other.class_ == self.class_
                and other.lang == self.lang and other.attributes == self.attributes)

    def __hash__(self):
        return hash((self.id, self.value, self.class_, self.lang, self.attributes))

    def to_json(self):
        return {
            'id': self.id,
            'value': self.value,
            'class': self.class_,
            'lang': self.lang,
            'attributes': [a.to_json() for a in self.attributes]
        }

    @staticmethod
    def from_json(object_json: Dict[str, Any]):
        return Object(
            object_json['id'],
            object_json['value'],
            object_json['class'],
            object_json['lang'],
            [Attribute.from_json(a) for a in object_json['attributes']]
        )


class OntRelation:
    object1_id: int
    name: str
    object2_id: int

    def __init__(self, object1_id: int, name: str, object2_id: int):
        self.object1_id = object1_id
        self.name = name
        self.object2_id = object2_id

    def __repr__(self):
        return f'Relation(object1_id={self.object1_id}, name={self.name}, object2_id={self.object2_id})'

    def __eq__(self, other):
        if not isinstance(other, OntRelation):
            return False

        return other.object1_id == self.object1_id and other.name == self.name and other.object2_id == self.object2_id

    def __hash__(self):
        return hash((self.object1_id, self.name, self.object2_id))

    def to_json(self):
        return {
            'object1_id': self.object1_id,
            'name': self.name,
            'object2_id': self.object2_id
        }

    @staticmethod
    def from_json(relation_json: Dict[str, Any]):
        return OntRelation(
            relation_json['object1_id'],
            relation_json['name'],
            relation_json['object2_id']
        )


def detect_lang(value: str, ad: AlphabetDetector) -> str:
    if ad.only_alphabet_chars(value, 'LATIN'):
        return 'en'
    else:
        return 'ru'


def term_to_attribute(
        term: Term,
        ad: AlphabetDetector
) -> Attribute:
    return Attribute(term.class_, term.value, detect_lang(term.value, ad))


def add_person_attrs(
        terms: Set[Term],
        attrs_by_term: Dict[Term, List[Attribute]],
        ad: AlphabetDetector
):
    persons = [t for t in terms if t.class_ == 'Person']

    if not persons:
        return

    from semantics_analysis.llm_agent import LLMAgent
    llm_agent = LLMAgent()

    with open('prompts/person.txt', 'r', encoding='utf-8') as f:
        prompt_template = f.read().strip()

    for person in persons:
        lang = detect_lang(person.value, ad)

        prompt = prompt_template.replace('{lang}', lang)
        prompt = prompt.replace('{context}', person.mentions[0].text)
        prompt = prompt.replace('{input}', person.mentions[0].value)

        response = llm_agent(prompt, max_new_tokens=128, stop_sequences=['.', 'Персона:', '```', '('])

        match = PERSON_PATTERN.match(response)

        if not match:
            continue

        name = match.group('name')
        surname = match.group('surname')

        attrs = []

        not_stated = '<не-указано>'

        if name != not_stated:
            attrs.append(Attribute('Name', name, lang))
        if surname != not_stated:
            attrs.append(Attribute('Surname', surname, lang))

        if person in attrs_by_term:
            attrs_by_term[person].extend(attrs)
        else:
            attrs_by_term[person] = attrs


def convert_to_ont_entities(
        terms: List[Term],
        relations: List[Relation]
) -> Tuple[List[Object], List[OntRelation]]:
    ad = AlphabetDetector()

    # consider only those terms that are in a relation

    considered_terms: Set[Term] = set()

    considered_relations: Set[Relation] = set()

    attrs_by_term: Dict[Term, List[Attribute]] = dict()

    for rel in relations:
        term1, term2 = rel.term1, rel.term2

        if term1.class_ in ATTR_CLASSES:
            attr = term_to_attribute(term1, ad)

            if term2 in attrs_by_term:
                attrs_by_term[term2].append(attr)
            else:
                attrs_by_term[term2] = [attr]

            considered_terms.add(term2)

        elif term2.class_ in ATTR_CLASSES:
            attr = term_to_attribute(term2, ad)

            if term1 in attrs_by_term:
                attrs_by_term[term1].append(attr)
            else:
                attrs_by_term[term1] = [attr]

            considered_terms.add(term1)

        else:
            considered_terms.add(term1)
            considered_terms.add(term2)

            considered_relations.add(rel)

    # we should also consider Alternative Name attribute
    # there can be group intersecting each other, we should unite them

    considered_term_mentions = [term for term in terms if len(term.mentions) >= 2]

    for term in considered_term_mentions:
        alt_name_attrs = [
            Attribute('Alternative Name', t.norm_value, detect_lang(t.value, ad))

            for t in term.mentions[1:]

            if t.norm_value != term.value
        ]

        if term in attrs_by_term:
            attrs_by_term[term].extend(alt_name_attrs)
        else:
            attrs_by_term[term] = alt_name_attrs

    # we should also add names and surnames as attrs for persons
    add_person_attrs(considered_terms, attrs_by_term, ad)

    # now we can turn terms into objects

    curr_obj_id = 1

    objects: List[Object] = []

    id_by_term: Dict[Term, int] = dict()

    for term in considered_terms:
        attrs = attrs_by_term.get(term, [])

        objects.append(
            Object(
                curr_obj_id,
                term.value,
                term.class_,
                detect_lang(term.value, ad),
                attrs
            )
        )

        id_by_term[term] = curr_obj_id

        curr_obj_id += 1

    ont_relations: List[OntRelation] = []

    for rel in considered_relations:
        obj1_id = id_by_term[rel.term1]
        obj2_id = id_by_term[rel.term2]

        if rel.predicate not in MAP_RELATIONS:
            predicate = rel.predicate
        else:
            predicate, reverse = MAP_RELATIONS[rel.predicate]

            if reverse:
                obj1_id, obj2_id = obj2_id, obj1_id

        rel_name = f'{rel.term1.class_}_{predicate}_{rel.term2.class_}'

        ont_relations.append(OntRelation(obj1_id, rel_name, obj2_id))

    return objects, ont_relations


def main():
    alice = Object(
        id=1,
        value='Алиса',
        class_='Приложение',
        lang='ru',
        attributes=[
            Attribute(
                name='Дата',
                value='2010',
                lang='ru'
            )
        ]
    )

    yandex = Object(
        id=2,
        value='Яндекс',
        class_='Организация',
        lang='ru',
        attributes=[
            Attribute(
                name='Альтернативное название',
                value='Yandex',
                lang='en'
            )
        ]
    )

    objects = [alice, yandex]
    relations = [OntRelation(alice.id, 'имеет автора', yandex.id)]

    result_json = {
        'objects': [o.to_json() for o in objects],
        'relations': [r.to_json() for r in relations]
    }

    with open('result.json', 'w', encoding='utf-8') as f:
        json.dump(result_json, f, ensure_ascii=False, indent=2)

    with open('result.json', 'r', encoding='utf-8') as f:
        result_json = json.load(f)

    objects = [Object.from_json(o) for o in result_json['objects']]

    relations = [OntRelation.from_json(r) for r in result_json['relations']]

    print('Objects:')
    for o in objects:
        print(f' - {o}')
    print()

    print('Relations:')
    for r in relations:
        print(f' - {r}')
    print()


if __name__ == '__main__':
    main()
