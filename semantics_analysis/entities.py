import json
from typing import Dict, List, Optional, Any, TypeVar, Generic, Iterator


class TermMention:
    def __init__(
            self,
            value: str,
            ontology_class: str,
            end_pos: int,
            text: str,
            norm_value: Optional[str] = None,
            source: str = 'roberta'
    ):
        self.value = value
        self.class_ = ontology_class
        self.start_pos = end_pos - len(value)
        self.end_pos = end_pos
        self.text = text
        self.norm_value = norm_value
        self.source = source

    def to_json(self):
        return {
            'value': self.value,
            'start_pos': self.start_pos
        }

    def __repr__(self):
        return f'TermMention(value={self.value}, start_pos={self.start_pos}, end_pos={self.end_pos})'

    def __eq__(self, other):
        if not isinstance(other, TermMention):
            return False

        return other.value == self.value and other.start_pos == self.start_pos and self.text == other.text

    def __hash__(self):
        return hash((self.value, self.start_pos, self.text))


class Term:
    def __init__(
            self,
            ontology_class: str,
            value: str,
            mentions: List[TermMention],
            source: str = 'roberta'
    ):
        self.value = value
        self.class_ = ontology_class
        self.source = source
        self.mentions = []

        prev_mentions = set()

        for mention in mentions:
            lower = mention.value.lower()

            if lower in prev_mentions:
                continue

            prev_mentions.add(lower)
            self.mentions.append(mention)

    def to_json(self):
        return {
            'class': self.class_,
            'value': self.value,
            'mentions': [m.to_json() for m in self.mentions]
        }

    @staticmethod
    def from_json(term_json: Dict[str, Any], text: str = ''):
        start_pos = term_json['start_pos']
        value = term_json['value']
        class_ = term_json['class']

        end_pos = start_pos + len(value)

        return Term(class_, value, mentions=[TermMention(value, class_, end_pos, text)])

    def __repr__(self):
        return f'Term(class={self.class_}, value={self.value}, source={self.source})'

    def __eq__(self, other):
        if not isinstance(other, Term):
            return False

        return other.class_ == self.class_ and other.value == self.value

    def __hash__(self):
        return hash((self.class_, self.value))


class Relation:
    def __init__(self, term1: Term, predicate: str, term2: Term):
        self.term1 = term1
        self.term2 = term2
        self.predicate = predicate
        self.id = f'{self.term1.class_}_{self.predicate}_{self.term2.class_}'

        if not predicate:
            raise ValueError('Got empty predicate.')

    def __repr__(self):
        return f'Relation(term1={self.term1}, predicate={self.predicate}, term2={self.term2})'

    def __eq__(self, other):
        if not isinstance(other, Relation):
            return False

        return self.term1 == other.term1 and self.term2 == other.term2 and self.predicate == other.predicate

    def __hash__(self):
        return hash((self.term1, self.predicate, self.term2))

    def as_str(self):
        return f'({self.term1.value}) {self.predicate} ({self.term2.value})'

    def to_json(self):
        return {
            'term1': self.term1.to_json(),
            'predicate': self.predicate,
            'term2': self.term2.to_json()
        }

    def inverse(self):
        return Relation(term1=self.term2, predicate=self.predicate, term2=self.term1)

    @staticmethod
    def from_json(rel_json: Dict[str, Any], text: str = ''):
        term1 = Term.from_json(rel_json['term1'], text)
        term2 = Term.from_json(rel_json['term2'], text)

        return Relation(
            term1=term1,
            predicate=rel_json['predicate'],
            term2=term2
        )


class Sentence:
    def __init__(self,
                 sent_id: int,
                 text: str,
                 terms: List[Term],
                 relations: List[Relation]):
        self.id = sent_id
        self.text = text
        self.terms = terms
        self.relations = relations

    def find_relation_by_id(self, relation_id: str) -> Optional[Relation]:
        for r in self.relations:
            if f'{r.term1.class_}_{r.predicate}_{r.term2.class_}' == relation_id:
                return r
        return None

    def find_term_by_class(self, class_id: str) -> Optional[Term]:
        for term in self.terms:
            if term.class_ == class_id:
                return term

        return None

    def find_terms_by_class(self, class_id: str) -> List[Term]:
        return [t for t in self.terms if t.class_ == class_id]

    def to_json(self):
        return {
            'id': self.id,
            'text': self.text,
            'terms': [t.to_json() for t in self.terms],
            'relations': [r.to_json() for r in self.relations]
        }

    @staticmethod
    def from_json(sent_json: Dict[str, Any]):
        text = sent_json['text']
        id = sent_json['id']
        terms = [Term.from_json(t, text) for t in sent_json['terms']]
        relations = [Relation.from_json(r, text) for r in sent_json['relations']]
        return Sentence(id, text, terms, relations)

    def __repr__(self):
        return f'Sentence(id={self.id}, text="{self.text}", terms={self.terms}, relations={self.relations})'

    def __hash__(self):
        return hash((self.text, self.terms, self.relations))


def read_sentences(file_name: str) -> List[Sentence]:
    with open(file_name, 'r', encoding='utf-8') as f:
        sentences_json = json.load(f)

        return [Sentence.from_json(s) for s in sentences_json['sentences']]


GenType = TypeVar('GenType')


class BoundedIterator(Generic[GenType]):
    total: int
    items: Iterator[GenType]

    def __init__(self, total: int, items: Iterator[GenType]):
        self.total = total
        self.items = items
