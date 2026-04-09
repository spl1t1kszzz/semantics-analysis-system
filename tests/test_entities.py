import pytest

from semantics_analysis.entities import TermMention, Term, Relation


class TestTermMention:
    def test_start_pos_calculated_from_end_pos(self):
        m = TermMention("BERT", "Method", end_pos=10, text="hello BERT world")
        assert m.start_pos == 6
        assert m.end_pos == 10

    def test_equality(self):
        m1 = TermMention("BERT", "Method", end_pos=10, text="text")
        m2 = TermMention("BERT", "Method", end_pos=10, text="text")
        assert m1 == m2

    def test_inequality_different_value(self):
        m1 = TermMention("BERT", "Method", end_pos=10, text="text")
        m2 = TermMention("GPT", "Method", end_pos=10, text="text")
        assert m1 != m2

    def test_hash_consistency(self):
        m1 = TermMention("BERT", "Method", end_pos=10, text="text")
        m2 = TermMention("BERT", "Method", end_pos=10, text="text")
        assert hash(m1) == hash(m2)

    def test_to_json(self):
        m = TermMention("BERT", "Method", end_pos=10, text="hello BERT world")
        j = m.to_json()
        assert j == {'value': 'BERT', 'start_pos': 6}


class TestTerm:
    def test_deduplicates_mentions(self):
        m1 = TermMention("BERT", "Method", end_pos=10, text="text")
        m2 = TermMention("bert", "Method", end_pos=20, text="text")
        t = Term("Method", "BERT", mentions=[m1, m2])
        assert len(t.mentions) == 1

    def test_equality(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Method", "BERT", mentions=[])
        assert t1 == t2

    def test_inequality(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Method", "GPT", mentions=[])
        assert t1 != t2

    def test_to_json(self):
        m = TermMention("BERT", "Method", end_pos=10, text="text")
        t = Term("Method", "BERT", mentions=[m])
        j = t.to_json()
        assert j['class'] == 'Method'
        assert j['value'] == 'BERT'
        assert len(j['mentions']) == 1


class TestRelation:
    def test_creation(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Task", "NER", mentions=[])
        r = Relation(t1, "solves", t2)
        assert r.predicate == "solves"
        assert r.id == "Method_solves_Task"

    def test_empty_predicate_raises(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Task", "NER", mentions=[])
        with pytest.raises(ValueError):
            Relation(t1, "", t2)

    def test_equality(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Task", "NER", mentions=[])
        r1 = Relation(t1, "solves", t2)
        r2 = Relation(t1, "solves", t2)
        assert r1 == r2

    def test_inverse(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Task", "NER", mentions=[])
        r = Relation(t1, "solves", t2)
        inv = r.inverse()
        assert inv.term1 == t2
        assert inv.term2 == t1

    def test_as_str(self):
        t1 = Term("Method", "BERT", mentions=[])
        t2 = Term("Task", "NER", mentions=[])
        r = Relation(t1, "solves", t2)
        assert r.as_str() == "(BERT) solves (NER)"
