from semantics_analysis.entities import Relation, Term, TermMention
from semantics_analysis.multi_agent.conflict_resolution import (
    detect_relation_conflicts,
    parse_conflict_choice,
    RelationConflictResolver,
)


def _term(value: str, class_: str = 'Method') -> Term:
    m = TermMention(value=value, ontology_class=class_, end_pos=len(value), text=value)
    return Term(class_, value, [m])


def _rel(pred: str, v1: str = 'A', v2: str = 'B') -> Relation:
    return Relation(_term(v1), pred, _term(v2, 'Task'))


class TestParseConflictChoice:
    def test_number(self):
        assert parse_conflict_choice('1', 3) == 1
        assert parse_conflict_choice('Ответ: 2', 3) == 2
        assert parse_conflict_choice('вариант 3', 3) == 3

    def test_no(self):
        assert parse_conflict_choice('нет', 3) is None
        assert parse_conflict_choice('Нет, ни один', 3) is None

    def test_out_of_range(self):
        assert parse_conflict_choice('5', 3) is None


class TestDetectConflicts:
    def test_same_pair_different_predicates(self):
        group = [_rel('solves'), _rel('isUsedForSolving')]
        conflicts = detect_relation_conflicts(group)
        assert len(conflicts) == 1
        assert len(conflicts[0]) == 2

    def test_same_predicate_not_conflict(self):
        group = [_rel('solves'), _rel('solves')]
        assert detect_relation_conflicts(group) == []


class TestResolveGroup:
    def test_single_verified_skips_llm(self):
        resolver = RelationConflictResolver(reverify_callback=lambda t, r: r.predicate == 'solves')
        group = [_rel('solves'), _rel('isUsedForSolving')]
        out = resolver.resolve_group('text', group)
        assert len(out) == 1
        assert out[0].predicate == 'solves'

    def test_llm_choice_mock(self):
        class FakeAgent:
            def __call__(self, prompt, **kwargs):
                return '2'

        resolver = RelationConflictResolver(
            llm_agent=FakeAgent(),
            reverify_callback=lambda t, r: False,
        )
        group = [_rel('solves'), _rel('isUsedForSolving')]
        out = resolver.resolve_group('text', group)
        assert len(out) == 1
        assert out[0].predicate == 'isUsedForSolving'

    def test_llm_no_returns_empty(self):
        class FakeAgent:
            def __call__(self, prompt, **kwargs):
                return 'нет'

        resolver = RelationConflictResolver(llm_agent=FakeAgent(), reverify_callback=lambda t, r: False)
        group = [_rel('solves'), _rel('isUsedForSolving')]
        assert resolver.resolve_group('text', group) == []
