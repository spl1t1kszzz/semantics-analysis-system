from semantics_analysis.entities import TermMention
from semantics_analysis.pipelines import (
    AnalysisResult,
    SequencePipeline,
    drop_empty_term_mentions,
    normalize_languages,
)


class TestAnalysisResult:
    def test_defaults(self):
        r = AnalysisResult("hello")
        assert r.text == "hello"
        assert r.term_mentions == []
        assert r.terms == []
        assert r.relations == []


class TestDropEmptyTermMentions:
    def test_drops_empty_values(self):
        m1 = TermMention("BERT", "Method", end_pos=4, text="BERT")
        m1.norm_value = "BERT"
        m2 = TermMention("", "Method", end_pos=4, text="")
        m2.norm_value = ""
        state = AnalysisResult("text", term_mentions=[m1, m2])
        result = drop_empty_term_mentions(state)
        assert len(result.term_mentions) == 1
        assert result.term_mentions[0].value == "BERT"

    def test_drops_none_norm_value(self):
        m = TermMention("test", "Method", end_pos=4, text="test")
        m.norm_value = None
        state = AnalysisResult("text", term_mentions=[m])
        result = drop_empty_term_mentions(state)
        assert len(result.term_mentions) == 0


class TestNormalizeLanguages:
    def test_strips_yazyk_suffix(self):
        m = TermMention("Python", "Lang", end_pos=6, text="Python")
        m.norm_value = "Python язык"
        state = AnalysisResult("text", term_mentions=[m])
        result = normalize_languages(state)
        assert result.term_mentions[0].norm_value == "Python"

    def test_no_change_for_non_lang(self):
        m = TermMention("BERT", "Method", end_pos=4, text="BERT")
        m.norm_value = "BERT язык"
        state = AnalysisResult("text", term_mentions=[m])
        result = normalize_languages(state)
        assert result.term_mentions[0].norm_value == "BERT язык"


class TestSequencePipeline:
    def test_runs_steps_in_order(self):
        log = []

        def step1(state):
            log.append("step1")
            return state

        def step2(state):
            log.append("step2")
            return state

        pipeline = SequencePipeline(step1, step2)
        pipeline(AnalysisResult("text"))
        assert log == ["step1", "step2"]
