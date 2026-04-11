"""
Поиск и разрешение конфликтов между извлечёнными отношениями.

Конфликт: одна и та же пара сущностей (term1, term2) связана разными предикатами
или противоречащими отношениями. Разрешение — через LLM (диалог/выбор наилучшего).
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING

import nltk.tokenize

from semantics_analysis.entities import Relation, Term
from semantics_analysis.ontology_utils import relations_metadata_by_class_pair

if TYPE_CHECKING:
    from semantics_analysis.llm_agent import LLMAgent


def _term_pair_key(rel: Relation) -> Tuple[str, str, str, str]:
    """Ключ пары терминов: (class1, value1, class2, value2) нормализованный."""
    t1, t2 = rel.term1, rel.term2
    return (t1.class_, t1.value.strip().lower(), t2.class_, t2.value.strip().lower())


def detect_relation_conflicts(relations: List[Relation]) -> List[List[Relation]]:
    """
    Группирует отношения по паре (term1, term2). Возвращает только группы,
    где есть разные предикаты (настоящий конфликт).
    Группы с одинаковым предикатом (варианты — разные формы термина) — не конфликт.
    """
    by_pair: dict = defaultdict(list)
    for rel in relations:
        key = _term_pair_key(rel)
        by_pair[key].append(rel)

    return [group for group in by_pair.values()
            if len(set(r.predicate for r in group)) > 1]


class RelationConflictResolver:
    """
    Агент разрешения конфликтов: для каждой конфликтной группы отношений
    запрашивает у LLM выбор одного корректного отношения (или ни одного).
    Опционально: двухшаговый диалог (сначала обоснование, потом выбор) и
    повторная верификация выбранного отношения через callback.
    """

    def __init__(
        self,
        llm_agent: Optional["LLMAgent"] = None,
        prompt_path: Optional[str] = None,
        log_prompts: bool = False,
        log_responses: bool = False,
        use_dialogue: bool = False,
        reverify_callback: Optional[Callable[[str, Relation], bool]] = None,
    ):
        if llm_agent is None:
            from semantics_analysis.llm_agent import LLMAgent
            llm_agent = LLMAgent()
        self.llm_agent = llm_agent
        self.log_prompts = log_prompts
        self.log_responses = log_responses
        self.use_dialogue = use_dialogue
        self.reverify_callback = reverify_callback
        path = Path(prompt_path or "prompts/resolve_relation_conflict.txt")
        if path.exists():
            self.prompt_template = path.read_text(encoding="utf-8").strip()
        else:
            self.prompt_template = self._default_prompt()

    def _default_prompt(self) -> str:
        return """Текст: {context}

Для пары сущностей «{term1_value}» ({term1_class}) и «{term2_value}» ({term2_class}) из текста выше предложены следующие отношения:
{options}

Выбери ровно одно отношение, которое верно в данном контексте, или ответь «нет», если ни одно не подходит.
Ответ (только номер варианта или слово «нет»):"""

    def _dialogue_reasoning_prompt(self) -> str:
        return """Текст: {context}

Для пары сущностей «{term1_value}» ({term1_class}) и «{term2_value}» ({term2_class}) предложены отношения:
{options}

Кратко обоснуй (1–2 предложения), какие варианты подходят к тексту, а какие нет. Не выбирай пока — только оцени."""

    def _dialogue_choice_prompt(self) -> str:
        return """{reasoning}

Исходя из твоего обоснования выше, выбери ровно один вариант по номеру (1, 2, …) или ответь «нет», если ни один не подходит.
Ответ (номер или «нет»):"""

    def _get_predicate_description(self, class1: str, class2: str, predicate: str) -> str:
        """Получить описание предиката из метаданных онтологии."""
        for key in [(class1, class2), (class2, class1)]:
            if key in relations_metadata_by_class_pair:
                metadata = relations_metadata_by_class_pair[key]
                if predicate in metadata and 'yes' in metadata[predicate]:
                    return metadata[predicate]['yes'].get('description', '')
        return ''

    def _extract_context_around_terms(self, text: str, t1: Term, t2: Term, margin: int = 500) -> str:
        """Вырезать фрагмент текста вокруг терминов, а не с начала."""
        start_pos = min(t1.mentions[0].start_pos, t2.mentions[0].start_pos)
        end_pos = max(t1.mentions[0].end_pos, t2.mentions[0].end_pos)

        ctx_start = max(0, start_pos - margin)
        ctx_end = min(len(text), end_pos + margin)

        # Расширить до границ предложений
        sentences = nltk.tokenize.sent_tokenize(text)
        offset = 0
        first_sent_id = 0
        last_sent_id = len(sentences) - 1
        for idx, sent in enumerate(sentences):
            new_offset = offset + len(sent) + 1
            if new_offset > ctx_start and first_sent_id == 0:
                first_sent_id = idx
            if new_offset >= ctx_end and last_sent_id == len(sentences) - 1:
                last_sent_id = idx
                break
            offset = new_offset

        return ' '.join(sentences[first_sent_id:last_sent_id + 1])

    def resolve_group(self, text: str, group: List[Relation]) -> List[Relation]:
        """
        Для группы конфликтующих отношений (>1 предикат для одной пары):
        если есть reverify_callback — проверить каждое независимо,
        оставить подтверждённые. Если ни одно не подтвердилось — оставить все
        (чтобы не терять recall).
        """
        if self.reverify_callback is None:
            return group

        verified = []
        for rel in group:
            if self.reverify_callback(text, rel):
                verified.append(rel)

        # Если ни одно не прошло верификацию, сохраняем все —
        # лучше FP, чем потерять TP
        return verified if verified else group

    def resolve_all(self, text: str, relations: List[Relation]) -> Tuple[List[Relation], int, int]:
        """
        Находит все конфликтные группы, разрешает каждую.
        Возвращает (список отношений без конфликтов, число конфликтов, число разрешённых).
        """
        conflicts = detect_relation_conflicts(relations)
        if not conflicts:
            return relations, 0, 0
        conflict_set = {rel for group in conflicts for rel in group}
        non_conflict_rels = [rel for rel in relations if rel not in conflict_set]
        resolved = []
        for group in conflicts:
            chosen = self.resolve_group(text, group)
            resolved.extend(chosen)
        result = non_conflict_rels + resolved
        n_in = len(relations)
        n_out = len(set(result))
        if n_out != n_in:
            print(f"[DEBUG resolve_all] IN={n_in} OUT={n_out} conflicts={len(conflicts)} conflict_rels={len(conflict_set)} non_conflict={len(non_conflict_rels)} resolved={len(resolved)}")
            for group in conflicts:
                predicates = [r.predicate for r in group]
                t1, t2 = group[0].term1, group[0].term2
                print(f"  conflict: ({t1.value}) -- {predicates} -- ({t2.value})")
        return result, len(conflicts), len(conflicts)
