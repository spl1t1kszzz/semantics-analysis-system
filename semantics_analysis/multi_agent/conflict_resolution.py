"""
Поиск и разрешение конфликтов между извлечёнными отношениями.

Конфликт: одна и та же пара сущностей (term1, term2) связана разными предикатами
или противоречащими отношениями. Разрешение — через LLM (диалог/выбор наилучшего).
"""

from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING

from semantics_analysis.entities import Relation, Term

if TYPE_CHECKING:
    from semantics_analysis.llm_agent import LLMAgent


def _term_pair_key(rel: Relation) -> Tuple[str, str, str, str]:
    """Ключ пары терминов: (class1, value1, class2, value2) нормализованный."""
    t1, t2 = rel.term1, rel.term2
    return (t1.class_, t1.value.strip().lower(), t2.class_, t2.value.strip().lower())


def detect_relation_conflicts(relations: List[Relation]) -> List[List[Relation]]:
    """
    Группирует отношения по паре (term1, term2). Возвращает только группы,
    где больше одного отношения (конфликт: несколько предикатов для одной пары).
    """
    by_pair: dict = defaultdict(list)
    for rel in relations:
        key = _term_pair_key(rel)
        by_pair[key].append(rel)

    return [group for group in by_pair.values() if len(group) > 1]


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
Ответ (только номер или «нет»):"""

    def resolve_group(self, text: str, group: List[Relation]) -> List[Relation]:
        """
        Для группы конфликтующих отношений возвращает список из 0 или 1 отношения,
        выбранного LLM. При use_dialogue — сначала запрос обоснования, затем выбор.
        При заданном reverify_callback выбранное отношение дополнительно верифицируется.
        """
        if len(group) <= 1:
            return group

        rel0 = group[0]
        t1, t2 = rel0.term1, rel0.term2
        options = "\n".join(
            f"{i+1}. {rel.predicate}" for i, rel in enumerate(group)
        )
        context_slice = text[:3000]

        if self.use_dialogue:
            reasoning_prompt = self._dialogue_reasoning_prompt().format(
                context=context_slice,
                term1_value=t1.value,
                term1_class=t1.class_,
                term2_value=t2.value,
                term2_class=t2.class_,
                options=options,
            )
            if self.log_prompts:
                from semantics_analysis.utils import log
                log(f"[CONFLICT REASONING PROMPT]: {reasoning_prompt}\n")
            reasoning = self.llm_agent(
                reasoning_prompt,
                max_new_tokens=150,
                stop_sequences=[],
            ).strip()
            if self.log_responses:
                from semantics_analysis.utils import log
                log(f"[CONFLICT REASONING]: {reasoning}\n")
            choice_prompt = self._dialogue_choice_prompt().format(reasoning=reasoning)
            prompt = choice_prompt
            max_tokens = 32
        else:
            prompt = self.prompt_template.format(
                context=context_slice,
                term1_value=t1.value,
                term1_class=t1.class_,
                term2_value=t2.value,
                term2_class=t2.class_,
                options=options,
            )
            max_tokens = 32

        if self.log_prompts and not self.use_dialogue:
            from semantics_analysis.utils import log
            log(f"[CONFLICT PROMPT]: {prompt}\n")

        response = self.llm_agent(
            prompt,
            max_new_tokens=max_tokens,
            stop_sequences=[".", "\n"],
        ).strip()
        if self.log_responses:
            from semantics_analysis.utils import log
            log(f"[CONFLICT RESPONSE]: {response}\n")

        response_lower = response.lower()
        if "нет" in response_lower or "none" in response_lower or "no" in response_lower:
            return []

        chosen: List[Relation] = []
        for i in range(1, 10):
            if str(i) in response:
                idx = i - 1
                if 0 <= idx < len(group):
                    chosen = [group[idx]]
                    break
        if not chosen and ("1" in response or "один" in response_lower):
            chosen = [group[0]]

        if chosen and self.reverify_callback:
            if not self.reverify_callback(text, chosen[0]):
                return []
        return chosen

    def resolve_all(self, text: str, relations: List[Relation]) -> Tuple[List[Relation], int, int]:
        """
        Находит все конфликтные группы, разрешает каждую через LLM.
        Возвращает (список отношений без конфликтов, число конфликтов, число разрешённых).
        """
        conflicts = detect_relation_conflicts(relations)
        conflict_set = {rel for group in conflicts for rel in group}
        non_conflict_rels = [rel for rel in relations if rel not in conflict_set]
        resolved = []
        for group in conflicts:
            chosen = self.resolve_group(text, group)
            resolved.extend(chosen)
        result = non_conflict_rels + resolved
        return result, len(conflicts), len(conflicts)
