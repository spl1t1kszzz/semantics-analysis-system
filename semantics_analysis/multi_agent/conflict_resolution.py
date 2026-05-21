"""
Поиск и разрешение конфликтов между извлечёнными отношениями.

Конфликт: одна и та же пара сущностей (term1, term2) связана разными предикатами.
Разрешение: re-verify кандидатов → при необходимости выбор одного через LLM.
"""

import re
from collections import defaultdict
from pathlib import Path
from typing import List, Tuple, Optional, Callable, TYPE_CHECKING

import nltk.tokenize

from semantics_analysis.entities import Relation, Term
from semantics_analysis.ontology_utils import relations_metadata_by_class_pair

if TYPE_CHECKING:
    from semantics_analysis.llm_agent import LLMAgent


def _term_pair_key(rel: Relation) -> Tuple[str, str, str, str]:
    t1, t2 = rel.term1, rel.term2
    return (t1.class_, t1.value.strip().lower(), t2.class_, t2.value.strip().lower())


def detect_relation_conflicts(relations: List[Relation]) -> List[List[Relation]]:
    by_pair: dict = defaultdict(list)
    for rel in relations:
        by_pair[_term_pair_key(rel)].append(rel)

    return [
        group for group in by_pair.values()
        if len(set(r.predicate for r in group)) > 1
    ]


def parse_conflict_choice(response: str, n_options: int) -> Optional[int]:
    """
    Индекс выбранного варианта 1..n_options или None («нет» / не распознано как выбор).
    """
    if n_options <= 0:
        return None

    text = response.strip()
    lower = text.lower()

    if re.match(r'^(нет|none|ни одн|ничего)\b', lower):
        return None

    for pattern in (
        r'(?:вариант|ответ|выбор|option)?\s*[#№]?\s*(\d+)',
        r'^(\d+)\s*[.:)\-]',
        r'\b(\d+)\b',
    ):
        match = re.search(pattern, lower)
        if match:
            idx = int(match.group(1))
            if 1 <= idx <= n_options:
                return idx

    return None


class RelationConflictResolver:
    """
    Для конфликтной группы (>1 предикат на одну пару терминов):
    1) re-verify каждого кандидата (если задан callback);
    2) 0 подтверждённых → LLM выбирает один или «нет»;
    3) 1 подтверждённый → возвращаем его;
    4) 2+ подтверждённых → LLM выбирает лучший среди них;
    5) опционально повторная verify выбранного отношения.
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

Для пары «{term1_value}» ({term1_class}) и «{term2_value}» ({term2_class}) предложены отношения:
{options}

Выбери ровно одно отношение или ответь «нет».
Ответ (номер или «нет»):"""

    def _log(self, label: str, content: str) -> None:
        if self.log_prompts or self.log_responses:
            from semantics_analysis.utils import log
            log(f'[{label}]: {content}\n')

    def _get_predicate_description(self, class1: str, class2: str, predicate: str) -> str:
        for key in [(class1, class2), (class2, class1)]:
            if key in relations_metadata_by_class_pair:
                metadata = relations_metadata_by_class_pair[key]
                if predicate in metadata and 'yes' in metadata[predicate]:
                    return metadata[predicate]['yes'].get('description', '')
        return ''

    def _format_options(self, relations: List[Relation]) -> str:
        lines = []
        for i, rel in enumerate(relations, 1):
            desc = self._get_predicate_description(
                rel.term1.class_, rel.term2.class_, rel.predicate
            )
            line = f'{i}. {rel.predicate} — «{rel.term1.value}» → «{rel.term2.value}»'
            if desc:
                line += f'\n   Смысл: {desc}'
            lines.append(line)
        return '\n'.join(lines)

    def _extract_context_around_terms(self, text: str, t1: Term, t2: Term, margin: int = 500) -> str:
        start_pos = min(t1.mentions[0].start_pos, t2.mentions[0].start_pos)
        end_pos = max(t1.mentions[0].end_pos, t2.mentions[0].end_pos)

        ctx_start = max(0, start_pos - margin)
        ctx_end = min(len(text), end_pos + margin)

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

    def _build_choice_prompt(
        self,
        context: str,
        t1: Term,
        t2: Term,
        candidates: List[Relation],
    ) -> str:
        prompt = self.prompt_template
        prompt = prompt.replace('{context}', context)
        prompt = prompt.replace('{term1_value}', t1.value)
        prompt = prompt.replace('{term1_class}', t1.class_)
        prompt = prompt.replace('{term2_value}', t2.value)
        prompt = prompt.replace('{term2_class}', t2.class_)
        prompt = prompt.replace('{options}', self._format_options(candidates))
        return prompt.strip()

    def _llm_choose_one(
        self,
        text: str,
        group: List[Relation],
        candidates: List[Relation],
    ) -> Optional[Relation]:
        if not candidates:
            return None

        t1, t2 = group[0].term1, group[0].term2
        context = self._extract_context_around_terms(text, t1, t2)

        if self.use_dialogue and len(candidates) > 1:
            reasoning_prompt = (
                f'Текст: {context}\n\n'
                f'Пара: «{t1.value}» ({t1.class_}) и «{t2.value}» ({t2.class_}).\n'
                f'Варианты:\n{self._format_options(candidates)}\n\n'
                'Кратко (1–2 предложения): какие варианты подходят к тексту, какие нет. '
                'Не выбирай номер — только оценка.'
            )
            self._log('CONFLICT REASONING PROMPT', reasoning_prompt)
            reasoning = self.llm_agent(reasoning_prompt, max_new_tokens=128, stop_sequences=[])
            self._log('CONFLICT REASONING RESPONSE', reasoning)

            choice_prompt = (
                f'{reasoning.strip()}\n\n'
                f'Исходя из оценки выше, выбери ровно один номер (1–{len(candidates)}) '
                f'или ответь «нет».\nОтвет:'
            )
        else:
            choice_prompt = self._build_choice_prompt(context, t1, t2, candidates)

        self._log('CONFLICT CHOICE PROMPT', choice_prompt)
        response = self.llm_agent(choice_prompt, max_new_tokens=16, stop_sequences=['\n\n'])
        self._log('CONFLICT CHOICE RESPONSE', response)

        idx = parse_conflict_choice(response, len(candidates))
        if idx is None:
            return None
        return candidates[idx - 1]

    def _filter_by_reverify(self, text: str, group: List[Relation]) -> List[Relation]:
        if self.reverify_callback is None:
            return list(group)
        return [rel for rel in group if self.reverify_callback(text, rel)]

    def resolve_group(self, text: str, group: List[Relation]) -> List[Relation]:
        if len(group) <= 1:
            return group

        verified = self._filter_by_reverify(text, group)

        chosen: Optional[Relation]
        if len(verified) == 1:
            chosen = verified[0]
        elif len(verified) == 0:
            chosen = self._llm_choose_one(text, group, group)
        else:
            chosen = self._llm_choose_one(text, group, verified)

        if chosen is None:
            return []

        if self.reverify_callback is not None and chosen not in verified:
            if not self.reverify_callback(text, chosen):
                return []

        return [chosen]

    def resolve_all(self, text: str, relations: List[Relation]) -> Tuple[List[Relation], int, int]:
        conflicts = detect_relation_conflicts(relations)
        if not conflicts:
            return relations, 0, 0

        conflict_set = {rel for group in conflicts for rel in group}
        non_conflict_rels = [rel for rel in relations if rel not in conflict_set]
        resolved: List[Relation] = []
        for group in conflicts:
            resolved.extend(self.resolve_group(text, group))

        return non_conflict_rels + resolved, len(conflicts), len(conflicts)
