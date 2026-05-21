import re
from typing import List, Optional, Iterator, Tuple

import nltk.tokenize

from semantics_analysis.entities import Relation, Term, BoundedIterator
from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.ontology_utils import predicates_by_class_pair, \
    relations_metadata_by_class_pair
from semantics_analysis.relation_extraction.relation_extractor import RelationExtractor
from semantics_analysis.utils import log

ru_by_en_predicate = {
    'isExampleOf': 'является примером',
    'isPartOf': 'является частью',
    'isModificationOf': 'является модификацией'
}


class LLMRelationExtractor(RelationExtractor):

    def __init__(self,
                 model: str = 'gpt-4o-mini',
                 show_explanation: bool = False,
                 log_prompts: bool = False,
                 log_llm_responses: bool = False,
                 max_term_distance: int = 300,
                 considered_class1: Optional[str] = None,
                 considered_class2: Optional[str] = None,
                 use_multi_probe: bool = False,
                 ):

        self.model = model
        self.token_idx = 0

        self.llm_agent = LLMAgent(model=model)

        with open('prompts/relation_extraction.txt', 'r', encoding='utf-8') as f:
            self.prompt_template = f.read().strip()

        with open('prompts/directed_relation_extraction.txt', 'r', encoding='utf-8') as f:
            self.same_class_prompt_template = f.read().strip()

        with open('prompts/verify_relation.txt', 'r', encoding='utf-8') as f:
            self.verification_prompt_template = f.read().strip()

        with open('prompts/relation_extraction_multi_probe.txt', 'r', encoding='utf-8') as f:
            self.multi_probe_prompt_template = f.read().strip()

        self.show_explanation = show_explanation
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses
        self.max_term_distance = max_term_distance
        self.considered_class1 = considered_class1
        self.considered_class2 = considered_class2
        self.use_multi_probe = use_multi_probe

    def __call__(self, text: str, terms: List[Term]) -> BoundedIterator[Optional[Relation]]:
        # we seek only for binary relations
        # so every pair of terms can be checked

        # we use the knowledge from ontology and check only those pairs
        # that can be in a relation

        # so its normal to check for relation between Task and Method,
        # but it does not make sense to check for relation between InfoResource and Metric

        total = sum((k - 1) for k in range(2, len(terms) + 1))

        return BoundedIterator(total, self._predict_relations(text, terms))

    def _predict_relations(self, text: str, terms: List[Term]) -> Iterator[Optional[Relation]]:
        curr = 0

        terms_count = len(terms)

        for i in range(terms_count):
            for j in range(i + 1, terms_count):
                term1, term2 = terms[i], terms[j]
                curr += 1

                class1, class2 = term1.class_, term2.class_

                if self.considered_class1 and self.considered_class1 != class1:
                    yield None
                    continue

                if self.considered_class2 and self.considered_class2 != class2:
                    yield None
                    continue

                start_pos = min(term1.mentions[0].start_pos, term2.mentions[0].start_pos)

                end_pos = max(term1.mentions[0].end_pos, term2.mentions[0].end_pos)

                if end_pos - start_pos >= self.max_term_distance:
                    yield None
                    continue

                if (class1, class2) in predicates_by_class_pair:
                    predicates = predicates_by_class_pair[(class1, class2)]
                elif (class2, class1) in predicates_by_class_pair:
                    predicates = predicates_by_class_pair[(class2, class1)]

                    term1, term2 = term2, term1
                else:
                    predicates = []

                if not predicates:
                    yield None
                    continue

                try:
                    results = self.detect_predicates(term1, term2, text)
                except Exception as e:
                    raise e

                # Multi-agent: если стандартное извлечение не нашло ничего,
                # пробуем multi-probe как fallback (ловит доп. отношения)
                if not results and self.use_multi_probe:
                    results = self.detect_predicates_multi_probe(term1, term2, text)

                if not results:
                    yield None
                    continue

                for predicate, should_reverse in results:
                    if should_reverse:
                        rel = Relation(term2, predicate, term1)
                    else:
                        rel = Relation(term1, predicate, term2)

                    if not self.verify_relation(text, rel):
                        continue

                    yield rel

    def verify_relation(self, text: str, rel: Relation) -> bool:
        predicate = ru_by_en_predicate.get(rel.predicate, rel.predicate)

        relation_str = f'{rel.term1.value} {predicate} {rel.term2.value}'

        # Get relation description from ontology metadata
        relation_description = ''
        class1, class2 = rel.term1.class_, rel.term2.class_
        for key in [(class1, class2), (class2, class1)]:
            if key in relations_metadata_by_class_pair:
                metadata = relations_metadata_by_class_pair[key]
                if rel.predicate in metadata and 'yes' in metadata[rel.predicate]:
                    relation_description = metadata[rel.predicate]['yes'].get('description', '')
                    break
        if not relation_description:
            relation_description = predicate

        prompt = self.verification_prompt_template
        prompt = prompt.replace('{input}', relation_str)
        prompt = prompt.replace('{context}', text)
        prompt = prompt.replace('{relation_description}', relation_description)

        response = self.llm_agent(
            prompt,
            max_new_tokens=8,
            stop_sequences=['.', '\n']
        )

        return 'да' in response.lower()

    def detect_predicates(self, term1: Term, term2: Term, text: str) -> List[Tuple[str, bool]]:
        """Return all matching predicates from the LLM response (not just the first one).

        This enables the multi-agent conflict resolver to detect and resolve
        cases where multiple relations are plausible for a single term pair.
        """
        predicates = predicates_by_class_pair[(term1.class_, term2.class_)]

        if not predicates:
            return []

        prompt = self.create_llm_prompt(term1, term2, text)

        if self.log_prompts:
            log(f'[INPUT PROMPT]: {prompt}\n')

        # Don't use stop tokens so the model can list multiple predicates
        max_new_tokens = 512 if self.show_explanation else 256

        response = self.llm_agent(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=[]
        )

        if self.log_llm_responses:
            log(f'[LLM RESPONSE]: {response}\n')

        # Check only if the response starts with a refusal, not if "нет" appears
        # somewhere in the middle (e.g. "нет, не isPartOf, но isModificationOf")
        response_stripped = response.strip().lower()
        if response_stripped.startswith(('нет', 'none', 'no ', 'not ')):
            return []

        # Check for "Ответ: нет" pattern
        if 'ответ: нет' in response_stripped or 'ответ:нет' in response_stripped:
            return []

        results = []
        response_lower = response.lower()

        for predicate in predicates:
            if predicate in response:
                if term1.class_ != term2.class_:
                    results.append((predicate, False))
                else:
                    term1_value = term1.value.lower()
                    term2_value = term2.value.lower()

                    if term1_value not in response_lower or term2_value not in response_lower:
                        continue

                    predicate_pos = response_lower.index(predicate.lower())
                    term1_pos = response_lower.index(term1_value)

                    if term1_pos > predicate_pos:
                        results.append((predicate, True))
                    else:
                        results.append((predicate, False))

        return results

    def detect_predicates_multi_probe(self, term1: Term, term2: Term, text: str) -> List[Tuple[str, bool]]:
        """Один LLM-вызов с бинарным да/нет по каждому предикату.

        Вместо того, чтобы просить модель выбрать один предикат,
        для каждого возможного предиката задаём вопрос «да/нет».
        Это позволяет обнаружить конфликты (несколько «да»).
        """
        class1, class2 = term1.class_, term2.class_
        predicates = predicates_by_class_pair.get((class1, class2), [])
        if not predicates:
            return []

        metadata = relations_metadata_by_class_pair.get((class1, class2), {})

        # Собираем контекст вокруг терминов
        sentences = nltk.tokenize.sent_tokenize(text)
        start_pos = min(term1.mentions[0].start_pos, term2.mentions[0].start_pos)
        end_pos = max(term1.mentions[0].end_pos, term2.mentions[0].end_pos)
        offset = 0
        first_sent_id = None
        last_sent_id = None
        for idx, sent in enumerate(sentences):
            new_offset = offset + len(sent) + 1
            if first_sent_id is None and new_offset > start_pos:
                first_sent_id = idx
            if last_sent_id is None and end_pos < new_offset:
                last_sent_id = idx
            offset = new_offset
        if last_sent_id is None:
            last_sent_id = len(sentences) - 1
        context = ' '.join(sentences[first_sent_id:last_sent_id + 1])

        # Собираем примеры из метаданных
        examples_list = ''
        counter = 1
        for predicate, pred_meta in metadata.items():
            for answer in ['yes', 'no']:
                if answer not in pred_meta:
                    continue
                description = pred_meta[answer]['description']
                examples_list += f'{counter}. В этих примерах {description}:\n'
                for example in pred_meta[answer].get('examples', []):
                    example_text = example.get('text', '')
                    example_t1 = example.get(class1, '')
                    example_t2 = example.get(class2, '')
                    reply = 'да' if answer == 'yes' else 'нет'
                    examples_list += f'```\nТекст: {example_text}\nТермин {class1}: {example_t1}\nТермин {class2}: {example_t2}\n{predicate}: {reply}\n```\n'
                    counter += 1

        relations_list = ''
        questions_list = ''
        probe_lines = []
        answer_lines = []
        for i, predicate in enumerate(predicates, 1):
            pred_meta = metadata.get(predicate, {})
            desc = pred_meta.get('yes', {}).get('description', predicate)
            relations_list += f' - {predicate}: {desc}\n'
            probe_lines.append(f'{i}. {predicate} — {desc}')
            answer_lines.append(f'{i}. {predicate}:')

            questions_list += f' - {predicate}:\n'
            for question in pred_meta.get('evaluation-questions', [])[:3]:
                questions_list += f'     • {question}\n'

        prompt = self.multi_probe_prompt_template
        prompt = prompt.replace('{text}', context)
        prompt = prompt.replace('{class1}', class1)
        prompt = prompt.replace('{class2}', class2)
        prompt = prompt.replace('{term1}', term1.value)
        prompt = prompt.replace('{term2}', term2.value)
        prompt = prompt.replace('{relations_list}', relations_list.strip())
        prompt = prompt.replace('{questions_list}', questions_list.strip() or '—')
        prompt = prompt.replace('{examples_list}', examples_list)
        prompt = prompt.replace('{probe_list}', '\n'.join(probe_lines))
        prompt = prompt.replace('{answer_format}', '\n'.join(answer_lines))

        if self.log_prompts:
            log(f'[MULTI-PROBE PROMPT]: {prompt}\n')

        response = self.llm_agent(
            prompt,
            max_new_tokens=256,
            stop_sequences=[]
        )

        if self.log_llm_responses:
            log(f'[MULTI-PROBE RESPONSE]: {response}\n')

        results = self._parse_multi_probe_response(response, predicates)
        if not results:
            results = self._parse_multi_probe_response_legacy(response, predicates)
        return results

    @staticmethod
    def _parse_multi_probe_response(response: str, predicates: List[str]) -> List[Tuple[str, bool]]:
        """Строки вида «1. solves: да» или «solves: нет»."""
        if not predicates:
            return []
        alt = '|'.join(re.escape(p) for p in predicates)
        pattern = re.compile(
            rf'(?:^\s*\d+\.\s*)?({alt})\s*:\s*(да|нет|yes|no)\b',
            re.IGNORECASE | re.MULTILINE,
        )
        yes_tokens = {'да', 'yes'}
        found: List[Tuple[str, bool]] = []
        seen = set()
        for match in pattern.finditer(response):
            predicate = match.group(1)
            if predicate in seen:
                continue
            seen.add(predicate)
            if match.group(2).lower() in yes_tokens:
                found.append((predicate, False))
        return found

    @staticmethod
    def _parse_multi_probe_response_legacy(response: str, predicates: List[str]) -> List[Tuple[str, bool]]:
        results = []
        response_lower = response.lower()
        for predicate in predicates:
            predicate_lower = predicate.lower()
            if predicate_lower not in response_lower:
                continue
            pos = response_lower.index(predicate_lower) + len(predicate_lower)
            snippet = response_lower[pos:pos + 24]
            if re.search(r'\b(да|yes)\b', snippet):
                results.append((predicate, False))
            elif re.search(r'\b(нет|no)\b', snippet):
                continue
            elif 'да' in snippet or 'yes' in snippet:
                results.append((predicate, False))
        return results

    def detect_predicate(self, term1: Term, term2: Term, text: str) -> (Optional[str], bool):
        predicates = predicates_by_class_pair[(term1.class_, term2.class_)]

        if not predicates:
            return None, False

        prompt = self.create_llm_prompt(term1, term2, text)

        if self.log_prompts:
            log(f'[INPUT PROMPT]: {prompt}\n')

        stop_tokens = ['.']

        if term1.class_ == term2.class_ and not self.show_explanation:
            stop_tokens.append(',')

        max_new_tokens = 512 if self.show_explanation else 256

        response = self.llm_agent(
            prompt,
            max_new_tokens=max_new_tokens,
            stop_sequences=stop_tokens
        )

        if self.log_llm_responses:
            log(f'[LLM RESPONSE]: {response}\n')

        no_answers = ['none', 'нет', 'Нет', ' no ', 'not', ' не ']

        if response.startswith('none'):
            return None, False

        for no in no_answers:
            if no in response:
                return None, False

        for predicate in predicates:
            if predicate in response:

                if term1.class_ != term2.class_:
                    return predicate, False
                else:
                    response = response.lower()

                    term1_value = term1.value.lower()
                    term2_value = term2.value.lower()

                    if term1_value not in response or term2_value not in response:
                        return None, False

                    predicate_pos = response.index(predicate.lower())

                    term1_pos = response.index(term1_value)

                    if term1_pos > predicate_pos:
                        return predicate, True
                    else:
                        return predicate, False

        return None, False

    def create_llm_prompt(self, term1: Term, term2: Term, text: str) -> str:
        class1, class2 = term1.class_, term2.class_

        sentences = nltk.tokenize.sent_tokenize(text)

        start_pos = min(term1.mentions[0].start_pos, term2.mentions[0].start_pos)

        end_pos = max(term1.mentions[0].end_pos, term2.mentions[0].end_pos)

        offset = 0

        first_sent_id = None
        last_sent_id = None
        for idx, sent in enumerate(sentences):
            new_offset = offset + len(sent) + 1

            if first_sent_id is None and new_offset > start_pos:
                first_sent_id = idx

            if last_sent_id is None and end_pos < new_offset:
                last_sent_id = idx

        if last_sent_id is None:
            last_sent_id = len(sentences) - 1

        text = ' '.join(sentences[first_sent_id:last_sent_id + 1])

        if class1 == class2:
            return self.create_llm_prompt_same_class(term1, term2, text)

        prompt_metadata = relations_metadata_by_class_pair[(class1, class2)]

        relations_list = ''
        examples_list = ''
        counter = 1

        predicates = []
        questions_list = ''

        for predicate, metadata in prompt_metadata.items():
            predicates.append(predicate)

            description = metadata['yes']['description']
            relations_list += f' - {predicate} : {description}\n'

            for answer in ['yes', 'no']:
                description = metadata[answer]['description']

                examples_list += f'{counter}. В этих примерах {description}:\n'

                for example in metadata[answer]['examples']:
                    example_text = example['text']

                    example_term1 = example[class1]
                    example_term2 = example[class2]

                    description = None if 'description' not in example else example['description']

                    reply = ''
                    if answer == 'no':
                        reply += 'Ответ: нет'
                    else:
                        reply += f'Ответ: да\n{predicate}'

                    if not description and not self.show_explanation:
                        reply += '.'
                    else:
                        reply += '\n'
                        if description:
                            description = description.replace('.', '')
                            reply += f'Объяснение: {description}.'
                        elif self.show_explanation:
                            reply += f'Объяснение: <объяснение>.'

                    examples_list += '```\n'
                    examples_list += (f'Текст: {example_text}\n'
                                      f'Термин {class1}: {example_term1}\n'
                                      f'Термин {class2}: {example_term2}\n'
                                      f'Есть ли подходящее отношение между терминами "{example_term1}" и "{example_term2}" в этом тексте?\n'
                                      f'{reply}\n')
                    examples_list += '```\n'

                    counter += 1

            questions_list += f' - Отношение {predicate}:\n'

            for question in prompt_metadata[predicate]['evaluation-questions']:
                questions_list += f'   - {question}\n'

        input_text = (f'Текст: {text}\n'
                      f'Термин {class1}: {term1.value}\n'
                      f'Термин {class2}: {term2.value}\n'
                      f'Есть ли подходящее отношение между терминами "{term1.value}" и "{term2.value}" в этом тексте?\n'
                      f'Ответ:')

        prompt = self.prompt_template
        if self.show_explanation:
            prompt = prompt.replace('{explanation}', '\nОбъяснение: <объяснение>')
        else:
            prompt = prompt.replace('{explanation}', '')

        prompt = prompt.replace('{class1}', class1)
        prompt = prompt.replace('{class2}', class2)
        prompt = prompt.replace('{relations_list}', relations_list)
        prompt = prompt.replace('{examples_list}', examples_list)
        prompt = prompt.replace('{questions_list}', questions_list)
        prompt = prompt.replace('{relation_names}', str(predicates))
        prompt = prompt.replace('{input}', input_text)

        return prompt.strip()

    def create_llm_prompt_same_class(self, term1: Term, term2: Term, text: str) -> str:
        assert term1.class_ == term2.class_

        class_ = term1.class_

        prompt_metadata = relations_metadata_by_class_pair[(class_, class_)]

        relations_list = ''
        examples_list = ''
        counter = 1

        predicates = []
        questions_list = ''

        for predicate, metadata in prompt_metadata.items():
            predicates.append(predicate)

            description = metadata['yes']['description']
            relations_list += f' - {predicate} : {description}\n'

            for answer in ['yes', 'no']:
                description = metadata[answer]['description']

                examples_list += f'{counter}. В этих примерах {description}:\n'

                for example in metadata[answer]['examples']:
                    example_text = example['text']

                    example_term1 = example[class_ + '_1']
                    example_term2 = example[class_ + '_2']

                    description = None if 'description' not in example else example['description']

                    reply = ''
                    if answer == 'no':
                        reply += 'Ответ: нет'
                    else:
                        reply += f'Ответ: да\n'
                        reply += f'{example_term1} {predicate} {example_term2}'

                    if not description and not self.show_explanation:
                        reply += '.'
                    else:
                        reply += '\n'
                        if description:
                            description = description.replace('.', '')
                            reply += f'Объяснение: {description}.'
                        elif self.show_explanation:
                            reply += f'Объяснение: <объяснение>.'

                    examples_list += '```\n'
                    examples_list += (f'Текст: {example_text}\n'
                                      f'Термины {class_}: {example_term1}, {example_term2}\n'
                                      f'Есть ли подходящее отношение между этими терминами в этом тексте?\n'
                                      f'{reply}\n')
                    examples_list += '```\n'

                    counter += 1

            questions_list += f' - Отношение {predicate}:\n'

            for question in prompt_metadata[predicate]['evaluation-questions']:
                questions_list += f'   - {question}\n'

        input_text = (f'Текст: {text}\n'
                      f'Термины {class_}: {term1.value}, {term2.value}\n'
                      f'Есть ли подходящее отношение между этими терминами в этом тексте?\n'
                      f'Ответ:')

        prompt = self.same_class_prompt_template
        if self.show_explanation:
            prompt = prompt.replace('{explanation}', '\nОбъяснение: <объяснение>')
        else:
            prompt = prompt.replace('{explanation}', '')

        prompt = prompt.replace('{class}', class_)
        prompt = prompt.replace('{relations_list}', relations_list)
        prompt = prompt.replace('{examples_list}', examples_list)
        prompt = prompt.replace('{questions_list}', questions_list)
        prompt = prompt.replace('{relation_names}', str(predicates))
        prompt = prompt.replace('{input}', input_text)

        return prompt.strip()
