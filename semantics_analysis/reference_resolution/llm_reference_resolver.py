from typing import List

from rich.progress import Progress

from semantics_analysis.entities import Term, TermMention
from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.ontology_utils import attribute_classes, CLASS_ALIASES
from semantics_analysis.reference_resolution.reference_resolver import ReferenceResolver
from semantics_analysis.utils import log


class LLMReferenceResolver(ReferenceResolver):

    def __init__(self,
                 progress: Progress,
                 model: str = 'gpt-4o-mini',
                 show_explanation: bool = False,
                 log_prompts: bool = False,
                 log_llm_responses: bool = False,
                 use_all_tokens: bool = False):
        with open('prompts/resolve_reference.txt', 'r', encoding='utf-8') as f:
            self.check_synonyms_prompt_template = f.read().strip()

        self.model = model
        self.llm_agent = LLMAgent(
            model=model,
            use_all_tokens=use_all_tokens
        )
        self.show_explanation = show_explanation
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses
        self.use_all_tokens = use_all_tokens
        self.token_idx = 0
        self.progress = progress

    def __call__(self, term_mentions: List[TermMention], text: str) -> List[Term]:
        term_mentions_by_class = {term.class_ : [] for term in term_mentions}

        for term in term_mentions:
            term_mentions_by_class[term.class_].append(term)

        terms = []
        for class_, term_mentions in term_mentions_by_class.items():
            group_by_term = {}

            curr_group_id = 0

            if class_ in attribute_classes:  # these classes should not have grouping
                terms.extend([
                    Term(class_, mention.value, mentions=[mention])

                    for mention in term_mentions if mention not in group_by_term
                ])
                continue

            total = sum((k - 1) for k in range(2, len(term_mentions) + 1))

            group_task = self.progress.add_task(description=f'Grouping terms 1/{total}', total=total)
            curr = 0

            for i in range(len(term_mentions)):
                for j in range(i + 1, len(term_mentions)):
                    term1, term2 = term_mentions[i], term_mentions[j]

                    if term1.value.lower() == term2.value.lower():
                        similar = True
                    elif term1.value.lower() in CLASS_ALIASES or term2.value.lower() in CLASS_ALIASES:
                        similar = False
                    else:
                        try:
                            similar = self.are_synonyms(term1, term2, text)
                        except Exception as e:
                            self.progress.remove_task(group_task)
                            raise e

                    curr += 1
                    if similar:
                        if term1 in group_by_term:
                            group_by_term[term2] = group_by_term[term1]
                        elif term2 in group_by_term:
                            group_by_term[term1] = group_by_term[term2]
                        else:
                            group_by_term[term1] = curr_group_id
                            group_by_term[term2] = curr_group_id
                            curr_group_id += 1

                    self.progress.update(group_task, description=f'Grouping terms {curr}/{total}', advance=1)

            self.progress.remove_task(group_task)

            # add single terms
            terms.extend([
                Term(class_, mention.value, mentions=[mention])

                for mention in term_mentions if mention not in group_by_term
            ])

            terms_by_group = {}

            for term, group in group_by_term.items():
                if group not in terms_by_group:
                    terms_by_group[group] = [term]
                else:
                    terms_by_group[group].append(term)

            terms.extend([
                Term(class_, mentions[0].value, mentions)

                for mentions in terms_by_group.values()
            ])
        return terms

    def are_synonyms(self, term1: TermMention, term2: TermMention, text: str) -> bool:
        if term1.class_ != term2.class_:
            return False

        input_text = (f'Текст: {text}\n'
                      f'Являются ли "{term1.value}" и "{term2.value}" названиями одной и той же сущности в этом тексте?\n'
                      f'Ответ:')

        prompt = self.check_synonyms_prompt_template
        prompt = prompt.replace('{input}', input_text)

        if self.log_prompts:
            log(f'[INPUT PROMPT]: {prompt}\n')

        response = self.llm_agent(prompt, max_new_tokens=16, stop_sequences=['.']).strip()

        if self.log_llm_responses:
            log(f'[ SYNONYMS ]: {response}\n')

        response = response.lower()

        return 'да' in response
