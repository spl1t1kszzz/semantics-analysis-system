from typing import Optional

from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.ontology_utils import term_metadata_by_class
from semantics_analysis.term_extraction.term_verifier import TermVerifier


class LLMTermVerifier(TermVerifier):
    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        if llm_agent:
            self.llm_agent = llm_agent
        else:
            self.llm_agent = LLMAgent(use_all_tokens=True)

        with open('prompts/verify_term.txt', 'r', encoding='utf-8') as f:
            self.verification_prompt_template = f.read().strip()

    def __call__(self, term_value: str, term_class: str, text: str) -> bool:
        desc = term_metadata_by_class[term_class]['description']
        class_name = term_metadata_by_class[term_class]['name']

        positive_list = ''
        for positive_example in term_metadata_by_class[term_class]['examples']['positive']:
            p_value = positive_example['value']
            p_desc = positive_example['description']

            positive_list += f' - {p_value}: {p_desc}\n'

        positive_list = positive_list.strip()

        negative_list = ''
        for negative_example in term_metadata_by_class[term_class]['examples']['negative']:
            p_value = negative_example['value']
            p_desc = negative_example['description']

            negative_list += f' - {p_value}: {p_desc}\n'

        negative_list = negative_list.strip()

        prompt = self.verification_prompt_template
        prompt = prompt.replace('{class}', class_name)
        prompt = prompt.replace('{description}', desc)
        prompt = prompt.replace('{text}', text)
        prompt = prompt.replace('{term}', term_value)
        prompt = prompt.replace('{positive}', positive_list)
        prompt = prompt.replace('{negative}', negative_list)

        response = self.llm_agent(
            prompt,
            max_new_tokens=8,
            stop_sequences=['.', '\n']
        )

        return 'да' in response.lower()
