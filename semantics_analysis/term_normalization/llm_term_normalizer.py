from typing import Optional

from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.term_normalization.term_normalizer import TermNormalizer


class LLMTermNormalizer(TermNormalizer):

    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        self.cached_results = {}

        if llm_agent:
            self.llm_agent = llm_agent
        else:
            self.llm_agent = LLMAgent(use_all_tokens=True)

        with open('prompts/normalization.txt', 'r', encoding='utf-8') as f:
            self.prompt_template = f.read()

    def __call__(self, term: str) -> str:
        if not term:
            return term

        cached_result = self.cached_results.get(term, None)

        if cached_result is not None:
            return cached_result

        prompt = self.prompt_template.replace('{input}', term)

        normalized = self.llm_agent(
            prompt,
            stop_sequences=['\n', '(', '.'],
            max_new_tokens=256
        ).replace('(', '').strip()

        while normalized.endswith('.'):
            normalized = normalized[:-1]

        if len(normalized) == 1:
            normalized = term

        normalized = normalized.strip()

        self.cached_results[term] = normalized
        return normalized


def main():
    term = "техникой вероятностного латентно-семантического индекса"

    normalizer = LLMTermNormalizer()

    normalized = normalizer(term)

    print(normalized)

    a = [1, 2, 3]

    print(a[5:])


if __name__ == '__main__':
    main()
