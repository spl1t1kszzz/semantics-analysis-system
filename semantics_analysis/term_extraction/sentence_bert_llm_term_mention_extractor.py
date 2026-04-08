import json
from typing import List, Optional, Dict, Tuple

import nltk
from rich.progress import Progress
from tqdm import tqdm

from semantics_analysis.entities import TermMention
from semantics_analysis.llm_agent import LLMAgent
from semantics_analysis.meaning_model.meaning import Meaning
from semantics_analysis.meaning_model.mpnet_meaning_model import MpnetMeaningModel
from semantics_analysis.ontology_utils import term_metadata_by_class
from semantics_analysis.phrase_extractor import PhraseExtractor
from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier

MIN_SIMILARITY = 0.3
EXAMPLES_PER_CLASS = 2
MAX_CLASSES_IN_PROMPT = 6


class SentenceBERTLLMTermExtractor:

    def __init__(self, llm_agent: Optional[LLMAgent] = None):
        if llm_agent:
            self.llm_agent = llm_agent
        else:
            self.llm_agent = LLMAgent(use_all_tokens=True)

        self.term_verifier = LLMTermVerifier(llm_agent)

        self.phrase_extractor = PhraseExtractor()

        self.meaning_model = MpnetMeaningModel()

        with open('prompts/classify_word.txt', 'r', encoding='utf-8') as f:
            self.word_classification_prompt_template = f.read().strip()

        with open('prompts/classify_phrase.txt', 'r', encoding='utf-8') as f:
            self.phrase_classification_prompt_template = f.read().strip()

        with open('metadata/terms_by_class.json', 'r', encoding='utf-8') as f:
            terms_by_class = json.load(f)

        self.class_by_term = {}

        try:
            with open('metadata/meaning_by_term.json', 'r', encoding='utf-8') as f:
                vector_by_term = json.load(f)
        except IOError as _:

            # file not found
            vector_by_term = {}
            for class_, terms in terms_by_class.items():

                for term in tqdm(terms):
                    vector_by_term[term] = self.meaning_model.get_meaning(term).values

            with open('metadata/meaning_by_term.json', 'w', encoding='utf-8') as wf:
                json.dump(vector_by_term, wf, indent=2, ensure_ascii=False)

        for class_, terms in terms_by_class.items():

            for term in terms:
                self.class_by_term[term] = class_

        self.meaning_by_term = {}
        for term, vector in vector_by_term.items():
            self.meaning_by_term[term] = Meaning(term, vector)

    def __call__(self, text: str, progress: Progress) -> List[TermMention]:

        sentences = nltk.tokenize.sent_tokenize(text)

        if len(sentences) > 1:
            all_terms = []
            total = len(sentences)
            sentences_task = progress.add_task(description=f'Sentence 0/{total}', total=total)

            for idx, sent in enumerate(sentences):
                try:
                    all_terms.extend(self(sent, progress))
                except Exception as e:
                    progress.remove_task(sentences_task)
                    raise e

                progress.update(sentences_task, description=f'Sentence {idx + 1}/{total}', advance=1)

            progress.remove_task(sentences_task)

            return list(set(all_terms))

        ids_by_phrase = self.phrase_extractor(text)

        phrases = list(ids_by_phrase.keys())

        # print(phrases)

        if not phrases:
            return []

        found_terms = []

        total = len(phrases)
        resolving = progress.add_task(description=f'Resolving terms 0/{total}', total=total)

        checked_phrases = set()

        for idx, phrase in enumerate(phrases):
            if phrase in checked_phrases or phrase.startswith('-') or phrase.endswith('-'):
                continue
            else:
                checked_phrases.add(phrase)

            examples_by_class = self._find_potential_classes(phrase)

            try:
                response = self._resolve_class(text, phrase, examples_by_class)
            except Exception as e:
                progress.remove_task(resolving)
                raise e

            progress.update(resolving, description=f'Resolving terms {idx + 1}/{total}', advance=1)

            if not response:
                continue

            class_, term = response

            try:
                verified = self.term_verifier(term, class_, text)
            except Exception as e:
                progress.remove_task(resolving)
                raise e

            if verified:
                found_terms.append(TermMention(class_, term, len(text), text))

        progress.remove_task(resolving)

        merged_terms = set()

        for i in range(len(found_terms)):
            for j in range(i + 1, len(found_terms)):
                term1, term2 = found_terms[i], found_terms[j]

                if term1.class_ != term2.class_:
                    continue

                ids1, ids2 = ids_by_phrase[term1.value], ids_by_phrase[term2.value]

                intersection = [n for n in ids1 if n in ids2]

                if not intersection:
                    continue

                if len(intersection) == len(ids1) or len(intersection) == len(ids2):
                    continue

                if ids2[0] < ids1[0]:
                    term1, term2 = term2, term1
                    ids1, ids2 = ids2, ids1

                words1 = term1.value.split()
                words2 = term2.value.split()

                taken_ids = set()
                words = words1 + words2

                ids = ids1 + ids2

                if len(words) != len(ids):
                    continue

                final_words = []
                for idx, word in zip(ids, words):
                    if idx in taken_ids:
                        continue
                    taken_ids.add(idx)
                    final_words.append(word)

                value = ' '.join(final_words)

                merged_terms.add(TermMention(term1.class_, value, len(text), text))

        for t in merged_terms:
            found_terms.append(t)

        sorted_terms = sorted(found_terms, key=lambda t: len(t.value))

        filtered_terms = set()

        for i in range(len(sorted_terms)):
            smaller_term = sorted_terms[i]
            is_part_of_bigger = False

            for j in range(i + 1, len(sorted_terms)):
                bigger_term = sorted_terms[j]

                if smaller_term.class_ == bigger_term.class_ and smaller_term.value in bigger_term.value:
                    is_part_of_bigger = True
                    break

            if not is_part_of_bigger:
                filtered_terms.add(smaller_term)

        return list(filtered_terms)

    def _resolve_class(self, text: str, term: str, examples_by_class: Dict[str, List[str]]) -> Optional[Tuple[str, str]]:
        class_descriptions = ''
        for class_, examples in examples_by_class.items():
            desc = term_metadata_by_class[class_]['description']
            name = term_metadata_by_class[class_]['name']

            examples = ', '.join(examples)

            class_descriptions += (f'- {class_}:\n'
                                   f'    Название: {name}\n'
                                   f'    Описание: {desc}\n'
                                   f'    Примеры: {examples}\n\n')

        class_descriptions = class_descriptions.strip()

        if ' ' in term:
            prompt = self.phrase_classification_prompt_template
        else:
            prompt = self.word_classification_prompt_template

        prompt = prompt.replace('{text}', text)
        prompt = prompt.replace('{term}', term)
        prompt = prompt.replace('{class_descriptions}', class_descriptions)
        prompt = prompt.replace('{classes}', ', '.join(examples_by_class))

        response = self.llm_agent(
            prompt,
            max_new_tokens=24,
            stop_sequences=['.', '(', 'не', '\n']
        ).replace('(', '').strip()

        # print(prompt)
        # print(term)
        # print(response)

        if response.startswith('не'):
            return None

        while response.endswith('.'):
            response = response[:len(response) - 1]

        for class_ in examples_by_class:
            if class_ in response:
                return class_, term

        # failed to get correct response from LLM
        return None

    def _find_potential_classes(self, phrase: str) -> Dict[str, List[str]]:
        phrase_meaning = self.meaning_model.get_meaning(phrase)

        candidates_classes = set()

        examples_by_class = {}

        similarity_by_term = {}

        for term, class_ in self.class_by_term.items():
            term_meaning = self.meaning_by_term.get(term, None)

            if not term_meaning:
                continue

            similarity = term_meaning.calculate_similarity(phrase_meaning)

            if similarity < MIN_SIMILARITY:
                continue

            similarity_by_term[term] = similarity

            candidates_classes.add(class_)

            if class_ not in examples_by_class:
                examples_by_class[class_] = [term]
            else:
                examples_by_class[class_].append(term)

        similarity_by_class = {}
        sorted_examples_by_class = {}
        for class_, examples in examples_by_class.items():
            sorted_examples_by_class[class_] = sorted(examples, key=lambda ex: similarity_by_term[ex], reverse=True)[:EXAMPLES_PER_CLASS]

            similarity_by_class[class_] = max(similarity_by_term[ex] for ex in sorted_examples_by_class[class_])

        sorted_classes = sorted(candidates_classes, key=lambda c: similarity_by_class[c], reverse=True)[:MAX_CLASSES_IN_PROMPT]

        return {class_: sorted_examples_by_class[class_] for class_ in sorted_classes}


def main():
    extractor = SentenceBERTLLMTermExtractor()

    print('Press q to stop.')

    while True:
        text = input("Enter text: ").strip()

        if text == 'q':
            break

        with Progress() as progress:
            found_terms = extractor(text, progress)

        for term in found_terms:
            print(f'{term.value} -> {term.class_}')

        print()


if __name__ == '__main__':
    main()
