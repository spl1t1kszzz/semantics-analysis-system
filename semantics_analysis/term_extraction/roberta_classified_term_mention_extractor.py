from typing import List

import nltk
import torch
from transformers import AutoTokenizer, AutoModelForSequenceClassification

from semantics_analysis.entities import TermMention
from semantics_analysis.term_extraction.roberta_unclassified_term_mention_extractor import \
    RobertaUnclassifiedTermExtractor
from semantics_analysis.term_extraction.term_mention_extractor import TermMentionExtractor, PredictionsType
from semantics_analysis.utils import log_term_predictions

DEFAULT_THRESHOLD = 0.7


from semantics_analysis.constants import TERM_CLASSES as LABEL_LIST, TERM_ID_TO_CLASS as id2label, TERM_CLASS_TO_ID as label2id


class RobertaTermExtractor(TermMentionExtractor):
    def __init__(
            self,
            device: str,
            class_threshold: float = DEFAULT_THRESHOLD,
            log_labels: bool = False,
            term_threshold: float = 0.1
    ):
        self.device = device

        self.unclassified_term_extractor = RobertaUnclassifiedTermExtractor(
            device=device,
            min_term_prob=term_threshold
        )
        self.model_path = 'aiwannafly/semantics-analysis-term-classifier-v.0.2'
        self.tokenizer = AutoTokenizer.from_pretrained(
            'ai-forever/ruRoberta-large'
        )
        self.model = AutoModelForSequenceClassification.from_pretrained(
            self.model_path,
            num_labels=len(LABEL_LIST),
            id2label=id2label,
            label2id=label2id
        )
        self.model.to(device)
        self.threshold = class_threshold
        self.log_labels = log_labels

    def __call__(self, text: str, predictions: PredictionsType = None) -> List[TermMention]:
        term_predictions = []

        term_mentions = self.unclassified_term_extractor(text, term_predictions)

        if self.log_labels:
            log_term_predictions(term_predictions)

        sentences = nltk.tokenize.sent_tokenize(text)

        sentence_idx_by_term = {}

        curr_term_idx = 0

        for sent_idx, sent in enumerate(sentences):
            text_offset = text.find(sent) + len(sent)

            for i in range(curr_term_idx, len(term_mentions)):
                if term_mentions[i].end_pos <= text_offset:
                    sentence_idx_by_term[term_mentions[i]] = sent_idx
                    curr_term_idx += 1
                else:
                    break

        terms_by_sent_idx = {}

        for term, sent_idx in sentence_idx_by_term.items():
            if sent_idx not in terms_by_sent_idx:
                terms_by_sent_idx[sent_idx] = [term]
            else:
                terms_by_sent_idx[sent_idx].append(term)

        labeled_term_mentions = []

        for sent_idx in range(0, len(sentences)):
            sent = sentences[sent_idx]

            if sent_idx not in terms_by_sent_idx:
                continue

            text_offset = text.find(sent)

            terms = terms_by_sent_idx[sent_idx]

            for term in terms:
                term.text = sent
                term.start_pos -= text_offset
                term.end_pos -= text_offset

                if term.value != sent[term.start_pos:term.end_pos]:
                    continue

                markup_text = text[:term.start_pos] + '<term>' + term.value + '</term>' + text[term.end_pos:]

                term.text = text
                term.start_pos += text_offset
                term.end_pos += text_offset

                with torch.inference_mode():
                    outputs = self.model.forward(**self.tokenizer(markup_text, return_tensors='pt').to(device=self.device))

                probs = torch.squeeze(torch.softmax(outputs.logits, dim=1))

                if predictions:
                    predictions[term] = [(class_, p.item()) for class_, p in zip(LABEL_LIST, probs)]

                if max(probs) < self.threshold:
                    continue

                idx = outputs.logits.argmax(dim=1)[0].item()

                class_ = id2label[idx]

                term.class_ = class_

                labeled_term_mentions.append(term)

        return labeled_term_mentions
