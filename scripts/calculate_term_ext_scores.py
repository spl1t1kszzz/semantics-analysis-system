import os
from typing import List, Any, Dict
import json

from rich.progress import Progress
from rich.table import Table
from rich.console import Console

from semantics_analysis.entities import (
    TermMention,
    Sentence,
    read_sentences,
)
from semantics_analysis.term_extraction.llm_term_verifier import LLMTermVerifier
from semantics_analysis.term_extraction.roberta_classified_term_mention_extractor import (
    RobertaTermExtractor,
)
from semantics_analysis.term_extraction.term_verifier import TermVerifier


# ============================================================
# NER METRICS (MENTION-LEVEL)
# ============================================================

def update_ner_scores_mentions(
        gold_mentions: List[TermMention],
        predicted_mentions: List[TermMention],
        scores: Dict[str, Any]
):
    """
    Считает TP / FP / FN для NER на уровне упоминаний (TermMention).
    Совпадение: (class, start_pos, end_pos)
    """
    gold_set = {(m.class_, m.start_pos, m.end_pos) for m in gold_mentions}
    pred_set = {(m.class_, m.start_pos, m.end_pos) for m in predicted_mentions}

    # инициализация классов
    for cls, _, _ in gold_set | pred_set:
        if cls not in scores:
            scores[cls] = {
                "predicted": {"correct": 0, "incorrect": 0},
                "expected": {"found": 0, "not_found": 0}
            }

    # FN / TP
    for m in gold_set:
        cls = m[0]
        if m in pred_set:
            scores[cls]["expected"]["found"] += 1
            scores[cls]["predicted"]["correct"] += 1
        else:
            scores[cls]["expected"]["not_found"] += 1

    # FP
    for m in pred_set:
        cls = m[0]
        if m not in gold_set:
            scores[cls]["predicted"]["incorrect"] += 1


# ============================================================
# TERM VERIFIER APPLICATION
# ============================================================

def apply_term_verifier(
        mentions: List[TermMention],
        verifier: TermVerifier
) -> List[TermMention]:
    """
    Применяет TermVerifier к списку TermMention.
    Оставляет только подтверждённые LLM упоминания.
    """
    return [
        m for m in verifier.filter_terms(mentions)
        if m is not None
    ]


# ============================================================
# MAIN EVALUATION LOOP
# ============================================================

def calculate_ner_scores(
        sentences: List[Sentence],
        ner_extractor,
        term_verifier: TermVerifier,
        scores: Dict[str, Any],
        progress: Progress
):
    total = len(sentences)

    task = progress.add_task(
        description=f"[green]Извлечение терминов (0/{total})",
        total=total
    )

    for i, sent in enumerate(sentences, start=1):

        # --- GOLD: TermMention ---
        gold_mentions: List[TermMention] = []
        for term in sent.terms:
            gold_mentions.extend(term.mentions)

        # --- PREDICTED: TermMention ---
        predicted_mentions: List[TermMention] = ner_extractor(sent.text)

        # --- LLM verification ---
        predicted_mentions = apply_term_verifier(
            mentions=predicted_mentions,
            verifier=term_verifier
        )

        # --- METRICS ---
        update_ner_scores_mentions(
            gold_mentions=gold_mentions,
            predicted_mentions=predicted_mentions,
            scores=scores
        )

        progress.update(
            task,
            advance=1,
            description=f"[green]NER evaluation ({i}/{total})"
        )

    progress.remove_task(task)

def compute_term_ext_metrics(scores: dict) -> dict:
    metrics = {}

    for cls, data in scores.items():
        tp = data["predicted"]["correct"]
        fp = data["predicted"]["incorrect"]
        fn = data["expected"]["not_found"]

        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = (
            2 * precision * recall / (precision + recall)
            if (precision + recall) > 0 else 0.0
        )

        metrics[cls] = {
            "TP": tp,
            "FP": fp,
            "FN": fn,
            "Precision": precision,
            "Recall": recall,
            "F1": f1,
        }

    return metrics

def load_scores(path: str) -> dict:
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def print_ner_metrics_table(metrics: dict):
    table = Table(title="NER metrics by class")

    table.add_column("Class", justify="left")
    table.add_column("TP", justify="right")
    table.add_column("FP", justify="right")
    table.add_column("FN", justify="right")
    table.add_column("Precision", justify="right")
    table.add_column("Recall", justify="right")
    table.add_column("F1", justify="right")

    for cls, m in sorted(metrics.items(), key=lambda x: x[1]["F1"], reverse=True):
        table.add_row(
            cls,
            str(m["TP"]),
            str(m["FP"]),
            str(m["FN"]),
            f"{m['Precision']:.3f}",
            f"{m['Recall']:.3f}",
            f"{m['F1']:.3f}",
        )

    Console().print(table)

def save_scores(scores: dict, path: str):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    with open(path, "w", encoding="utf-8") as f:
        json.dump(scores, f, ensure_ascii=False, indent=2)


# ============================================================
# ENTRY POINT
# ============================================================

if __name__ == "__main__":
    sentences_file = "tests/short_sent.json"
    scores_path = "results/short_term_ext_new_prompts.json"
    print(f"Sentences file: {sentences_file}")
    print(f"Scores file: {scores_path}")

    sentences = read_sentences(sentences_file)

    # --- NER extractor ---
    term_extractor = RobertaTermExtractor(
        device="cpu",
        term_threshold=0.2,
        class_threshold=0.5
    )

    # --- LLM verifier ---
    term_verifier = LLMTermVerifier()

    scores: Dict[str, Any] = {}

    with Progress() as progress:
        calculate_ner_scores(
            sentences=sentences,
            ner_extractor=term_extractor,
            term_verifier=term_verifier,
            scores=scores,
            progress=progress
        )

    # ---------- SAVE RAW SCORES ----------
    save_scores(scores, scores_path)

    print(f"\nScores saved to: {scores_path}")
    # ---------- LOAD + METRICS ----------
    loaded_scores = load_scores(scores_path)
    metrics = compute_term_ext_metrics(loaded_scores)

    print_ner_metrics_table(metrics)