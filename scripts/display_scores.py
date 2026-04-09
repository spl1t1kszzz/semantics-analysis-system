import json
import sys
from typing import Tuple, List, Dict, Optional

from colorama import Style, init
from rich.progress import Progress, TextColumn, BarColumn

from semantics_analysis.ontology_utils import loaded_relation_ids


def parse_scores(
        scores_path_: str,
        relations_to_consider: Optional[List[str]],
        storage: Dict[str, float]
) -> List[Tuple[str, int, Dict[str, float]]]:
    with open(scores_path_, 'r', encoding='utf-8') as f:
        scores_by_relation = json.load(f)

    avg_recall = 0
    avg_precision = 0
    count = 0

    result = []

    for relation, scores in scores_by_relation.items():
        if relations_to_consider and relation not in relations_to_consider:
            continue

        class1, _, class2 = relation.split('_')

        correct = scores['predicted']['correct']['count']
        incorrect = scores['predicted']['incorrect']['count']

        if correct + incorrect > 0:
            precision = correct / (correct + incorrect)
        else:
            precision = 0

        found = scores['expected']['found']['count']
        not_found = scores['expected']['not_found']['count']

        if found + not_found > 0:
            recall = found / (found + not_found)
        else:
            recall = 0

        if found + not_found == 0:
            continue

        recall = int(recall * 100) / 100.0
        precision = int(precision * 100) / 100.0

        occurrences_count_ = max(found + not_found, correct + incorrect)

        avg_recall += recall
        avg_precision += precision
        count += 1

        result.append((relation, occurrences_count_, {'Recall': recall, 'Precision': precision}))

    storage['avg_recall'] = avg_recall
    storage['avg_precision'] = avg_precision
    storage['count'] = count

    return result

def save_scores_to_md(
        md_path: str,
        rel_cnt_scores: List[Tuple[str, int, Dict[str, float]]],
        storage: Dict[str, float]
):
    lines = []

    lines.append("# Relation Extraction Metrics\n")
    lines.append("| Relation | Occurrences | Precision | Recall |")
    lines.append("|----------|-------------|-----------|--------|")

    for relation, occ_cnt, scores in rel_cnt_scores:
        precision = scores["Precision"]
        recall = scores["Recall"]

        lines.append(
            f"| {relation} | {occ_cnt} | {precision:.2f} | {recall:.2f} |"
        )

    # --- AVERAGE ---
    count = storage["count"]
    avg_precision = storage["avg_precision"] / count if count > 0 else 0.0
    avg_recall = storage["avg_recall"] / count if count > 0 else 0.0

    lines.append("| **AVERAGE** | — | "
                 f"**{avg_precision:.2f}** | **{avg_recall:.2f}** |")

    with open(md_path, "w", encoding="utf-8") as f:
        f.write("\n".join(lines))



def main():
    color_by_metric = {
        'Recall': 'green',
        'Precision': 'magenta'
    }

    description_length = 20

    init()

    relations_to_consider = []

    for arg in sys.argv[1:]:
        if arg in loaded_relation_ids:
            relations_to_consider.append(arg)
        else:
            print(f'Invalid relation id: {arg}')
            exit(0)

    if relations_to_consider:
        total_id = '_and_'.join(relations_to_consider)

        scores_path = f'tests/{total_id}/scores.json'
    else:
        scores_path = 'results/shot_rel_new_prompts.json'

    storage = {}

    rel_cnt_scores = parse_scores(scores_path, relations_to_consider, storage)

    rel_cnt_scores = sorted(rel_cnt_scores, key=lambda x: x[2]['Precision'], reverse=True)

    md_path = "results/shot_rel_new_prompts_metrics.md"
    save_scores_to_md(md_path, rel_cnt_scores, storage)

    print(f"\nMarkdown report saved to {md_path}")

    for relation, occurrences_count, scores in rel_cnt_scores:
        print()
        print(f'{Style.DIM}—————————————————————————————————————————————————————————————{Style.RESET_ALL}')
        print()
        print(f'[{occurrences_count}] {Style.BRIGHT}{relation}{Style.RESET_ALL}\n')

        for metric, value in scores.items():
            color = color_by_metric[metric]

            diff = description_length - len(f'{metric}: {value}')

            description = f'[{color}]{metric}: {value}'

            if diff > 0:
                description = ' ' * diff + description

            with Progress(TextColumn(text_format=description, justify='right'),
                          BarColumn(complete_style=color)) as progress:
                metric_task = progress.add_task(total=1.0, description='')

                progress.update(metric_task, description='', advance=value)

        print()

    if relations_to_consider:
        exit(0)

    count = storage['count']
    avg_recall = storage['avg_recall']
    avg_precision = storage['avg_precision']

    print(f'{Style.DIM}—————————————————————————————————————————————————————————————{Style.RESET_ALL}')
    print()
    print(f'{Style.BRIGHT}[AVERAGE SCORES]{Style.RESET_ALL} for {count} relation classes:\n')

    scores = {
        'Recall': int(100 * avg_recall / count) / 100,
        'Precision': int(100 * avg_precision / count) / 100
    }

    description_length = 16

    for metric, value in scores.items():
        color = color_by_metric[metric]

        diff = description_length - len(f'{metric}: {value}')

        description = f'[{color}]{metric}: {value}'

        if diff > 0:
            description = ' ' * diff + description

        with Progress(TextColumn(text_format=description, justify='right'),
                      BarColumn(complete_style=color)) as progress:
            metric_task = progress.add_task(total=1.0, description='')

            progress.update(metric_task, description='', advance=value)


if __name__ == '__main__':
    main()
