import json

import inquirer
from colorama import init as colorama_init, Style

from semantics_analysis.config import load_config
from semantics_analysis.factory import build_pipeline
from semantics_analysis.ontology_entities import convert_to_ont_entities
from semantics_analysis.pipelines import AnalysisResult
from semantics_analysis.utils import log_found_relations, AlignedProgress
from semantics_analysis.visualization import display_relation_graph

LOG_STYLE = Style.DIM


def analyze_text(app_config):
    text = input(f'{LOG_STYLE}[      INPUT     ]{Style.RESET_ALL}: ')
    print()

    with AlignedProgress() as progress:
        pipeline = build_pipeline(app_config, progress)
        result = pipeline(AnalysisResult(text))

    objects, ont_relations = convert_to_ont_entities(result.terms, result.relations)
    ont_entities_json = {
        'objects': [o.to_json() for o in objects],
        'relations': [r.to_json() for r in ont_relations]
    }

    with open('ont_entities.json', 'w', encoding='utf-8') as wf:
        json.dump(ont_entities_json, wf, ensure_ascii=False, indent=2)

    if result.relations and app_config.display_graph:
        display_relation_graph(result.terms, result.relations)
    else:
        log_found_relations(result.relations)


def main():
    colorama_init()
    app_config = load_config('config.yml')

    while True:
        analyze_text(app_config)

        question = inquirer.questions.List(
            name='answer',
            message='Продолжить анализировать тексты дальше',
            choices=['Нет.', 'Да.']
        )

        answer = inquirer.prompt([question])['answer']

        if answer != 'Да.':
            break


if __name__ == '__main__':
    main()
