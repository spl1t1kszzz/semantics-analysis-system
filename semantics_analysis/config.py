from dataclasses import dataclass, field
from pathlib import Path
from typing import Optional

import yaml


@dataclass
class Config:
    # Term extraction
    use_dict: bool = True
    device: str = 'cpu'
    term_threshold: float = 0.2
    class_threshold: float = 0.5

    # LLM
    llm: str = ''

    # Relation extraction
    max_term_distance: int = 300

    # Multi-agent
    use_multi_agent: bool = True
    use_conflict_dialogue: bool = False
    use_reverify_after_resolve: bool = False

    # Display
    display_graph: bool = True
    show_term_predictions: bool = False
    show_class_predictions: bool = False
    show_explanation: bool = False
    split_on_sentences: bool = False

    # Logging
    log_prompts: bool = False
    log_llm_responses: bool = False

    def __post_init__(self):
        if not self.llm:
            raise ValueError("Модель LLM не задана. Укажите 'llm' в config.yml.")
        if self.show_explanation:
            self.log_llm_responses = True


def load_config(file_path: str = 'config.yml') -> Config:
    path = Path(file_path)
    if not path.exists():
        return Config()

    with open(path, 'r') as stream:
        config_dict = yaml.safe_load(stream).get('app-config', {})

    field_map = {
        'use-dict': 'use_dict',
        'device': 'device',
        'term-threshold': 'term_threshold',
        'class-threshold': 'class_threshold',
        'llm': 'llm',
        'max-term-distance': 'max_term_distance',
        'use-multi-agent': 'use_multi_agent',
        'use-conflict-dialogue': 'use_conflict_dialogue',
        'use-reverify-after-resolve': 'use_reverify_after_resolve',
        'display-graph': 'display_graph',
        'show-term-predictions': 'show_term_predictions',
        'show-class-predictions': 'show_class_predictions',
        'show-explanation': 'show_explanation',
        'split-on-sentences': 'split_on_sentences',
        'log-prompts': 'log_prompts',
        'log-llm-responses': 'log_llm_responses',
    }

    kwargs = {}
    for yaml_key, attr_name in field_map.items():
        if yaml_key in config_dict:
            kwargs[attr_name] = config_dict[yaml_key]

    return Config(**kwargs)
