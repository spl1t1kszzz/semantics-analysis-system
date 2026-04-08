import yaml


class Config:
    use_dict: bool
    display_graph: bool
    show_term_predictions: bool
    show_class_predictions: bool
    llm: str
    device: str
    show_explanation: bool
    split_on_sentences: bool
    log_prompts: bool
    log_llm_responses: bool
    use_multi_agent: bool
    use_conflict_dialogue: bool
    use_reverify_after_resolve: bool

    def __init__(
            self,
            use_dict: bool,
            display_graph: bool,
            show_term_predictions: bool,
            show_class_predictions: bool,
            llm: str,
            device: str,
            show_explanation: bool,
            split_on_sentences: bool,
            log_prompts: bool,
            log_llm_responses: bool,
            use_multi_agent: bool = True,
            use_conflict_dialogue: bool = False,
            use_reverify_after_resolve: bool = False,
    ):
        self.use_dict = use_dict
        self.display_graph = display_graph
        self.show_term_predictions = show_term_predictions
        self.show_class_predictions = show_class_predictions
        self.llm = llm
        self.device = device
        self.show_explanation = show_explanation
        self.split_on_sentences = split_on_sentences
        self.log_prompts = log_prompts
        self.log_llm_responses = log_llm_responses
        self.use_multi_agent = use_multi_agent
        self.use_conflict_dialogue = use_conflict_dialogue
        self.use_reverify_after_resolve = use_reverify_after_resolve

        if self.show_explanation:
            self.log_llm_responses = True


def load_config(file_path: str) -> Config:
    with open(file_path, 'r') as stream:
        config_dict = yaml.safe_load(stream)['app-config']

    return Config(
        config_dict['use-dict'],
        config_dict['display-graph'],
        config_dict['show-term-predictions'],
        config_dict['show-class-predictions'],
        config_dict['llm'],
        config_dict['device'],
        config_dict['show-explanation'],
        config_dict['split-on-sentences'],
        config_dict['log-prompts'],
        config_dict['log-llm-responses'],
        config_dict.get('use-multi-agent', True),
        config_dict.get('use-conflict-dialogue', False),
        config_dict.get('use-reverify-after-resolve', False),
    )
