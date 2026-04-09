import tempfile
from pathlib import Path

from semantics_analysis.config import Config, load_config


class TestConfig:
    def test_defaults(self):
        c = Config()
        assert c.use_dict is True
        assert c.device == 'cpu'
        assert c.llm == 'gpt-4o-mini'
        assert c.max_term_distance == 300
        assert c.use_multi_agent is True
        assert c.log_prompts is False

    def test_show_explanation_enables_log_responses(self):
        c = Config(show_explanation=True)
        assert c.log_llm_responses is True

    def test_load_from_yaml(self):
        yaml_content = """
app-config:
  use-dict: false
  device: 'cuda'
  llm: 'gpt-4'
  display-graph: false
  show-term-predictions: false
  show-class-predictions: false
  show-explanation: false
  split-on-sentences: false
  log-prompts: true
  log-llm-responses: false
"""
        with tempfile.NamedTemporaryFile(mode='w', suffix='.yml', delete=False) as f:
            f.write(yaml_content)
            f.flush()
            c = load_config(f.name)

        assert c.use_dict is False
        assert c.device == 'cuda'
        assert c.llm == 'gpt-4'
        assert c.log_prompts is True

    def test_load_missing_file_returns_defaults(self):
        c = load_config('/nonexistent/path/config.yml')
        assert c.use_dict is True
        assert c.device == 'cpu'
