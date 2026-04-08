import os
from typing import List, Optional

from openai import OpenAI

from semantics_analysis.values import values

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

# По умолчанию используется OpenAI API (нужен OPENAI_API_KEY в .env или окружении).
# OPENAI_API_BASE задаётся только для прокси или другого endpoint.
def _get_openai_client():
    api_key = os.environ.get("OPENAI_API_KEY", "").strip()
    base_url = os.environ.get("OPENAI_API_BASE", "").strip() or None
    if not api_key:
        raise RuntimeError(
            "Задайте OPENAI_API_KEY в файле .env или переменных окружения. "
            "Создайте .env по образцу .env.example и укажите ключ OpenAI."
        )
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    return OpenAI(**kwargs)


class LLMAgent:
    def __init__(
            self,
            model: str = 'gpt-4o-mini',
            use_all_tokens: bool = False,
    ):
        self.model = model
        self.llm = _get_openai_client()
        self.use_all_tokens = use_all_tokens

    def __call__(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        attempt = 0

        while True:
            try:
                messages = []
                messages.append({"role": "user", "content": prompt})
                # temperature не передаём: часть моделей (например o1) поддерживает только значение по умолчанию
                return self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=1,
                    max_completion_tokens=max_new_tokens,
                ).choices[0].message.content.strip()
            except Exception as e:
                print(e)
                if attempt >= len(values):
                    raise e

                attempt += 1
                continue
