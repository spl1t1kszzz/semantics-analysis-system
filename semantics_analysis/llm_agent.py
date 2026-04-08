import os
import time
from typing import List

from openai import OpenAI

try:
    from dotenv import load_dotenv
    load_dotenv()
except ImportError:
    pass

MAX_RETRIES = 5
RETRY_DELAY_SECONDS = 2


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
    ):
        self.model = model
        self.llm = _get_openai_client()

    def __call__(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": prompt}]
                return self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=1,
                    max_completion_tokens=max_new_tokens,
                ).choices[0].message.content.strip()
            except Exception as e:
                print(e)
                if attempt >= MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
