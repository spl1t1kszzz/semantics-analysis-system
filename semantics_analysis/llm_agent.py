import hashlib
import os
import time
from typing import List

import httpx
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
    proxy = os.environ.get("OPENAI_PROXY", "").strip() or None
    if not api_key:
        raise RuntimeError(
            "Задайте OPENAI_API_KEY в файле .env или переменных окружения. "
            "Создайте .env по образцу .env.example и укажите ключ OpenAI."
        )
    kwargs = {"api_key": api_key}
    if base_url:
        kwargs["base_url"] = base_url
    if proxy:
        kwargs["http_client"] = httpx.Client(proxy=proxy)
    return OpenAI(**kwargs)


class LLMAgent:
    def __init__(
            self,
            model: str = 'gpt-4o-mini',
            enable_cache: bool = True,
    ):
        self.model = model
        self.llm = _get_openai_client()
        self._cache = {} if enable_cache else None

    def _cache_key(self, prompt: str, max_new_tokens: int) -> str:
        raw = f"{self.model}:{max_new_tokens}:{prompt}"
        return hashlib.sha256(raw.encode()).hexdigest()

    def __call__(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        if self._cache is not None:
            key = self._cache_key(prompt, max_new_tokens)
            if key in self._cache:
                return self._cache[key]

        for attempt in range(MAX_RETRIES):
            try:
                messages = [{"role": "user", "content": prompt}]
                result = self.llm.chat.completions.create(
                    model=self.model,
                    messages=messages,
                    n=1,
                    max_completion_tokens=max_new_tokens,
                ).choices[0].message.content.strip()

                if self._cache is not None:
                    self._cache[key] = result

                return result
            except Exception as e:
                print(e)
                if attempt >= MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))
