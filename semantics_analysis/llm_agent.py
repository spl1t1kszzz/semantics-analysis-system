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

# Модели, которые маршрутизируются через Anthropic API
ANTHROPIC_MODELS = {
    'claude-sonnet-4-20250514',
    'claude-haiku-4-5-20251001',
    'claude-opus-4-20250514',
}


def _is_anthropic_model(model: str) -> bool:
    return model in ANTHROPIC_MODELS or model.startswith('claude-')


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


def _get_anthropic_client():
    import anthropic
    api_key = os.environ.get("ANTHROPIC_API_KEY", "").strip()
    if not api_key:
        raise RuntimeError(
            "Задайте ANTHROPIC_API_KEY в файле .env или переменных окружения."
        )
    return anthropic.Anthropic(api_key=api_key)


class LLMAgent:
    def __init__(
            self,
            model: str = '',
    ):
        self.model = model
        self._use_anthropic = _is_anthropic_model(model)
        if self._use_anthropic:
            self.llm = _get_anthropic_client()
        else:
            self.llm = _get_openai_client()

    def __call__(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        for attempt in range(MAX_RETRIES):
            try:
                if self._use_anthropic:
                    return self._call_anthropic(prompt, stop_sequences, max_new_tokens)
                else:
                    return self._call_openai(prompt, stop_sequences, max_new_tokens)
            except Exception as e:
                print(e)
                if attempt >= MAX_RETRIES - 1:
                    raise
                time.sleep(RETRY_DELAY_SECONDS * (attempt + 1))

    def _call_openai(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        messages = [{"role": "user", "content": prompt}]
        result = self.llm.chat.completions.create(
            model=self.model,
            messages=messages,
            n=1,
            temperature=0,
            max_completion_tokens=max_new_tokens,
        ).choices[0].message.content.strip()
        return result

    def _call_anthropic(self, prompt: str, stop_sequences: List[str], max_new_tokens: int) -> str:
        kwargs = {
            "model": self.model,
            "max_tokens": max_new_tokens,
            "temperature": 0,
            "messages": [{"role": "user", "content": prompt}],
        }
        if stop_sequences:
            kwargs["stop_sequences"] = stop_sequences
        result = self.llm.messages.create(**kwargs)
        return result.content[0].text.strip()
