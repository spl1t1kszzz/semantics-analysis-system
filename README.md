# Система семантического анализа русских IT-текстов

Извлекает термины, классифицирует их по 16 онтологическим классам и находит семантические отношения между ними. Использует RoBERTa-модели для извлечения терминов и LLM (OpenAI / Anthropic / совместимые провайдеры) для верификации, нормализации и извлечения отношений.

## Возможности

- Извлечение и классификация терминов через RoBERTa + словарь
- LLM-верификация, нормализация и разрешение кореференций
- Извлечение семантических отношений из онтологии
- Мультиагентное разрешение конфликтов при извлечении отношений (multi-probe стратегия)
- Поддержка OpenAI, Anthropic (Claude) и OpenAI-совместимых провайдеров (Ollama, Groq, OpenRouter)
- Визуализация графа знаний

## Установка

Требуется Python 3.10+ и [Poetry](https://python-poetry.org/).

```bash
pip install poetry
poetry install
```

## Настройка

Скопируйте `.env.example` в `.env` и укажите ключ API:

```bash
cp .env.example .env
```

```env
# OpenAI или совместимый провайдер
OPENAI_API_KEY=sk-...
OPENAI_API_BASE=http://localhost:11434/v1   # для Ollama (опционально)

# Anthropic (опционально)
ANTHROPIC_API_KEY=sk-ant-...

# Прокси (опционально)
OPENAI_PROXY=http://127.0.0.1:8080
```

Модель и параметры задаются в `config.yml`:

```yaml
app-config:
  llm: 'qwen2.5:7b'        # имя модели
  use-multi-agent: true     # мультиагентное разрешение конфликтов
  max-term-distance: 300    # макс. дистанция между терминами для отношений
```

## Запуск

**Интерактивный CLI** — вводите текст, получаете граф знаний:

```bash
poetry run python run_analysis.py
```

**Анализ статьи с Habr** по URL:

```bash
poetry run python analyze_article.py
```

**Оценка качества** на размеченном датасете:

```bash
poetry run python scripts/evaluate_on_dataset_kristina.py [sentences.json] [--limit N]
```

**Сравнение baseline vs мультиагентный пайплайн:**

```bash
poetry run python scripts/compare_pipeline_multiagent.py [sentences.json] [--limit N]
```

## Пример

![image](https://github.com/aiwannafly/semantics-analysis-system/assets/90191819/19ebca4e-59fd-4555-92ef-8c9d8ac18f30)

## Документация

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — архитектура системы
- [`docs/MULTI_AGENT.md`](docs/MULTI_AGENT.md) — мультиагентная архитектура
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — список изменений
- [`results/experiments/EXPERIMENTS.md`](results/experiments/EXPERIMENTS.md) — результаты экспериментов
