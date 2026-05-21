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
  use-multi-agent: false    # false = baseline (эталон); true = multi-probe + конфликты
  max-term-distance: 300    # макс. дистанция между терминами для отношений
```

Конфиги: `config.yml` (baseline по умолчанию), `config.baseline.yml`, `config.multiagent.yml`. Эталонные метрики: [`docs/BASELINE.md`](docs/BASELINE.md).

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

## Результаты

Оценка на 100 предложениях, модель `qwen2.5:7b`, 114 эталонных отношений:

| Метрика | Базовый | Мультиагентный | Δ |
|---------|---------|----------------|---|
| Precision (micro) | 0.712 | 0.715 | +0.003 |
| Recall (micro) | 0.824 | 0.861 | +0.037 |
| **F1 (micro)** | **0.764** | **0.782** | **+0.018** |
| **F1 (macro)** | **0.743** | **0.794** | **+0.051** |

Подробности: [`results/compare_baseline_vs_multiagent_qwen2.5_7b_100sent.md`](results/compare_baseline_vs_multiagent_qwen2.5_7b_100sent.md)

## Пример

![image](https://github.com/aiwannafly/semantics-analysis-system/assets/90191819/19ebca4e-59fd-4555-92ef-8c9d8ac18f30)

## Документация

- [`docs/ARCHITECTURE.md`](docs/ARCHITECTURE.md) — архитектура системы
- [`docs/MULTI_AGENT.md`](docs/MULTI_AGENT.md) — мультиагентная архитектура
- [`docs/CHANGELOG.md`](docs/CHANGELOG.md) — список изменений
