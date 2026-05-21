# Архитектура системы семантического анализа

## Обзор

Система извлекает термины, классифицирует их по 16 онтологическим классам и находит семантические отношения между ними из текстов на русском языке (IT-тематика). Используется комбинация RoBERTa-моделей и LLM (OpenAI API).

## Структура проекта

```
semantics-analysis-system/
|
|-- run_analysis.py              # Основной entry point (интерактивный CLI)
|-- analyze_article.py           # Анализ статей с Habr
|-- config.yml                   # Конфигурация приложения
|
|-- semantics_analysis/          # Основной пакет
|   |-- config.py                # @dataclass Config + load_config()
|   |-- constants.py             # TERM_CLASSES, маппинги id<->class
|   |-- entities.py              # TermMention, Term, Relation, Sentence
|   |-- factory.py               # build_pipeline(), build_term_extractor()
|   |-- pipelines.py             # Pipeline ABC, SequencePipeline, шаги пайплайна
|   |-- llm_agent.py             # LLMAgent (OpenAI API wrapper + retry)
|   |
|   |-- term_extraction/         # Извлечение терминов
|   |   |-- term_mention_extractor.py        # ABC TermMentionExtractor
|   |   |-- dict_term_mention_extractor.py   # Словарный экстрактор
|   |   |-- roberta_unclassified_term_mention_extractor.py  # RoBERTa BIO-тегирование
|   |   |-- roberta_classified_term_mention_extractor.py    # RoBERTa классификация
|   |   |-- hybrid_term_mention_extractor.py # CombinedTermExtractor (ансамбль)
|   |   |-- term_verifier.py                 # ABC TermVerifier
|   |   |-- llm_term_verifier.py             # LLM-верификация терминов
|   |
|   |-- term_normalization/      # Нормализация (лемматизация)
|   |   |-- term_normalizer.py              # ABC TermNormalizer
|   |   |-- llm_term_normalizer.py          # LLM-нормализация
|   |
|   |-- term_post_processing/    # Постобработка терминов
|   |   |-- term_post_processor.py          # ABC TermPostProcessor
|   |   |-- computer_science_term_post_processor.py  # ResolveLibraries
|   |   |-- merge_close_term_post_processor.py       # MergeCloseTerms
|   |
|   |-- reference_resolution/    # Группировка упоминаний в термины
|   |   |-- reference_resolver.py           # ABC ReferenceResolver
|   |   |-- llm_reference_resolver.py       # LLM-проверка синонимии
|   |
|   |-- relation_extraction/     # Извлечение семантических отношений
|   |   |-- relation_extractor.py           # ABC RelationExtractor
|   |   |-- llm_relation_extractor.py       # LLM-извлечение отношений
|   |
|   |-- multi_agent/             # Мультиагентное разрешение конфликтов
|   |   |-- base.py                         # AgentRole, SharedState
|   |   |-- conflict_resolution.py          # RelationConflictResolver
|   |   |-- orchestrator.py                 # MultiAgentOrchestrator (legacy)
|   |
|   |-- knowledge_graph/         # Построение графа знаний
|   |   |-- kg_builder.py                   # build_knowledge_graph()
|   |
|   |-- ontology_entities.py     # Object, OntRelation, convert_to_ont_entities()
|   |-- ontology_utils.py        # Метаданные онтологии: предикаты, классы, алиасы
|   |-- phrase_extractor.py      # Spacy-based извлечение фраз
|   |-- habr_parser.py           # Парсер статей Habr
|   |-- visualization.py         # Граф отношений через Pyvis
|   |-- utils.py                 # Progress bars, логирование
|
|-- scripts/                     # Утилитарные скрипты
|   |-- evaluate_on_dataset_kristina.py     # Оценка на датасете
|   |-- compare_pipeline_multiagent.py      # Сравнение базовый vs мультиагентный
|   |-- calculate_scores.py                 # Метрики качества
|   |-- ...
|
|-- prompts/                     # LLM-промпт шаблоны
|   |-- verify_term.txt
|   |-- normalization.txt
|   |-- relation_extraction.txt
|   |-- relation_extraction_multi_probe.txt  # Multi-probe: бинарный да/нет по каждому предикату
|   |-- directed_relation_extraction.txt
|   |-- verify_relation.txt
|   |-- resolve_reference.txt
|   |-- resolve_relation_conflict.txt
|
|-- metadata/                    # Метаданные онтологии
|   |-- new_term_classes.json    # Классы терминов с примерами
|   |-- relations.json           # Типы отношений между классами
|   |-- terms_by_class.json      # Словарь терминов по классам
|
|-- tests/                       # Юнит-тесты
|   |-- test_entities.py
|   |-- test_config.py
|   |-- test_pipelines.py
```

## Архитектура пайплайна

Система построена на паттерне **Pipeline** -- последовательность шагов, каждый из которых трансформирует `AnalysisResult`:

```
Текст
  |
  v
[PredictTerms]          -- RoBERTa + словарь -> TermMention[]
  |
  v
[PreprocessTerms]       -- ResolveLibraries, MergeCloseTerms
  |
  v
[VerifyTerms]           -- LLM фильтрует ложные срабатывания
  |
  v
[NormalizeTerms]        -- LLM лемматизация
  |
  v
[drop_empty / normalize_languages]  -- очистка
  |
  v
[ResolveReference]      -- LLM группирует синонимы -> Term[]
  |
  v
[PredictSemanticRelations]  -- LLM извлекает отношения -> Relation[]
  |
  v
[ResolveRelationConflicts]  -- (опционально) мультиагентное разрешение
  |
  v
AnalysisResult { terms, relations }
```

### Сборка пайплайна

Пайплайн собирается **фабрикой** `build_pipeline(config, progress)` в [factory.py](../semantics_analysis/factory.py):

- Создаёт единый `LLMAgent` и передаёт его во все компоненты
- Конфигурирует пороги, модели, логирование из `Config`
- Опционально добавляет шаг разрешения конфликтов

### Entry points

- **`run_analysis.py`** -- интерактивный CLI: вводишь текст, получаешь граф
- **`analyze_article.py`** -- парсит статью с Habr и анализирует по параграфам

Оба используют `build_pipeline()` для сборки.

## Ключевые сущности

### TermMention
Единичное упоминание термина в тексте. Поля: `value`, `class_`, `start_pos`, `end_pos`, `text`, `norm_value`, `source`.

### Term
Группа упоминаний, ссылающихся на одну сущность (после reference resolution). Поля: `value`, `class_`, `mentions[]`.

### Relation
Бинарное отношение: `term1 --predicate--> term2`. Предикаты определены в `metadata/relations.json`.

## 16 онтологических классов

Определены в [constants.py](../semantics_analysis/constants.py):

| Класс | Описание |
|-------|----------|
| Method | Методы, алгоритмы (BERT, SVM, градиентный спуск) |
| Activity | Виды деятельности (обучение, разработка) |
| Science | Науки и области знаний (NLP, машинное обучение) |
| Object | Абстрактные объекты (данные, модель, граф) |
| Person | Люди (Илон Маск, исследователь) |
| InfoResource | Источники информации (статья, датасет MNIST) |
| Task | Задачи (классификация, NER) |
| Organization | Организации (Google, MIT) |
| Environment | Среды и платформы (Python, TensorFlow) |
| Model | Конкретные модели (GPT-4, ResNet-50) |
| Metric | Метрики (F1-score, accuracy) |
| Value | Числовые значения (95%, 1000 эпох) |
| Application | Приложения (ChatGPT, Google Translate) |
| Date | Даты (2023 год) |
| Lang | Языки (русский, английский) |
| Dataset | Датасеты (ImageNet, SQuAD) |

## Извлечение терминов

Три стратегии, комбинируемые через `CombinedTermExtractor`:

1. **DictTermExtractor** -- словарный поиск по `metadata/terms_by_class.json`
2. **RobertaUnclassifiedTermExtractor** -- BIO-тегирование (модель `aiwannafly/semantics-analysis-term-extractor`)
3. **RobertaTermExtractor** -- классификация в 16 классов (модель `aiwannafly/semantics-analysis-term-classifier-v.0.2`)

Результаты объединяются с приоритетом наибольшего совпадения.

## LLM-интеграция

Все LLM-вызовы идут через [LLMAgent](../semantics_analysis/llm_agent.py):

- **OpenAI API**: модели GPT (gpt-4o-mini, gpt-4o, gpt-5.2 и др.)
- **Anthropic API**: модели Claude (claude-sonnet-4, claude-opus-4 и др.) — выбор провайдера автоматический по имени модели
- **OpenAI-совместимые провайдеры**: Groq, OpenRouter и др. — через `OPENAI_API_BASE`
- **Retry**: экспоненциальная задержка, до 5 попыток

Промпт-шаблоны хранятся в `prompts/` и подставляются компонентами.

## Конфигурация

Файл [config.yml](../config.yml), загружается через `load_config()` в `Config` dataclass:

```yaml
app-config:
  use-dict: true           # Использовать словарный экстрактор
  device: 'cpu'            # cpu / cuda
  llm: 'gpt-4o-mini'      # Модель LLM (gpt-4o-mini, claude-sonnet-4, и др.)
  term-threshold: 0.2      # Порог для RoBERTa term extraction
  class-threshold: 0.5     # Порог для RoBERTa classification
  max-term-distance: 300   # Макс. расстояние между терминами для поиска отношений
  use-multi-agent: false   # false = baseline; true = multi-probe + конфликты (см. config.multiagent.yml)
```

Эталон baseline: [BASELINE.md](BASELINE.md). Секреты хранятся в `.env`:
- `OPENAI_API_KEY` — для OpenAI / OpenAI-совместимых провайдеров
- `ANTHROPIC_API_KEY` — для моделей Claude
- `OPENAI_API_BASE` — альтернативный endpoint (Groq, OpenRouter)
- `OPENAI_PROXY` — HTTP/SOCKS5 прокси

## Мультиагентное извлечение отношений

Подробное описание: [MULTI_AGENT.md](MULTI_AGENT.md)

### Multi-probe fallback

При включённом мультиагентном режиме (`use-multi-agent: true`) используется двухуровневая стратегия извлечения:

1. **Стандартное извлечение** (`detect_predicates()`) — свободный промпт с few-shot примерами, модель выбирает один предикат
2. **Multi-probe fallback** (`detect_predicates_multi_probe()`) — запускается только если стандартное извлечение не нашло ничего. Бинарный «да/нет» по каждому предикату независимо

Все кандидаты проходят верификацию через `verify_relation()` с описанием семантики из онтологии.

### Разрешение конфликтов

Если для одной пары терминов обнаружено >1 предиката, [RelationConflictResolver](../semantics_analysis/multi_agent/conflict_resolution.py) проверяет каждое через re-verify и оставляет подтверждённые.

## Зависимости

| Группа | Пакеты |
|--------|--------|
| Core | PyYAML, colorama, rich, python-dotenv, openai |
| NLP | spacy, nltk, alphabet-detector, conllu |
| ML | transformers, torch |
| Visualization | pyvis, matplotlib |
| CLI | inquirer |
| Dev | pytest |
