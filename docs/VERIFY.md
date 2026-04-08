# Как проверить работу системы

## 1. Зависимости

Убедитесь, что окружение установлено:

```bash
poetry install
```

Если в `pyproject.toml` добавлен `openai`, но его ещё нет в окружении:

```bash
poetry add openai
# или
pip install openai python-dotenv
```

Для работы с **собственным API OpenAI** создайте файл `.env` в корне проекта (см. `.env.example`):

```
OPENAI_API_KEY=sk-ваш-ключ
```

Либо задайте переменную в терминале:

- Windows (PowerShell): `$env:OPENAI_API_KEY = "sk-..."`
- Linux/macOS: `export OPENAI_API_KEY=sk-...`

---

## 2. Быстрая проверка (без ввода текста)

Из **корня проекта**:

```bash
poetry run python scripts/check_work.py
```

Скрипт проверяет:

- импорты (multi_agent, knowledge_graph, entities);
- загрузку конфига (`config.yml`);
- наличие `OPENAI_API_KEY`;
- детекцию конфликтов (без вызова LLM);
- построение ГЗ с дедупликацией.

Полный прогон по короткому тексту с LLM:

```bash
poetry run python scripts/check_work.py --full
```

---

## 3. Интерактивный анализ текста

Запуск основного сценария (ввод текста с клавиатуры, сохранение `ont_entities.json`, опционально граф):

```bash
poetry run python run_analysis.py
```

Введите русскоязычный текст (например, про BERT и NER). В конце появятся термины и отношения, будет создан `ont_entities.json` и при `display-graph: true` — HTML-граф.

Мультиагентный режим (разрешение конфликтов) включается в `config.yml`:

```yaml
use-multi-agent: true
# Двухшаговый диалог при конфликте: сначала обоснование, потом выбор
use-conflict-dialogue: false
# Повторная верификация выбранного отношения после разрешения конфликта
use-reverify-after-resolve: false
```

Подробнее про варианты улучшения мультиагентности — в [docs/MULTIAGENT_IMPROVEMENTS.md](MULTIAGENT_IMPROVEMENTS.md).

---

## 4. Анализ статьи по ID (Habr)

Если используется парсер статей:

```bash
poetry run python analyze_article.py <article_id>
```

Результат: `ont_entities.json` и файл `article_<id>.html` с графом.

---

## 5. Что проверить по шагам

| Что проверить              | Как |
|----------------------------|-----|
| Конфиг и модель LLM        | `scripts/check_work.py` (шаг 2) |
| Детекция конфликтов        | `scripts/check_work.py` (шаг 4) |
| Построение ГЗ              | `scripts/check_work.py` (шаг 5) |
| Полный пайплайн с LLM      | `scripts/check_work.py --full` или `run_analysis.py` |
| Свой ключ OpenAI           | Задать `OPENAI_API_KEY`, запустить `run_analysis.py` |

При ошибках смотрите вывод в консоль; при проблемах с API — что ключ верный и при необходимости в `config.yml` указана поддерживаемая модель (для OpenAI: `gpt-4o`, `gpt-4o-mini` и т.д.).

---

## 6. Оценка на датасете Кристины (Dataset_Kristina)

Датасет лежит в папке `Dataset_Kristina/5. Датасет/dataset_for_extraction/` (рядом с проектом или внутри него). Формат: `dataset_entity/dataset_entity_*.txt` (conll-подобный с `# text =` и `# relations =`).

### Шаг 1: Конвертация в JSON

Из корня проекта:

```bash
poetry run python scripts/convert_dataset_kristina_to_sentences.py
```

По умолчанию ищет папку `Dataset_Kristina` в корне проекта или в родительской папке, результат сохраняется в `tests/dataset_kristina_sentences.json`. Можно указать пути явно:

```bash
poetry run python scripts/convert_dataset_kristina_to_sentences.py "путь/к/dataset_entity" "tests/dataset_kristina_sentences.json"
```

Классы датасета маппятся на онтологию: `Technology`, `App_system` → `Application`; `Subject` → `Object`. Предикат `IsAlternativeNameFor` приводится к `isAlternativeNameFor`.

### Шаг 2: Запуск оценки

```bash
poetry run python scripts/evaluate_on_dataset_kristina.py
```

Используется файл `tests/dataset_kristina_sentences.json`. Опции:

- `--multi-agent` — включить разрешение конфликтов между отношениями перед подсчётом метрик;
- `--limit N` — обработать только первые N предложений (для отладки);
- первый аргумент — путь к другому JSON с предложениями.

Пример с мультиагентным режимом и лимитом 20 предложений:

```bash
poetry run python scripts/evaluate_on_dataset_kristina.py --multi-agent --limit 20
```

Результаты:

- **results/dataset_kristina_scores.json** — метрики по типам отношений и micro/macro (precision, recall, F1);
- **results/dataset_kristina_metrics.md** — краткий отчёт в виде таблицы.

### Шаг 3: Метрики по построению графа (тройки и структура)

Метрики на уровне **графа** (совпадение троек «сущность — отношение — сущность» и структура):

1. **Вариант A: по результатам оценки на датасете Кристины**

   Запустите оценку с сохранением предсказанных троек:

   ```bash
   poetry run python scripts/evaluate_on_dataset_kristina.py --save-triples [--limit N]
   ```

   Будет создан файл **results/dataset_kristina_predicted_triples.json**. Затем:

   ```bash
   poetry run python calculate_graph_metrics.py tests/dataset_kristina_sentences.json results/dataset_kristina_predicted_triples.json
   ```

2. **Вариант B: эталонные предложения + построенный ГЗ (ont_entities.json)**

   Если у вас уже есть файл **ont_entities.json** (построенный, например, через `run_analysis.py` или `analyze_article.py`):

   ```bash
   poetry run python calculate_graph_metrics.py tests/short_sent.json ont_entities.json
   ```

   Скрипт сам определит формат предсказаний (список троек, shot_rel_*.json или objects+relations).

3. **Что выводится**

   - **Тройки:** Precision, Recall, F1 (по совпадению троек (value1, predicate, value2)).
   - **Структура:** число узлов и рёбер в эталонном и предсказанном графах.
   - Результаты сохраняются в **results/graph_metrics.json** и **results/graph_metrics.md**.

---

## 7. Сравнение базового пайплайна и мультиагентного

Скрипт **scripts/compare_pipeline_multiagent.py** один раз прогоняет датасет через оценку **без** разрешения конфликтов и один раз **с** агентом разрешения конфликтов, затем выводит сравнение метрик.

```bash
poetry run python scripts/compare_pipeline_multiagent.py [sentences.json] [--limit N]
```

По умолчанию: `tests/dataset_kristina_sentences.json`. Результаты:

- **results/compare_baseline_vs_multiagent.json** — метрики обоих вариантов и разницы (diff).
- **results/compare_baseline_vs_multiagent.md** — таблицы: micro (Precision/Recall/F1), macro, по типам отношений, количество предсказанных троек.
