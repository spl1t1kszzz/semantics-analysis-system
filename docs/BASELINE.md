# Baseline-пайплайн (эталон)

Зафиксированный baseline — стандартное извлечение отношений без мультиагентных шагов. Используется как точка отсчёта при улучшении метрик.

## Что входит в baseline

Полный пайплайн (`build_pipeline` / `run_evaluation`):

1. RoBERTa + словарь → термины  
2. LLM: verify, normalize, reference resolution  
3. **Отношения:** `LLMRelationExtractor` с `use_multi_probe=false`  
   - `detect_predicates()` (few-shot, один свободный ответ)  
   - `verify_relation()` для каждого кандидата  
4. **Без** `ResolveRelationConflicts`

Конфиг: [`config.baseline.yml`](../config.baseline.yml) или `config.yml` (по умолчанию baseline).

## Эталонные метрики (100 предложений, `qwen2.5:7b`)

Датасет: `tests/dataset_kristina_sentences.json`, `--limit 100`.  
Зафиксированный эталон (не менять): [`benchmarks/frozen_baseline_qwen2.5_7b_100sent.md`](../benchmarks/frozen_baseline_qwen2.5_7b_100sent.md) и `.json`.

| Метрика | Baseline (эталон) | Мультиагент (для сравнения) |
|---------|-------------------|-----------------------------|
| Micro P / R / F1 | 0.712 / 0.824 / **0.764** | 0.715 / 0.861 / 0.782 |
| Macro F1 | **0.743** | 0.794 |
| TP / FP / FN | 89 / 36 / 19 | 93 / 37 / 15 |
| Graph triples F1 | **0.574** | 0.593 |

114 эталонных отношений в разметке; 104 уникальные gold-тройки для graph-метрик.

## Воспроизведение

```bash
# Только baseline-оценка (конфиг по умолчанию = baseline)
poetry run python scripts/evaluate_on_dataset_kristina.py tests/dataset_kristina_sentences.json --limit 100

# Только multi-agent, сравнение с frozen baseline (один прогон)
poetry run python scripts/compare_pipeline_multiagent.py tests/dataset_kristina_sentences.json --limit 100

# Полный пересчёт baseline + multi-agent (два прогона)
poetry run python scripts/compare_pipeline_multiagent.py tests/dataset_kristina_sentences.json --limit 100 --run-both
```

Результаты экспериментов: `results/compare_vs_frozen_baseline_*.md` (не путать с frozen эталоном в `benchmarks/`).

Мультиагентный прогон вручную:

```bash
# Временно: скопировать config.multiagent.yml → config.yml
# или задать use-multi-agent: true в config.yml
poetry run python scripts/evaluate_on_dataset_kristina.py tests/dataset_kristina_sentences.json --limit 100 --multi-agent
```

## Критерий улучшений

Новые изменения сравниваем с baseline micro F1 **0.764** (и при необходимости graph triples F1 **0.574**). Мультиагентный режим — отдельная ветка экспериментов (`config.multiagent.yml`), не подменяет эталон baseline.
