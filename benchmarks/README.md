# Зафиксированные эталоны

## `frozen_baseline_qwen2.5_7b_100sent.*`

Неизменяемая точка отсчёта для baseline-пайплайна (100 предложений, модель `qwen2.5:7b`).

- **JSON:** `frozen_baseline_qwen2.5_7b_100sent.json` — секция `baseline`, graph baseline, `meta`
- **Markdown:** `frozen_baseline_qwen2.5_7b_100sent.md` — полный отчёт сравнения на момент фиксации

Источник: `results/compare_baseline_vs_multiagent_qwen2.5_7b_100sent.*` (коммит фиксации baseline).

**Не перезаписывать** при экспериментах. Новые прогоны мультиагента сравниваются с этим файлом:

```bash
poetry run python scripts/compare_pipeline_multiagent.py tests/dataset_kristina_sentences.json --limit 100
```

Полный пересчёт baseline + multi-agent (два прогона): флаг `--run-both`.
