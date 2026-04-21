# Эксперименты по улучшению мультиагентного пайплайна

Модель: qwen2.5:7b | Тестовая выборка: 100 предложений | Эталонных отношений: 114

## Сводка результатов

| Версия | Baseline F1 | Multiagent F1 | Δ F1 (MA-BL) | TP/FP/FN (MA) | Описание |
|--------|------------|---------------|--------------|---------------|----------|
| v0 | 0.7640 | 0.7586 | -0.0054 | 88/36/20 | Исходный: LLM-арбитр в conflict resolution |
| v1 | 0.7640 | 0.7792 | +0.0152 | 90/33/18 | verify-each + RelationVerifier agent |
| v2 | 0.7764 | 0.7764 | +0.0000 | 92/37/16 | Улучшенные промпты + keep_all conflicts |

## v0 — Исходное состояние
- Файлы: `v0_baseline_100sent.json`, `v0_baseline_100sent.md`
- Conflict resolution: LLM-арбитр выбирает одно отношение
- Multi-probe до конфликтов: F1=0.7754 (TP=88, FP=36, FN=15)
- После конфликтов: F1=0.7586 — теряет 5 FN

## v1 — Независимая верификация + RelationVerifier agent
- Файлы: `v1_verify_agent_100sent.json`, `v1_verify_agent_100sent.md`
- Conflict resolution заменён на verify-each (reverify_callback)
- Добавлен RelationVerifier — проверяет все cross-class отношения
- Добавлены cross-class примеры в verify_relation.txt
- **Baseline F1 не изменился** (0.7640), **Multiagent F1 +0.015** (0.7792)
- FP снизились 36→33, TP вырос 89→90

## v2 — Улучшенные промпты извлечения + keep_all
- Файлы: `v2_improved_prompts_keepall_100sent.json`, `v2_improved_prompts_keepall_100sent.md`
- Убран "нет"-bias из relation_extraction.txt и directed_relation_extraction.txt
- Conflict resolution = keep_all (return group)
- RelationVerifier отключен (verifier=None)
- **Baseline F1 вырос** 0.7640→0.7764 (+0.012) — промпты помогли
- Multiagent = baseline (conflict resolution keep_all не добавляет ценности)
- Multi-probe до конфликтов: F1=0.7863, но 3 FN теряются при resolve_all

## Текущее состояние (в работе)
- Промпты улучшены (v2)
- Conflict resolution = keep_all
- RelationVerifier = отключен
- Диагностика: resolve_all теряет 3 отношения при keep_all — debug в процессе
