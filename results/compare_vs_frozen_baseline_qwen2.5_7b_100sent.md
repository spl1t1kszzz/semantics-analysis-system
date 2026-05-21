# Сравнение: frozen baseline vs мультиагентный (текущий прогон)

**Baseline:** frozen: frozen_baseline_qwen2.5_7b_100sent.json  
**Модель прогона:** `qwen2.5:7b` | **Предложений:** 100 | **Эталонных отношений:** 114

## 1. Сводка (micro)
| Метрика | Baseline | Мультиагентный | Δ |
|---------|----------|----------------|---|
| Precision | 0.7120 | 0.6901 | -0.0219 |
| Recall | 0.8241 | 0.9074 | +0.0833 |
| F1 | 0.7640 | 0.7840 | +0.0200 |

TP=89/98, FP=36/44, FN=19/10 (baseline/multiagent)

## 2. Macro
| Метрика | Baseline | Мультиагентный | Δ |
|---------|----------|----------------|---|
| Precision | 0.7194 | 0.7337 | +0.0143 |
| Recall | 0.7686 | 0.8838 | +0.1152 |
| F1 | 0.7432 | 0.8018 | +0.0586 |

## 3. По типам отношений
| Relation | Support | Baseline P/R/F1 | Multiagent P/R/F1 | Δ F1 |
|----------|---------|-----------------|-------------------|------|
| Application_isUsedForSolving_Task | 15 | 0.74/0.93/0.82 | 0.70/0.93/0.80 | -0.02 |
| Application_hasAuthor_Organization | 12 | 0.73/0.92/0.81 | 0.67/1.00/0.80 | -0.01 |
| Method_solves_Task | 11 | 0.88/0.64/0.74 | 0.89/0.73/0.80 | +0.06 |
| Model_isUsedIn_Application | 9 | 0.90/1.00/0.95 | 0.82/1.00/0.90 | -0.05 |
| Metric_hasValue_Value | 7 | 1.00/0.86/0.92 | 1.00/1.00/1.00 | +0.08 |
| Object_isUsedInSolving_Task | 6 | 1.00/0.83/0.91 | 1.00/0.83/0.91 | +0.00 |
| Method_isAlternativeNameFor_Method | 6 | 0.55/1.00/0.71 | 0.55/1.00/0.71 | +0.00 |
| Date_isDateOf_Method | 6 | 1.00/0.50/0.67 | 1.00/0.67/0.80 | +0.13 |
| Task_isAlternativeNameFor_Task | 5 | 0.71/1.00/0.83 | 0.71/1.00/0.83 | +0.00 |
| Metric_isUsedFor_Model | 4 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | +0.00 |
| Application_isUsedIn_Science | 4 | 0.67/0.50/0.57 | 0.80/1.00/0.89 | +0.32 |
| Model_isUsedForSolving_Task | 3 | 1.00/0.33/0.50 | 1.00/1.00/1.00 | +0.50 |
| Task_isSolvedIn_Science | 3 | 0.38/1.00/0.55 | 0.38/1.00/0.55 | +0.00 |
| Application_isAlternativeNameFor_Application | 3 | 0.33/0.67/0.44 | 0.33/0.67/0.44 | +0.00 |
| Metric_isUsedIn_Task | 2 | 1.00/1.00/1.00 | 0.50/1.00/0.67 | -0.33 |
| Application_hasAuthor_Person | 2 | 0.33/1.00/0.50 | 0.33/1.00/0.50 | +0.00 |
| Method_hasAuthor_Person | 2 | 1.00/0.50/0.67 | 1.00/0.50/0.67 | +0.00 |
| Metric_isAlternativeNameFor_Metric | 2 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | +0.00 |
| Science_isAlternativeNameFor_Science | 2 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | +0.00 |
| Model_hasAuthor_Organization | 1 | 0.33/1.00/0.50 | 0.20/1.00/0.33 | -0.17 |
| Method_isUsedIn_Science | 1 | 0.00/0.00/0.00 | 1.00/1.00/1.00 | +1.00 |
| Method_isUsedIn_Application | 1 | 0.00/0.00/0.00 | 0.00/0.00/0.00 | +0.00 |
| Activity_hasAuthor_Organization | 1 | 1.00/1.00/1.00 | 1.00/1.00/1.00 | +0.00 |

## 4. Предсказанные тройки (граф)
| Метрика | Baseline | Мультиагентный | Δ |
|---------|----------|----------------|---|
| Precision | 0.5113 | 0.4935 | -0.0178 |
| Recall | 0.6538 | 0.7308 | +0.0770 |
| F1 | 0.5738 | 0.5891 | +0.0153 |

- Baseline (frozen): 133 троек
- Мультиагентный (прогон): 154 троек
- Эталон: 104 троек

## 5. Поэтапное сравнение

| Этап | Precision | Recall | F1 | TP | FP | FN |
|------|-----------|--------|-----|----|----|-----|
| Baseline (frozen) | 0.7120 | 0.8241 | 0.7640 | 89 | 36 | 19 |
| Multi-agent: до конфликтов | 0.6901 | 0.9245 | 0.7903 | 98 | 44 | 8 |
| Multi-agent: после конфликтов | 0.6901 | 0.9074 | 0.7840 | 98 | 44 | 10 |