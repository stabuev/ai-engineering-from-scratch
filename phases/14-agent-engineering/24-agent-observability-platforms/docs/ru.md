# Наблюдаемость агентов: Langfuse, Phoenix, Opik

> В 2026 году доминируют три open-source платформы наблюдаемости агентов. Langfuse (MIT) — 6M+ установок в месяц, tracing + управление prompt + evals + session replay. Arize Phoenix (Elastic 2.0) — глубокие agent-specific evals, релевантность RAG, автоинструментация OpenInference. Comet Opik (Apache 2.0) — автоматическая оптимизация prompt, guardrails, обнаружение галлюцинаций LLM-judge.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 23 (OTel GenAI)
**Время:** ~45 минут

## Цели обучения

- Назвать три ведущие open-source платформы наблюдаемости агентов и их лицензии.
- Различать, в чем каждая сильнее: Langfuse (prompt mgmt + sessions), Phoenix (RAG + auto-instrumentation), Opik (optimization + guardrails).
- Объяснить, почему 89% организаций сообщают, что к 2026 году у них есть agent observability.
- Реализовать stdlib trace-to-dashboard pipeline с LLM-judge evaluation.

## Проблема

OTel GenAI (Урок 23) дает схему. Но вам все еще нужна платформа, которая принимает spans, запускает evaluations, хранит prompt versions и показывает regressions. Три претендента делают акцент на разных частях lifecycle.

## Концепция

### Langfuse (MIT)

- 6M+ SDK installs/month, 19k+ GitHub stars.
- Возможности: tracing, prompt management with versioning + playground, evaluations (LLM-as-judge, user feedback, custom), session replays.
- Июнь 2025: бывшие commercial modules (LLM-as-a-judge, annotation queues, prompt experiments, Playground) open-sourced под MIT.
- Сильнее всего для: end-to-end observability с тесным prompt-management loop.

### Arize Phoenix (Elastic License 2.0)

- Более глубокая agent-specific evaluation: trace clustering, anomaly detection, retrieval relevancy for RAG.
- Нативная OpenInference auto-instrumentation.
- Сочетается с managed Arize AX для production.
- Нет prompt versioning — позиционируется как инструмент drift/behavioral-regression рядом с более широкими платформами.
- Сильнее всего для: RAG relevancy, behavioral drift, anomaly detection.

### Comet Opik (Apache 2.0)

- Automated prompt optimization через A/B experiments.
- Guardrails (PII redaction, topical constraints).
- LLM-judge hallucination detection.
- Benchmark из собственных измерений Comet: Opik logs + evals за 23.44s против Langfuse 327.15s (~14x gap) — воспринимайте vendor benchmarks как ориентир.
- Сильнее всего для: optimization loop, automated experimentation, guardrail enforcement.

### Отраслевые данные

По Maxim (2026 field analysis): у 89% организаций есть agent observability; quality issues - главный production barrier (их называют 32% respondents).

### Выбор платформы

| Потребность | Выбор |
|------|------|
| All-in-one с prompt management | Langfuse |
| Глубокая RAG evaluation + drift | Phoenix |
| Автоматическая оптимизация + guardrails | Opik |
| Открытая лицензия, без ELv2 | Langfuse (MIT) или Opik (Apache 2.0) |
| Интеграция с Datadog / New Relic | Любая — все экспортируют OTel |

### Где этот паттерн ломается

- **Нет eval strategy.** Tracing без evaluation - это просто дорогой logging.
- **Self-rolled LLM-judge без grounding.** Применяется паттерн CRITIC (Урок 05) — judges нужны external tools для factual verification.
- **Prompt versions не связаны с traces.** Когда prod регрессирует, вы не можете bisect до prompt, который это вызвал.

## Соберите это

`code/main.py` реализует stdlib trace collector + LLM-judge evaluator:

- Ingest GenAI-shaped spans.
- Группировка по session, теги failed runs (guardrail trips, low-confidence evals).
- Scripted LLM-judge, который оценивает agent responses по rubric.
- Dashboard-like summary: failure rate, top failure reasons, eval score distribution.

Запустите:

```
python3 code/main.py
```

Output: per-session eval scores и failure categorization, соответствующие тому, что показали бы Langfuse/Phoenix/Opik.

## Используйте это

- **Langfuse** self-hosted или cloud; подключайте через OTel или их SDK.
- **Arize Phoenix** self-hosted; auto-instrument OpenInference.
- **Comet Opik** self-hosted или cloud; automated optimization loop.
- **Datadog LLM Observability** для смешанных ops+ML teams, которые уже используют Datadog.

## Отправьте в работу

`outputs/skill-obs-platform-wiring.md` выбирает платформу и подключает traces + evals + prompt versions к существующему агенту.

## Упражнения

1. Экспортируйте неделю OTel traces в Langfuse cloud (free tier). Какие sessions failed? Почему?
2. Напишите LLM-judge rubric для своего домена (factual correctness, tone, scope adherence). Проверьте на 50 traces.
3. Сравните Langfuse prompt versioning с Phoenix trace clustering. Что быстрее говорит, что сломалось?
4. Прочитайте guardrail docs Opik. Подключите PII redaction guardrail к одному из ваших agent runs.
5. Проведите benchmark трех платформ на вашем corpus. Игнорируйте vendor-published numbers; измерьте сами.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Tracing | "Сборщик spans" | Прием OTel / SDK spans; индексирование по сессиям |
| Prompt management | "Prompt CMS" | Версионированные prompt, связанные с трассами |
| LLM-as-judge | "Автоматическая eval" | Отдельная LLM оценивает вывод агента по рубрике |
| Session replay | "Воспроизведение трассы" | Пошаговый просмотр прошлых запусков для отладки |
| RAG relevancy | "Качество retrieval" | Соответствует ли извлеченный контекст запросу |
| Trace clustering | "Группировка поведения" | Кластеризация похожих запусков для обнаружения drift |
| Guardrail enforcement | "Политика во время логирования" | Проверки PII/toxicity/scope для залогированного контента |

## Дополнительное чтение

- [Langfuse docs](https://langfuse.com/) — tracing, evals, управление prompt
- [Arize Phoenix docs](https://docs.arize.com/phoenix) — автоинструментация, drift
- [Comet Opik](https://www.comet.com/site/products/opik/) — optimization + guardrails
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — схема, которую потребляют все три
