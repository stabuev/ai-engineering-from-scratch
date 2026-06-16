# Capstone 11 — LLM Observability & Eval Dashboard

> Langfuse перешел на open-core. Arize Phoenix опубликовал mappings GenAI semconv 2026 года. Helicone и Braintrust усилили per-user cost attribution. OpenLLMetry от Traceloop стал de-facto SDK instrumentation. Продакшен-форма: ClickHouse для traces, Postgres для metadata, Next.js для UI и небольшой парк eval jobs (DeepEval, RAGAS, LLM-judge), работающих по sampled traces. Постройте self-hosted вариант, принимайте ingest как минимум из четырех SDK families и покажите, что injected regression ловится меньше чем за пять минут.

**Тип:** Capstone
**Языки:** TypeScript (UI), Python / TypeScript (ingest + evals), SQL (ClickHouse)
**Предварительные требования:** Phase 11 (LLM engineering), Phase 13 (tools), Phase 17 (infrastructure), Phase 18 (safety)
**Отрабатываемые фазы:** P11 · P13 · P17 · P18
**Время:** 25 часов

## Цели обучения

- Построить дашборд observability и оценки LLM с трассировкой по семантическим конвенциям GenAI.
- Атрибутировать стоимость по пользователю и по запросу.
- Подключить онлайн-оценки и алертинг на регрессии качества.

## Проблема

Каждая AI-команда, обслуживающая production traffic в 2026 году, держит observability plane рядом с model. Атрибуция затрат. Обнаружение галлюцинаций. Мониторинг drift. Сигнал jailbreak. SLO dashboards. Alerts об утечках PII. Open-source references - Langfuse, Phoenix, OpenLLMetry - сошлись на OpenTelemetry GenAI semantic conventions как ingest schema. Теперь можно инструментировать OpenAI, Anthropic, Google, LangChain, LlamaIndex и vLLM одним SDK и отправлять совместимые spans.

Вы построите self-hosted dashboard, который принимает ingest как минимум из четырех SDK families, запускает небольшой набор eval jobs по sampled traces, обнаруживает drift и отправляет alerts. Планка измерения: при намеренно injected regression (prompt начинает генерировать PII) dashboard должен обнаружить это и отправить alert меньше чем за пять минут.

## Концепция

Ingest идет через OTLP HTTP. SDK создает GenAI-semconv spans: `gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`, `gen_ai.response.id`, `llm.prompts`, `llm.completions`. Spans попадают в ClickHouse для columnar analytics; metadata (users, sessions, apps) попадает в Postgres.

Evals запускаются как batch jobs по sampled traces. DeepEval оценивает faithfulness, toxicity и answer relevance. RAGAS оценивает retrieval metrics, когда trace содержит retrieval context. Custom LLM-judges запускают проверки для конкретного домена (PII leak, off-policy response). Eval runs записываются обратно в тот же ClickHouse как eval spans, связанные с parent trace.

Drift detection отслеживает embedding-space distributions во времени (PSI или KL divergence на prompt embeddings) плюс trends eval-score. Alerts идут в Prometheus Alertmanager, затем в Slack / PagerDuty. UI - Next.js 15 с Recharts.

## Архитектура

```
production apps:
  OpenAI SDK  +  Anthropic SDK  +  Google GenAI SDK
  LangChain + LlamaIndex + vLLM
       |
       v
  OpenTelemetry SDK with GenAI semconv
       |
       v  OTLP HTTP
  collector (ingest, sample, fan-out)
       |
       +-------------+-----------+
       v             v           v
   ClickHouse    Postgres    S3 archive
   (spans)       (metadata)  (raw events)
       |
       +---> eval jobs (DeepEval, RAGAS, LLM-judge)
       |     sampled or all-trace
       |     write eval spans back
       |
       +---> drift detector (PSI / KL on prompt embeddings)
       |
       +---> Prometheus metrics -> Alertmanager -> Slack / PagerDuty
       |
       v
   Next.js 15 dashboard (Recharts)
```

## Стек

- Ingest: OpenTelemetry SDKs + GenAI semantic conventions; OTLP HTTP transport
- Collector: OpenTelemetry Collector с tail-sampling processor (для контроля cost)
- Storage: ClickHouse для spans, Postgres для metadata, S3 для raw event archive
- Evals: DeepEval, RAGAS 0.2, Arize Phoenix evaluator pack, custom LLM-judge
- Drift: PSI / KL на pooled prompt embeddings (sentence-transformers) еженедельно
- Alerting: Prometheus Alertmanager -> Slack / PagerDuty
- UI: Next.js 15 App Router + Recharts + server actions
- SDKs, поддерживаемые из коробки: OpenAI, Anthropic, Google GenAI, LangChain, LlamaIndex, vLLM

## Соберите

1. **Collector config.** OpenTelemetry Collector с OTLP HTTP receiver, tail-sampler, который сохраняет 100% errored traces и 10% successes, и exporters в ClickHouse и S3.

2. **ClickHouse schema.** Таблица `spans` с columns, отражающими GenAI semconv: `gen_ai_system`, `gen_ai_request_model`, `input_tokens`, `output_tokens`, `latency_ms`, `prompt_hash`, `trace_id`, `parent_span_id`, плюс JSON bag для long payloads. Добавьте secondary indexes по user_id и app_id.

3. **SDK coverage test.** Напишите небольшой client app с каждым SDK (OpenAI, Anthropic, Google, LangChain, LlamaIndex, vLLM) с OpenLLMetry auto-instrument. Проверьте, что каждый создает canonical GenAI spans, которые попадают в ClickHouse.

4. **Eval jobs.** Scheduled job читает sampled traces за последние 15 минут и запускает DeepEval faithfulness, toxicity и answer relevance. Outputs - eval spans, связанные с parent trace.

5. **Custom LLM-judge.** PII-leak judge: по response, вызовите guard LLM, чтобы оценить вероятность PII leak. Responses с высоким score попадают в triage queue.

6. **Drift detection.** Weekly job вычисляет PSI между pooled prompt embeddings текущей недели и trailing 4-week baseline. Если PSI выше threshold, отправьте alert.

7. **Dashboard.** Next.js 15 со страницами: overview (spans/sec, cost/user, p95 latency), traces (search + waterfall), evals (faithfulness trend, toxicity), drift (PSI over time), alerts.

8. **Alerting chain.** Prometheus exporter читает eval score aggregates и latency percentiles; Alertmanager направляет warnings в Slack, а critical breaches в PagerDuty.

9. **Regression probe.** Внедрите bug: evaluated chatbot начинает утекать fake SSNs в 1% случаев. Измерьте MTTR: от deployment bug до Slack alert.

## Используйте

```
$ curl -X POST https://my-otel-collector/v1/traces -d @trace.json
[collector]  accepted 1 trace, 3 spans
[clickhouse] inserted 3 spans (app=chat, user=u_42)
[eval]       DeepEval faithfulness 0.82, toxicity 0.03
[drift]      weekly PSI 0.08 (below 0.2 threshold)
[ui]         live at https://obs.example.com
```

## Сдайте

`outputs/skill-llm-observability.md` - deliverable. Для заданного LLM application dashboard принимает traces этого приложения, запускает evals, отправляет alerts при drift и показывает cost/user breakdown в Next.js.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Trace-schema coverage | Число SDK families, создающих canonical GenAI spans (target: 6+) |
| 20 | Eval correctness | DeepEval / RAGAS scores vs hand-labeled set |
| 20 | Dashboard UX | MTTR на injected regression (цель - меньше 5 минут) |
| 20 | Cost / scale | Устойчивый ingest 1k spans/sec без backlog |
| 15 | Alerting + drift detection | Цепочка Prometheus/Alertmanager проверена end to end |
| **100** | | |

## Упражнения

1. Добавьте custom instrumentation для Haystack framework. Проверьте, что canonical spans попадают в ClickHouse с корректными `gen_ai.*` attributes.

2. Замените DeepEval на Phoenix evaluators на тех же traces. Измерьте score drift между двумя eval engines.

3. Усильте drift detector: вычисляйте PSI per app-id, а не globally. Покажите per-app drift trails.

4. Добавьте страницу "user impact": cost-per-user и failure-rate-per-user со sparklines.

5. Постройте tail-sampling policy, которая сохраняет 100% traces с toxicity > 0.5 плюс 10% stratified sample остальных. Измерьте внесенный sampling bias.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| GenAI semconv | "OTel LLM attributes" | Спецификация OpenTelemetry 2025 для LLM span attributes (system, model, tokens) |
| Tail sampling | "Post-trace sample" | Collector решает сохранить или отбросить trace после завершения (может учитывать errors) |
| PSI | "Population stability index" | Drift metric comparing two distributions; > 0.2 обычно означает meaningful drift |
| LLM-judge | "Eval as model" | LLM, оценивающая output другой LLM по rubric (faithfulness, toxicity, PII) |
| Tail-sampling policy | "Keep-rule" | Rule, которая решает, какие traces persist vs drop; errored + sample-rate |
| Eval span | "Linked eval trace" | Child span с eval score, связанный с original LLM call span |
| Cost per user | "Unit economics" | Dollar cost, отнесенный к user_id за window; ключевая product metric |

## Дополнительное чтение

- [Langfuse](https://github.com/langfuse/langfuse) — reference open-core observability platform
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — альтернативный reference с сильной поддержкой drift
- [OpenLLMetry (Traceloop)](https://github.com/traceloop/openllmetry) — auto-instrumentation SDK family
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — ingest schema
- [Helicone](https://www.helicone.ai) — альтернативная hosted observability
- [Braintrust](https://www.braintrust.dev) — альтернативная eval-first platform
- [ClickHouse documentation](https://clickhouse.com/docs) — columnar хранилище spans
- [DeepEval](https://github.com/confident-ai/deepeval) — evaluator library
