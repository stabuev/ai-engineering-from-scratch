# OpenTelemetry GenAI — трассировка вызовов инструментов end-to-end

> Агент вызывает пять инструментов, три MCP server и двух sub-agents. Вам нужен один trace через все это. Семантические соглашения OpenTelemetry GenAI (стабильные атрибуты в v1.37 и выше) — стандарт 2026 года, нативно поддерживаемый Datadog, Langfuse, Arize Phoenix, OpenLLMetry и AgentOps. Этот урок называет обязательные атрибуты, проходит по иерархии spans (agent → LLM → tool) и поставляет emitter для spans на stdlib, который можно подключить к любому OTel exporter.

**Тип:** Сборка
**Языки:** Python (stdlib, emitter для OTel spans)
**Предварительные требования:** Фаза 13 · 07 (MCP server), Фаза 13 · 08 (MCP client)
**Время:** ~75 минут

## Цели обучения

- Назвать обязательные атрибуты OTel GenAI для LLM span и span выполнения инструмента.
- Построить иерархию trace, покрывающую agent loop, LLM call, tool call и dispatch MCP client.
- Решить, какой контент захватывать (opt-in), а какой редактировать/скрывать по умолчанию.
- Отправлять spans в локальный collector (Jaeger, Langfuse) без переписывания кода инструментов.

## Проблема

Отладочный случай из февраля 2026 года: пользователь сообщает: "мой агент иногда отвечает за 30 секунд, а иногда за 3 секунды." Trace нет. Логи показывают LLM call, но не dispatch tool, не round-trip MCP server, не sub-agent. Вы гадаете. В итоге находите: один MCP server иногда зависает на cold-start.

Без end-to-end tracing это не найти. OTel GenAI решает проблему.

Соглашения закрепились в 2025-2026 годах в группе OpenTelemetry semantic-conventions. Они определяют стабильные имена атрибутов, чтобы Datadog, Langfuse, Phoenix, OpenLLMetry и AgentOps одинаково разбирали одни и те же spans. Инструментируете один раз; отправляете в любой backend.

## Концепция

### Иерархия spans

```
agent.invoke_agent  (top, INTERNAL span)
 ├── llm.chat       (CLIENT span)
 ├── tool.execute   (INTERNAL)
 │    └── mcp.call  (CLIENT span)
 ├── llm.chat       (CLIENT span)
 └── subagent.invoke (INTERNAL)
```

Все это вложено под один trace id. Span ids связывают отношения parent-child.

### Обязательные атрибуты

Согласно semconv 2025-2026:

- `gen_ai.operation.name` — `"chat"`, `"text_completion"`, `"embeddings"`, `"execute_tool"`, `"invoke_agent"`.
- `gen_ai.provider.name` — `"openai"`, `"anthropic"`, `"google"`, `"azure_openai"`.
- `gen_ai.request.model` — запрошенная строка модели (например, `"gpt-4o-2024-08-06"`).
- `gen_ai.response.model` — модель, которая фактически обслужила запрос.
- `gen_ai.usage.input_tokens` / `gen_ai.usage.output_tokens`.
- `gen_ai.response.id` — provider response id для корреляции.

Для tool spans:

- `gen_ai.tool.name` — идентификатор инструмента.
- `gen_ai.tool.call.id` — id конкретного вызова.
- `gen_ai.tool.description` — описание инструмента (опционально).

Для agent spans:

- `gen_ai.agent.name` / `gen_ai.agent.id` / `gen_ai.agent.description`.

### Виды spans

- `SpanKind.CLIENT` для вызовов, пересекающих границу процесса (LLM provider, MCP server).
- `SpanKind.INTERNAL` для собственных шагов agent loop и выполнения инструмента.

### Opt-in захват контента

По умолчанию spans несут метрики и timing — не prompts и completions. Большие payloads и PII по умолчанию выключены. Установите `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` и специальные env vars для content-capture, чтобы включить контент. Перед включением в prod внимательно проверьте риски.

### Events на spans

События уровня токенов можно добавлять как span events:

- `gen_ai.content.prompt` — входные сообщения.
- `gen_ai.content.completion` — выходные сообщения.
- `gen_ai.content.tool_call` — записанный вызов инструмента.

Events упорядочены во времени внутри span для детального воспроизведения.

### Exporters

OTel spans экспортируются в:

- **Jaeger / Tempo.** OSS, on-prem.
- **Langfuse.** Специализирован для LLM observability; визуализирует token usage.
- **Arize Phoenix.** Evals + tracing вместе.
- **Datadog.** Коммерческий; нативно разбирает атрибуты `gen_ai.*`.
- **Honeycomb.** Column-oriented; удобен для запросов.

Все говорят на OTLP как wire format. Вашему коду все равно.

### Propagation через MCP

Когда MCP client вызывает server, внедрите W3C traceparent header в запрос. Streamable HTTP поддерживает стандартные headers. Stdio нативно не переносит HTTP headers; roadmap спецификации на 2026 год обсуждает добавление поля `_meta.traceparent` в JSON-RPC calls.

Пока это не поставлено: включайте traceparent в `_meta` каждого запроса вручную. Server логирует trace id.

### Метрики

Наряду со spans, GenAI semconv определяет metrics:

- `gen_ai.client.token.usage` — histogram.
- `gen_ai.client.operation.duration` — histogram.
- `gen_ai.tool.execution.duration` — histogram.

Используйте их для dashboards, которым не нужна детализация каждого вызова.

### Слой AgentOps

AgentOps (основан в 2024 году) специализируется на GenAI observability. Он оборачивает популярные фреймворки (LangGraph, Pydantic AI, CrewAI), чтобы автоматически испускать OTel spans. Полезно, если ваш stack использует поддерживаемый фреймворк; иначе используйте ручную instrumentation.

## Используйте

`code/main.py` выводит spans в форме OTel в stdout (в формате, похожем на OTLP JSON) для агента, который вызывает LLM, отправляет два tools и делает один MCP round-trip. Реального exporter нет — урок сфокусирован на форме span и наборе атрибутов. Вставьте вывод в OTLP-совместимый viewer или просто прочитайте его.

На что обратить внимание:

- Trace id общий для всех spans.
- Связи parent-child закодированы через `parentSpanId`.
- Обязательные атрибуты `gen_ai.*` заполнены.
- Content capture по умолчанию выключен; один сценарий включает его через env var.

## Отгрузите

Этот урок создает `outputs/skill-otel-genai-instrumentation.md`. Для кодовой базы агента skill создает план instrumentation: где добавить spans, какие атрибуты заполнять и на какие exporters нацелиться.

## Упражнения

1. Запустите `code/main.py`. Посчитайте spans и определите, какие из них CLIENT, а какие INTERNAL.

2. Включите content capture (env var) и подтвердите, что events `gen_ai.content.prompt` и `gen_ai.content.completion` появляются. Отметьте последствия для PII.

3. Добавьте метрику выполнения инструмента `gen_ai.tool.execution.duration` и отправляйте ее как histogram sample для каждого вызова.

4. Прокиньте traceparent из parent agent span в поле `_meta.traceparent` MCP request. Проверьте, что MCP server увидел бы тот же trace id.

5. Прочитайте спецификацию OTel GenAI semconv. Определите один атрибут из semconv, который код этого урока НЕ отправляет. Добавьте его.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| OTel | "OpenTelemetry" | Открытый стандарт для traces, metrics, logs |
| GenAI semconv | "Семантические соглашения GenAI" | Стабильные имена атрибутов для LLM / tool / agent spans |
| `gen_ai.*` | "Namespace атрибутов" | Все атрибуты GenAI используют этот префикс |
| Span | "Операция во времени" | Единица работы с началом, концом и атрибутами |
| Trace | "Родство между spans" | Дерево spans с общим trace id |
| SpanKind | "CLIENT / SERVER / INTERNAL" | Подсказки о направлении span |
| OTLP | "OpenTelemetry Line Protocol" | Wire format для exporters |
| Opt-in content | "Захват prompt / completion" | По умолчанию выключен; включается env var |
| traceparent | "Header W3C" | Передает trace context между сервисами |
| Exporter | "Shipper для конкретного backend" | Компонент, который отправляет spans в Jaeger / Datadog / etc. |

## Дополнительное чтение

- [OpenTelemetry — GenAI semconv](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — канонические соглашения для GenAI spans, metrics и events
- [OpenTelemetry — GenAI spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-spans/) — список атрибутов LLM и tool-execution spans
- [OpenTelemetry — GenAI agent spans](https://opentelemetry.io/docs/specs/semconv/gen-ai/gen-ai-agent-spans/) — agent-level span `invoke_agent`
- [open-telemetry/semantic-conventions — GenAI spans](https://github.com/open-telemetry/semantic-conventions/blob/main/docs/gen-ai/gen-ai-spans.md) — source of truth на GitHub
- [Datadog — LLM OTel semantic convention](https://www.datadoghq.com/blog/llm-otel-semantic-convention/) — walkthrough production integration
