# OpenTelemetry GenAI Semantic Conventions

> GenAI SIG OpenTelemetry (запущена в апреле 2024 года) определяет стандартную схему для телеметрии агентов. Имена span, атрибуты и правила захвата контента сходятся между вендорами, чтобы трассы агентов означали одно и то же в Datadog, Grafana, Jaeger и Honeycomb.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 13 (LangGraph), Фаза 14 · 24 (Observability Platforms)
**Время:** ~60 минут

## Цели обучения

- Назвать категории GenAI spans: model/client, agent, tool.
- Отличать `invoke_agent` CLIENT от INTERNAL spans и понимать, когда применяется каждый вариант.
- Перечислить top-level GenAI attributes: provider name, request model, data-source ID.
- Объяснить контракт content-capture: opt-in, `OTEL_SEMCONV_STABILITY_OPT_IN`, рекомендация external-reference.

## Проблема

Каждый vendor придумывает собственные span names. Ops-команды в итоге строят dashboards под каждый framework отдельно. GenAI SIG OpenTelemetry исправляет это, определяя единый стандарт, на который ориентируется вся экосистема.

## Концепция

### Категории spans

1. **Model / client spans.** Покрывают сырые LLM-вызовы. Их эмитят provider SDKs (Anthropic, OpenAI, Bedrock) и framework model adapters.
2. **Agent spans.** `create_agent` (когда агент создается) и `invoke_agent` (когда он запускается).
3. **Tool spans.** Один span на каждый вызов инструмента; связан с agent span отношением parent-child.

### Именование agent spans

- Имя span: `invoke_agent {gen_ai.agent.name}`, если имя задано; fallback - `invoke_agent`.
- Вид span:
  - **CLIENT** — для remote agent services (OpenAI Assistants API, Bedrock Agents).
  - **INTERNAL** — для in-process agent frameworks (LangChain, CrewAI, local ReAct).

### Ключевые attributes

- `gen_ai.provider.name` — `anthropic`, `openai`, `aws.bedrock`, `google.vertex`.
- `gen_ai.request.model` — model ID.
- `gen_ai.response.model` — resolved model (может отличаться от request из-за routing).
- `gen_ai.agent.name` — идентификатор агента.
- `gen_ai.operation.name` — `chat`, `completion`, `invoke_agent`, `tool_call`.
- `gen_ai.data_source.id` — для RAG: какой corpus или store был использован.

Для Anthropic, Azure AI Inference, AWS Bedrock и OpenAI существуют technology-specific conventions.

### Захват контента

Правило по умолчанию: instrumentations SHOULD NOT захватывать inputs/outputs по умолчанию. Capture включается через opt-in:

- `gen_ai.system_instructions`
- `gen_ai.input.messages`
- `gen_ai.output.messages`

Рекомендуемый production pattern: хранить content внешне (S3, ваш log store), а в spans записывать references (pointer IDs, а не prose). Это защита от content-poisoning из Урока 27, встроенная в observability.

### Stability

Большинство conventions являются experimental по состоянию на март 2026 года. Включите stable preview через:

```
OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental
```

Datadog v1.37+ нативно отображает атрибуты GenAI в свою схему LLM Observability. Другие бэкенды (Grafana, Honeycomb, Jaeger) поддерживают сырые атрибуты.

### Где этот паттерн ломается

- **Захват полных prompts в spans.** PII, secrets и customer data оказываются в traces, которые может читать ops. Храните внешне.
- **Нет `gen_ai.provider.name`.** Multi-provider dashboards ломаются, когда отсутствует attribution.
- **Spans без parent links.** Осиротевшие tool spans. Всегда распространяйте context.
- **Не задан stability opt-in.** Ваши attributes могут быть переименованы при backend upgrade.

## Соберите это

`code/main.py` реализует stdlib span emitter, соответствующий GenAI conventions:

- `Span` со схемой GenAI attributes.
- `Tracer` с `start_span`, nested contexts.
- Скриптовый запуск агента, который эмитит: `create_agent`, `invoke_agent` (INTERNAL), span для каждого инструмента, `chat` spans для вызовов LLM.
- Режим content-capture, который хранит prompts внешне и записывает IDs в spans.

Запустите:

```
python3 code/main.py
```

Output: span tree со всеми обязательными GenAI attributes и "external store", показывающий opt-in content references.

## Используйте это

- **Datadog LLM Observability** (v1.37+) нативно мапит attributes.
- **Langfuse / Phoenix / Opik** (Урок 24) — auto-instrument the ecosystem.
- **Jaeger / Honeycomb / Grafana Tempo** — raw OTel traces; стройте dashboards из GenAI attributes.
- **Self-hosted** — запустите OTel Collector с GenAI processor.

## Отправьте в работу

`outputs/skill-otel-genai.md` подключает OTel GenAI spans к существующему агенту с content-capture defaults и external-reference storage.

## Упражнения

1. Инструментируйте ваш ReAct loop из Урока 01 с `invoke_agent` (INTERNAL) + per-tool spans. Отправьте в Jaeger instance.
2. Добавьте content capture в режиме "references only": prompts в SQLite, span attributes несут только row IDs.
3. Прочитайте spec для `gen_ai.data_source.id`. Подключите его к поиску Mem0 из Урока 09.
4. Задайте `OTEL_SEMCONV_STABILITY_OPT_IN=gen_ai_latest_experimental` и проверьте, что collector не переименовывает ваши attributes.
5. Постройте dashboard: "which tool errors correlate with which models" только из GenAI attributes.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| GenAI SIG | "OpenTelemetry GenAI group" | Рабочая группа OTel, определяющая схему |
| invoke_agent | "Span агента" | Имя span, представляющего запуск агента |
| CLIENT span | "Удаленный вызов" | Span для вызова удаленного агентского сервиса |
| INTERNAL span | "In-process" | Span для in-process agent run |
| gen_ai.provider.name | "Provider" | anthropic / openai / aws.bedrock / google.vertex |
| gen_ai.data_source.id | "RAG source" | В каком corpus/store был retrieval hit |
| Content capture | "Логирование prompt" | Захват сообщений по явному включению; в prod храните внешне |
| Stability opt-in | "Preview mode" | Переменная окружения для фиксации экспериментальных conventions |

## Дополнительное чтение

- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — спецификация
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — GenAI spans by default
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — OTel spans built in
- [Claude Agent SDK](https://platform.claude.com/docs/en/agent-sdk/overview) — W3C trace context propagation
