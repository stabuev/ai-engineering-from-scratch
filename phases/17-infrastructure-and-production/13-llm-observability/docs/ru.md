# LLM Observability Stack Selection

> Рынок observability 2026 делится на две категории. Development platforms (LangSmith, Langfuse, Comet Opik) объединяют monitoring с evals, prompt management, session replays. Gateway/instrumentation tools (Helicone, SigNoz, OpenLLMetry, Phoenix) фокусируются на telemetry. Langfuse — MIT-licensed core с сильным OSS balance (50K events/month free cloud). Phoenix — OpenTelemetry-native под Elastic License 2.0, отличный для drift/RAG visualization, но не persistent production backend. Arize AX использует zero-copy Iceberg/Parquet integration и заявляет 100x cheaper than monolithic observability. LangSmith лидирует для LangChain/LangGraph, $39/user/mo, self-host только в Enterprise. Helicone proxy-based с 15-30 min setup, 100K req/mo free, но меньше глубины по agent traces. Частый production pattern: Gateway (Helicone/Portkey) + eval platform (Phoenix/TruLens), склеенные OpenTelemetry.

**Тип:** Learn
**Языки:** Python (stdlib, toy trace-sampling simulator)
**Предварительные требования:** Phase 17 · 08 (Inference Metrics), Phase 14 (Agent Engineering)
**Время:** ~60 minutes

## Цели обучения

- Отличить development platforms (bundled: evals + prompts + sessions) от gateway/telemetry tools (traces + metrics only).
- Сопоставить шесть major tools (Langfuse, LangSmith, Phoenix, Arize AX, Helicone, Opik) с их licensing, pricing и sweet-spot use cases.
- Объяснить OpenTelemetry-glue pattern, который позволяет сочетать gateway tool с отдельной eval platform.
- Назвать cost differentiator 2026 (zero-copy approach Arize AX vs monolithic ingest) и указать примерный multiplier 100x.

## Проблема

Вы ship LLM feature. Он работает. У вас нет видимости в prompt failures, tool loops, latency regressions, cost spikes или prompt-cache hit rate. Вы гуглите "LLM observability" и получаете восемь tools, каждый утверждает, что решает ту же проблему, но в трех разных price points.

Они не решают одну и ту же проблему. LangSmith отвечает на "why did this LangGraph run fail?" Phoenix отвечает на "is my RAG pipeline drifting?" Helicone отвечает на "which app is burning tokens?" Langfuse отвечает на "can I self-host the whole thing?" Разные tools, разные audiences.

Выбор включает четыре оси: stack (LangChain? raw SDK? multi-vendor?), license tolerance (MIT only? Elastic OK? commercial fine?), budget (free tier? $100/mo? $1000/mo?) и self-host (must? nice-to-have? never?).

## Концепция

### Две категории

**Development platforms** объединяют observability с evals, prompt management, dataset versioning, session replay. Вы запускаете experiments, видите, какой prompt сработал, dataset-regression нового prompt против old winners. LangSmith, Langfuse, Comet Opik.

**Gateway/telemetry tools** инструментируют inference calls — prompt, response, tokens, latency, model, cost. Helicone, SigNoz, OpenLLMetry, Phoenix. Минималистично. Можно сочетать с отдельным eval tool через OpenTelemetry.

### Langfuse — OSS balance

- Core Apache / MIT licensed; self-host via Docker.
- Cloud free tier: 50K events/month. Paid: $29/mo for team.
- Evals, prompt management, traces, datasets. Reasonable coverage of all four dev-platform features.
- Sweet spot: нужны LangSmith-class features, но требуется self-host или OSS license.

### Phoenix (Arize) — telemetry-first, OpenTelemetry-native

- Elastic License 2.0; self-host trivial.
- Отличен для RAG и drift visualization. Embedding-space scatter plots поставляются как first-class.
- Не спроектирован как persistent production backend — прежде всего development-time observability.
- Sweet spot: RAG pipeline development, drift debugging, работает в паре с separate gateway for production.

### Arize AX — ставка на scale

- Commercial. Zero-copy data lake integration via Iceberg/Parquet.
- Заявляет ~100x cheaper than monolithic observability (Datadog-class) at scale. Математика: traces хранятся в вашем Parquet on S3; Arize читает напрямую.
- Sweet spot: >10M traces/day, existing data lake, нужны LLM-specific dashboards без Datadog pricing.

### LangSmith — LangChain/LangGraph first

- Commercial, $39/user/month. Self-host only on Enterprise.
- Best-in-class для LangChain и LangGraph stacks. Если вы не на них, менее compelling.
- Sweet spot: команда committed to LangChain, willing to pay.

### Helicone — proxy-based minimum viable

- 15-30 minute setup через замену `OPENAI_API_BASE` на Helicone proxy.
- MIT licensed; 100K req/mo free, paid $20/mo+.
- Includes failover, caching, rate limits — also acts as gateway.
- Меньше глубины по agent / multi-step traces.
- Sweet spot: quick start, single-stack app, need gateway + observability in one.

### Opik (Comet) — OSS dev platform

- Apache 2.0, fully OSS.
- Similar feature set to Langfuse with Comet heritage.
- Sweet spot: ML teams already on Comet, want LLM observability in the same pane.

### SigNoz — OpenTelemetry-first full APM

- Apache 2.0. Handles general APM plus LLM via OpenTelemetry.
- Sweet spot: unified observability across services and LLM calls.

### Glue: OpenTelemetry + GenAI semantic conventions

OpenTelemetry published GenAI semantic conventions в late 2025 (`gen_ai.system`, `gen_ai.request.model`, `gen_ai.usage.input_tokens`). Tools, которые consume OTel, могут interoperate. Emerging production pattern:

1. Emit OTel with GenAI conventions from every LLM call.
2. Route to gateway (Helicone / Portkey) for day-to-day.
3. Dual-ship to eval platform (Phoenix / Langfuse) for regressions.
4. Archive in data lake (Iceberg) for long-term analysis via Arize AX or DuckDB.

### Ловушка: instrumentation на неправильном layer

Instrumentation внутри agent framework (например, adding LangSmith traces) связывает вас с этим framework. Instrumenting at the HTTP/OpenAI-SDK layer (через OpenLLMetry или gateway) portable.

### Sampling — нельзя хранить все

При >1M requests/day full-trace retention стоит дороже, чем LLM calls. Sample by rules: 100% errors, 100% high-cost, 5% success. Keep aggregates always; keep raw for the long tail.

### Числа, которые нужно помнить

- Langfuse free cloud: 50K events/month.
- LangSmith: $39/user/month.
- Helicone free: 100K req/month.
- Arize AX claim: ~100x cheaper than monolithic at scale.
- OpenTelemetry GenAI conventions: 2025 shipping, 2026 widely adopted.

## Используйте это

`code/main.py` симулирует 1M-trace day across retention strategies (100% ingest, sampling, sampling + errors). Сообщает storage cost и что теряется в каждом варианте.

## Отправьте в прод

Этот урок создает `outputs/skill-observability-stack.md`. По stack, scale, budget и license posture выбирает tool(s).

## Упражнения

1. Ваша команда на LangChain хочет OSS self-hosted observability. Выберите Langfuse или Opik и обоснуйте.
2. При 5M traces/day и Datadog quotes $150K/month вычислите break-even для Arize AX.
3. Спроектируйте OpenTelemetry GenAI attribute set, который guideline вашей org должен требовать для каждого LLM call.
4. Аргументируйте, достаточно ли Phoenix alone для production. Когда его недостаточно?
5. Helicone adds 20ms proxy overhead. При P99 TTFT 300 ms это приемлемо? А если SLA is 100 ms?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| OpenLLMetry | "OTel for LLMs" | Open-source OpenTelemetry instrumentation for LLMs |
| GenAI conventions | "OTel attributes" | Standard OTel attribute names for LLM calls |
| LangSmith | "LangChain observability" | Commercial platform bundled with LangChain ecosystem |
| Langfuse | "OSS LangSmith" | MIT OSS with similar feature set |
| Phoenix | "Arize dev tool" | OpenTelemetry-native dev/eval platform |
| Arize AX | "scale observability" | Commercial zero-copy Iceberg/Parquet observability |
| Helicone | "proxy observability" | HTTP proxy collecting LLM telemetry + gateway features |
| Opik | "Comet LLM" | Apache 2.0 OSS dev platform from Comet |
| Session replay | "trace rerun" | Replay a full agent session with tool calls |
| Eval | "offline test" | Running candidate model/prompt over labeled dataset |

## Дополнительное чтение

- [SigNoz — Top LLM Observability Tools 2026](https://signoz.io/comparisons/llm-observability-tools/)
- [Langfuse — Arize AX Alternative analysis](https://langfuse.com/faq/all/best-phoenix-arize-alternatives)
- [PremAI — Setting Up Langfuse, LangSmith, Helicone, Phoenix](https://blog.premai.io/llm-observability-setting-up-langfuse-langsmith-helicone-phoenix/)
- [OpenTelemetry GenAI Semantic Conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/)
- [Arize Phoenix docs](https://docs.arize.com/phoenix)
- [Helicone docs](https://docs.helicone.ai/)
