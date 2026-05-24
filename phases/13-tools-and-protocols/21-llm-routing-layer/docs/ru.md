# Слой LLM routing — LiteLLM, OpenRouter, Portkey

> Provider lock-in обходится дорого. Разным workloads с tool-calling подходят разные модели. Routing gateways дают одну API surface, retries, failover, cost tracking и guardrails. В 2026 году доминируют три архетипа: LiteLLM (open-source self-hosted), OpenRouter (managed SaaS), Portkey (production-grade, open-sourced в марте 2026 года). Этот урок называет критерии выбора и проходит через routing gateway на stdlib.

**Тип:** Изучение
**Языки:** Python (stdlib, routing + failover + tracker стоимости)
**Предварительные требования:** Фаза 13 · 02 (function calling), Фаза 13 · 17 (gateways)
**Время:** ~45 минут

## Цели обучения

- Отличать self-hosted, managed и production-grade варианты routing.
- Реализовать fallback chain, который повторяет попытки при сбоях provider в заданном порядке приоритета.
- Отслеживать per-request cost и token usage по providers.
- Выбирать между LiteLLM, OpenRouter и Portkey для заданного production-ограничения.

## Проблема

Сценарии, где provider routing важен:

1. **Стоимость.** Claude Sonnet стоит в 3 раза дороже Haiku. Для triage task достаточно Haiku; для synthesis task Sonnet оправдан. Маршрутизируйте per-request.

2. **Failover.** У OpenAI плохой час. Каждый запрос падает. Нужен автоматический fallback на Anthropic без redeploy.

3. **Latency.** Live chat UI нужен быстрый time-to-first-token. Batch summarizer — нет. Маршрутизируйте по latency SLA.

4. **Compliance.** EU users должны оставаться в EU regions. Маршрутизируйте по region.

5. **Experimentation.** A/B двух models на одном workload. Маршрутизируйте по test bucket.

Ручное кодирование всего этого для каждой интеграции повторяется. Routing gateway дает один OpenAI-compatible API и берет остальное на себя.

## Концепция

### Форма OpenAI-compatible proxy

Все говорят в OpenAI-shape. Routing gateway предоставляет `/v1/chat/completions`, принимает OpenAI schema и внутри проксирует в Anthropic / Gemini / Cohere / Ollama / anything. Клиенту все равно.

### Aliases моделей

Вместо `claude-3-5-sonnet-20251022` ваш код говорит `our_smart_model`. Gateway сопоставляет aliases с реальными models. Когда Anthropic выпускает Claude 4, вы меняете alias на стороне сервера; код не трогаете.

### Цепочки fallback

```
primary: openai/gpt-4o
on 5xx: anthropic/claude-3-5-sonnet
on 5xx: google/gemini-1.5-pro
on 5xx: refuse
```

Gateways задают это в config. Retries учитываются в budget, чтобы каскады fallback не взрывали стоимость.

### Semantic caching

Идентичные или почти идентичные prompts попадают в cache вместо provider. Экономия на повторяющихся agent loops может составлять от 30 до 60 процентов. Keys основаны на embeddings; почти идентичные prompts делят cache slot.

### Guardrails

На уровне gateway:

- **PII redaction.** Regex или ML-based pass перед отправкой prompts.
- **Policy violations.** Отклонение prompts с запрещенным content.
- **Output filters.** Очистка completions от утечек.

Portkey и Kong поставляют opinionated guardrails. LiteLLM оставляет их optional.

### Rate limits per key

Один API key = одна team. Per-key budgets не дают одной team съесть общую quota. Большинство gateways это поддерживают.

### Trade-offs self-hosted и managed

| Фактор | LiteLLM (self-hosted) | OpenRouter (managed) | Portkey (production) |
|--------|----------------------|----------------------|----------------------|
| Код | Open source, Python | Managed SaaS | Open source (март 2026 года) + managed |
| Setup | Развернуть proxy | Зарегистрироваться | Любой вариант |
| Providers | 100+ | 300+ | 100+ |
| Billing | Ваши собственные keys | OpenRouter credits | Ваши собственные keys |
| Observability | OpenTelemetry | Dashboard | Полный OTel + PII redaction |
| Лучше всего для | Команд, которым нужен полный контроль | Быстрого прототипирования | Production с compliance |

LiteLLM выигрывает, когда у вас есть SRE team и нужна data sovereignty. OpenRouter выигрывает, когда нужна единая подписка и никакой infra. Portkey выигрывает, когда guardrails и compliance нужны out of the box.

### Отслеживание стоимости

Каждый запрос несет `provider`, `model`, `input_tokens`, `output_tokens`. Умножьте на цены per-model per-token (берутся из pricing sheet, который поддерживает gateway). Агрегация по per-user / per-team / per-project.

### MCP плюс routing

Gateway может маршрутизировать и LLM calls, и MCP sampling requests. Когда modelPreferences в sampling request предпочитают конкретную model, gateway переводит это в нужный backend. Здесь Фаза 13 · 17 (MCP gateway) и routing gateway из этого урока иногда сливаются в один service.

### Стратегии routing

- **Static priority.** Первый в списке; fallback при ошибке.
- **Load balancing.** Round-robin или weighted.
- **Cost-aware.** Выбрать самую дешевую model, удовлетворяющую latency / quality.
- **Latency-aware.** Выбрать самую быструю model за последние N минут.
- **Task-aware.** Prompt classifier маршрутизирует coding в одну model, summarization — в другую.

## Используйте

`code/main.py` реализует routing gateway примерно в 150 строк: принимает requests в форме OpenAI, переводит их в per-provider stubs, запускает priority fallback chain, отслеживает per-request cost и применяет PII redaction pass к inputs. Запустите его с тремя сценариями: обычный request, outage primary-provider с fallback, утечка PII, пойманная redaction.

На что обратить внимание:

- `ROUTES` dict: alias -> упорядоченный по приоритету список конкретных providers.
- Loop fallback повторяет попытки на 5xx.
- Cost tracker умножает token usage на per-model rates.
- PII redactor очищает patterns в форме SSN перед forwarding.

## Отгрузите

Этот урок создает `outputs/skill-routing-config-designer.md`. Для профиля workload (latency, cost, compliance) skill выбирает LiteLLM / OpenRouter / Portkey и создает routing config.

## Упражнения

1. Запустите `code/main.py`. Вызовите outage scenario; подтвердите, что fallback попадает на второго provider и cost атрибутируется корректно.

2. Добавьте semantic caching: SHA256 от prompt — lookup key; cache hits возвращаются мгновенно. Измерьте cost savings на повторном вызове.

3. Добавьте prompt classifier, который маршрутизирует prompts вида "code ..." в alias, отдающий приоритет intelligence, а prompts вида "summarize ..." — в alias, отдающий приоритет speed.

4. Спроектируйте per-team budgets: у каждой team есть monthly spend cap; gateway отказывает в requests после достижения cap. Выберите enforcement granularity (per-request или windowed).

5. Прочитайте docs LiteLLM, OpenRouter и Portkey side by side. Назовите одну feature каждого, которой нет у двух других.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Routing gateway | "LLM proxy" | Слой с одной API surface перед множеством providers |
| OpenAI-compatible | "Говорит на схеме OpenAI" | Принимает форму `/v1/chat/completions`, переводит в любой backend |
| Model alias | "our_smart_model" | Имя в вашем коде, которое gateway сопоставляет с concrete model |
| Fallback chain | "Список retry" | Упорядоченный список providers, которые пробуются при failure |
| Semantic caching | "Prompt-embedding cache" | Key — embedding prompt; near-duplicates дают cache hit |
| Guardrails | "Фильтры input/output" | Редактируют PII, отклоняют policy violations |
| Per-key rate limit | "Бюджет team" | Quota, scoped to API key |
| Cost tracking | "Расход на запрос" | Агрегация token usage x price per model |
| LiteLLM | "Открытый proxy" | Self-hostable OSS routing gateway |
| OpenRouter | "Managed SaaS" | Hosted gateway с credit-based billing |
| Portkey | "Production-вариант" | Open-source + managed со встроенными guardrails |

## Дополнительное чтение

- [LiteLLM — docs](https://docs.litellm.ai/) — self-hosted routing gateway
- [OpenRouter — quickstart](https://openrouter.ai/docs/quickstart) — managed routing SaaS
- [Portkey — docs](https://portkey.ai/docs) — production routing с guardrails
- [TrueFoundry — LiteLLM vs OpenRouter](https://www.truefoundry.com/blog/litellm-vs-openrouter) — guide по выбору
- [Relayplane — LLM gateway comparison 2026](https://relayplane.com/blog/llm-gateway-comparison-2026) — vendor survey
