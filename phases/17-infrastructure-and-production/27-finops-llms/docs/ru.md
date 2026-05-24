# FinOps for LLMs — Unit Economics and Multi-Tenant Attribution

> Traditional FinOps ломается на LLM spend. Costs — это token-transactions, а не resource-uptime. Tags не мапятся — API call является transaction, не asset. Engineering decisions (prompt design, context window, output length) являются financial decisions. Playbook 2026 имеет три attribution dimensions, которые нужно instrument on day one: per-user (`user_id`) для seat pricing and expansion, per-task (`task_id` + `route`) для product surface cost and prioritization, per-tenant (`tenant_id`) для unit economics and renewal. Четыре token layers — prompt, tool, memory, response — один bucket скрывает spend. Enforcement ladder для multi-tenant products: rate limits per tenant (2-3x expected peak, clear 429 + retry-after); daily spend cap (1.5-3x contracted ceiling; triggers rate tightening + alert); kill switches при spend z-score > 4 (auto-pause + page on-call). Attribution patterns: tag-and-aggregate, telemetry-joiner (trace-ID → billing; highest accuracy), sampling-and-extrapolation, model-based allocation, event-sourced, real-time streaming. Unit metric: cost per resolved query, cost per generated artifact — not $/M tokens. Retroactive tagging always misses; instrument at request creation.

**Тип:** Learn
**Языки:** Python (stdlib, toy cost-attribution simulator with kill switch)
**Предварительные требования:** Phase 17 · 13 (Observability), Phase 17 · 14 (Caching)
**Время:** ~60 minutes

## Цели обучения

- Объяснить, почему traditional FinOps (tags + tiers) ломается на LLM spend, и назвать три new attribution dimensions.
- Перечислить четыре token layers (prompt, tool, memory, response) и почему single-bucket billing скрывает cost.
- Спроектировать enforcement ladder (rate → spend cap → kill switch) для multi-tenant product.
- Выбрать unit metric (cost per resolved query / artifact) вместо $/M tokens.

## Проблема

Ваш bill показывает $40,000. Вы не знаете:
- Какой tenant их потратил.
- Какая product feature это вызвала.
- Был ли какой-то individual user abusive.
- Был ли culprit prompt bloat, tool calls или memory amplification.

Tag-and-aggregate на provider-side работает для cloud resources (EC2, S3), где tags проходят в line items. LLM API calls не auto-tag — вам нужно проставлять user/task/tenant в call site и протаскивать дальше. Retroactive attribution always misses edge cases.

## Концепция

### Три attribution dimensions

**Per-user** (`user_id`): кто сколько стоит. Драйвит seat pricing, expansion conversations, выявляет power users.

**Per-task** (`task_id` + `route`): какая product surface сколько стоит. Драйвит feature prioritization, решения kill-expensive-features.

**Per-tenant** (`tenant_id`): какой customer profitable. Драйвит unit economics, renewal pricing, tier thresholds.

Instrument all three at call site on day one. Retroactive is always worse.

### Четыре token layers

| Layer | Example | Typical % of total |
|-------|---------|---------------------|
| Prompt | system + user input | 40-60% |
| Tool | tool-call results fed back | 20-40% (agent workloads) |
| Memory | prior conversation / retrieved docs | 10-30% |
| Response | model output | 10-30% |

Сведение всех четырех в один bucket делает optimization слепой. Разделяйте их в attribution schema.

### Enforcement ladder

1. **Rate limit** per tenant. 2-3x expected peak. Возвращайте 429 с `Retry-After`. Tenant видит friction; surprise bill не возникает.

2. **Daily spend cap** per tenant. 1.5-3x contracted ceiling. Trigger: tighten rate limit + alert customer-success.

3. **Kill switch** на spend z-score > 4 относительно tenant baseline. Auto-pause tenant; page on-call; escalate to ops + CS.

### Attribution patterns

- **Tag-and-aggregate**: stamp metadata headers; aggregate later. Simple; rough.
- **Telemetry joiner**: join traces to billing via trace IDs. Highest accuracy. What mature teams do.
- **Sampling + extrapolation**: sample 5-10%, multiply. Cost-effective for rough spend; misses tails.
- **Model-based allocation**: regression to infer cost driver. For legacy data without tags.
- **Event-sourced**: cost as events in a stream (Kafka / Kinesis). Real-time.
- **Real-time streaming**: dashboard updates sub-second.

### Cost per X — unit metric

$/M tokens — язык vendors. Product metrics:

- Cost per resolved support ticket.
- Cost per generated article.
- Cost per successful agent task.
- Cost per user-session-minute.

Привязывайте cost к product outcome. Иначе optimization не имеет якоря.

### Cost attribution trace shape

```
trace_id: abc123
  user_id: u_42
  tenant_id: t_7
  task_id: task_classify_doc
  route: model_haiku
  layers:
    prompt_tokens: 1800
    tool_tokens: 600
    memory_tokens: 400
    response_tokens: 150
  cost_usd: 0.0135
  cached_input: true
  batch: false
```

Emit on every call. Store in data lake. Aggregate per dimension. Phase 17 · 13 observability stack — место, где это живет.

### The compounded-savings stack

Stack: cache + batch + route + gateway. With all four:
- Cache L2 (Phase 17 · 14): ~10x cheaper input.
- Batch (Phase 17 · 15): 50% off.
- Route to cheap model (Phase 17 · 16): 60% cost reduction.
- Gateway efficiency (Phase 17 · 19): redundancy + retries.

Best-case stacked: ~5-10% of naive baseline. У большинства команд включены 2-3 levers; немногие складывают все четыре.

### Числа, которые стоит запомнить

- Attribution dimensions: per-user, per-task, per-tenant.
- Four token layers: prompt, tool, memory, response.
- Kill switch: spend z-score > 4.
- Unit metric: cost per resolved query, not $/M tokens.
- Stacked optimizations: ~5-10% of baseline possible.

## Используйте это

`code/main.py` симулирует multi-tenant LLM service с three-tier enforcement ladder. Внедряет abusive tenant и демонстрирует срабатывание kill switch.

## Отгрузите это

Этот урок создает `outputs/skill-finops-plan.md`. По product and scale проектирует attribution schema и enforcement ladder.

## Упражнения

1. Запустите `code/main.py`. При каком z-score срабатывает kill switch? Как выбрать threshold?
2. Спроектируйте per-tenant, per-task cost dashboard. Какие 5 views построите первыми?
3. Ваш largest tenant unit-economics-negative. Предложите три interventions, ordered by customer impact.
4. Посчитайте cost per resolved ticket для support product: 3M tokens/ticket, ~800 tickets/day, GPT-5 cached rate.
5. Аргументируйте, может ли retroactive tagging когда-либо работать. Когда он приемлем?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Per-user attribution | "user-level cost" | `user_id` stamped on every call |
| Per-task attribution | "feature cost" | `task_id` + `route` identify product surface |
| Per-tenant attribution | "customer cost" | `tenant_id`; drives unit economics |
| Four token layers | "cost layers" | prompt + tool + memory + response |
| Rate limit | "429 guard" | Per-tenant ceiling enforced at gateway |
| Daily spend cap | "daily ceiling" | Tenant-scoped budget with alert |
| Kill switch | "auto-pause" | Spend z-score > 4 triggers auto-suspension |
| Cost per resolved | "product unit metric" | Cost tied to product outcome, not tokens |
| Telemetry joiner | "trace-to-billing" | Highest-accuracy attribution pattern |
| Stacked optimization | "cache+batch+route+gateway" | Compounding savings to ~5-10% baseline |

## Дополнительное чтение

- [FinOps Foundation — FinOps for AI Overview](https://www.finops.org/wg/finops-for-ai-overview/)
- [FinOps School — Cost per Unit 2026 Guide](https://finopsschool.com/blog/cost-per-unit/)
- [Digital Applied — LLM Agent Cost Attribution 2026](https://www.digitalapplied.com/blog/llm-agent-cost-attribution-guide-production-2026)
- [PointFive — Managed LLMs in Azure OpenAI](https://www.pointfive.co/blog/finops-for-ai-economics-of-managed-llms-in-azure-open-ai)
