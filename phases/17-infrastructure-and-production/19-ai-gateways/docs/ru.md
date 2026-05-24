# AI Gateways — LiteLLM, Portkey, Kong AI Gateway, Bifrost

> Gateway находится между вашими apps и model providers. Core features: provider routing, fallback, retries, rate limiting, secret references, observability, guardrails. Market split in 2026: **LiteLLM** — MIT OSS с 100+ providers, OpenAI-compatible, но ломается около ~2000 RPS (8 GB memory, cascading failures в published benchmarks); лучше всего для Python, <500 RPS, dev/prototyping. **Portkey** позиционируется как control plane (guardrails, PII redaction, jailbreak detection, audit trails), стал Apache 2.0 open-source в March 2026, 20-40 ms latency overhead, production tier за $49/mo. **Kong AI Gateway** построен на Kong Gateway — собственный benchmark Kong на тех же 12 CPUs: на 228% быстрее Portkey, на 859% быстрее LiteLLM; pricing $100/model/month (max 5 на Plus tier); enterprise-fit, если вы уже на Kong. **Bifrost** (Maxim AI) — automatic retries with configurable backoff, fallback to Anthropic on OpenAI 429. **Cloudflare / Vercel AI Gateways** — managed, zero-ops, basic retry. Data residency определяет self-host decision; Portkey и Kong находятся посередине с OSS + optional managed.

**Тип:** Learn
**Языки:** Python (stdlib, toy gateway-routing simulator)
**Предварительные требования:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 16 (Model Routing)
**Время:** ~60 minutes

## Цели обучения

- Перечислить шесть core gateway features (routing, fallback, retries, rate limits, secrets, observability, guardrails).
- Сопоставить четыре gateways 2026 года (LiteLLM, Portkey, Kong AI, Bifrost) с scale ceilings и use cases.
- Процитировать Kong benchmark (228% vs Portkey, 859% vs LiteLLM) и объяснить, почему это важно для >500 RPS.
- Выбрать self-hosted vs managed с учетом data residency и ops budget.

## Проблема

Ваш продукт вызывает OpenAI, Anthropic и self-hosted Llama. У каждого провайдера свой SDK, error model, rate limit и auth scheme. Вам нужны failover (если OpenAI возвращает 429, попробовать Anthropic), единое credential store, unified observability и rate limits per tenant.

Изобретать это на app layer означает связать каждый service с каждым provider. Gateway layer консолидирует это в одном process с одним API (обычно OpenAI-compatible), который fan out к providers.

## Концепция

### Six core features

1. **Provider routing** — OpenAI, Anthropic, Gemini, self-hosted и т.д. за одним API.
2. **Fallback** — при 429, 5xx или quality failure повторить elsewhere.
3. **Retries** — exponential backoff, bounded attempts.
4. **Rate limits** — per-tenant, per-key, per-model.
5. **Secret references** — получать credentials из vault at runtime (никогда в app).
6. **Observability** — OTel + GenAI attributes (Phase 17 · 13) + cost attribution.
7. **Guardrails** — PII redaction, jailbreak detection, allowed-topics filters.

### LiteLLM — MIT OSS, Python

- 100+ providers, OpenAI-compatible, router config, fallback, basic observability.
- Ломается около 2000 RPS в benchmark Kong; 8 GB memory footprint, cascading failures under sustained load.
- Best fit: Python app, <500 RPS, dev/staging gateways, experimental routing.
- Cost: $0 for OSS; cloud free tier exists.

### Portkey — control plane positioning

- Apache 2.0 OSS с March 2026. Guardrails, PII redaction, jailbreak detection, audit trails.
- 20-40 ms per-request latency overhead.
- $49/mo for production tier with retention + SLA.
- Best fit: regulated industries needing guardrails + observability bundled.

### Kong AI Gateway — the scale play

- Построен на Kong Gateway (mature API gateway product, lua+OpenResty).
- Собственный benchmark Kong на 12-CPU equivalent: 228% быстрее Portkey, 859% быстрее LiteLLM.
- Pricing: $100/model/month, max 5 on Plus tier.
- Best fit: already on Kong; >1000 RPS; willing to license.

### Bifrost (Maxim AI)

- Automatic retries with configurable backoff.
- Fallback to Anthropic on OpenAI 429 — canonical recipe.
- Newer entrant; commercial.

### Cloudflare AI Gateway / Vercel AI Gateway

- Managed, zero-ops. Basic retry and observability.
- Best fit: Edge-serving JavaScript apps on Cloudflare/Vercel.
- Limited compared to Kong/Portkey on guardrails and rate limits.

### Self-hosted vs managed

Data residency — forcing function. Healthcare and finance default self-host (LiteLLM or Portkey OSS or Kong). Consumer products default managed (Cloudflare AI Gateway) или middle-tier (Portkey managed). Hybrid: self-hosted for regulated tenant, managed for others.

### Latency budget

- LiteLLM: 5-15 ms overhead typical.
- Portkey: 20-40 ms overhead.
- Kong: 3-8 ms overhead.
- Cloudflare/Vercel: 1-3 ms overhead (edge advantage).

Gateway latency напрямую добавляется к TTFT. Для TTFT P99 < 100 ms SLA — Kong или Cloudflare. Для P99 < 500 ms — любой.

### Rate-limit semantics matter

Simple token-bucket работает до moderate scale. Multi-tenant требует sliding-window + burst allowance + per-tenant tiering. LiteLLM поставляет token-bucket; Kong — sliding-window; Portkey — tiered.

### Gateway + observability + routing compose

Phase 17 · 13 (observability) + 16 (model routing) + 19 (gateways) — это один слой в production. Выберите один tool, который покрывает все три, или аккуратно соедините их: большинство deployments 2026 года объединяют Helicone (observability) или Portkey (guardrails) с Kong (scale) для split roles.

### Numbers you should remember

- LiteLLM: breaks at ~2000 RPS, 8 GB memory.
- Portkey: 20-40 ms overhead; Apache 2.0 since March 2026.
- Kong: 228% faster than Portkey, 859% faster than LiteLLM.
- Kong pricing: $100/model/month, 5 max on Plus tier.
- Cloudflare/Vercel: 1-3 ms overhead at the edge.

## Используйте это

`code/main.py` симулирует gateway routing с fallback по 3 providers при injection 429/5xx. Показывает latency, retry rate и fallback hit rate.

## Доведите до результата

Этот урок создает `outputs/skill-gateway-picker.md`. По scale, ops posture, compliance и latency budget он выбирает gateway.

## Упражнения

1. Запустите `code/main.py`. Configure fallback from OpenAI→Anthropic→self-hosted. Какой expected hit rate при 5% provider error rate?
2. Ваш SLA — TTFT P99 < 200 ms на baseline 300 ms. Какие gateways остаются within budget?
3. Healthcare customer требует self-hosted + PII redaction + audit. Выберите Portkey OSS или Kong.
4. Сравните LiteLLM vs Kong: при каком RPS ceiling команде стоит migrate?
5. Спроектируйте rate-limit policy для multi-tenant SaaS: free tier, trial tier, paid tier. Token-bucket или sliding-window?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Gateway | "API broker" | process между apps и providers |
| LiteLLM | "the MIT one" | Python OSS, 100+ providers, ломается на 2K RPS |
| Portkey | "guardrails gateway" | control plane + observability, Apache 2.0 |
| Kong AI Gateway | "the scale one" | построен на Kong Gateway, benchmark leader |
| Bifrost | "Maxim's gateway" | retries + Anthropic fallback recipe |
| Cloudflare AI Gateway | "edge managed" | edge-deployed managed gateway, zero-ops |
| PII redaction | "data scrub" | Regex + NER mask перед отправкой в model |
| Jailbreak detection | "prompt injection guard" | classifier on user input |
| Audit trail | "regulated log" | immutable record каждого LLM call |
| Token-bucket | "simple rate limit" | refill-based rate limiter |
| Sliding-window | "precise rate limit" | time-windowed rate limiter; better fairness |

## Дополнительное чтение

- [Kong AI Gateway Benchmark](https://konghq.com/blog/engineering/ai-gateway-benchmark-kong-ai-gateway-portkey-litellm)
- [TrueFoundry — AI Gateways 2026 Comparison](https://www.truefoundry.com/blog/a-definitive-guide-to-ai-gateways-in-2026-competitive-landscape-comparison)
- [Techsy — Top LLM Gateway Tools 2026](https://techsy.io/en/blog/best-llm-gateway-tools)
- [LiteLLM GitHub](https://github.com/BerriAI/litellm)
- [Portkey GitHub](https://github.com/Portkey-AI/gateway)
- [Kong AI Gateway docs](https://docs.konghq.com/gateway/latest/ai-gateway/)
