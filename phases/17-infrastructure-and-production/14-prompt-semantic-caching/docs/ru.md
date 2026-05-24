# Prompt Caching and Semantic Caching Economics

> **Pricing snapshot dated 2026-04.** Numeric claims ниже отражают vendor rate cards, зафиксированные при публикации урока; проверяйте linked docs перед цитированием дальше.

> Caching происходит на двух слоях. L2 (provider-level) prompt/prefix caching переиспользует attention KV для повторяющихся prefixes — docs Anthropic prompt-caching заявляют до 90% cost reduction и 85% latency reduction на long prompts; для Claude 3.5 Sonnet cache reads стоят $0.30/M vs $3.00/M fresh с 5-minute TTL и 2x write premium для 1-hour TTL option (docs.anthropic.com, 2026-04). OpenAI prompt caching применяется автоматически для prompts ≥1024 tokens и оценивает cached input примерно с 90% discount vs fresh (platform.openai.com, 2026-04); точная per-model cached rate зависит от live rate card. L1 (app-level) semantic caching полностью пропускает LLM при embedding similarity hits. Vendor "95% accuracy" означает correctness match, а не hit rate — production hit rates по отчетам варьируются от 10% (open-ended chat) до 70% (structured FAQ); ни один provider не публикует official baseline, поэтому считайте это community telemetry, а не guarantees. Production pitfalls: parallelization kills caching (N parallel requests, отправленных до первой cache write, могут увеличить spend в несколько раз), а dynamic content внутри prefix полностью предотвращает cache hits. ProjectDiscovery reported moving from 7% to 74% hit rate (2025-11) by moving dynamic text out of the cacheable prefix.

**Тип:** Learn
**Языки:** Python (stdlib, toy two-layer cache simulator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 06 (SGLang RadixAttention)
**Время:** ~60 minutes

## Цели обучения

- Отличить L2 prompt/prefix caching (KV reuse at provider) от L1 semantic caching (LLM bypass on similar prompts).
- Объяснить explicit marking Anthropic `cache_control` и два TTL options (5-min vs 1-hour) с их price multipliers.
- Вычислить expected monthly savings по hit rate, prompt/response mix и token prices.
- Назвать parallelization anti-pattern, который увеличивает bills by 5-10x, и dynamic-content anti-pattern, который обрушивает hit rate.

## Проблема

Вы добавляете prompt caching в RAG service. Bill не меняется. Вы измеряете hit rate; он 7%. Prompts выглядят static, но это не так — system prompt включает current date formatted to the minute, request ID и randomized example reorder for diversity. Каждый request пишет новый cache entry, читает ноль.

Отдельно ваш agent запускает ten parallel tool calls per user question. Все десять приходят к provider до завершения первой cache write. Десять writes, zero reads. Ваш bill в 5-10x выше, чем должен был быть "with caching".

Caching — это protocol, а не flag. Два слоя, два разных failure modes.

## Концепция

### L2 — provider prompt/prefix caching

Provider хранит attention KV для cacheable prefix и переиспользует его в следующем request с совпадающим prefix. Вы один раз платите write cost, reads почти бесплатны.

**Anthropic (Claude 3.5 / 3.7 / 4 series)**: explicit `cache_control` marker в request. Вы помечаете, какие blocks cacheable. TTL: 5-minute (write costs 1.25x base) или 1-hour (write costs 2x base). Cache reads: $0.30/M on Claude 3.5 Sonnet vs $3.00/M fresh — 10x cheaper (docs.anthropic.com, as of 2026-04). Rates differ per model (Opus/Haiku published separately); always cross-check the live pricing page.

**OpenAI**: automatic caching for prompts ≥1024 tokens (platform.openai.com, 2026-04). No explicit flag. Cached input примерно 10x cheaper than fresh on current gpt-4o/gpt-5 rate cards. Ни docs, ни release notes не публикуют official hit-rate baseline; community reports cluster around 30–60% with careful prompt design. Monitor `usage.cached_tokens` to measure your own.

**Google (Gemini)**: context caching via explicit API; 1M-token context значит, что caching pays even more.

**Self-hosted (vLLM, SGLang)**: Phase 17 · 06 covers RadixAttention — тот же pattern on your own compute.

### L1 — app-level semantic caching

До вызова LLM hash the prompt, embed it и ищите similar cached request (cosine similarity above threshold, обычно 0.95+). On hit возвращайте cached response. On miss вызывайте LLM и cache the result.

Open-source: Redis Vector Similarity, GPTCache, Qdrant. Commercial: Portkey Cache, Helicone Cache.

Vendor accuracy claims означают, насколько часто returned cached response был semantically appropriate — не как часто вы hit. Production hit rates:

- Open-ended chat: 10-15%.
- Structured FAQ / support: 40-70%.
- Code questions: 20-30% (small variants kill hits).
- Voice agents repeating prompts: 50-80% (voice normalization fixed set).

### Parallelization anti-pattern

Ваш agent делает 10 tool calls in parallel. Все 10 имеют один и тот же 4K-token system prompt. Anthropic cache writes per-request; первая cache-write завершается примерно через 300 ms после того, как provider видит prompt. Requests 2-10 приходят в тот же millisecond window, и каждый получает cache miss. Вы платите 10 write premiums, 0 read discounts.

Fix: batch with sequential-first — сделайте request 1 alone, затем fire 2-10, когда 1's cache has populated. Добавляет 300 ms к первому tool call; saves 5-10x the bill.

### Dynamic content anti-pattern

Ваш system prompt выглядит так:

```
You are a helpful assistant. The current time is 14:32:17.
User ID: abc123. Today is Tuesday...
```

Каждый request уникален. Каждый request writes. Zero hits.

Fix: переместите все действительно static в cacheable prefix; добавьте dynamic content после cache boundary:

```
[cacheable]
You are a helpful assistant. [rules, examples, instructions]
[/cacheable]
[dynamic, not cached]
Current time: 14:32:17. User: abc123.
```

ProjectDiscovery подняли cache hit rate с 7% до 74% таким способом и опубликовали anatomy.

### Stack batch + cache для overnight workloads

Batch APIs (Phase 17 · 15) дают 50% discount при 24-hour turnaround. Cached input сверху дает еще ~10x. Overnight classification, labeling и report generation workloads могут снизиться до ~10% synchronous-uncached cost за счет stacking.

### Числа, которые нужно помнить

Pricing points captured 2026-04 из linked vendor docs и drift every few months — перепроверяйте перед использованием.

- Anthropic cached read: $0.30/M on Claude 3.5 Sonnet, roughly 10x cheaper than fresh input (docs.anthropic.com).
- Anthropic cache write premium: 1.25x (5-min TTL) or 2x (1-hour TTL).
- OpenAI auto-cache: applies to prompts ≥1024 tokens; cached input priced at roughly 10% of fresh input on current rate cards (platform.openai.com).
- Semantic cache hit rate (community-reported): ~10% open chat; up to ~70% structured FAQ. Not a vendor-documented baseline.
- ProjectDiscovery: 7% → 74% hit rate by moving dynamic out of prefix (project blog, 2025-11).
- Parallelization anti-pattern: typical reports of 5–10x bill inflation when N parallel requests miss the first cache write.

## Используйте это

`code/main.py` симулирует L1 + L2 caching на mixed workloads. Сообщает hit rates, bill и показывает parallelization penalty.

## Отправьте в прод

Этот урок создает `outputs/skill-cache-auditor.md`. По prompt template и traffic он audits cacheability и рекомендует restructure.

## Упражнения

1. Запустите `code/main.py`. Toggle the parallelization flag. Насколько изменится bill?
2. В вашем system prompt есть date. Переместите ее наружу. Покажите before/after hit rate math.
3. Рассчитайте break-even для 1-hour TTL (2x write) vs 5-minute TTL (1.25x write) given your request arrival rate.
4. Semantic cache at 0.95 threshold hits 20%. At 0.85 it hits 50%, but you see incorrect cached responses. Выберите правильный threshold и обоснуйте.
5. Вы batch 10 parallel sub-queries per user question. Перепишите для cache-friendliness без добавления end-to-end latency.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| L2 prompt cache | "prefix cache" | Provider stores KV for repeated prefix |
| `cache_control` | "Anthropic cache marker" | Explicit attribute marking cacheable blocks |
| Cache write premium | "write tax" | Extra cost for first miss-to-cache (1.25x or 2x) |
| L1 semantic cache | "embedding cache" | App-level hash-and-embed before calling LLM |
| GPTCache | "LLM caching lib" | Popular OSS L1 cache library |
| Cache hit rate | "hits / total" | Fraction of requests served from cache |
| Parallelization anti-pattern | "the N-write trap" | N parallel requests miss cache N times |
| Dynamic content trap | "the time-in-prompt trap" | Dynamic bytes in prefix kill hit rate |
| RadixAttention | "intra-replica cache" | SGLang's prefix-cache implementation |

## Дополнительное чтение

- [Anthropic Prompt Caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — official `cache_control` semantics and TTLs.
- [OpenAI Prompt Caching](https://platform.openai.com/docs/guides/prompt-caching) — automatic caching behavior and eligibility.
- [TianPan — Semantic Caching for LLMs Production](https://tianpan.co/blog/2026-04-10-semantic-caching-llm-production)
- [ProjectDiscovery — Cut LLM Costs 59% With Prompt Caching](https://projectdiscovery.io/blog/how-we-cut-llm-cost-with-prompt-caching)
- [DigitalOcean / Anthropic — Prompt Caching](https://www.digitalocean.com/blog/prompt-caching-with-digital-ocean)
