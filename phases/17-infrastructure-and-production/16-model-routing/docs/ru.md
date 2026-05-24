# Model Routing как примитив снижения стоимости

> Динамический broker оценивает каждый request (task type, token length, embedding similarity, confidence) и отправляет простые queries в дешевую модель, повышая сложные до frontier model. Это также называют model cascading. Production case studies показывают снижение стоимости на 20-60% при iso-quality в US/UK/EU deployments; улучшение routing efficiency на 30% в высоконагруженном SaaS превращается в шестизначную годовую экономию. Контекст 2026 года: цены LLM inference падали примерно в 10x в год — token уровня GPT-4 прошел путь от $20/M до ~$0.40/M с конца 2022 до 2026. Большая часть падения — это более эффективные serving stacks (Phase 17 · 04-09), а не hardware. Routing — способ превратить это падение цены в margin без product regression. Failure mode — cheap-model drift: route отправляет 40% в более слабую модель, качество падает на 3-5% на reasoning tasks, и никто не замечает целый квартал. Контролируйте routes online quality metrics, а не только offline eval sets.

**Тип:** Learn
**Языки:** Python (stdlib, toy cascading router simulator)
**Предварительные требования:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 19 (AI Gateways)
**Время:** ~60 minutes

## Цели обучения

- Объяснить model cascading: cheap-first с confidence check, escalation при низкой confidence.
- Перечислить четыре routing signals (task classification, prompt length, embedding similarity to known-hard set, self-confidence from first-pass).
- Посчитать expected blended cost при целевом routing split и допустимой quality loss.
- Назвать drift-monitoring metric (online quality gate), который ловит cheap-model creep.

## Проблема

Ваш сервис стоит $80k/month на GPT-5. Analytics показывает, что 70% queries простые: "what time is it in Paris?" "rephrase this sentence." Модель Haiku-class идеально справляется с ними за 3% стоимости. 30% требуют reasoning GPT-5 — coding, math, multi-step planning.

Если маршрутизировать 70% в cheap и 30% в expensive, счет падает примерно на 65% при том же product quality. Это routing. Сложность — построить broker без регрессии качества.

## Концепция

### Four routing signals

1. **Task classification**: simple/complex/codegen/math/chat. Это может быть rules-based classifier, small LLM (Haiku-class at $0.25/M) или embedding similarity to labeled buckets. Output: route = cheap / balanced / frontier.

2. **Prompt length**: prompts >4K tokens часто требуют frontier для coherence. Prompts <500 tokens обычно не требуют.

3. **Embedding similarity to known-hard set**: если query близок (cosine > 0.88) к known-hard bucket, сразу escalate to frontier.

4. **Self-confidence from first-pass**: отправить в cheap; если log-probs модели показывают low confidence ИЛИ она refuses ИЛИ выдает hedging language, повторить на frontier. Добавляет P95 latency примерно на 10% traffic, но экономит 50%+ на остальных 90%.

### Three patterns

**Pre-route** (classifier up front): добавляет ~5-10ms latency; самый быстрый в целом.

**Cascade** (cheap-first, escalate on low confidence): ~1.2x median latency (cheap run плюс verify), ~2x на escalated. Лучший quality floor.

**Ensemble route** (run cheap and frontier in parallel for a sample, reward-model pick): максимальное качество, максимальная стоимость; используйте только для критичных A/B.

### Implementation

AI gateways (Phase 17 · 19) предоставляют routing. В LiteLLM есть `router` config с fallback и cost-routing. Portkey имеет guards + routing. Kong AI Gateway использует plugin-based routing. Model marketplace OpenRouter предоставляет recommendation API.

Open-source: RouteLLM (LMSYS), Not Diamond (commercial), Prompt Mule.

### The 2026 price curve

| Model class | Late 2022 | 2026 | Change |
|-------------|-----------|------|--------|
| GPT-4-level quality | ~$20/M | ~$0.40/M | в 50x дешевле |
| Frontier (GPT-5, Claude 4) | — | ~$3-10/M | новый tier |

Большая часть улучшения — serving efficiency: ключевые уроки Phase 17 · 04-09 превратились в снижение provider-side cost. Routing позволяет забрать эти gains на app layer, не ожидая, пока все пользователи перейдут на cheap tier.

### Drift is the real risk

Ваш route отправляет 40% в cheap model. За шесть месяцев distribution tasks смещается (пользователи становятся более sophisticated, задают более длинные вопросы). Router не замечает, потому что classifier был обучен на Q1 data. Качество тихо падает. Никто не жалуется достаточно громко. Вы узнаете об этом из competitor benchmark, где проиграли.

Контролируйте routes online quality metrics:

- User thumbs-up / thumbs-down per route.
- Automated LLM-judge on a held-out sample (5%) per route.
- Escalation rate: если cascade поднимает вверх >30%, cheap model over-routed.
- Refusal rate per route.

### Numbers you should remember

- Экономия routing в 2026 при iso-quality: 20-60% по case studies.
- Падение цен LLM 2022-2026: ~10x в год aggregate.
- GPT-4-level 2022 vs 2026: ~$20/M → ~$0.40/M.
- Влияние cascade на latency: ~1.2x median, ~2x escalated (~10% traffic).

## Используйте это

`code/main.py` симулирует pre-route, cascade и ensemble на mixed workload. Показывает blended cost, quality loss и escalation rate.

## Доведите до результата

Этот урок создает `outputs/skill-router-plan.md`. По workload и quality budget он выбирает routing pattern и signals.

## Упражнения

1. Запустите `code/main.py`. При каком accuracy floor cascade выигрывает у pre-route?
2. Ваша user base: 30% enterprise (complex queries), 70% free tier (simple). Спроектируйте routing split. Какая online metric его контролирует?
3. Route снижает качество на 2%, но экономит 40%. Ship или нет? Зависит от продукта — аргументируйте обе стороны.
4. Реализуйте confidence check с использованием logprobs из OpenAI / Anthropic APIs. С какого threshold начать?
5. За шесть месяцев escalation rate вырос с 8% до 22%. Диагностируйте три причины и fix для каждой.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Model routing | "cost broker" | динамический выбор модели на request |
| Model cascade | "cheap-first escalate" | запустить cheap, перейти к frontier при low confidence |
| Pre-route | "classify first" | classifier upfront; без повторного запуска |
| Ensemble route | "parallel pick" | запустить несколько, reward-model выбирает лучший |
| Escalation rate | "uprouted %" | доля cascade requests, которые escalated |
| RouteLLM | "LMSYS router" | OSS router library |
| Not Diamond | "commercial router" | SaaS-продукт для model routing |
| Drift | "cheap creep" | distribution shift, который router не замечает |
| Online quality gate | "live check" | automated LLM-judge sampling live traffic |

## Дополнительное чтение

- [AbhyashSuchi — Model Routing LLM 2026 Best Practices](https://abhyashsuchi.in/model-routing-llm-2026-best-practices/)
- [Lukas Brunner — Rise of Inference Optimization 2026](https://dev.to/lukas_brunner/the-rise-of-inference-optimization-the-real-llm-infra-trend-shaping-2026-4e4o)
- [RouteLLM paper / code](https://github.com/lm-sys/RouteLLM)
- [Not Diamond — model routing](https://www.notdiamond.ai/)
- [OpenRouter](https://openrouter.ai/) — multi-model gateway with routing primitives.
