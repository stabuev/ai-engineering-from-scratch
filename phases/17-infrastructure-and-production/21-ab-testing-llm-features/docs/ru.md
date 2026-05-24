# A/B Testing LLM Features — GrowthBook, Statsig и проблема Vibes

> Traditional A/B testing не был построен для non-deterministic LLMs. Критическое различие: evals отвечают "can the model do the job?" A/B tests отвечают "do users care?" Нужны оба; shipping on vibe checks закончился. Что тестировать в 2026: prompt engineering (wording), model selection (GPT-4 vs GPT-3.5 vs OSS; accuracy vs cost vs latency), generation parameters (temperature, top-p). Реальные cases: chatbot reward-model variant дал +70% conversation length и +30% retention; Nextdoor AI subject-line experiments дали +1% CTR после reward-function refinement; Khan Academy Khanmigo итерировал по оси latency-vs-math-accuracy. Platform split: **Statsig** (acquired by OpenAI for $1.1B in September 2025) — sequential testing, CUPED, all-in-one. **GrowthBook** — open-source, warehouse-native, Bayesian + Frequentist + Sequential engines, CUPED, SRM checks, Benjamini-Hochberg + Bonferroni corrections. Вы выбираете на основе warehouse-SQL preference и того, важно ли вашей организации "acquired by OpenAI".

**Тип:** Learn
**Языки:** Python (stdlib, toy sequential test simulator)
**Предварительные требования:** Phase 17 · 13 (Observability), Phase 17 · 20 (Progressive Deployment)
**Время:** ~60 minutes

## Цели обучения

- Отличить evals ("can the model do the job") от A/B tests ("do users care").
- Перечислить три testable axes (prompt, model, parameters) и выбрать metric для каждой.
- Объяснить CUPED, sequential testing и Benjamini-Hochberg multiple-comparison corrections.
- Выбрать Statsig или GrowthBook по warehouse-SQL posture и corporate acquisition stance.

## Проблема

Вы вручную tuned system prompt. Он кажется лучше. Вы ship. Conversion меняется в пределах noise. Вы вините metric. Или вы выпустили новую model, а conversion не сдвинулся — модель деградировала или change слишком мал для detection? Вы не знаете, потому что shipped without an A/B.

Evals отвечают, может ли model выполнить task на labeled set. Они не отвечают, предпочитают ли users output. Это показывает только controlled online experiment, и только если у experiment достаточно power, он контролирует non-determinism и корректирует multiple comparisons.

## Концепция

### Evals vs A/B tests

**Evals** — offline, labeled set, judge (rubric or LLM-as-judge or human). Ответ: "Is the output correct / helpful / safe on this fixed distribution?"

**A/B test** — online, live users, randomized. Ответ: "Does the new variant move the user-level metric that matters?"

Нужны оба. Evals ловят regressions до exposure; A/B подтверждает product impact после.

### What to test

1. **Prompt engineering** — wording, system-prompt structure, examples. Metric: task success, user retention, cost/request.
2. **Model selection** — GPT-4 vs GPT-3.5-Turbo vs Llama-OSS. Metric: accuracy (task) + cost/request + latency P99. Multi-objective.
3. **Generation parameters** — temperature, top-p, max_tokens. Metric: task-specific (output diversity vs determinism).

### CUPED — variance reduction

Controlled-experiments Using Pre-Experiment Data. Вычитает pre-period variance через regression до сравнения post-period. Типичное variance reduction: 30-70%. Effective sample size растет бесплатно.

Implementation: и Statsig, и GrowthBook implement.

### Sequential testing

Classical A/B предполагает fixed sample size. Sequential tests ("peek-and-decide") контролируют false-positive rate при repeated looks. Always-valid sequential procedures (mSPRT, Howard's confidence sequences) позволяют stop early на clear winners.

### Multiple-comparison corrections

Запуск 20 A/B tests при 95% confidence дает один false positive случайно. Bonferroni correction ужесточает α per-test; Benjamini-Hochberg контролирует false-discovery rate. GrowthBook implements both.

### SRM — sample ratio mismatch

Assignment hash randomizes users to variants. Если split 50/50 дает 47/53, что-то сломано — SRM check это отмечает. Обе платформы implement.

### Statsig vs GrowthBook

**Statsig**:
- Acquired by OpenAI for $1.1B (September 2025). Hosted, SaaS.
- Sequential testing, CUPED, held-out populations.
- All-in-one: feature flags + experimentation + observability.
- Best fit: team already wants a bundled product, doesn't care about OpenAI ownership.

**GrowthBook**:
- Open-source (MIT); warehouse-native (читает напрямую из Snowflake/BigQuery/Redshift).
- Multiple engines: Bayesian, Frequentist, Sequential.
- CUPED, SRM, Bonferroni, BH corrections.
- Self-host or managed cloud.
- Best fit: warehouse-SQL shop, data team controls the metric layer, wants OSS.

### Non-determinism complicates power

Один и тот же prompt дает разные outputs. Traditional power calculations предполагают IID observations. С LLM non-determinism effective sample size ниже nominal. Умножайте required sample size примерно на 1.3-1.5x как safety margin.

### Real case outcomes

- Chatbot reward model variant: +70% conversation length, +30% retention.
- Nextdoor subject lines: +1% CTR after reward-function refinement.
- Khan Academy Khanmigo: iterative latency-vs-math-accuracy trade.

### The anti-pattern: shipping on vibes

Каждый senior engineer может назвать feature, которую ship, потому что "it feels better", без A/B. Большинство таких features ухудшили product metrics, что команда не замечала месяцами. A/B — forcing function.

### Numbers you should remember

- Statsig acquired by OpenAI: $1.1B, September 2025.
- GrowthBook: open-source MIT; Bayesian + Frequentist + Sequential.
- CUPED variance reduction: 30-70%.
- LLM non-determinism → +30-50% sample-size buffer.

## Используйте это

`code/main.py` симулирует sequential A/B test with fixed and sequential boundaries. Показывает, как sequential позволяет stop early.

## Доведите до результата

Этот урок создает `outputs/skill-ab-plan.md`. По feature change, workload и baseline он выбирает platform, gates, sample size.

## Упражнения

1. Запустите `code/main.py`. For an expected 5% lift with baseline 3% conversion, what sample size to 80% power?
2. Выберите Statsig или GrowthBook для healthcare-regulated on-prem customer.
3. Спроектируйте A/B, который тестирует GPT-4 vs GPT-3.5 on cost-per-resolved-ticket. Что primary metric, guardrail metric, secondary?
4. Canary проходит, но A/B показывает -1.2% conversion. Ship или нет? Запишите escalation criteria.
5. Примените CUPED к pre-period with 60% of the variance of post. Посчитайте effective-sample-size boost.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Eval | "offline test" | labeled-set evaluation of model capability |
| A/B test | "experiment" | live randomized comparison on users |
| CUPED | "variance reduction" | pre-period regression to reduce variance |
| Sequential test | "peek-ok test" | always-valid procedure allowing early stop |
| Multiple comparison | "the family error" | запуск многих tests повышает false positives |
| Bonferroni | "tight correction" | разделить α на number of tests |
| Benjamini-Hochberg | "BH FDR" | false-discovery-rate control, less conservative |
| SRM | "bad split" | sample ratio mismatch; assignment bug |
| Statsig | "OpenAI owned" | commercial all-in-one, acquired 2025 |
| GrowthBook | "the OSS one" | MIT warehouse-native platform |
| mSPRT | "sequential probability ratio test" | classical sequential procedure |

## Дополнительное чтение

- [GrowthBook — How to A/B Test AI](https://blog.growthbook.io/how-to-a-b-test-ai-a-practical-guide/)
- [Statsig — Beyond Prompts: Data-Driven LLM Optimization](https://www.statsig.com/blog/llm-optimization-online-experimentation)
- [Statsig vs GrowthBook comparison](https://www.statsig.com/perspectives/ab-testing-feature-flags-comparison-tools)
- [Deng et al. — CUPED](https://www.exp-platform.com/Documents/2013-02-CUPED-ImprovingSensitivityOfControlledExperiments.pdf)
- [Howard — Confidence Sequences](https://arxiv.org/abs/1810.08240)
