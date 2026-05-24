# Shadow Traffic, Canary Rollout и Progressive Deployment для LLMs

> LLM rollouts объединяют самые сложные части software deployment: нет unit tests, failure modes размыты, signals приходят с задержкой. Последовательность такая: (1) shadow mode — дублировать prod requests в candidate model, логировать, сравнивать без влияния на пользователя; ловит очевидные distribution issues, но не гарантирует quality; (2) canary rollout — progressive traffic shift 10% → 25% → 50% → 75% → 100% с gates на каждом шаге; отслеживать latency percentiles, cost/request, error/refusal rate, output length distribution, user-feedback rate; (3) A/B testing для distinct alternatives после подтверждения stability. Non-determinism неустраним — до 15% accuracy variation между запусками с identical inputs из-за GPU FP non-associativity плюс batch-size variance. Cost — переменная, а не константа: модель на 20% лучше может быть в 3x дороже за call. Rollback speed решает: если rollback требует redeploy, вы слишком медленные. Policy живет в config/flags; model живет в registry with pinned digests; rollback = flip policy + revert threshold + pin old model за секунды.

**Тип:** Learn
**Языки:** Python (stdlib, toy canary-progression simulator)
**Предварительные требования:** Phase 17 · 13 (Observability), Phase 17 · 21 (A/B Testing)
**Время:** ~60 minutes

## Цели обучения

- Отличить shadow mode (zero-impact compare), canary (live traffic progressive) и A/B (stability-confirmed comparison).
- Перечислить пять LLM-specific canary metrics (latency, cost/request, error/refusal, output-length distribution, user feedback).
- Объяснить, почему LLM non-determinism (до 15%) меняет смысл "stable" в rollout.
- Спроектировать rollback path, который занимает секунды (policy flip), а не часы (redeploy).

## Проблема

Вы выкатываете новую model. Offline evals показывают прирост accuracy на 3%. Вы включаете ее в production. За 24 часа cost вырос на 40%, user thumbs-down вырос на 8%, три customer tickets сообщают "weird answers." Вы откатываетесь. Redeploy занимает 3 часа. Выходные испорчены.

Каждая часть этого была предотвратима. Shadow mode поймал бы 40% cost spike до того, как его увидел пользователь. Canary остановился бы на 10%, когда сдвинулся thumbs-down. Policy-flag rollback занял бы 30 секунд. Дисциплина закрывает промежуток между "offline evals look good" и "real users are happy."

## Концепция

### Shadow mode

Candidate получает те же requests, что и production; outputs логируются, но не возвращаются пользователям. Нулевое влияние на пользователя. Логируйте:

- Output content (diff against production).
- Token counts (cost delta).
- Latency.
- Refusal and error.

Ловит: cost blow-ups, length regressions, obvious refusal changes, hard errors. НЕ ловит: quality delta, которую восприняли бы users. Shadow — smoke test, не quality test.

### Canary rollout

Progressive traffic shift with gates. Типичная progression: 1% → 10% → 25% → 50% → 75% → 100%. Gate на 5 metrics на каждом шаге:

1. **Latency percentiles** — P50, P95, P99. Breach: canary has P99 > 1.5x baseline.
2. **Cost per request** — blended $. Breach: >20% above baseline.
3. **Error / refusal rate** — 5xx plus explicit refusals. Breach: 2x baseline.
4. **Output length distribution** — mean + P99. Breach: distributional shift.
5. **User-feedback rate** — thumbs-down / ticket filings. Breach: 1.5x baseline.

### Non-determinism is the new variance

Identical inputs produce non-identical outputs. Причины:

- GPU FP non-associativity (floating-point reduction order varies by batch).
- Batch-size variance (same prompt in a batch of 128 vs batch of 16).
- Sampling (temperature > 0).

Измерено: до 15% accuracy variation run-to-run на identical eval sets. "Stable" в rollout означает, что metrics находятся within expected variance, а не identical to baseline. Устанавливайте gates выше noise floor.

### Cost is a variable

Модель на 20% лучше может быть в 3x дороже за call. Cost/request — один из пяти gates. Выпуск "better" model, которая ломает unit economics, — причина для rollback.

### Rollback is the weapon

- Policy flag (feature flag system): flip percentage in config; занимает секунды.
- Model pinning (registry digest): pinned model does not auto-upgrade.
- Rollback = revert flag + set pinned digest to previous. Seconds, not hours.

Если вашему stack нужен redeploy для rollback, исправьте это до rollout.

### Tooling

**Argo Rollouts** / **Flagger** — Kubernetes progressive delivery controllers. Интегрируются с Istio/Linkerd weighted routing.

**Istio weighted routing** — service-mesh-level traffic split.

**KServe / Seldon Core** — model serving with built-in canary.

**Feature flags** — LaunchDarkly, Flagsmith, Unleash. Policy-level flip, без redeploy.

### Metrics cadence

Canary gates проверяются каждые 5-15 минут в зависимости от traffic volume. 1% traffic при 10 req/min дает 50-150 data points per window — достаточно для latency, но noisy для user feedback. 10% дает примерно в 10x больше. Progressions должны останавливаться достаточно надолго, чтобы накопить enough samples на каждом шаге.

### The A/B step is optional

Если новая model явно отличается (different behavior, different cost curve, different tone), проведите A/B test на 50% после прохождения canary. Если это просто improved version, переходите к 100%, когда canary gates пройдены.

### Numbers you should remember

- Canary progression: 1% → 10% → 25% → 50% → 75% → 100%.
- Non-determinism ceiling: до 15% run-to-run variance on identical inputs.
- Five canary metrics: latency, cost, error/refusal, output length, user feedback.
- Cost gate: >20% above baseline is a breach.
- Rollback: seconds, not hours.

## Используйте это

`code/main.py` симулирует canary rollout with injected regressions. Показывает, на какой stage rollout останавливается и какой gate сработал.

## Доведите до результата

Этот урок создает `outputs/skill-rollout-runbook.md`. По candidate model, baseline и risk tolerance он проектирует план shadow→canary→100%.

## Упражнения

1. Запустите `code/main.py`. Inject a 25% cost regression. На каком stage canary остановится?
2. Новая model имеет 3% accuracy gain offline, но cost/request +18%. Ship или нет? Зависит от policy — запишите оба пути.
3. Спроектируйте rollback, который занимает under 60 seconds end-to-end. Перечислите required infrastructure.
4. Non-determinism показывает ±7% на вашем eval. Настройте canary gates так, чтобы не получать false alarm. Какие multipliers использовать?
5. Shadow mode ловит 40% cost spike до canary. Запишите alert rule, который сработает в shadow.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Shadow mode | "duplicate to new" | zero-impact send-to-candidate for logging |
| Canary | "progressive traffic" | постепенный user-exposed rollout with gates |
| Gates | "rollout checks" | metric thresholds, блокирующие progression |
| Non-determinism | "LLM variance" | неустранимые run-to-run differences |
| Policy flag | "flag flip rollback" | config-level rollback, seconds not hours |
| Model pin | "registry digest" | immutable reference to a model version |
| Argo Rollouts | "K8s progressive" | Kubernetes-native canary/rollback controller |
| KServe | "inference K8s" | model serving with canary primitives |
| Istio weighted | "mesh split" | service-mesh traffic splitter |

## Дополнительное чтение

- [TianPan — Releasing AI Features Without Breaking Production](https://tianpan.co/blog/2026-04-09-llm-gradual-rollout-shadow-canary-ab-testing)
- [MarkTechPost — Safely Deploying ML Models](https://www.marktechpost.com/2026/03/21/safely-deploying-ml-models-to-production-four-controlled-strategies-a-b-canary-interleaved-shadow-testing/)
- [APXML — Advanced LLM Deployment Patterns](https://apxml.com/courses/mlops-for-large-models-llmops/chapter-4-llm-deployment-serving-optimization/advanced-llm-deployment-patterns)
- [Argo Rollouts docs](https://argo-rollouts.readthedocs.io/)
- [Flagger docs](https://docs.flagger.app/)
