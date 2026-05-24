# SRE for AI — Multi-Agent Incident Response, Runbooks, Predictive Detection

> AI SRE использует LLM, grounded in infrastructure data (logs, runbooks, service topology), через RAG для автоматизации фаз расследования, документирования и координации. Архитектурный паттерн 2026 — multi-agent orchestration: специализированные агенты (logs, metrics, runbooks), координируемые supervisor; AI предлагает гипотезы и queries, люди утверждают judgment calls. Datadog Bits AI и Azure SRE Agent поставляют это как managed products. Runbooks эволюционируют: NeuBird Hawkeye использует adversarial evaluation (две модели анализируют один incident; agreement = confidence, disagreement = uncertainty); operational memory сохраняется при смене команды. Auto-remediation остается осторожной: AI предлагает, люди утверждают. Полностью автономные действия узкие (restart pod, rollback specific deploy) с жесткими guardrails — все, кто продает "set it and forget it", переобещают. Новый фронтир: pre-incident prediction. Исследование MIT сообщает, что LLM, обученная на historical logs + GPU temps + API error patterns, предсказала 89% outages за 10-15 min. Прогноз: у 95% enterprise LLM будет automated failover к концу 2026.

**Тип:** Learn
**Языки:** Python (stdlib, toy multi-agent incident triage simulator)
**Предварительные требования:** Phase 17 · 13 (Observability), Phase 17 · 24 (Chaos Engineering)
**Время:** ~60 minutes

## Цели обучения

- Нарисовать multi-agent AI SRE architecture: supervisor + specialized agents (logs, metrics, runbooks) + human approval gate.
- Объяснить, почему auto-remediation узкая (restart pod, revert deploy), а не широкая (re-architect service).
- Назвать adversarial evaluation pattern (NeuBird Hawkeye): две модели согласны = confidence; не согласны = escalate.
- Процитировать результат MIT 89% early-detection и операционное ограничение: predictions without actuation are just dashboards.

## Проблема

On-call engineer получает page в 3 a.m. "High error rate in checkout." Он проверяет Datadog, Loki, три runbooks, deploy log. Через 30 минут понимает, что root cause — vLLM OOM из-за spike KV cache. Он перезапускает pod; ошибка исчезает.

В 2026 первые 20 минут такого расследования автоматизируемы. Группировка logs по service, корреляция с recent deploys, сопоставление с runbooks — все это RAG + tool-use. Supervised agent может выполнить first-pass triage и представить гипотезу до того, как человек откроет Datadog.

Полностью автономная remediation — другая проблема. Restart pod: безопасно. Scale GPU pool: безопасно, если policy разрешает. Re-architect the service: категорически нет. Дисциплина в том, чтобы провести узкую границу.

## Концепция

### Multi-agent architecture

```
          Incident
             │
             ▼
        Supervisor
        /    |    \
       ▼     ▼     ▼
  Log agent  Metric agent  Runbook agent
       │     │     │
       └─────┴─────┘
             │
             ▼
        Hypothesis + evidence
             │
             ▼
        Human approval
             │
             ▼
        Action (narrow set)
```

Supervisor разбивает incident на sub-queries. Specialized agents имеют доступ к инструментам (log search, PromQL, doc retrieval). Supervisor синтезирует результат, показывает hypothesis + evidence человеку. Человек утверждает или перенаправляет.

### Область auto-remediation

**Safe (narrow)**: restart pod, revert specific deploy, scale pool within pre-approved bounds, enable pre-approved feature flag.

**Not safe (broad)**: change service topology, modify resource limits, deploy new code, change IAM, alter databases.

Все, кто продает "set it and forget it", переобещают. Безопасный набор растет по мере взросления AI SRE, но граница реальна.

### Adversarial evaluation (NeuBird Hawkeye)

Две модели независимо анализируют один incident. Если они согласны по root cause, confidence высокий. Если не согласны, escalate to human с обеими гипотезами на виду. Простой паттерн, эффективный фильтр против hallucinated root causes.

### Operational memory

Team turnover — тихий убийца традиционного SRE: tribal knowledge уходит. AI SRE хранит runbooks + post-mortems в vector DB; agents извлекают их при каждом новом incident. Когда приходят новые engineers, у AI есть вся история.

### Pre-incident prediction

Исследование MIT 2025: LLM, обученная на historical logs, GPU temperatures, API error patterns, предсказала 89% outages за 10-15 минут до их возникновения на test set.

Reality check: predictions without actuation are dashboards. Операционный вопрос: "когда мы предсказываем, что делаем?" Pre-emptive drain? Pager? Auto-scale? Ответ зависит от policy.

### Products in 2026

- **Datadog Bits AI** — managed SRE copilot внутри Datadog.
- **Azure SRE Agent** — Azure-native.
- **NeuBird Hawkeye** — adversarial eval + operational memory.
- **PagerDuty AIOps** — triage + deduplication.
- **Incident.io Autopilot** — incident commander + coordination.

### Runbooks as code

Runbooks эволюционируют из Confluence pages в versioned markdown со structured sections (symptom, hypothesis, verify, act). Структурированные runbooks дают лучший RAG retrieval. Любой AI-SRE rollout начинайте с превращения неструктурированных runbooks в structured.

### Числа, которые стоит запомнить

- MIT early-detection: 89% outages, 10-15 min lead time.
- Multi-agent triage: supervisor + (logs, metrics, runbooks) + human.
- Safe auto-remediation set: restart pod, revert deploy, scale within bounds.
- Adversarial eval: two models independent; agreement = confidence.

## Используйте это

`code/main.py` симулирует multi-agent triage: log agent находит error, metric agent находит CPU spike, runbook agent сопоставляет с known issue. Supervisor ранжирует гипотезы.

## Отгрузите это

Этот урок создает `outputs/skill-ai-sre-plan.md`. По current on-call, incident volume, team maturity проектирует AI SRE rollout.

## Упражнения

1. Запустите `code/main.py`. Что если log и metric agents не согласны? Как supervisor разрешает конфликт?
2. Определите три "safe" auto-remediation actions для вашего service. Обоснуйте каждую.
3. Напишите structured runbook template: sections, required fields, verification commands.
4. Predictive detection срабатывает при 12 min lead. Ваша policy — pager, pre-drain или оба?
5. Аргументируйте, стоит ли команде из 3 человек внедрять AI SRE в 2026 или ждать. Учитывайте maturity, volume, risk.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| AI SRE | "agent for on-call" | Incident investigation + coordination на базе LLM |
| Supervisor agent | "the orchestrator" | Top-level agent, разбивающий incidents на sub-queries |
| Specialized agent | "domain agent" | Sub-agent с tool access (logs, metrics, runbooks) |
| Auto-remediation | "AI fixes it" | Узкое pre-approved действие; НЕ широкая re-architecture |
| Operational memory | "vector runbooks" | Post-mortems + runbooks в vector DB для RAG |
| Adversarial eval | "two-model check" | Независимый анализ; agreement = confidence |
| NeuBird Hawkeye | "the adversarial one" | Product с adversarial-eval + memory pattern |
| Bits AI | "Datadog's SRE agent" | AI SRE под управлением Datadog |
| Pre-incident prediction | "early detection" | 10-15 min lead time в outage prediction |

## Дополнительное чтение

- [incident.io — AI SRE Complete Guide 2026](https://incident.io/blog/what-is-ai-sre-complete-guide-2026)
- [InfoQ — Human-Centred AI for SRE](https://www.infoq.com/news/2026/01/opsworker-ai-sre/)
- [DZone — AI in SRE 2026](https://dzone.com/articles/ai-in-sre-whats-actually-coming-in-2026)
- [Datadog Bits AI](https://www.datadoghq.com/product/bits-ai/)
- [NeuBird Hawkeye](https://www.neubird.ai/)
- [awesome-ai-sre](https://github.com/agamm/awesome-ai-sre)
