# Chaos Engineering for LLM Production

> Chaos engineering для LLM в 2026 — отдельная дисциплина. Предпосылки перед запуском экспериментов в production: defined SLI/SLO, trace+metric+log observability, automated rollback, runbooks, on-call. Архитектура имеет четыре planes: control (experiment scheduler), target (services, infra, data stores), safety (guards + abort + traffic filters), observability (metrics + traces + logs), feedback (into SLO adjustments). Guardrails обязательны: burn-rate alerts ставят эксперименты на pause, если daily error-budget burn > 2x expected; suppression windows + trace-ID correlation dedupe alert noise. Cadence: weekly small canary + SLO review; monthly game day + postmortem; quarterly cross-team resilience audit + dependency mapping. LLM-specific experiments: memory overload, network failures, provider outages, malformed prompts, KV cache eviction storms. Tooling: Harness Chaos Engineering (LLM-derived recommendations, blast-radius downscaling, MCP tool integration); LitmusChaos (CNCF); Chaos Mesh (CNCF Kubernetes-native).

**Тип:** Learn
**Языки:** Python (stdlib, toy chaos experiment runner)
**Предварительные требования:** Phase 17 · 23 (SRE for AI), Phase 17 · 13 (Observability)
**Время:** ~60 minutes

## Цели обучения

- Назвать пять prerequisites для chaos engineering (SLI/SLO, observability, rollback, runbooks, on-call) и объяснить, почему пропуск любого ломает практику.
- Нарисовать четыре planes (control, target, safety, observability) и feedback loop в SLO.
- Перечислить пять LLM-specific experiments (memory overload, network fail, provider outage, malformed prompt, KV eviction storm).
- Выбрать инструмент — Harness, LitmusChaos, Chaos Mesh — под заданный stack.

## Проблема

Chaos testing в традиционных стеках уже устоялся. LLM stacks добавляют новые failure modes. 4K-token prompt с poison character останавливает tokenizer на 12 секунд. Upstream provider возвращает 429; ваш gateway делает retries; ваш service получает OOM из-за retry-amplified concurrency. KV cache eviction storm под burst load вызывает re-prefill cascades, которые насыщают compute.

Ничего из этого не проявится в unit tests. Chaos engineering — способ обнаружить это раньше пользователей.

## Концепция

### Prerequisites

Не запускайте chaos в production без:

1. **SLI/SLO** — defined service-level indicators and objectives.
2. **Observability** — traces, metrics, logs, подключенные к dashboards.
3. **Automated rollback** — Phase 17 · 20 policy-flag rollback.
4. **Runbooks** — structured, Phase 17 · 23.
5. **On-call** — кто-то, кто реагирует.

Отсутствие любого пункта превращает chaos в реальный incident.

### Four planes + feedback

**Control plane** — experiment scheduler (Litmus workflow, Chaos Mesh schedule, Harness UI).

**Target plane** — services, pods, nodes, load balancers, data stores.

**Safety plane** — kill switch, suppression windows, blast-radius limits, error-budget gates.

**Observability plane** — обычные metrics + trace-ID correlation, чтобы отличать chaos-induced failures от natural failures.

**Feedback loop** — findings возвращаются в SLO adjustment, runbook updates, code fixes.

### Guardrails обязательны

- **Burn-rate alert**: pause experiment, если daily error-budget burn превышает 2x expected.
- **Suppression windows**: заглушают non-experiment alerts в blast radius на время experiment.
- **Trace-ID correlation**: все experiment-induced errors несут tag, чтобы on-call мог dedupe.

### Пять LLM-specific experiments

1. **Memory overload** — вызовите KV cache preemption storm, отправляя long-context requests с высокой concurrency. Наблюдайте: service graceful shed или crash?

2. **Network failure** — разорвите connectivity между inference gateway и provider. Наблюдайте: fallback включается within SLA? (Phase 17 · 19)

3. **Provider outage simulation** — 100% 429 от OpenAI. Наблюдайте: routing failover to Anthropic? (Phase 17 · 16, 19)

4. **Malformed prompt** — внедрите tokenizer-stalling payload (e.g., deeply nested unicode, huge UTF-8 codepoint). Наблюдайте: один request блокирует worker?

5. **KV eviction storm** — принудительно вызовите eviction, насытив vLLM block budget. Наблюдайте: LMCache восстанавливается или service деградирует?

### Cadence

- **Weekly** — small canary experiments в staging, возможно 5% prod.
- **Monthly** — scheduled game day по конкретному scenario; cross-team attendance; postmortem.
- **Quarterly** — cross-team resilience audit; dependency map update.

### Tooling

- **Harness Chaos Engineering** — commercial; AI-derived experiment recommendations; blast-radius downscaling; MCP tool integration.
- **LitmusChaos** — CNCF graduated; Kubernetes workflow-based.
- **Chaos Mesh** — CNCF sandbox; Kubernetes-native CRD style.
- **Gremlin** — commercial; broad support.
- **AWS FIS** / **Azure Chaos Studio** — managed cloud offerings.

### Starting small

Первый experiment: pod-kill одной decode replica под steady traffic. Наблюдайте rerouting and recovery. Если это работает и выглядит безопасно, переходите к network chaos.

Первый LLM-specific experiment: inject one provider 429 for 5 minutes. Наблюдайте fallback. Большинство команд обнаруживают, что их fallback не был полностью протестирован.

### Числа, которые стоит запомнить

- Four planes: control, target, safety, observability.
- Burn-rate pause: 2x expected daily budget burn.
- Cadence: weekly canary, monthly game day, quarterly audit.
- Five LLM experiments: memory, network, provider, malformed prompt, KV storm.

## Используйте это

`code/main.py` симулирует три chaos experiments с safety plane gates. Сообщает, какие experiments сработали бы burn-rate abort.

## Отгрузите это

Этот урок создает `outputs/skill-chaos-plan.md`. По stack и maturity выбирает первые три experiments и tooling.

## Упражнения

1. Запустите `code/main.py`. Какой experiment срабатывает burn-rate gate и почему?
2. Спроектируйте первые пять chaos experiments для vLLM-based RAG service. Включите success criteria.
3. Ваш burn-rate alert поставил experiment на pause. Как определить root cause — chaos или natural?
4. Аргументируйте, должен ли chaos запускаться в production или только staging. Когда production — правильный ответ?
5. Назовите три LLM-specific failure modes, которые generic network-chaos не может воспроизвести.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| SLI / SLO | "service targets" | Indicator + objective; обязательный prerequisite |
| Blast radius | "scope" | Набор services / users, затронутых experiment |
| Burn-rate alert | "budget gate" | Срабатывает, когда error-budget burn rate > 2x expected |
| Game day | "monthly drill" | Scheduled cross-team chaos exercise |
| LitmusChaos | "CNCF workflow" | Graduated CNCF Kubernetes chaos tool |
| Chaos Mesh | "CNCF CRD" | CNCF sandbox Kubernetes-native chaos |
| Harness CE | "commercial AI-assisted" | Harness chaos с AI recommendations |
| Malformed prompt | "tokenizer bomb" | Input, который стопорит tokenization |
| KV eviction storm | "preemption cascade" | Массовый eviction, запускающий re-prefills |

## Дополнительное чтение

- [DevSecOps School — Chaos Engineering 2026 Guide](https://devsecopsschool.com/blog/chaos-engineering/)
- [Ankush Sharma — Observability for LLMs (book)](https://www.amazon.com/Observability-Large-Language-Models-Engineering-ebook/dp/B0DJSR65TR)
- [LitmusChaos (CNCF)](https://litmuschaos.io/)
- [Chaos Mesh (CNCF)](https://chaos-mesh.org/)
- [Harness Chaos Engineering](https://www.harness.io/products/chaos-engineering)
- [AWS FIS](https://aws.amazon.com/fis/)
