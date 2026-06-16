# Failure Modes — MAST, groupthink, monoculture, cascading errors

> Эталонная таксономия на 2026 год — **MAST** (Cemri et al., NeurIPS 2025, arXiv:2503.13657), выведенная из 1642 execution traces по 7 state-of-the-art open-source MAS и показывающая **41–86.7% failure rate**. Три корневые категории: **Specification Problems** (41.77%) — role ambiguity, неясные task definitions; **Coordination Failures** (36.94%) — communication breakdowns, state desync; **Verification Gaps** (21.30%) — missing validation, absent quality checks. Семейство **Groupthink** (arXiv:2508.05687) добавляет: monoculture collapse (одна base model → correlated failures), conformity bias (агенты усиливают ошибки друг друга), deficient theory of mind, mixed-motive dynamics, cascading reliability failures. Пример cascade: retry storms, где payment failure вызывает order retries, которые вызывают inventory retries, которые перегружают inventory service (10x load за секунды — нужны circuit breakers). Memory poisoning: галлюцинация одного агента попадает в shared memory, downstream agents воспринимают ее как факт; accuracy постепенно падает, что делает root-cause diagnosis болезненной. **STRATUS** (NeurIPS 2025) сообщает о 1.5x improvement mitigation success через specialized detection / diagnosis / validation agents. Этот урок рассматривает failure modes как первоклассные engineering targets.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 16 · 13 (Shared Memory), Phase 16 · 14 (Consensus and BFT), Phase 16 · 15 (Voting and Debate Topology)
**Время:** ~75 минут

## Цели обучения

- Использовать таксономию MAST для классификации мультиагентных отказов.
- Распознавать groupthink, каскадные retry-штормы и отравление памяти.
- Встраивать агентов обнаружения отказов (STRATUS) в цикл.

## Проблема

Multi-agent systems fail 41-86.7% of the time на реальных задачах (Cemri et al. 2025 измерили это на 7 open-source MAS). Это нельзя отладить подходом "just add more agents." У failures есть структурные причины. Таксономия MAST дает категории. Этот урок сопоставляет каждую категорию с конкретным detection, diagnosis и mitigation pattern, чтобы числа перестали выглядеть произвольными.

Production practice 2026 года — рассматривать failure modes как design inputs. Ваша architecture не "good enough", пока вы не можете указать на каждую категорию MAST и назвать mitigation, который развернули.

## Концепция

### Категории MAST

**Specification Problems (41.77% of failures).** Задача агента была определена недостаточно строго. Примеры:

- Role ambiguity: два агента оба считают, что они reviewer.
- Task underspecified: "summarize this", когда пользователь хотел конкретный angle.
- Success criteria implicit: агент не может понять, succeeded ли он.

Mitigations:
- Пишите explicit role contracts. Prompt каждого агента говорит, что он делает *и чего не делает*.
- Acceptance tests per task. До старта агента определите "done looks like X."
- Pre-flight spec check: отдельный агент проверяет task definition перед dispatch.

**Coordination Failures (36.94%).** Communication или state breakdowns.

Примеры:
- Два агента обновляют shared state без synchronization.
- Message lost between agents (queue failure, timeout).
- State drift: agent A считает задачу завершенной; agent B все еще executing.

Mitigations:
- Versioned shared state with optimistic concurrency.
- Explicit acknowledgment для critical messages (retry until acked).
- Periodic state-sync checkpoints; detect drift early.

**Verification Gaps (21.30%).** Нет независимой проверки outputs.

Примеры:
- Один агент заявляет success; никто не verifies.
- Chain of agents каждый доверяет output предыдущего.
- Test coverage missing on the emergent composed behavior.

Mitigations:
- Independent verifier agent (Lesson 13). Только чтение, независимый доступ к source.
- Explicit handoff contract: "A's output must pass checker C before B starts."
- Outcome logging for post-hoc analysis.

### Groupthink family (arXiv:2508.05687)

Пять связанных failures, когда агенты homogenize или mimic each other:

**Monoculture collapse.** Одна base model или training data → correlated errors. Когда три агента используют одну LLM, они разделяют ее hallucinations.

**Conformity bias.** Агенты смещаются к самому громкому или most-confident peer, даже когда он ошибается.

**Deficient ToM.** Агенты не моделируют beliefs друг друга; coordination разваливается (Lesson 18).

**Mixed-motive dynamics.** Агенты с partially-aligned incentives дрейфуют к compromise-middle, который не удовлетворяет никого.

**Cascading reliability failures.** Error pattern одного компонента запускает error patterns в зависимых компонентах.

### Cascading example — retry storm

Классический incident pattern 2026 года:

```
payment service fails 10% of requests
   ↓
order agent retries payment (exponential backoff but naive)
   ↓
each retry is a new order-inventory check
   ↓
inventory service sees 2x normal load
   ↓
inventory service starts timing out
   ↓
every order retries inventory check
   ↓
inventory service sees 10x normal load
   ↓
cluster goes down
```

Исправление классическое: **circuit breakers**. Когда downstream error rate превышает threshold, short-circuit с cached или default results. Плюс capped retry budgets per request.

Circuit breakers — один из немногих multi-agent failure mitigations, которые вы напрямую заимствуете из distributed systems без модификаций.

### Memory poisoning (revisited)

Из Lesson 13: hallucination одного агента становится shared-memory fact; downstream agents рассуждают на основе poisoned fact. В терминах MAST это verification gap на shared-memory layer.

Gradual accuracy decay — симптом. Вы не получаете crash; вы получаете slow drift, который трудно root-cause.

Mitigation: append-only log, provenance, unwritable verifier. Уже разобрано в Lesson 13.

### STRATUS — специализированные агенты для failure detection

STRATUS (NeurIPS 2025) сообщает о 1.5x mitigation-success improvement, когда вы разворачиваете:

- **Detection agent.** Наблюдает symptom patterns (high disagreement, retry spikes, accuracy drift).
- **Diagnosis agent.** По symptoms выводит likely root cause из таксономии MAST.
- **Validation agent.** После применения mitigation проверяет, что symptoms clear.

Это SRE-style incident response, примененный к agent systems. Все три роли могут быть LLM agents со специализированными prompts.

### Failure-mode audit

Best practice 2026 года — annual (или per-major-release) failure-mode audit:

1. **Trace sample.** Соберите ~1000 real execution traces.
2. **Categorize.** Для failures в каждом trace сопоставьте MAST + Groupthink categories.
3. **Compute failure-by-category rate.** Какие categories доминируют в вашей системе?
4. **Rank mitigations.** Какой fix устранит больше всего failures?
5. **Pick 2-3 mitigations.** Реализуйте; re-audit next quarter.

Дисциплина важнее конкретных choices. Без audits failures сливаются в noise и никогда не решаются системно.

### Когда systems fail silently

Самая опасная category failures — silent correctness failure. Систему, которая fails loudly (crash, exception, alert), можно мониторить. Систему, которая выдает plausible-but-wrong outputs, нельзя обнаружить по exception logs. Поэтому verification gaps — самая дорогая category per-failure, хотя по count это только 21.30%.

Инвестируйте в:
- Sample-based human review.
- Golden-dataset regression tests.
- Cross-agent cross-checking on important outputs.

### Failure vs slow failure

Некоторые failures немедленные; некоторые медленные. Immediate failures (timeout, schema mismatch, auth error) дешевы в detection. Slow failures (memory poisoning, monoculture drift, role ambiguity) дороги в detection и prevention.

Engineering move 2026 года: instrument slow-failure proxies, чтобы ловить drift до того, как он станет visible error. Agreement rate, retry rate, output-length distribution и edit-distance between consecutive agent versions — полезные proxies.

## Соберите

`code/main.py` реализует:

- `FailureTaxonomy` — categorizes simulated incidents into MAST + Groupthink categories.
- `CircuitBreaker` — classic pattern; opens when error rate exceeds threshold.
- `RetryStormSimulator` — показывает cascading failure; toggles circuit breaker on / off.
- `DetectionAgent` — scripted STRATUS-style symptom matcher.

Запуск:

```
python3 code/main.py
```

Ожидаемый output:
- retry storm без circuit breaker: inventory errors blow up (simulated).
- с circuit breaker: cap at threshold; degraded-mode responses served.
- detection agent flags the pattern and names the MAST category.

## Используйте

`outputs/skill-mast-auditor.md` запускает MAST-style failure-mode audit на multi-agent system. Traces → categorization → mitigation ranking.

## Запустите в production

Failure-mode discipline in production:

- **MAST audit per quarter.** Не annual. Categories shift as your system grows.
- **Circuit breakers everywhere.** Каждый outbound call к любому dependent service. Default open threshold на 5-10% error rate.
- **Golden datasets.** Маленькие, high-quality, hand-audited. Regression-test against them weekly.
- **STRATUS trio.** Detection + Diagnosis + Validation agents monitoring production. Начните только с detection agent; добавьте diagnosis, когда symptoms noisy.
- **Failure budget.** Explicit SLO для failure rate by category. Exceeding budget triggers a stop-shipping conversation.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что circuit breaker caps the retry storm. Меняйте failure threshold и наблюдайте tradeoff.
2. Реализуйте **slow-failure proxy**: agreement rate across 3 parallel agents. Когда он резко падает, trigger an alert. Симулируйте monoculture drift, постепенно коррелируя agent outputs.
3. Прочитайте Cemri et al. (arXiv:2503.13657). Выберите одну из их 7 MAS systems и сопоставьте ее top 3 failure categories. Как они сравниваются с тем, что предсказывает MAST?
4. Прочитайте Groupthink paper (arXiv:2508.05687). Определите, какой из пяти patterns труднее всего detect in production. Предложите proxy metric.
5. Спроектируйте STRATUS-style detection-diagnosis-validation trio для конкретной multi-agent system, которую знаете. За какими symptoms следит detection? Какие mitigations рекомендует diagnosis? Как validation подтверждает, что они работают?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| MAST | "The 2026 taxonomy" | Cemri 2025; 3 root categories + 14 sub-types of failures. |
| Specification Problem | "Role ambiguity" | Task или role under-defined; agents do not know what to do. |
| Coordination Failure | "State drift" | Communication или sync breakdown между agents. |
| Verification Gap | "No one checked" | Outputs accepted without independent validation. |
| Groupthink family | "Homogeneity failures" | Monoculture, conformity, deficient ToM, mixed-motive, cascading. |
| Monoculture collapse | "Same model, same hallucinations" | Correlated errors from shared base model or training data. |
| Retry storm | "Cascading error amplification" | Один failure triggers retries, которые amplify load downstream. |
| Circuit breaker | "Fail fast on error rate" | Open when error rate exceeds threshold; short-circuit with default. |
| STRATUS | "Incident response trio" | Detection + diagnosis + validation agents. 1.5x mitigation success. |
| Memory poisoning | "Hallucinations propagate" | Shared-memory fact tainted; downstream agents reason on poison. |

## Дополнительное чтение

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy, NeurIPS 2025
- [Groupthink failures in multi-agent LLMs](https://arxiv.org/abs/2508.05687) — monoculture, conformity, and the five-family taxonomy
- [STRATUS — specialized agents for MAS incident response](https://neurips.cc/) — NeurIPS 2025 proceedings entry (detection + diagnosis + validation)
- [Release It! — stability patterns (Nygard)](https://pragprog.com/titles/mnee2/release-it-second-edition/) — the canonical circuit-breaker reference
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production failure-mode notes
