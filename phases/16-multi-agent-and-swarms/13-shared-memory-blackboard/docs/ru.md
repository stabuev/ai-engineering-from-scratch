# Shared Memory and Blackboard Patterns

> В multi-agent systems 2026 сосуществуют два подхода: **message pool** (everyone sees everyone's messages, как в AutoGen GroupChat или MetaGPT) и **blackboard with subscription** (agents subscribe to relevant events, как в Context-Aware MCP или Matrix framework). Оба являются единственной stateful-частью multi-agent system — значит, именно там живут самые интересные bugs. Эталонный failure mode — **memory poisoning**: один agent hallucinate "fact," другие agents treat it as verified, и accuracy постепенно деградирует так, что это намного сложнее debug, чем immediate crash. Этот lesson строит обе структуры на stdlib, injects poisoning attack и показывает три mitigations, которые реально работают в production.

**Тип:** Learn + Build
**Языки:** Python (stdlib, `threading`)
**Предварительные требования:** Phase 16 · 04 (Primitive Model), Phase 16 · 09 (Parallel Swarm Networks)
**Время:** ~75 minutes

## Цели обучения

- Противопоставлять топологии памяти message-pool и blackboard.
- Определять, когда каждая выигрывает и как происходит отравление памяти.
- Прослеживать прецедент blackboard (Hayes-Roth, 1985).

## Проблема

Multi-agent systems нужно место, где agents могут делиться facts. Буквальный вариант — "pass everything in messages" — но это заново изобретает shared state с лишним copying. Другой вариант — "give everyone a global log" — но global logs растут без ограничений и легко poison. Третий — "project a view per agent" — scalable, но schema-heavy.

Когда один из agents hallucinate и записывает hallucination в shared state, каждый downstream agent, который читает этот state, принимает hallucination как fact. К моменту, когда human это замечает, reasoning chain уже в пяти шагах, а root cause — третье сообщение, когда-либо записанное. Debugging multi-agent accuracy decay сложнее, чем debugging crash.

Это memory poisoning. Это вторая по документированности failure family в MAST taxonomy (Cemri et al., arXiv:2503.13657), и она structural: любой shared-memory design без provenance и unwritable verifier рано или поздно проявит ее.

## Концепция

### The two main topologies

**Full message pool.** Каждый agent читает каждое message. AutoGen GroupChat и MetaGPT используют это. Просто, прозрачно, inspectable, но не масштабируется дальше ~10 agents, потому что context каждого agent заполняется работой других agents.

```
agent-A ──write──▶ ┌────────────────┐ ◀──read── agent-D
                   │ message pool   │
agent-B ──write──▶ │                │ ◀──read── agent-E
                   │ (global log)   │
agent-C ──write──▶ └────────────────┘ ◀──read── agent-F
```

**Blackboard with subscription.** Agents declare interest in topics; substrate routes only relevant messages. CA-MCP (arXiv:2601.11595) и Matrix decentralized framework (arXiv:2511.21686) используют это. Масштабируется дальше, но требует upfront schema design, чтобы subscriptions были meaningful.

```
                   ┌─ topic: prices ──┐
agent-A ──pub────▶ │                  │ ──▶ agent-D (subscribed)
                   ├─ topic: orders ──┤
agent-B ──pub────▶ │                  │ ──▶ agent-E (subscribed)
                   ├─ topic: alerts ──┤
agent-C ──pub────▶ │                  │ ──▶ agent-F (subscribed)
                   └──────────────────┘
```

### When each wins

- **Full pool** выигрывает, когда agents мало (< 10), они heterogeneous, а conversation short-horizon. Reasoning о том, кто что сказал, тривиален, когда everyone sees everything.
- **Blackboard** выигрывает, когда agents много, они homogeneous по role, но многочисленны по instance (swarms), а conversation long-running. Routing экономит token cost и уменьшает context pollution.

Production systems часто смешивают: небольшой full pool наверху (planning layer), blackboards ниже (worker layer).

### Memory poisoning, in one scenario

Три agents работают над research task. Agent A — retrieval agent. Agent B — summarizer. Agent C — analyst.

1. A fetches a page and writes a message to shared state: "The study reports a 42% accuracy improvement."
2. На полученной странице на самом деле было "4.2% improvement." A галлюцинировал десятичный разряд.
3. B, reading shared state, writes: "Large 42% accuracy gain reported (source: A)."
4. C, reading shared state, writes: "Recommend adoption — 42% lift is transformative."
5. Финальный отчет цитирует число 42%, которого никогда не существовало.

Ни один agent не crashed. Ни один test не failed. System "worked." Hallucination перешла из context одного agent в reasoning каждого downstream agent через shared state.

### Why this is structural

Без shared state hallucination agent A остается в context A. Downstream agents могли бы re-fetch или re-derive и, возможно, поймали бы error. С naive shared state context A становится context каждого, и hallucination отмывается в fact.

Проблема не в shared state per se — проблема в shared state **without provenance and without an independent verifier**. Три mitigations решают это:

1. **Attribute provenance on every write.** Каждая entry в shared state записывает, кто ее wrote, when, under what prompt, и (если применимо) какой source agent cited. Downstream agents читают со skepticism, keyed to provenance.
2. **Version writes; treat them as append-only.** Correction — это new entry, которая supersedes old, а не in-place update. Audit trail сохраняется.
3. **Keep at least one agent that cannot write to shared state.** Read-only verifier agent samples entries, re-fetches sources и flags inconsistencies. Поскольку он не может писать в pool, он не может быть poisoned by the pool.

### Blackboard precedent (Hayes-Roth, 1985)

Blackboard pattern старше LLM agents на четыре десятилетия. Hayes-Roth (1985, "A Blackboard Architecture for Control") описал specialist Knowledge Sources, которые observe global blackboard, contribute partial solutions и trigger other sources. Blackboard 2026 (CA-MCP, Matrix) — тот же pattern с LLM agents как Knowledge Sources и JSON blobs как partial solutions. Старая литература уже задокументировала solutions to write contention, opportunistic control и consistency, которые modern systems rediscover.

### Projection vs full view

Pure blackboard дает каждому subscriber одну и ту же projection (topic-scoped). Более aggressive design — **per-agent projection**: каждый agent получает view, customized to its role. LangGraph state reducers — canonical 2026 implementation: reducer function folds global state into a role-specific slice.

Per-agent projection масштабируется дальше, но требует schema. Без нее вы rebuild ad-hoc projection в prompt каждого agent.

### Write-contention patterns

Multiple agents writing simultaneously — это concurrency problem, а не только LLM problem. Работают три patterns:

- **Sequential writer (single producer).** Все writes проходят через одного coordinator agent, который serializes. Просто, но bottleneck.
- **Optimistic concurrency with versioning.** У каждой entry есть version; writers fail on version mismatch and retry. Classic database technique.
- **Topic partitioning.** Разные agents own different topics. Нет cross-topic contention. Требует designed partition boundaries.

Большинство frameworks 2026 по умолчанию используют sequential writer, потому что LLM calls достаточно медленные, contention редок, и bottleneck не hurts.

### The unwritable verifier

Самая load-bearing mitigation — read-only verifier. Implementation rules:

- Verifier shares state with the team (reads the blackboard or pool).
- Verifier has no write handle to shared state — only to a separate verification channel.
- Verifier independently fetches sources cited in writes. Flags disagreement.
- Verifier's own outputs are routed to a human or a separate decision agent, never fed back into the pool.

Без этого separation outputs verifier становятся new entries in the pool, то есть poisoned pool poisons verifier, а тот poisons its verifications.

## Соберите

`code/main.py` реализует обе topologies на stdlib Python плюс toy poisoning attack и три mitigations.

- `MessagePool` — thread-safe append-only log with full read-out.
- `Blackboard` — topic-keyed pub/sub with per-agent subscriptions.
- `ProvenanceEntry` — every write records (writer, timestamp, prompt_hash, source_uri).
- `PoisoningScenario` — runs a three-agent research task where agent A hallucinates a decimal. Prints final report.
- `Verifier` — a read-only agent that re-fetches sources and flags inconsistencies. Runs the same scenario with the verifier present.

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод:
- Run 1 (no verifier): hallucinated 42% propagates to the final report.
- Run 2 (with verifier): verifier flags the inconsistency, pool labeled "flagged", final report includes a retraction.

## Используйте

`outputs/skill-memory-auditor.md` — skill, который audits shared-memory design любой multi-agent system на provenance, versioning и verifier separation. Запускайте его на новых multi-agent architectures перед production.

## Запустите в production

For any shared-memory design:

- Record provenance on every write: `(writer, timestamp, prompt_hash, tool_calls_cited, source_uri)`.
- Make the log append-only. Corrections are new entries that reference the superseded one.
- Deploy at least one read-only verifier agent with independent source access.
- Route verifier output to a separate channel, not back into the shared pool.
- Log the ratio of writes that are supersessions — rising ratio is early evidence of hallucination patterns.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что run 1 propagates hallucination, а run 2 catches it.
2. Добавьте second hallucination: agent B invents a dataset size. Verifier should catch both without being hand-tuned for either.
3. Переключите full pool на blackboard with topic partitions (`prices`, `summaries`, `analyses`). Какие poisoning scenarios topic partitioning makes harder to pull off, а с какими не помогает?
4. Прочитайте Hayes-Roth (1985, "A Blackboard Architecture for Control"). Найдите два control patterns из paper, не обсужденные в этом lesson, которые были бы полезны systems 2026.
5. Прочитайте CA-MCP (arXiv:2601.11595). Сопоставьте его Shared Context Store либо с MessagePool, либо с Blackboard class в `code/main.py`. Какие primitives CA-MCP добавляет поверх?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Message pool | "Shared chat history" | Append-only log, который читает каждый agent. Full transparency, poor scaling. |
| Blackboard | "Shared workspace" | Topic-keyed pub/sub. Агенты подписываются на релевантные topics. Масштабируется дальше. |
| Provenance | "Who wrote what" | Metadata on each write: writer, timestamp, prompt, sources. |
| Memory poisoning | "Hallucinations spreading" | Error одного agent попадает в shared state, downstream agents adopt it as fact. |
| Append-only | "No in-place updates" | Corrections are new entries that supersede. Preserves audit trail. |
| Unwritable verifier | "Independent auditor" | Read-only agent, который re-fetches sources and flags inconsistencies. |
| Projection | "Scoped view" | Per-agent view computed from global state. LangGraph reducers are the canonical case. |
| Knowledge Source | "Specialist agent" | Термин Hayes-Roth 1985 для участника blackboard. |

## Дополнительное чтение

- [Cemri et al. — Why Do Multi-Agent LLM Systems Fail?](https://arxiv.org/abs/2503.13657) — MAST taxonomy; memory poisoning is a coordination-failure sub-family
- [CA-MCP — Context-Aware Multi-Server MCP](https://arxiv.org/abs/2601.11595) — Shared Context Store for coordinated MCP servers
- [Matrix — decentralized multi-agent framework](https://arxiv.org/abs/2511.21686) — message-queue-based blackboard without a central orchestrator
- [LangGraph state and reducers](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — per-agent projection pattern in production
- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — provenance and verification notes from a production deployment
