# A2A — The Agent-to-Agent Protocol

> Google announced A2A в апреле 2025; к апрелю 2026 spec находится на https://a2a-protocol.org/latest/specification/ и его поддерживают 150+ organizations. A2A — horizontal complement к MCP (Lesson 13): где MCP vertical (agent ↔ tools), A2A peer-to-peer (agent ↔ agent). Он определяет Agent Cards (discovery), tasks with artifacts (text, structured data, video), opaque task lifecycles и auth. Production systems все чаще сочетают MCP с A2A. Google Cloud добавил поддержку A2A в Vertex AI Agent Builder в течение 2025-2026.

**Тип:** Learn + Build
**Языки:** Python (stdlib, `http.server`, `json`)
**Предварительные требования:** Phase 16 · 04 (Primitive Model)
**Время:** ~75 minutes

## Проблема

Вашему agent нужно вызвать другого agent в другой системе. Как? Можно открыть HTTP endpoint, определить bespoke JSON schema и надеяться, что другая сторона говорит на нем. Каждая пара agents становится custom integration.

A2A — universal wire protocol для такого вызова. Standard discovery, standard task model, standard transport, standard artifacts. Как HTTP+REST, но для agents как first-class citizens.

## Концепция

### The four elements

**Agent Card.** JSON document по адресу `/.well-known/agent.json`, описывающий agent: name, skills, endpoints, supported modalities, auth requirements. Discovery происходит чтением card.

```
GET https://agent.example.com/.well-known/agent.json
→ {
    "name": "code-review-agent",
    "skills": ["review-python", "review-typescript"],
    "endpoints": {
      "tasks": "https://agent.example.com/tasks"
    },
    "auth": {"type": "bearer"},
    "modalities": ["text", "structured"]
  }
```

**Task.** Единица работы. Async, stateful object с lifecycle: `submitted → working → completed / failed / canceled`. Client отправляет task, затем polls или subscribes for updates.

**Artifact.** Result type, который производит task. Text, structured JSON, image, video, audio. Artifacts типизированы, чтобы разные modalities были first-class.

**Opaque lifecycle.** A2A не предписывает, *как* remote agent решает task. Client видит state transitions и artifacts; implementation свободна использовать любой framework.

### The MCP/A2A split

- **MCP** (Lesson 13): agent ↔ tool. Agent читает/пишет через JSON-RPC к tool server. Stateless по умолчанию.
- **A2A**: agent ↔ agent. Peer protocol; обе стороны — agents со своим reasoning.

Production multi-agent systems используют оба. A2A peer вызывает MCP tools на своей стороне. Такое разделение сохраняет две concerns чистыми.

### Discovery flow

```
Client                     Agent server
  ├──GET /.well-known/agent.json──>
  <──Agent Card JSON─────────────
  ├──POST /tasks {skill, input}──>
  <──201 task_id, state=submitted
  ├──GET /tasks/{id}──────────────>
  <──state=working, 42% done──────
  ├──GET /tasks/{id}──────────────>
  <──state=completed, artifacts──
```

Или со streaming: SSE subscription на `/tasks/{id}/events` для push updates.

### Auth

A2A поддерживает три common patterns:

- **Bearer token** — OAuth2 или opaque.
- **mTLS** — mutual TLS; organizations доказывают identity друг другу.
- **Signed requests** — HMAC по payload.

Auth объявляется в Agent Card; clients discover and comply.

### 150+ organizations by April 2026

Enterprise adoption привело A2A к масштабу. Главная идея: A2A стал способом, которым enterprise agent systems пересекают trust boundaries. Google Cloud shipped Vertex AI Agent Builder A2A support; Microsoft Agent Framework supports it; большинство major frameworks (LangGraph, CrewAI, AutoGen) ship A2A adapters.

### Where A2A wins

- **Cross-organization calls.** Agent в company A вызывает agent в company B. Без A2A каждая пара — bespoke contract.
- **Heterogeneous frameworks.** LangGraph agent вызывает CrewAI agent, тот вызывает custom Python agent. A2A normalizes.
- **Typed artifacts.** Video result, structured JSON, audio — все first-class.
- **Long-running tasks.** Opaque lifecycle + polling делает tasks длиной в часы straightforward.

### Where A2A struggles

- **Latency-sensitive micro-calls.** Lifecycle A2A async. Sub-millisecond agent-to-agent не подходит; используйте direct RPC.
- **Tight-coupled in-process agents.** Если оба agents работают в одном Python process, HTTP round-trip A2A — overkill.
- **Small teams.** Spec overhead реален; internal-only agents могут не нуждаться в такой формальности.

### A2A vs ACP, ANP, NLIP

В 2024-2026 появилось несколько related specs:

- **ACP** (IBM/Linux Foundation) — predecessor to A2A, narrower scope.
- **ANP** (Agent Network Protocol) — peer-discovery-heavy, decentralized-first.
- **NLIP** (Ecma Natural Language Interaction Protocol, standardized December 2025) — natural-language content type.

A2A — самый adopted peer protocol по состоянию на апрель 2026. См. arXiv:2505.02279 (Liu et al., "A Survey of Agent Interoperability Protocols") для сравнения.

## Соберите

`code/main.py` реализует A2A-minimal server и client с использованием `http.server` и JSON. Server:

- exposes `/.well-known/agent.json`,
- accepts `POST /tasks`,
- manages task state,
- returns artifacts on `GET /tasks/{id}`.

Client:

- fetches the Agent Card,
- submits a task,
- polls until completion,
- reads the artifact.

Запуск:

```
python3 code/main.py
```

Script запускает server в background thread, затем запускает client against it. Вы видите полный flow: discovery, submit, poll, artifact.

## Используйте

`outputs/skill-a2a-integrator.md` проектирует A2A integration: contents Agent Card, task schemas, auth choice, streaming vs polling.

## Запустите в production

Чеклист:

- **Pin the spec version.** A2A все еще evolves; Agent Card should declare protocol version.
- **Idempotent task creation.** Duplicate submissions (network retries) should produce one task.
- **Artifact schemas.** Declare what shapes the agent returns; consumers should validate.
- **Rate limits + auth.** A2A public-facing; применяйте standard web security.
- **Dead-letter for failed tasks.** Inspect patterns over time for recurring failure types.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что client discovers server и receives the correct artifact.
2. Добавьте second skill к server (например, "summarize"). Обновите Agent Card. Напишите client, который выбирает skill based on task type.
3. Реализуйте SSE streaming endpoint: `/tasks/{id}/events`, который emits state changes. Что client должен делать иначе?
4. Прочитайте A2A spec (https://a2a-protocol.org/latest/specification/). Найдите три вещи, которые spec mandates, но эта demo не реализует.
5. Сравните A2A (Agent Card discovery) с MCP (server-side capability listing через `listTools`). Какой tradeoff между self-describing agents и capability-probing?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| A2A | "Agent-to-agent" | Peer protocol для agents, чтобы вызывать других agents across systems. Google 2025. |
| Agent Card | "The agent's business card" | JSON по адресу `/.well-known/agent.json`, описывающий skills, endpoints, auth. |
| Task | "The unit of work" | Async stateful object с lifecycle; artifacts создаются on completion. |
| Artifact | "The result" | Typed output: text, structured JSON, image, video, audio. First-class media. |
| Opaque lifecycle | "How it's solved is the agent's business" | Client видит state transitions; server свободен выбирать framework/tools. |
| Discovery | "Finding the agent" | `GET /.well-known/agent.json` возвращает card. |
| MCP vs A2A | "Tools vs peers" | MCP: vertical agent ↔ tool. A2A: horizontal agent ↔ agent. |
| ACP / ANP / NLIP | "Sibling protocols" | Adjacent specs; A2A is the most-adopted 2026. |

## Дополнительное чтение

- [A2A specification](https://a2a-protocol.org/latest/specification/) — canonical spec
- [Google Developers Blog — A2A announcement](https://developers.googleblog.com/en/a2a-a-new-era-of-agent-interoperability/) — April 2025 launch post
- [A2A GitHub repo](https://github.com/a2aproject/A2A) — reference implementations and SDKs
- [Liu et al. — A Survey of Agent Interoperability Protocols](https://arxiv.org/html/2505.02279v1) — MCP, ACP, A2A, ANP comparison
