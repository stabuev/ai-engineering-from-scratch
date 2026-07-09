# LangGraph: графы с состоянием и надежное выполнение

> LangGraph — эталон 2026 года для low-level stateful orchestration. Agent — это state machine; nodes — functions; edges — transitions; state immutable, а checkpoint создается после каждого step. Resume после любого failure происходит ровно с того места, где остановились.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 12 (Workflow Patterns)
**Время:** ~75 минут

## Цели обучения

- Описать core model LangGraph: state machine с immutable state, function nodes, conditional edges и checkpoints после каждого шага.
- Назвать четыре возможности, которые выделяет документация: durable execution, streaming, human-in-the-loop, comprehensive memory.
- Объяснить три orchestration topologies, которые поддерживает LangGraph: supervisor, peer-to-peer (swarm), hierarchical (nested subgraphs).
- Реализовать stdlib state graph с immutable state, conditional edges и циклом checkpoint/resume.

## Проблема

У agents и workflows общая проблема: когда 40-step run падает на step 38, вы хотите resume from step 38, а не начинать заново. Второсортные state models заставляют operators городить retries вокруг библиотеки, которая предполагает fresh runs.

Ответ дизайна LangGraph: state — first-class typed object, mutations explicit, а checkpoints сохраняются после каждого node. Resume — это вызов `load_state(session_id)`.

## Концепция

### Graph

Graph определяется через:

- **State type.** Typed dict (или Pydantic model), который каждый node читает и мутирует.
- **Nodes.** Чистые функции `(state) -> state_update`. Обновления сливаются в state после возврата.
- **Edges.** Conditional или direct transitions между nodes.
- **Entry and exit.** Sentinel nodes `START` и `END` отмечают границу.

Пример: agent с nodes `classify`, `refund`, `bug`, `sales`, `done` — routing workflow как graph.

### Durable execution

После возврата каждого node runtime сериализует state и записывает его в checkpointer (SQLite, Postgres, Redis, custom). При failure на step N runtime может сделать `resume(session_id)` и продолжить с step N+1 с точным state.

Документация LangGraph явно выделяет production users, для которых это важно: Klarna, Uber, J.P. Morgan. Утверждение не в форме graph само по себе; оно в том, что graph shape плюс checkpointing делают recovery дешевым.

### Streaming

Каждый node может yield partial output. Graph передает вызывающему коду stream событий per-node-delta, чтобы UIs обновлялись во время выполнения graph.

### Human-in-the-loop

Inspect and modify state between nodes. Реализации: pause before critical node, показать state человеку, принять modifications, resume. Checkpointer делает это простым, потому что state уже сериализован.

### Memory

Short-term (within a run — conversation history in state) и long-term (across runs — persistent через checkpointer plus separate long-term store). LangGraph интегрируется с external memory systems (Mem0, custom) через tools.

### Три топологии

1. **Supervisor.** Central router LLM dispatches to specialist subagents. `create_supervisor()` в `langgraph-supervisor` (хотя команда LangChain в 2026 рекомендует делать это через tool calls напрямую для большего context control).
2. **Swarm / peer-to-peer.** Agents hand off directly через shared tool surface. Нет central router.
3. **Hierarchical.** Supervisors managing sub-supervisors, реализованные как nested subgraphs.

### Где этот паттерн ломается

- **Слишком маленькие checkpoints.** Если checkpoint сохраняет только conversation turns, tool state и memory writes остаются unrecoverable. Full state must serialize.
- **Non-deterministic nodes.** Resume предполагает, что node inputs производят тот же state update. Random seeds, wall-clock, external APIs должны быть captured.
- **Over-use of conditional edges.** Graph, где каждое edge conditional, — state machine, о которой невозможно reason. Предпочитайте linear chains с occasional branches.

## Соберите это

`code/main.py` реализует stdlib stateful graph:

- `State` — typed dict с `messages`, `step`, `route`, `output`, `human_approval`.
- `Node` — callable, который принимает state и возвращает update dict.
- `StateGraph` — nodes + edges + conditional edges + run + resume.
- `SQLiteCheckpointer` (in-memory fake) — сериализует state после каждого node; `load(session_id)` восстанавливает.
- Demo graph: classify -> branch(refund / bug / sales) -> human gate -> send.

Запустите:

```
python3 code/main.py
```

Трасса показывает первый run, падающий на human gate, persistence, затем resume, который производит final output.

## Используйте это

- **LangGraph** — эталон, production-ready. Используйте `create_react_agent`, `create_supervisor` или строите свой graph.
- **AutoGen v0.4** (Lesson 14) — actor model alternative для high-concurrency scenarios.
- **Claude Agent SDK** (Lesson 17) — managed harness со встроенным session store.
- **Custom** — когда нужен точный контроль над state shape или checkpointer backend.

## Доведите до продакшена

`outputs/skill-state-graph.md` генерирует LangGraph-образный state graph в любом target runtime с подключенными checkpointing и resume.

## Упражнения

1. Добавьте conditional edge из `classify` в `end`, когда classification confidence ниже threshold. Resume run после того, как человек вручную установит `route`.
2. Замените SQLite-like fake на реальный SQLite checkpointer. Измерьте per-step serialization overhead.
3. Реализуйте parallel edges: два nodes run concurrently, merge через custom reducer. Что здесь дает immutable state?
4. Прочитайте reference `langgraph-supervisor`. Перенесите игрушку на `create_supervisor`. Сравните формы трасс.
5. Добавьте streaming: каждый node yields partial state while it runs. Печатайте deltas по мере arrival.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| State graph | "Agent как state machine" | Typed state + nodes + edges + reducers |
| Checkpointer | "Persistence backend" | Сериализует state после каждого node; enables resume |
| Reducer | "Слияние state" | Function, combining current state with a node's update |
| Conditional edge | "Branch" | Edge, выбранное function of state |
| Subgraph | "Nested graph" | Graph, используемый как node внутри другого graph |
| Durable execution | "Возобновление после failure" | Restart at last successful node with exact state |
| Supervisor | "Router LLM" | Central dispatcher для specialist subagents |
| Swarm | "P2P agents" | Agents hand off через shared tools; no central router |

## Дополнительное чтение

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — reference docs
- [langgraph-supervisor reference](https://reference.langchain.com/python/langgraph/supervisor/) — API паттерна supervisor
- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model alternative
- [Claude Agent SDK overview](https://platform.claude.com/docs/en/agent-sdk/overview) — session store и subagents
