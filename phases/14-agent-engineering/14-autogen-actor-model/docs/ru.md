# AutoGen v0.4: акторная модель и агентный фреймворк

> AutoGen v0.4 (Microsoft Research, Jan 2025) перепроектировал orchestration агентов вокруг actor model. Асинхронный обмен сообщениями, event-driven agents, fault isolation, естественная конкуррентность. Framework теперь в maintenance mode, пока Microsoft Agent Framework (public preview Oct 2025) становится преемником.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 12 (Workflow Patterns)
**Время:** ~75 минут

## Цели обучения

- Описать actor model: agents как actors, messages как единственный IPC, failure isolation для каждого actor.
- Назвать три API layers AutoGen v0.4 — Core, AgentChat, Extensions — и назначение каждого.
- Объяснить, почему decoupling доставки сообщений и обработки даёт fault isolation и natural concurrency.
- Реализовать stdlib actor runtime на Python и перенести на него two-agent code-review flow.

## Проблема

Большинство agent frameworks синхронны: один agent производит, один agent потребляет, всё внутри call stack. Failures рушат stack. Concurrency прикручена сбоку. Distribution требует переписывания.

Ответ AutoGen v0.4: actor model. Каждый agent — actor с private inbox. Messages — единственный способ взаимодействия. Runtime отделяет доставку от обработки. Failures изолируются в одном actor. Concurrency встроена нативно. Distribution — просто другой transport.

## Концепция

### Actors

У actor есть:

- Private state (никогда напрямую не трогается извне).
- Inbox (message queue).
- Handler: `receive(message) -> effects`, где effects могут быть "reply", "send to other actor", "spawn new actor", "update state", "stop self".

Два actors не могут разделять память. Они могут только отправлять сообщения.

### Три API layers в AutoGen v0.4

1. **Core.** Low-level actor framework. `AgentRuntime`, `Agent`, `Message`, `Topic`. Асинхронный обмен сообщениями, event-driven.
2. **AgentChat.** Task-driven high-level API (замена ConversableAgent из v0.2). `AssistantAgent`, `UserProxyAgent`, `RoundRobinGroupChat`, `SelectorGroupChat`.
3. **Extensions.** Integrations — OpenAI, Anthropic, Azure, tools, memory.

### Почему decoupling важен

В модели v0.2 вызов `agent_a.chat(agent_b)` синхронно блокирует agent_a, пока agent_b не вернется. В v0.4 `send(agent_b, msg)` кладет message в inbox agent_b и возвращается. Runtime доставляет позже. Три следствия:

- **Fault isolation.** Crash Agent B не рушит Agent A — runtime ловит failure в handler B и решает, что делать (log, retry, dead-letter).
- **Natural concurrency.** Много messages in flight одновременно; actors обрабатывают inbox concurrently.
- **Distribution-ready.** Inbox + transport — одна и та же абстракция, независимо от того, actor in-process или на другом host.

### Topologies

- **RoundRobinGroupChat.** Agents ходят по очереди в fixed rotation.
- **SelectorGroupChat.** Selector agent выбирает, кто идёт дальше, на основе conversation context.
- **Magentic-One.** Reference multi-agent team для web browsing, code execution, file handling. Построена на AgentChat.

### Observability

OpenTelemetry support встроен. Каждый message emits span; tool calls несут attributes `gen_ai.*` согласно OTel GenAI semantic conventions 2026 года (Lesson 23).

### Статус: maintenance mode

Начало 2026 года: AutoGen v0.7.x стабилен для research и prototyping. Microsoft перенесла active development в Microsoft Agent Framework (public preview Oct 1 2025; 1.0 GA targeted end of Q1 2026). Паттерны AutoGen чисто переносятся вперёд — actor model является durable idea.

## Соберите это

`code/main.py` реализует stdlib actor runtime:

- `Message` — typed payload с `sender`, `recipient`, `topic`, `body`.
- `Actor` — abstract с `receive(message, runtime)`.
- `Runtime` — event loop с shared queue, delivery, failure isolation.
- Two-actor demo: `ReviewerAgent` reviews code, `ChecklistAgent` runs a checklist; они обмениваются messages до consensus.

Запустите:

```
python3 code/main.py
```

Трасса показывает доставку сообщений, simulated failure в одном actor, который не рушит другого, и convergence к shared verdict.

## Используйте это

- **AutoGen v0.4/v0.7** (maintenance) — стабилен для research, prototyping, multi-agent patterns.
- **Microsoft Agent Framework** (public preview) — forward path; те же actor-model ideas в обновленном API.
- **LangGraph swarm topology** (Lesson 13) — похожий паттерн через shared-tool handoffs.
- **Custom actor runtime** — когда нужен specific transport (NATS, RabbitMQ, gRPC).

## Доведите до продакшена

`outputs/skill-actor-runtime.md` генерирует minimal actor runtime плюс team template (RoundRobin или Selector) для заданной multi-agent task.

## Упражнения

1. Добавьте dead-letter queue: когда handler raises, помещайте failing message для human inspection. Как часто DLQ срабатывает в игрушке?
2. Реализуйте `SelectorGroupChat`: selector actor выбирает, кто обработает next message, на основе conversation state.
3. Добавьте distributed transport: замените in-process queue на JSON-over-HTTP server, чтобы actors могли run in separate processes.
4. Подключите OTel span per message (или no-op stand-in). Emit `gen_ai.agent.name`, `gen_ai.operation.name` согласно Lesson 23.
5. Прочитайте architecture post AutoGen v0.4. Перенесите игрушку на реальный API `autogen_core`. Что вы пропустили из того, что важно in production?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Actor | "Agent" | Private state + inbox + handler; без shared memory |
| Message | "Event" | Typed payload; единственный способ взаимодействия между actors |
| Inbox | "Mailbox" | Per-actor queue pending messages |
| Runtime | "Agent host" | Event loop, который routes messages и isolates failures |
| Topic | "Channel" | Named publish-subscribe route между actors |
| Fault isolation | "Let it crash" | Failure одного actor не рушит others |
| RoundRobinGroupChat | "Fixed-rotation team" | Agents ходят по очереди |
| SelectorGroupChat | "Context-routed team" | Selector выбирает, кто идет дальше |
| Magentic-One | "Reference team" | Multi-agent squad для web + code + files |

## Дополнительное чтение

- [AutoGen v0.4, Microsoft Research](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — пост о redesign
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — graph-shaped alternative
- [OpenTelemetry GenAI semantic conventions](https://opentelemetry.io/docs/specs/semconv/gen-ai/) — spans, которые AutoGen emits by default
