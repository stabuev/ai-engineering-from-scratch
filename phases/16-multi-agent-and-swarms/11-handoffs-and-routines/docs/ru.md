# Handoffs and Routines — Stateless Orchestration

> OpenAI Swarm (октябрь 2024) свел multi-agent orchestration к двум примитивам: **routines** (инструкции + инструменты как system prompt) и **handoffs** (инструмент, который возвращает другого Agent). Без state machine, без branching DSL — LLM маршрутизирует, вызывая нужный handoff tool. OpenAI Agents SDK (март 2025) — production-преемник. Сам Swarm остается самой чистой концептуальной ссылкой — весь его исходный код помещается в несколько сотен строк. Паттерн стал вирусным, потому что поверхность API примерно такая: "agent = prompt + tools; handoff = function returning agent." Ограничение: stateless, поэтому memory — проблема вызывающей стороны.

**Тип:** Learn + Build
**Языки:** Python (stdlib)
**Предварительные требования:** Phase 16 · 04 (Primitive Model)
**Время:** ~60 minutes

## Цели обучения

- Реализовывать два примитива OpenAI Swarm — routines и handoffs.
- Объяснять stateless-компромисс, делающий handoffs популярными.
- Определять, когда handoffs подходят, а когда буксуют.

## Проблема

Каждый multi-agent framework хочет, чтобы вы выучили его DSL: nodes и edges в LangGraph, crews и tasks в CrewAI, GroupChat и managers в AutoGen. Эти DSL — реальные абстракции, но из-за них вещь ощущается тяжелее, чем должна.

Swarm движется в противоположную сторону: используйте tool-calling capability, которая уже есть у модели. Handoffs становятся tool calls. Orchestrator — это тот agent, который сейчас удерживает conversation. State machine неявно задана в system prompts агентов.

## Концепция

### Two primitives

**Routine.** System prompt, который определяет роль агента и доступные tools. Думайте об этом как о scoped-наборе инструкций: "you are a triage agent; if the user asks about refunds, hand off to the refund agent."

**Handoff.** Tool, который агент может вызвать и который возвращает новый объект Agent. Swarm runtime обнаруживает возвращенное значение Agent и переключает active agent на следующий turn.

Это вся абстракция.

```
def transfer_to_refunds():
    return refund_agent  # Swarm sees Agent return → switch active agent

triage_agent = Agent(
    name="triage",
    instructions="Route the user to the right specialist.",
    functions=[transfer_to_refunds, transfer_to_sales, transfer_to_support],
)
```

System prompt triage agent заставляет его выбрать правильный handoff на основе сообщения пользователя. Tool-calling LLM выполняет маршрутизацию.

### Why it is viral

- **Small API.** Нужно выучить две концепции.
- **Uses what the model already does.** Tool calling уже production-grade у разных providers.
- **No state-machine burden.** Вы не описываете graph; prompts агентов описывают, кому они делают hand off.

### The stateless trade

Swarm явно stateless между runs. Framework держит message history во время run, но ничего не сохраняет. Memory, continuity, long-running tasks — все это проблема вызывающей стороны.

В production (OpenAI Agents SDK, март 2025) это было одной из главных вещей, которые изменились: SDK добавляет built-in session management, guardrails и tracing, сохраняя handoff primitive.

### When Swarm/handoffs fit

- **Triage patterns.** Front-line agent маршрутизирует пользователя к specialist.
- **Skill-based handoffs.** "If the task needs code, call the coder; if it needs research, call the researcher."
- **Short, bounded conversations.** Customer support, FAQ-to-ticket, простые workflows.

### When Swarm struggles

- **Long sessions with shared memory.** Handoffs сбрасывают conversation state к prompt нового агента плюс history. Нет persistent state между агентами без caller-managed memory.
- **Parallel execution.** Handoff идет one-at-a-time — active agent переключается. Parallelism требует, чтобы caller orchestrated несколько Swarm runs.
- **Audit and replay.** Stateless runs сложно точно replay; выбор handoff со стороны LLM недетерминирован.

### OpenAI Agents SDK (March 2025)

Production-преемник добавляет:

- **Session state.** Persistent thread между runs.
- **Guardrails.** Hooks для input/output validation.
- **Tracing.** Каждый tool call и handoff логируется.
- **Handoff filters.** Контроль того, какой context переносится при handoff.

Handoff primitive сохраняется; production ergonomics добавляются вокруг него.

### Swarm vs GroupChat

Оба используют LLM-driven routing, но различаются тем, **кто выбирает следующего**:

- GroupChat: selector (function или LLM) выбирает next speaker извне.
- Swarm: current agent выбирает successor, вызывая handoff tool.

Swarm — это "agent decides what's next"; GroupChat — "manager decides what's next." Решение Swarm живет в tool call активного агента; решение GroupChat живет в `GroupChatManager`.

## Соберите

`code/main.py` реализует Swarm from scratch: dataclass Agent, механизм handoff (tool возвращает Agent) и run loop, который обнаруживает agent switches.

Демо: triage agent маршрутизирует к специалистам refund, sales или support. У каждого specialist есть свои tools. Цикл запуска печатает каждый handoff.

Запуск:

```
python3 code/main.py
```

## Используйте

`outputs/skill-handoff-designer.md` проектирует handoff topology для заданной задачи: какие agents существуют, какие handoffs они могут вызывать, какой context переносится.

## Запустите в production

Чеклист:

- **Handoff logging.** Каждый handoff пишет trace event с from-agent, to-agent, context snapshot.
- **Context transfer rules.** Решите, что переносится при handoff: full history (дорого), last N messages или summary.
- **Guardrail on handoff.** Handoff к specialist с другими tool permissions должен быть authenticated — иначе prompt injection может принудить нежелательные handoffs.
- **Loop detection.** Два agents, передающие друг другу управление туда-сюда, — частый failure; обнаруживайте это простым last-K ring check.
- **Fallback agent.** Если handoff target не существует, fallback к safe default.

## Упражнения

1. Запустите `code/main.py`, выполните triage к refund agent. Подтвердите, что active agent второго turn — refund.
2. Добавьте правило loop-detection: если одна и та же пара agents сделала hand off 3 раза подряд, принудительно выйти. Спроектируйте fallback.
3. Прочитайте docs OpenAI Agents SDK про handoff filters. Реализуйте версию "summarize-on-handoff": outgoing agent сжимает context до bullet summary перед тем, как incoming agent берет управление.
4. Сравните Swarm handoff с selector в GroupChatManager. Какой pattern ухудшает prompt injection и почему?
5. Прочитайте Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Найдите одно явное design decision в Swarm, которое OpenAI Agents SDK изменил или сохранил.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Routine | "The agent prompt" | System prompt + tool list. Определяет роль и доступные handoffs. |
| Handoff | "Transfer to another agent" | Tool, который active agent может вызвать и который возвращает новый Agent. Runtime переключает active agent. |
| Stateless | "No memory between runs" | Swarm ничего не сохраняет; memory — ответственность caller. |
| Active agent | "Who's speaking now" | Agent, который сейчас удерживает conversation. Handoff меняет его. |
| Context transfer | "What moves on handoff" | Policy для того, какую history видит incoming agent: full, last N или summarized. |
| Handoff loop | "Agents ping-pong" | Failure mode, при котором два agents продолжают передавать управление друг другу. |
| OpenAI Agents SDK | "Production Swarm" | Преемник марта 2025; добавляет sessions, guardrails, tracing поверх handoff primitive. |
| Handoff filter | "Gate on transfer" | Feature SDK для inspect и modify context на handoff boundary. |

## Дополнительное чтение

- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — reference articulation
- [OpenAI Swarm repo](https://github.com/openai/swarm) — original implementation, kept as conceptual reference
- [OpenAI Agents SDK docs](https://openai.github.io/openai-agents-python/) — production successor with sessions and tracing
- [Anthropic handoff-in-Claude notes](https://docs.anthropic.com/en/docs/claude-code) — как Claude Code subagents используют handoff-like pattern через `Task`
