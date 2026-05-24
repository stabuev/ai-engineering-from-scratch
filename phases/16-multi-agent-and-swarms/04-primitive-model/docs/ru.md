# Модель примитивов multi-agent систем

> Каждый multi-agent фреймворк, поставляемый в 2026 году — AutoGen, LangGraph, CrewAI, OpenAI Agents SDK, Microsoft Agent Framework — это точка в четырехмерном пространстве проектирования. Четыре примитива, и ничего больше: агент, handoff, shared state, orchestrator. В этом уроке мы построим их с нуля, запустим игрушечную систему на всех четырех, а затем сопоставим каждый крупный фреймворк с теми же осями, чтобы вы могли понять любой новый релиз по одному абзацу.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 (Agent Engineering), Фаза 16 · 01 (Why Multi-Agent)
**Время:** ~60 минут

## Проблема

Каждые шесть месяцев выходит новый multi-agent фреймворк. AutoGen в 2023. CrewAI в 2024. LangGraph и OpenAI Swarm в 2024. Google ADK в апреле 2025. Microsoft Agent Framework RC в феврале 2026. Каждый пресс-релиз утверждает, что это "правильная абстракция".

Если пытаться изучать их по одному, вы быстро выгорите. API выглядят по-разному. Документация расходится в том, что такое "agent". Один фреймворк называет общую память "blackboard", другой — "message pool", третий — "StateGraph". Начинает казаться, что область просто бесконечно перетряхивается.

Это не так. Под маркетингом четыре примитива остаются стабильными. Выучите их один раз — и будете понимать каждый новый фреймворк по одному абзацу.

## Концепция

### Четыре примитива

1. **Agent** — системный промпт плюс список инструментов. Stateless; каждый запуск начинается с системного промпта и текущей истории сообщений.
2. **Handoff** — структурированная передача управления от одного агента к другому. Механически это tool call, который возвращает нового агента, или ребро графа, выбираемое по условию.
3. **Shared state** — любая структура данных, которую может читать больше одного агента (иногда и писать). Message pool, blackboard, key-value store, vector memory.
4. **Orchestrator** — тот, кто решает, кто говорит следующим. Варианты: явный граф (детерминированный), LLM speaker-selector (мягкий), handoff call последнего говорящего (OpenAI Swarm) или scheduler над очередью (swarm architecture).

Это и есть все пространство проектирования. Каждый фреймворк выбирает значения по умолчанию для каждой оси; остальное — поверхностный синтаксис.

### Как каждый фреймворк 2026 года ложится на эту модель

| Framework | Agent | Handoff | Shared state | Orchestrator |
|-----------|-------|---------|--------------|--------------|
| OpenAI Swarm / Agents SDK | `Agent(instructions, tools)` | tool returns Agent | caller's problem | the LLM's next handoff call |
| AutoGen v0.4 / AG2 | `ConversableAgent` | speaker-selector on GroupChat | message pool | selector function (LLM or round-robin) |
| CrewAI | `Agent(role, goal, backstory)` | `Process.Sequential / Hierarchical` | Task outputs chained | manager LLM or static order |
| LangGraph | node function | graph edge + condition | `StateGraph` reducer | the graph, deterministic |
| Microsoft Agent Framework | agent + orchestration patterns | pattern-specific | thread / context | pattern-specific |
| Google ADK | agent + A2A card | A2A task | A2A artifacts | host decides |

Поверхностные различия выглядят огромными. Под ними: те же четыре ручки настройки.

### Почему это важно

Когда вы видите примитивы, сравнение фреймворков превращается в короткий checklist:

- Доверяет ли orchestrator LLM маршрутизацию (Swarm) или фиксирует routing в коде (LangGraph)?
- Shared state хранит полную историю (GroupChat) или проекцию (StateGraph reducer)?
- Могут ли агенты изменять промпты друг друга (CrewAI manager) или только передавать управление (Swarm)?

Эти три вопроса дают 80% ответа, какой фреймворк подходит конкретной задаче. Вы перестаете искать "лучший multi-agent фреймворк" и начинаете проектировать под ось, которая действительно важна.

### Stateless-инсайт

Каждый примитив, кроме shared state, stateless. Agent — это функция от (prompt, tools). Handoff — это function call. Orchestrator — это scheduler. **Единственная stateful часть системы — shared state.** Именно там живут все интересные баги: memory poisoning (Lesson 15), порядок сообщений, версионирование, конкуренция записей.

Фреймворки, которые скрывают shared state (Swarm), перекладывают проблему на вызывающую сторону. Фреймворки, которые централизуют его (LangGraph checkpoint, AutoGen pool), делают его наблюдаемым, но переносят координационные издержки в реализацию shared-state.

### Анатомия одного примитива

#### Agent

```
Agent = (system_prompt, tools, model, optional_name)
```

Нет памяти. Нет состояния. Два агента с одинаковым system prompt и tools взаимозаменяемы. Все, что выглядит как per-agent state, на самом деле находится в shared state или в handoff protocol.

#### Handoff

```
Handoff = (from_agent, to_agent, reason, payload)
```

Доминируют три реализации:

- **Function return** — tool возвращает следующего агента. Это паттерн OpenAI Swarm. Агенты несут routing в своих tool schemas.
- **Graph edge** — LangGraph. Ребра декларативны. LLM производит значение; условие выбирает следующий node.
- **Speaker selection** — AutoGen GroupChat. Selector function (иногда сам LLM call) читает pool и выбирает, кто говорит следующим.

#### Shared state

```
SharedState = { messages: [], artifacts: {}, context: {} }
```

Минимум — список сообщений. Часто больше: structured artifacts (CrewAI Task outputs), typed context (LangGraph reducers), external memory (MCP, vector DB).

Две топологии: **full pool** (каждый агент видит каждое сообщение) и **projected** (агенты видят role-scoped view). Full pools просты и плохо масштабируются. Projected pools масштабируются, но требуют предварительного проектирования схемы.

#### Orchestrator

```
Orchestrator = ({state, last_speaker}) -> next_agent
```

Четыре разновидности:

- **Static** — граф фиксирован во время сборки (LangGraph deterministic, CrewAI Sequential).
- **LLM-selected** — LLM читает pool и выбирает следующего speaker (AutoGen, CrewAI Hierarchical).
- **Handoff-driven** — текущий агент решает, вызывая handoff tool (Swarm).
- **Queue-driven** — workers забирают задачи из общей очереди; явного next-speaker нет (swarm architectures, Matrix).

### Что меняется между фреймворками

Когда примитивы зафиксированы, остаются такие проектные решения:

- **Memory strategy** — ephemeral vs durable checkpointing (LangGraph checkpointer).
- **Safety boundary** — кто может approve handoff (human-in-the-loop).
- **Cost accounting** — token budgets по агентам.
- **Observability** — tracing handoffs, сохранение state для replay.

Все это реализуется поверх примитивов. Ничто из этого не является новым примитивом.

## Соберите это

`code/main.py` реализует четыре примитива примерно в 150 строках stdlib Python. Реальной LLM нет — каждый agent является scripted policy, чтобы фокус оставался на координационной структуре.

Файл экспортирует:

- `Agent` — dataclass из name, system prompt, tools, policy function.
- `Handoff` — функция, которая возвращает нового агента.
- `SharedState` — thread-safe message pool.
- `Orchestrator` — три варианта: `StaticOrchestrator`, `HandoffOrchestrator`, `LLMSelectorOrchestrator` (simulated).

Демо прогоняет один и тот же трехагентный pipeline (research → write → review) через все три типа orchestrator и в конце печатает message pool. Вы увидите, что outputs различаются только тем, *кто выбирает следующего*; agents и shared state идентичны между запусками.

Запустите:

```
python3 code/main.py
```

Ожидаемый output: три запуска orchestrator, по одному на pattern. Каждый печатает финальный message pool. Handoff-driven запуск достигает меньшего числа агентов, если researcher решает, что работа завершена рано — это миниатюрная версия tradeoff LLM-routing.

## Используйте это

`outputs/skill-primitive-mapper.md` — это skill, который читает любую multi-agent codebase или framework doc и возвращает mapping на четыре примитива. Запустите его на новом релизе фреймворка, чтобы получить понимание в один абзац до глубокого чтения документации.

## Доведите до production

Перед внедрением нового фреймворка напишите для него primitive mapping. Если не можете, документация неполна или фреймворк изобретает пятый примитив (редко — проверьте, не является ли это разновидностью shared-state, которую вы еще не видели).

Зафиксируйте mapping в architecture doc. Когда в команду приходит новый участник, отправьте ему mapping до API docs. Когда меняются версии фреймворка, diff-айте mapping, а не changelog.

## Упражнения

1. Запустите `code/main.py` три раза с разными agent policies. Понаблюдайте, как выбор orchestrator меняет то, какие agents запускаются.
2. Реализуйте четвертый тип orchestrator: queue-driven вариант, где agents опрашивают shared state на наличие работы. Какой deadlock может возникнуть и как его обнаружить?
3. Возьмите LangGraph quickstart (https://docs.langchain.com/oss/python/langgraph/workflows-agents) и перепишите его как четыре примитива. Какие абстракции LangGraph отображаются 1:1, а какие являются convenience wrappers?
4. Прочитайте OpenAI Swarm cookbook (https://developers.openai.com/cookbook/examples/orchestrating_agents). Определите, какой из четырех примитивов Swarm делает самым эргономичным, а какой перекладывает на caller.
5. Найдите в этой таблице один фреймворк, который полностью скрывает shared state. Объясните, что ломается, когда agents должны координироваться между handoffs без повторного чтения history.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Agent | "LLM с tools" | Тройка `(system_prompt, tools, model)`. Stateless. |
| Handoff | "Передача управления" | Структурированный call, который называет следующего агента и optional payload. Три реализации: function return, graph edge, speaker selection. |
| Shared state | "Memory" / "context" | Единственная stateful часть multi-agent системы. Message pool или blackboard. |
| Orchestrator | "Coordinator" | Тот, кто решает, кто запускается следующим. Static graph, LLM selector, handoff-driven или queue-driven. |
| Primitive | "Abstraction" | Одна из четырех осей, которые параметризует каждый framework. Не feature фреймворка. |
| Message pool | "Shared chat history" | Shared state с полной историей. Прост для рассуждения, плохо масштабируется. |
| Projected state | "Scoped view" | Role-specific view в shared state. Масштабируется, требует проектирования схемы. |
| Speaker selection | "Who talks next" | Orchestrator pattern, где функция (часто LLM) выбирает следующего agent из группы. |

## Дополнительное чтение

- [OpenAI cookbook: Orchestrating Agents — Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — самое ясное изложение handoff-driven orchestration
- [AutoGen stable docs](https://microsoft.github.io/autogen/stable/) — GroupChat + speaker selection как референс для LLM-selected orchestration
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — graph-edge orchestration и reducer-based shared state
- [CrewAI introduction](https://docs.crewai.com/en/introduction) — role-goal-backstory agents, Sequential / Hierarchical processes
- [AG2 (community AutoGen continuation)](https://github.com/ag2ai/ag2) — живая линия AutoGen v0.2 после того, как Microsoft перевела v0.4 в maintenance
