# Паттерны оркестрации: Supervisor, Swarm, Hierarchical

> Четыре orchestration patterns повторяются во фреймворках 2026 года: supervisor-worker, swarm / peer-to-peer, hierarchical, debate. Рекомендация Anthropic: "It's about building the right system for your needs." Начинайте просто; добавляйте topology только когда single agent plus five workflow patterns недостаточен.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 12 (Workflow Patterns), Фаза 14 · 25 (Multi-Agent Debate)
**Время:** ~60 минут

## Цели обучения

- Назвать четыре повторяющихся orchestration patterns и когда подходит каждый.
- Описать рекомендацию LangChain 2026 года: tool-call-based supervision vs supervisor libraries.
- Объяснить правило Anthropic "build the right system" и как оно ограничивает выбор topology.
- Реализовать все четыре в stdlib поверх common scripted LLM.

## Проблема

Команды хватаются за "multi-agent" раньше, чем он им нужен. Четыре patterns повторяются во фреймворках; когда вы можете их назвать, вы можете выбрать правильный — или вообще отказаться от topology.

## Концепция

### Supervisor-worker

- Central routing LLM направляет запросы specialist agents.
- Решает: loop back to self, hand off to specialist, terminate.
- Specialists не общаются друг с другом; весь routing идет через supervisor.

Frameworks: LangGraph `create_supervisor`, Anthropic orchestrator-workers, CrewAI Hierarchical Process.

**Рекомендация LangChain 2026 года:** делайте supervision через direct tool calls, а не через `create_supervisor`. Это дает более тонкий context engineering control: вы сами решаете, что именно видит каждый specialist.

### Swarm / peer-to-peer

- Agents напрямую hand off через shared tool surface.
- Нет central router.
- Меньшая latency, чем у supervisor (меньше hops).
- Сложнее reasoning about (нет единой точки контроля).

Frameworks: LangGraph swarm topology, OpenAI Agents SDK handoffs (когда все agents могут hand off ко всем остальным).

### Hierarchical

- Supervisors manage sub-supervisors, которые manage workers.
- Реализуется как nested subgraphs в LangGraph; nested crews в CrewAI.
- Масштабируется до больших agent populations ценой operational complexity.

Когда это нужно: когда context budget одного supervisor не может вместить descriptions всех specialists.

### Debate

- Parallel proposers + iterative cross-critique (Урок 25).
- Это не совсем orchestration, а скорее verification, но появляется во фреймворках как topology choice.

### CrewAI Crew vs Flow

CrewAI формализует два deployment modes:

- **Flow** для deterministic event-driven automation (рекомендуемая starting point for production).
- **Crew** для autonomous role-based collaboration.

Это orthogonal к четырем patterns выше, но мапится на topology: Flow обычно supervisor или hierarchical; Crew обычно supervisor с LLM router.

### Рекомендация Anthropic

"Success in the LLM space isn't about building the most sophisticated system. It's about building the right system for your needs."

Порядок решений:

1. Single agent + workflow patterns (Урок 12) — начинайте здесь.
2. Supervisor-worker — когда у вас 2-4 specialists.
3. Swarm — когда latency важнее reasoning clarity.
4. Hierarchical — только когда supervisor context budget fails.
5. Debate — когда accuracy важнее cost.

### Где этот паттерн ломается

- **Topology-first thinking.** "We need multi-agent" до определения, какую проблему multi-agent решает.
- **Bouncing handoffs in swarm.** A -> B -> A -> B. Используйте hop counters.
- **Fake hierarchy.** Три слоя, потому что "enterprise"; две реальные команды. Collapse.

## Соберите это

`code/main.py` реализует все четыре patterns в stdlib поверх scripted LLM:

- `Supervisor` — central router.
- `Swarm` — peer-to-peer with direct handoffs.
- `Hierarchical` — supervisors of supervisors.
- `Debate` — parallel proposers + critique.

Каждый pattern обрабатывает одну и ту же three-intent task (refund / bug / sales). Trace shapes различаются.

Запустите:

```
python3 code/main.py
```

Output: trace + op count для каждого pattern. Supervisor - самый clean; swarm - shortest; hierarchical - deepest; debate - most expensive.

## Используйте это

- **LangGraph** для supervisor и hierarchical (nested subgraphs).
- **OpenAI Agents SDK** для handoffs-as-tools (supervisor-shaped).
- **CrewAI Flow** для production deterministic.
- **Custom** для debate или когда нужен exact control.

## Отправьте в работу

`outputs/skill-orchestration-picker.md` выбирает topology и реализует ее.

## Упражнения

1. Преобразуйте supervisor-worker в swarm, убрав router. Что ломается? Что улучшается?
2. Добавьте hop counter в swarm: отказывать после 3 handoffs. Ловит ли он A->B->A bouncing?
3. Постройте two-level hierarchical system для domain с 12 specialists. Где context budget ломается без nesting?
4. Профилируйте четыре patterns на production-shaped workload. Кто выигрывает по каким metric (latency, cost, accuracy, debuggability)?
5. Прочитайте пост Anthropic "Building Effective Agents". Сопоставьте каждый ваш production flow с одним из четырех. Есть ли такие, которые плохо сопоставляются?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Supervisor-worker | "Router + specialists" | Central LLM dispatches to specialists; они не общаются друг с другом |
| Swarm | "Peer-to-peer" | Прямые handoff через общие инструменты; без центрального router |
| Hierarchical | "Супервизоры супервизоров" | Вложенные subgraph для больших популяций |
| Debate | "Предложение + критика" | Параллельные proposer, перекрестная критика (Урок 25) |
| Tool-call-based supervision | "Supervisor без библиотеки" | Реализация supervisor как прямых tool calls для контроля контекста |
| Crew | "Автономная команда" | Ролевой режим совместной работы в CrewAI |
| Flow | "Детерминированный workflow" | Event-driven production mode в CrewAI |

## Дополнительное чтение

- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — five patterns + agent vs workflow
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — supervisor, swarm, hierarchical
- [CrewAI docs](https://docs.crewai.com/en/introduction) — Crew vs Flow
- [Du et al., Society of Minds (arXiv:2305.14325)](https://arxiv.org/abs/2305.14325) — debate pattern
