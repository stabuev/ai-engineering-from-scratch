# Parallel / Swarm / Networked Architectures

> Контраст с supervisor: нет центрального решателя. Агенты читают общий event bus, асинхронно подхватывают работу, записывают результаты обратно. LangGraph явно поддерживает "Swarm Architecture" для децентрализованных, динамических environments. Matrix (arXiv:2511.21686) представляет и control flow, и data flow как serialized messages, передаваемые через distributed queues, чтобы устранить orchestrator bottleneck. Компромисс явный: determinism и traceability меняются на scalability. Swarm подходит задачам со многими независимыми подзадач; он не подходит задачам, которым нужен единый согласованный план.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib, `threading`, `queue`)
**Пререквизиты:** Phase 16 · 05 (Supervisor Pattern), Phase 16 · 04 (Primitive Model)
**Время:** ~75 минут

## Цели обучения

- Строить рой без центрального решателя: агенты читают общую шину событий и асинхронно берут работу.
- Определять, когда рой подходит, а когда проваливается.
- Противопоставлять топологию роя паттерну supervisor.

## Проблема

Supervisor масштабируется до нескольких workers. А что насчет сотен? Сам supervisor становится bottleneck: каждое решение о том, кто что делает, проходит через одного agent. Один медленный plan step задерживает всю систему.

Swarm architectures переворачивают дизайн. Вместо того чтобы central planner рассылал work, workers берут work из shared queue. "Coordination" зашита в semantics event bus. Нет orchestrator; система масштабируется до предела queue.

## Концепция

### Форма

```
                ┌──── shared queue ────┐
                │                      │
       ┌────────┼────────┐  ◄──────┬───┘
       ▼        ▼        ▼         │
     Worker  Worker  Worker   Worker
      A       B       C        D
       │        │        │         │
       └────────┴────────┴─────────┘
                 │
                 ▼
            results pool
```

Нет orchestrator. Каждый worker повторяет: взять task, обработать, записать result (и опционально поставить follow-ups в очередь).

### Когда swarm подходит

- **Много независимых задач.** Scraping, transforming, classifying. Tasks не зависят друг от друга.
- **Работа с переменной длительностью.** Если одни tasks занимают 100ms, а другие 10s, swarm автоматически балансирует нагрузку - быстрые workers берут следующие jobs. Supervisor должен заранее предугадывать duration.
- **Throughput важнее determinism.** Вам важно общее completion time, а не строгий ordering.

### Когда swarm проваливается

- **Упорядоченные workflows.** Если step 3 нужен output step 2, swarm рискует запустить step 3 до завершения step 2.
- **Задачи с global plan.** Сложные research questions выигрывают от planner. Swarm researchers производит независимые facts, а не coherent report.
- **Debugging.** Без central log и с asynchronous work воспроизводить bug дорого.

### Matrix (arXiv:2511.21686)

Matrix - статья 2025 года, которая доводит swarm до естественного предела: и control flow, и data flow являются serialized messages в distributed queues. Нет central coordinator. Fault tolerance возникает из message durability. Scalability становится проблемой message broker, а не самой системы.

Вклад: programming model, где multi-agent coordination формулируется как "what message topic does this agent subscribe to?", а не "which agent does the supervisor pick next?" Из-за этого система выглядит как pub/sub event mesh.

### LangGraph's Swarm Architecture

Документация LangGraph 2025 явно описывает "Swarm Architecture" как один из multi-agent patterns: agents - это nodes, но edges образуют directed graph with cycles, и любой node может активироваться из pool. Worker выбирает из available work по condition, а не по supervisor assignment.

### Режим отказа: starvation and hot-spotting

Если все workers берут fastest-available task, long-running tasks могут не выбираться, пока не останутся единственными. Классическое queue starvation.

Mitigations:
- Priority queues with explicit aging (increase priority with wait time).
- Worker specialization: some workers only take "long" tasks.
- Back-pressure: limit how many fast tasks enter the queue.

### Связь с content-based routing

Swarm естественно сочетается с content-based routing (Lesson 22). Вместо generic queue заведите одну queue на каждый message type. Specialist workers подписываются только на свой type. Это основа message-bus architectures, которые масштабируются до тысяч agents.

## Соберите это

`code/main.py` реализует swarm из 4 worker threads, которые берут задачи из shared `queue.Queue`. Tasks имеют variable durations (одни быстрые, другие медленные). Демо сравнивает:

- **Sequential baseline:** один worker обрабатывает все tasks serially.
- **Fixed assignment:** каждая task заранее assigned конкретному worker (supervisor-style).
- **Swarm:** workers берут задачи из shared queue.

Swarm автоматически балансирует load; fixed assignment оставляет быстрых workers idle, когда назначенная им task медленная.

Запуск:

```
python3 code/main.py
```

Вывод показывает per-worker task counts (swarm распределяет unevenly but optimally) и wall-clock times.

## Используйте это

`outputs/skill-swarm-fit.md` оценивает, стоит ли задаче использовать swarm vs supervisor. Входы: task independence, duration variance, ordering requirements, debuggability needs.

## Доведите до production

Чеклист:

- **Priority queue with aging.** Предотвращайте long-task starvation.
- **Worker idempotency.** Task может быть pulled больше одного раза, если worker падает mid-run. Workers должны быть idempotent.
- **Durable queue.** Используйте Kafka, Redis Streams или database-backed queue для production. `queue.Queue` только in-memory.
- **Observability per task.** У каждой task есть trace ID; каждый worker логирует start/end с ним.
- **Back-pressure.** Если queue растет быстрее, чем workers ее drain, замедляйте producer.

## Упражнения

1. Запустите `code/main.py`. Насколько swarm быстрее sequential на variable-duration workload? Насколько быстрее fixed assignment?
2. Добавьте вариант priority queue (используйте `queue.PriorityQueue`). Назначьте priority по полю task "importance". Посмотрите, starve ли low-priority tasks при continuous load.
3. Реализуйте hot-spot detector: логируйте, когда любой worker обрабатывает в 3× больше tasks, чем самый медленный worker. Что это говорит о task-duration distribution?
4. Прочитайте abstract и Section 3 статьи Matrix (arXiv:2511.21686). Назовите один конкретный tradeoff, который принимает Matrix (scalability gain), и один, от которого он отказывается (traceability, determinism).
5. Переделайте swarm demo так, чтобы использовать `queue.Queue` из tuples (task_type, payload), с workers, подписанными только на specific types. Какие routing rules имеют смысл, когда tasks heterogeneous?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Swarm architecture | "Децентрализованные agents" | Workers берут задачи из shared queue; нет central orchestrator. |
| Event bus | "Agents subscribe to topics" | Message broker, который routes tasks к workers по type или content. |
| Starvation | "Task never runs" | Low-priority task никогда не выбирается, потому что higher-priority work поступает непрерывно. |
| Hot-spotting | "Один worker захлебывается" | Load imbalance, когда один worker получает большинство tasks. |
| Back-pressure | "Замедлить producer" | Механизм, который сигнализирует upstream прекратить production, когда queue заполняется. |
| Idempotent worker | "Safe to re-run" | Task, обработанная дважды, дает тот же result. Требуется, потому что workers могут падать mid-run. |
| Durable queue | "Переживает crashes" | Queue, backed by disk или replicated storage; tasks не теряются при падении worker. |
| Matrix framework | "Full message-passing swarm" | И data, и control flow являются serialized messages в distributed queues. |

## Дополнительное чтение

- [LangGraph workflows and agents — Swarm Architecture](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — явная поддержка swarm
- [Matrix — A Decentralized Framework for Multi-Agent Systems](https://arxiv.org/abs/2511.21686) — полный message-passing swarm
- [Anthropic engineering — why supervisor not swarm in Research](https://www.anthropic.com/engineering/multi-agent-research-system) — почему конкретная production system явно выбрала supervisor вместо swarm
- [AutoGen v0.4 actor-model docs](https://microsoft.github.io/autogen/stable/) — event-driven actor rewrite, ближе к swarm, чем GroupChat в v0.2
