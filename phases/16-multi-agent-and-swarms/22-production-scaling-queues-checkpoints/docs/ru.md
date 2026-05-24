# Production Scaling — очереди, checkpoints, durability

> Масштабирование multi-agent систем до тысяч concurrent runs требует **durable execution**. Runtime LangGraph записывает checkpoint после каждого super-step с ключом `thread_id` (по умолчанию Postgres); падение worker освобождает lease, и другой worker продолжает выполнение. Агенты могут спать неопределенно долго, ожидая human input. **MegaAgent** (arXiv:2408.09955) использовал per-agent producer-consumer queue с тремя состояниями (Idle / Processing / Response) и двухслойную координацию (intra-group chat + inter-group admin chat). **Fiber/async** выигрывает у thread-per-job для LLM streaming: threads 99% времени простаивают, ожидая tokens, а fibers кооперативно уступают управление на I/O. Контраргумент: Ashpreet Bedi в "Scaling Agentic Software" предлагает **FastAPI + Postgres + nothing else**, пока нагрузка не докажет обратное — простые архитектуры работают дальше, чем ожидают. В этом уроке мы строим durable checkpoint log, per-agent work queue с transitions состояний, demo async-vs-thread и закрепляем прагматичное правило "start simple".

**Тип:** Изучение + Build
**Языки:** Python (stdlib, `asyncio`, `sqlite3`)
**Предварительные требования:** Phase 16 · 09 (Parallel Swarm Networks), Phase 16 · 13 (Shared Memory)
**Время:** ~75 минут

## Проблема

Прототип multi-agent системы работает на одном laptop с тремя агентами в in-memory event loop. Вы переносите его в production:

- Агенты иногда работают часами (длинные исследования, ожидания human-in-the-loop).
- Worker processes падают. Restart теряет state.
- Peak load в 10 раз выше average; нужно horizontal scaling.
- Пользователи платят за agent-run; нужны exactly-once semantics для billing.

In-memory event loop не делает ничего из этого. Под ним нужен durable execution layer. Канонические варианты 2026 года:

1. Workflow engine с checkpoints (Temporal, LangGraph runtime).
2. Message queue со state store (Postgres + SQS/RabbitMQ).
3. Actor-model frameworks (per-agent producer-consumer в MegaAgent).
4. Hand-rolled FastAPI + Postgres (аргумент Bedi).

Этот урок строит миниатюру каждого подхода.

## Концепция

### Durable execution, паттерн

Durable-execution engine сохраняет полный program state после каждого "step" (super-step, в терминах LangGraph). При падении:

```
worker crashes mid-step
  -> lease timeout
  -> another worker picks up the thread_id
  -> resumes from last checkpoint
  -> no duplicate side effects
```

Требования, чтобы это работало:

- **Serializable state.** Весь agent state должен быть persistable. Function closures с живыми database connections не переживают восстановление.
- **Deterministic resume.** При том же state и тех же inputs агент производит те же actions (или делегирует внешнему deterministic oracle для LLM calls).
- **Idempotent side effects.** External calls (tool calls, payments) должны быть idempotent или использовать deduplication key.

LangGraph пишет checkpoint после каждого super-step; Temporal пишет после каждой activity; Restate использует event-sourced journals. Все три реализуют один паттерн.

### Runtime LangGraph

У каждого агента есть `thread_id`; state — typed dict; каждый super-step пишет строку в checkpoints table. При resume runtime воспроизводит выполнение с последнего checkpoint, а не с нуля. Агенты могут вызывать `interrupt()`, ожидая human input; runtime сохраняет state и освобождает worker. Когда input приходит, любой worker может resume.

Это эталонный production design в апреле 2026.

### Per-agent queue в MegaAgent

arXiv:2408.09955 описывает масштабный эксперимент: тысячи concurrent agents в одном cluster. Архитектура:

```
agent i:
  state ∈ {Idle, Processing, Response}
  in_queue   <- messages addressed to agent i
  out_queue  -> replies + side effects

coordinators:
  intra-group chat  (agents in the same group)
  inter-group admin chat  (high-level routing)
```

Двухслойная координация позволяет intra-group conversation быть плотной, а inter-group оставаться разреженной — паттерн для удержания cost линейным при тысячах агентов.

### Async vs thread-per-job

LLM calls — I/O-bound. Thread, ожидающий следующий token, 99% времени idle. Threads стоят ~1MB RAM каждый; при 10,000 concurrent calls это 10GB только на stacks.

Fibers (Python `asyncio`, Go goroutines, Rust `tokio`) кооперативно уступают управление на I/O. Те же 10,000 calls комфортно помещаются в process. В масштабе LLM-agent async — не optimization, а architecture.

Исключение: CPU-bound post-processing (embedding, tokenizer tricks) все еще хочет threads или processes. Разделяйте I/O layer и CPU layer.

### Контраргумент Bedi

"Scaling Agentic Software" (Ashpreet Bedi, 2026) утверждает, что большинство команд over-engineer до измерения нагрузки. Прагматичный default:

- FastAPI + Postgres.
- Каждый agent run — строка; state обновляется in-place с optimistic concurrency.
- Background jobs через `pg_notify` или простой Celery worker.
- Retry policy в application code.

Для нагрузок ниже ~100 concurrent agent-runs на управляемых задачах этого часто достаточно. Обновляйте архитектуру, когда измерите, что она не справляется.

Правило: принимайте durable-execution frameworks, когда сталкиваетесь с конкретной проблемой, которую простые архитектуры не решают. Преждевременное внедрение сжигает время на ceremonies, которые не окупаются.

### Exactly-once semantics

Для платных agent runs нужно "exactly-once effective" (at-least-once delivery + idempotent consumer). Инженерные приемы:

- **Dedup key per run.** Включайте его в каждый side-effect call.
- **Outbox pattern.** Side effects сначала пишутся в table, затем отдельный process выполняет их. Оба шага idempotent.
- **Compensating transactions.** Когда side effect успешен, но tracking write не удался, планируйте compensate.

Это database-engineering patterns, а не LLM-specific. LLM tax только в том, что LLM calls медленные; все остальное — стандартные distributed systems.

### Rainbow deployment

Multi-agent research system Anthropic использует "rainbow deployments": несколько версий agent runtime работают одновременно, чтобы long-running agents не приходилось убивать при каждом code deploy. Canary новые версии на доле traffic; retire старые версии, когда их agents заканчивают.

Это стандартно для long-running stateful systems; адаптация 2026 года в том, что agents могут жить часами, поэтому deploy cycles должны это учитывать.

### Канонический production checklist

- Durable state (checkpoints, snapshots или outbox + replayable log).
- Idempotent side effects.
- Async I/O layer для LLM calls.
- At-least-once delivery with dedup.
- Rainbow/canary deployment для stateful workloads.
- Observability: per-agent traces, super-step audit, retry counter.

## Соберите

`code/main.py` реализует:

- `CheckpointStore` — checkpoint log на SQLite с ключами thread-id. Каждый super-step append row.
- `run_with_checkpoint(agent, thread_id)` — симулирует crash mid-run; второй worker resume с последнего checkpoint.
- `AgentQueue` — per-agent state machine Idle / Processing / Response с небольшой work queue.
- `demo_async_vs_threads()` — запускает 500 concurrent simulated "LLM calls" через asyncio и через threads; сообщает wall-clock и peak memory (approximated).

Запуск:

```
python3 code/main.py
```

Ожидаемый output: checkpoint resume успешно проходит после simulated crash; async version обрабатывает 500 concurrent calls за < 1s; thread version занимает несколько секунд и использует на порядки больше memory per concurrent unit.

## Используйте

`outputs/skill-scaling-advisor.md` советует выбор durable-execution: FastAPI + Postgres, LangGraph runtime, Temporal или custom. Калибруется по load, state-retention needs и deploy frequency.

## Запустите в production

Каноническое production hardening:

- **Start simple (Bedi's rule).** FastAPI + Postgres, пока измерения не покажут, что это не справляется.
- **Instrument everything before optimizing.** Per-run latency histogram, per-step time, retry count, failure categorization.
- **Outbox pattern for side effects.** Особенно payments и external API calls.
- **Rainbow deploys.** Никогда не убивайте in-flight agent runs во время deploys.
- **Adopt durable-execution engines (Temporal / LangGraph / Restate) when** вы столкнулись с конкретными проблемами: hour-long human-in-the-loop waits, cross-region coordination, complex retry/compensation policies.
- **Async for the I/O layer.** Threads только для CPU-bound post-processing.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что checkpoint resume работает; измерьте разницу async vs thread concurrency.
2. Реализуйте **outbox** table: каждый tool call сначала пишет в outbox, затем отдельная goroutine/task выполняет. Проверьте idempotency, запустив tool call дважды.
3. Симулируйте **rainbow deploy**: две concurrent runtime versions; route половину новых thread_ids в каждую; подтвердите, что in-flight threads на старой версии не interrupted.
4. Прочитайте runtime doc LangGraph (ссылка ниже). Определите, какие features runtime дольше всего воспроизводить в hand-rolled FastAPI + Postgres version. Это причина adopt, или можно defer?
5. Прочитайте MegaAgent (arXiv:2408.09955) Section 3. Двухслойная координация (intra-group + inter-group admin chat) явная. Набросайте, как бы вы отобразили это на message queue с двумя queue families.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Durable execution | "Persist the program state" | Engine writes state after each super-step; crash recovery is deterministic. |
| Super-step | "Transactional boundary" | Unit of work between checkpoints. Термин LangGraph. |
| thread_id | "Agent run identifier" | Key, связывающий checkpoints и resume logic. |
| Idempotency | "Safe to retry" | Повтор side effect дает тот же результат, что одна попытка. |
| Outbox pattern | "Decouple side effects" | Write intent to a table; отдельный executor performs and marks done. |
| At-least-once delivery | "Possible duplicates" | Message queue semantics; dedup key делает consumer effective-once. |
| Rainbow deploy | "Overlapping versions" | Multiple runtime versions concurrent during long-running workloads. |
| Async fiber | "Cooperative yielding" | User-mode concurrency; дешевле threads для I/O-bound loads. |
| Checkpoint | "State snapshot" | Serialized state на границе super-step; key for resume. |

## Дополнительное чтение

- [LangChain — The runtime behind production deep agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — LangGraph runtime design
- [MegaAgent](https://arxiv.org/abs/2408.09955) — per-agent producer-consumer queue; two-layer coordination at thousands of concurrent agents
- [Matrix](https://arxiv.org/abs/2511.21686) — decentralized framework with message queues as the coordination substrate
- [Temporal docs](https://docs.temporal.io/) — the reference workflow engine for durable execution
- [Anthropic — Multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production lessons including rainbow deployment
