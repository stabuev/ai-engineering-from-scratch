# Продакшен-рантаймы: queue, event, cron

> Production-агенты работают в шести формах runtime: request-response, streaming, durable execution, queue-based background, event-driven и scheduled. Сначала выбирайте форму runtime, и только потом framework. Observability является несущей частью в каждой форме.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 13 (LangGraph), Фаза 14 · 22 (Voice)
**Время:** ~60 минут

## Цели обучения

- Назвать шесть форм production runtime и сопоставить каждую с паттерном framework / product.
- Объяснить, почему durable execution (LangGraph) важно для long-horizon задач.
- Описать event-driven runtime и случаи, где подходит Claude Managed Agents.
- Объяснить утверждение observability-as-load-bearing для многошаговых агентов.

## Проблема

Production-агенты ломаются так, как Jupyter notebook не показывает: network timeout на шаге 37, пользователь кладет трубку посреди voice call, cron job умирает при перезагрузке машины, background worker исчерпывает память. Форма runtime определяет, какие сбои переживаемы.

## Концепция

### Request-response

- Синхронный HTTP. Пользователь ждет завершения.
- Подходит только для коротких задач (<30s).
- Стеки: Agno (Python + FastAPI), Mastra (TypeScript + Express/Hono/Fastify/Koa).
- Observability: стандартные HTTP access logs + OTel spans.

### Streaming

- SSE или WebSocket для постепенной выдачи результата.
- LiveKit расширяет это до WebRTC для voice/video (Урок 22).
- Стеки: любой framework с поддержкой streaming + frontend, который обрабатывает SSE/WS.
- Observability: per-chunk timing, first-token latency, tail latency.

### Durable execution

- State checkpointed после каждого шага; auto-resumes при сбое.
- AutoGen v0.4 actor model изолирует сбои до одного агента (Урок 14).
- Ключевой дифференциатор LangGraph (Урок 13).
- Необходимо, когда число шагов неизвестно, а цена восстановления высока.

### Queue-based / background

- Job попадает в queue, workers забирают его, результаты возвращаются через webhooks или pub/sub.
- Необходимо для long-horizon агентов (десятки-сотни шагов на задачу, по announcement Anthropic о computer use).
- Стеки: Celery (Python), BullMQ (Node), SQS + Lambda (AWS), custom.
- Observability: queue depth, per-job latency distribution, DLQ size.

### Event-driven

- Агенты подписываются на triggers: new email, PR opened, cron fire.
- Claude Managed Agents покрывает это из коробки (Урок 17).
- CrewAI Flows (Урок 15) структурирует event-driven deterministic workflows.
- Observability: trigger source, event-to-start latency, agent latency.

### Scheduled

- Cron-shaped агенты, которые запускаются периодически.
- Комбинируйте с durable execution, чтобы failing nightly run продолжался на следующем tick.
- Стеки: Kubernetes CronJob + durable framework; hosted (Render cron, Vercel cron).

### Паттерны развертывания 2026 года

- **CrewAI Flows** для event-driven production.
- **Agno** stateless FastAPI для Python microservices.
- **Mastra** server adapters (Express, Hono, Fastify, Koa) для embedding.
- **Pipecat Cloud / LiveKit Cloud** для managed voice (Урок 22).
- **Claude Managed Agents** для hosted long-running async.

### Observability как несущая часть

Без OpenTelemetry GenAI spans (Урок 23) плюс backend Langfuse/Phoenix/Opik (Урок 24) вы не сможете отладить многошагового агента, который упал на шаге 40. В production это не опция. Это разница между "мы быстро отлаживаем" и "мы переигрываем с нуля с большим количеством logging."

### Где production runtimes ломаются

- **Wrong shape choice.** Выбор request-response для 5-минутной задачи. Пользователи отключаются; workers накапливаются; retries усугубляют проблему.
- **No DLQ.** Queue workers без dead-letter. Failed jobs исчезают.
- **Opaque background work.** Background agent запускается без trace export. Сбои невидимы, пока пользователь о них не сообщит.
- **Skipping durable state.** Любой run > 30 seconds, где нельзя позволить restart, нуждается в durable execution.

## Практика

`code/main.py` — stdlib multi-shape demo:

- Request-response endpoint (plain function).
- Streaming handler (generator).
- Queue-based worker with DLQ.
- Event trigger registry.
- Cron-shaped scheduler.

Запустите:

```bash
python3 code/main.py
```

Output: пять traces, показывающих поведение каждой формы на одной и той же задаче. Та же agent logic, разные внешние оболочки. Durable execution (шестая форма) намеренно покрыта в Уроке 13 с LangGraph checkpointing.

## Как использовать

- **Request-response** для chat-style UX.
- **Streaming** для progressive responses.
- **Durable** для long-horizon tasks.
- **Queue** для batch / async / long-running.
- **Event** для agent reactivity.
- **Cron** для housekeeping (memory consolidation, evals, cost reports).

## Что подготовить

`outputs/skill-runtime-shape.md` выбирает runtime shape для задачи и связывает requirements по observability.

## Упражнения

1. Перенесите ваш ReAct loop из Урока 01 во все шесть форм в вашем stack. Какая форма подходит какому product surface?
2. Добавьте DLQ в queue-based demo. Смоделируйте 10% job failure; выведите DLQ size.
3. Напишите cron-triggered eval agent, который nightly запускается против ваших top 20 traces за день.
4. Реализуйте streaming with backpressure: если client медленный, поставьте agent на паузу. Как это взаимодействует с turn budget?
5. Прочитайте Claude Managed Agents docs. Когда вы перенесли бы self-hosted long-horizon agent в managed?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Request-response | "Synchronous" | Пользователь ждет; только короткие задачи |
| Streaming | "SSE / WS" | Постепенный output; лучший UX; latency наблюдаема по chunk |
| Durable execution | "Возобновление после сбоя" | Состояние с checkpoint; restart с последнего шага |
| Queue-based | "Фоновые задания" | Producer / worker pool / DLQ |
| Event-driven | "Trigger-based" | Агент реагирует на внешние события |
| DLQ | "Dead-letter queue" | Место ожидания для failed jobs |
| Claude Managed Agents | "Hosted harness" | Долгоживущий async, размещенный у Anthropic, с caching + compaction |

## Дополнительное чтение

- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — детали durable execution
- [Claude Managed Agents overview](https://platform.claude.com/docs/en/managed-agents/overview) — hosted long-running async
- [Anthropic, Introducing computer use](https://www.anthropic.com/news/3-5-models-and-computer-use) — "dozens-to-hundreds of steps per task"
- [AutoGen v0.4 (Microsoft Research)](https://www.microsoft.com/en-us/research/articles/autogen-v0-4-reimagining-the-foundation-of-agentic-ai-for-scale-extensibility-and-robustness/) — actor-model fault isolation
