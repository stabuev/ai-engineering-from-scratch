# Долго работающие фоновые агенты: Durable Execution

> Production long-horizon agents не запускаются в `while True`. Каждый LLM-вызов становится activity с checkpoint, retry и replay. Интеграция Temporal's OpenAI Agents SDK стала GA в марте 2026. Claude Code Routines (Anthropic) запускает запланированные вызовы Claude Code без постоянного локального процесса. Сессии приостанавливаются на human-input, переживают deploys и возобновляются с последнего checkpoint, ключуемого `thread_id`. За новой эргономикой стоит старый паттерн — workflow orchestration — с одним новым входом: LLM-вызовы как недетерминированные activities, которые должны детерминированно воспроизводиться при восстановлении.

**Тип:** Обучение
**Языки:** Python (stdlib, минимальная машина состояний для durable execution)
**Предварительные требования:** Phase 15 · 10 (Permission modes), Phase 15 · 01 (Long-horizon agents)
**Время:** ~60 минут

## Цели обучения

- Объяснять, почему длинногоризонтным агентам нужно durable-исполнение (чекпоинт, ретрай, реплей), а не while-цикл.
- Моделировать каждый вызов LLM как durable-активность.
- Называть стек durable-исполнения (Temporal) и что он гарантирует.

## Проблема

Рассмотрим агента, который работает четыре часа. Он вызывает три инструмента, дважды спрашивает пользователя и делает сорок LLM-вызовов. На середине host, на котором он запущен, перезагружается. Что происходит?

- В наивном цикле `while True`: все потеряно. Run начинается с нуля. Три вызова инструментов (с реальными side effects) выполняются снова. Пользователя снова спрашивают о вещах, которые он уже одобрил. Сорок LLM-вызовов оплачиваются повторно.
- С durable execution: run возобновляется с самого свежего checkpoint. Уже завершенные activities не выполняются повторно; их результаты воспроизводятся из durable log. Пользователь не одобряет повторно то, что уже одобрил. Уже сделанные LLM-вызовы не оплачиваются повторно.

Это тот же паттерн, который workflow engines поставляют десятилетие (Temporal, Cadence, Uber's Cherami). Новое в том, что LLM-вызовы теперь являются видом activity — недетерминированной, дорогой, с side effects — и хорошо ложатся в этот паттерн.

Сквозная тема урока: long-horizon reliability деградирует (METR наблюдает «35-minute degradation» — success rate падает примерно квадратично с горизонтом). Durable execution позволяет runs, которые длиннее, чем поддерживает профиль надежности; это новый способ отказывать безопасно, если дизайн правильный, и небезопасно, если дизайн неправильный.

## Концепция

### Activities, workflows и replay

- **Workflow**: детерминированный orchestration code. Определяет последовательность activities, branches, waits. Должен быть детерминированным, чтобы его можно было воспроизвести из event log без неожиданного расхождения.
- **Activity**: недетерминированная, потенциально падающая единица работы. LLM call, tool call, file write, HTTP request. Каждая activity логируется со своими inputs и (после завершения) outputs.
- **Event log**: durable backing store. Записываются каждый activity start, complete, fail, retry и каждое workflow decision.
- **Replay**: при восстановлении workflow code повторно запускается с начала; каждая уже завершенная activity возвращает свой залогированный результат без повторного выполнения. Фактически запускаются только activities, которые не были завершены.

Это та же форма, что React re-rendering через virtual DOM или Git, восстанавливающий working tree из commits. Детерминизм orchestrator делает durability дешевой.

### Почему LLM-вызовы подходят паттерну

LLM-вызовы:
- Недетерминированны (temperature > 0; даже temperature 0 смещается между версиями модели).
- Дороги (деньги и latency).
- Потенциально падают (rate limits, timeouts).
- Имеют side effects (если они вызывают инструменты).

Это ровно профиль activity. Оборачивание каждого LLM-вызова как activity дает retry с exponential backoff, checkpointing между рестартами и воспроизводимый trace для debugging.

### Checkpoints, ключуемые `thread_id`

LangGraph, Microsoft Agent Framework, Cloudflare Durable Objects и Claude Code Routines сошлись на одной форме API: `thread_id` (или эквивалент) идентифицирует сессию; каждый state transition сохраняется в backend (PostgreSQL по умолчанию, SQLite для dev, Redis для cache); resume читает последний checkpoint.

Выбор backend важен:

- **PostgreSQL**: durable, queryable, переживает deploys. Default для LangGraph.
- **SQLite**: только local-dev; теряет данные между hosts.
- **Redis**: быстрый, но ephemeral, если AOF/snapshot не настроены.
- **Cloudflare Durable Objects**: прозрачно distributed; scoped by a unique key; survives for hours to weeks.

### Human-input как first-class state

Propose-then-commit (Lesson 15) требует durable состояния «waiting on human». Workflow приостанавливается, внешняя queue хранит pending request, и approval возобновляет выполнение ровно с этой точки. Без durability это best-effort; с ним overnight approval приходит, и workflow продолжает утром.

### 35-minute degradation

METR наблюдала, что у каждого измеренного класса агентов надежность падает за пределами ~35 минут непрерывной работы. Удвоение длительности задачи примерно учетверяет failure rate. Durable execution этого не исправляет; оно позволяет работать дольше, чем поддерживает профиль надежности. Безопасный паттерн — сочетать durability с checkpoints, которые требуют fresh HITL on re-entry, и с budget kill switches (Lesson 13), ограничивающими total compute независимо от wall-clock time.

### Когда durable execution — неправильный ответ

- Runs короче нескольких минут без human input. Overhead > benefit.
- Строго read-only information retrieval.
- Задачи, где корректность требует end-to-end выполнения в одном context window (некоторые reasoning tasks; некоторые one-shot generation).

## Использование

`code/main.py` реализует минимальный durable-execution engine на stdlib Python. Он поддерживает:

- декоратор `@activity`, который логирует inputs и outputs в JSON event log.
- workflow function, которая упорядочивает activities.
- функцию `run_or_replay(workflow, event_log)`, которая replay завершенные activities без их повторного выполнения.

Driver симулирует workflow из трех activities, падает на середине и показывает (a) наивный retry, выполняющий все заново, против (b) replay, запускающего только недостающую activity.

## Результат

`outputs/skill-durable-execution-review.md` проверяет предлагаемое развертывание long-running agent на правильную форму durable-execution: activities, determinism, checkpoint backend, human-input state и HITL-on-resume policy.

## Упражнения

1. Запустите `code/main.py`. Наблюдайте разницу в activity-execution count между naive retry и replay. Измените crash point и покажите, что replay count меняется соответственно.

2. Переделайте toy engine так, чтобы он явно использовал `thread_id`. Симулируйте две concurrent sessions, которые делят engine, и подтвердите, что их event logs не конфликтуют.

3. Возьмите одну activity в toy engine. Внесите недетерминизм (wall-clock timestamp внутри workflow decision). Продемонстрируйте расхождение при replay. Объясните, как real engines с этим работают (side-effect registration, `Workflow.now()` APIs).

4. Прочитайте пост LangChain "Runtime behind production deep agents". Перечислите каждое состояние, которое runtime сохраняет, и назовите, какой failure mode каждое покрывает.

5. Спроектируйте checkpoint policy для 6-hour autonomous coding task. Где вы делаете checkpoint? Как выглядит resume-on-crash? Что требует fresh HITL?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| Workflow | «Скрипт агента» | Детерминированный orchestration code; воспроизводится из event log |
| Activity | «Шаг» | Недетерминированная единица (LLM call, tool call); логируется до и после |
| Event log | «Backing store» | Durable запись каждого state transition |
| Replay | «Resume» | Повторный запуск workflow; завершенные activities возвращают залогированные результаты без повторного выполнения |
| Checkpoint | «Save point» | Persisted state, ключуемый thread_id; latest-wins при resume |
| thread_id | «Session key» | Идентификатор, задающий scope durable state |
| 35-minute degradation | «Reliability decay» | METR: success rate падает ~квадратично с горизонтом |
| Non-determinism | «Drift on replay» | Wall clock, random, LLM output; должен регистрироваться как side effect |

## Дополнительное чтение

- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — budget, turns и resume semantics.
- [Microsoft — Agent Framework: human-in-the-loop and checkpointing](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — форма RequestInfoEvent.
- [LangChain — The Runtime Behind Production Deep Agents](https://www.langchain.com/conceptual-guides/runtime-behind-production-deep-agents) — конкретные runtime requirements.
- [OpenAI Agents SDK + Temporal integration (Trigger.dev announcement)](https://trigger.dev) — форма activity для LLM calls.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — ссылка на 35-minute degradation.
