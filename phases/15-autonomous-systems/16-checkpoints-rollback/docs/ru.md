# Checkpoints and Rollback

> Каждый graph-state transition сохраняется. Когда worker падает, его lease истекает, и другой worker подхватывает выполнение с последнего checkpoint. Cloudflare Durable Objects удерживают состояние часами или неделями. Propose-then-commit (Урок 15) определяет rollback plan для каждого действия. Post-action verification замыкает цикл. EU AI Act Article 14 делает effective human oversight обязательным для high-risk systems — на практике это означает, что checkpoints должны быть доступны для запросов, rollbacks должны быть отрепетированы, а audit trail должен переживать deploy. Острый режим отказа: без idempotency keys и precondition checks retry после transient failure может double-execute уже одобренное действие. Post-action verification — то, что это ловит.

**Тип:** Изучение
**Языки:** Python (stdlib, state machine checkpoint and rollback)
**Предварительные требования:** Фаза 15 · 12 (Надежное выполнение), Фаза 15 · 15 (Propose-then-commit)
**Время:** ~60 минут

## Цели обучения

- Объяснять, как сохранение каждого перехода состояния графа дает восстановление после сбоя и откат.
- Реализовывать передачу работы между воркерами по lease на последнем чекпоинте.
- Рассуждать о семантике exactly-once для действий агента.

## Проблема

Durable execution (Урок 12) делает упавшего агента возобновляемым. Propose-then-commit (Урок 15) делает одобренное действие аудируемым. Этот урок соединяет их: что происходит, когда одобренное действие выполняется частично, падает и возобновляется? Когда запускается rollback и относительно какого состояния?

Реальные системы связывают это по-разному:

- **LangGraph** сохраняет checkpoint для каждого graph-state transition в PostgreSQL. При падении worker lease освобождается, и другой worker возобновляет выполнение с последнего checkpoint. Workflows приостанавливаются на `interrupt()`, который сам сохраняется.
- **Cloudflare Durable Objects** хранят per-key state часами или неделями. Размещайте вычисление рядом с storage для одобренного действия.
- **Microsoft Agent Framework** предоставляет primitives `Checkpoint` в workflow API; replay плюс idempotency покрывают retries.

В каждом случае комбинация, которая действительно работает: idempotency key (предотвращает double-execute) + precondition check (state все еще тот, против которого мы дали approval) + post-action verify (side effect действительно произошел) + rollback on verify-fail.

## Концепция

### Каждый transition сохраняется

Graph-state transition — это любой шаг, переводящий workflow из одного именованного состояния в другое. Наивные реализации сохраняются только в конкретных commit points; production implementations сохраняют каждый transition. Стоимость (несколько дополнительных writes) мала относительно выигрыша в надежности (replay попадает куда угодно, lease recovery точен).

### Lease recovery

Когда worker падает, workflow не теряется; lease (краткоживущая claim, что этот worker выполняет этот run) просто истекает. Другой worker подхватывает последний checkpoint и продолжает. Lease mechanism позволяет production systems переживать rolling deploys без потери in-flight work.

### Idempotency плюс preconditions

Одной idempotency недостаточно. Рассмотрим: workflow одобрен для "перевести $100 из A в B, когда баланс > $1000." Workflow committed, падает в середине выполнения и возобновляется. Если проверяется только idempotency key, и выполнение возобновляется, transfer выполняется один раз (корректно). Но предположим, что между crash и resume баланс A падает до $500 через другой workflow. Idempotency check все еще проходит; precondition — нет. Без precondition check мы отправляем overdraft.

Каждому consequential action нужны оба:

- **Idempotency key**: предотвращает double-execute.
- **Precondition check**: подтверждает, что state все еще согласован с одобренным action.

### Post-action verification

"Tool returned 200" — это не verification. Настоящая verification заново читает целевое состояние и подтверждает, что side effect действительно произошел. Паттерны:

- Database update: `UPDATE ... RETURNING *`, затем assert, что returned row соответствует intended state.
- Email send: проверить папку отправленных по message ID после submission.
- File write: прочитать файл обратно и захэшировать его.
- API call: последующий `GET` по target resource.

Если verify fails, workflow находится в известном плохом состоянии. Включается rollback.

### Rollback plans

Каждое consequential action в propose-then-commit (Урок 15) несет rollback plan. Типы:

- **In-band rollback**: напрямую обратить side effect (`DELETE` after `INSERT`, `Send-correction-email` after send).
- **Compensating transaction**: новое действие, которое нейтрализует исходное (стандартный SAGA pattern).
- **Out-of-band rollback**: alert human, pause workflow, leave the bad state for investigation.

No-op rollback ("we cannot undo this") должен быть назван в proposal. Actions без rollback требуют более строгого HITL во время commit (Урок 15 challenge-and-response).

### Операционное прочтение EU AI Act Article 14

Article 14 требует "effective human oversight" для high-risk systems. В операционных терминах implementers читают это так:

- Checkpoints доступны аудитору для запросов.
- Rollbacks отрепетированы (tested end-to-end хотя бы один раз).
- Audit trail переживает deploy (checkpoint backend не ephemeral).
- Failed verifications вызывают alert, а не молча log.

Workflow, который падает mid-commit, возобновляется и завершает side effect без verify + rollback pathway, не проходит тест Article 14.

### Острый режим отказа: double-execute

Самый распространенный production incident в этой области:

1. Action approved, idempotency key k.
2. Commit starts, executes, returns 200.
3. Workflow crashes before persisting the "committed" status.
4. Workflow resumes; sees "approved but not committed"; re-executes.
5. Side effect fires twice.

Mitigation: сохранять "in-flight" intent до выполнения, выполнять с idempotency key, затем помечать "committed" только после успешной post-action verification. Если action fires, а status write fails, вы знаете, что нужно verify и (если нужно) re-fire. Если status write succeeds, а action fails, вы verify и fire exactly once через recovery path.

## Используйте это

`code/main.py` реализует checkpointed workflow с idempotency, preconditions, verify и rollback. Driver симулирует четыре сценария: clean run, retry after crash (idempotency catches), precondition fail (workflow aborts without firing), verify fail (rollback fires).

## Отгрузите это

`outputs/skill-rollback-rehearsal.md` проектирует rollback-rehearsal test для предлагаемого workflow и аудирует checkpoint backend на persistence audit-trail.

## Упражнения

1. Запустите `code/main.py`. Проверьте четыре сценария. Для случая crash-during-commit убедитесь, что action fires exactly once across retries.

2. Измените паттерн "сначала пометить выполненным, затем выполнить" так, чтобы status write срабатывал после action. Перезапустите crash scenario. Измерьте, сколько duplicate actions fires.

3. Спроектируйте rollback plan для конкретного production action (например, "post to a Slack channel"). Классифицируйте как in-band, compensating или out-of-band. Обоснуйте выбор.

4. Возьмите один знакомый вам workflow. Найдите каждое state transition. Пометьте каждое durability requirement (persist / do not persist). Посчитайте те, которые вы сейчас не сохраняете.

5. Rehearsed-rollback test: спроектируйте end-to-end test, который запускает real workflow, crashes it и подтверждает, что rollback path fires. Что проверяет test?

## Ключевые термины

| Термин | Как обычно говорят | Что это означает на самом деле |
|---|---|---|
| Checkpoint | "Точка сохранения" | Каждое graph-state transition сохраняется в durable store |
| Lease | "Claim worker" | Краткоживущая claim, что worker выполняет run; истекает при crash |
| Precondition | "Шлюз состояния" | Assertion, что state все еще согласован с approved action |
| Post-action verify | "Повторное чтение" | Подтвердить, что side effect действительно произошел в target system |
| In-band rollback | "Прямой откат" | Обратить side effect обратной операцией |
| Compensating transaction | "SAGA-откат" | Новое action, которое нейтрализует original |
| Mark-as-done-first | "Порядок записи статуса" | Persist committed status before returning from commit |
| Article 14 | "Человеческий надзор по EU AI Act" | Операционно: queryable checkpoints, rehearsed rollbacks, auditable trail |

## Дополнительное чтение

- [Microsoft Agent Framework — Checkpointing and HITL](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — checkpoint primitives и lease recovery.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — Durable Objects как state substrate.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — reliability framing для long-horizon workflows.
- [Anthropic — Claude Code Agent SDK: agent loop](https://code.claude.com/docs/en/agent-sdk/agent-loop) — workflow shape для Claude Code Routines.
