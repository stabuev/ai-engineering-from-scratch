# Human-in-the-Loop: Propose-Then-Commit

> Консенсус 2026 года по HITL конкретен. Это не "агент спрашивает, пользователь нажимает Approve." Это propose-then-commit: предлагаемое действие сохраняется в durable store с idempotency key; показывается reviewer вместе с intent, data lineage, затронутыми permissions, blast radius и rollback plan; commit выполняется только после positive acknowledgement; после выполнения идет verification, чтобы подтвердить, что side effect действительно произошел. `interrupt()` в LangGraph плюс PostgreSQL checkpointing, `RequestInfoEvent` в Microsoft Agent Framework и `waitForApproval()` в Cloudflare реализуют одну и ту же форму. Канонический режим отказа — rubber-stamp approval: "Approve?" кликают без review. Задокументированная mitigation — challenge-and-response с явным checklist.

**Тип:** Изучение
**Языки:** Python (stdlib, state machine propose-then-commit с idempotency)
**Предварительные требования:** Фаза 15 · 12 (Надежное выполнение), Фаза 15 · 14 (Tripwires)
**Время:** ~60 минут

## Цели обучения

- Объяснять паттерн HITL propose-then-commit и почему он лучше «спросить-и-кликнуть».
- Сохранять предложенное действие в durable-состояние до коммита.
- Рассуждать об аудируемости предложенных и закоммиченных действий.

## Проблема

Агент выполняет действие. Пользователь должен решить: одобрить или нет. Если решение мгновенное, это, вероятно, не review. Если решение структурированное, оно медленное, но заслуживает доверия. Инженерный вопрос в том, как сделать структурированное review путем наименьшего сопротивления.

HITL-паттерн эпохи 2023 года был синхронным prompt: "Агент хочет отправить письмо X с текстом Y — одобрить?" Пользователь нажимает Approve. Всем кажется, что система безопасна. На практике эту поверхность часто rubber-stamp: пользователи одобряют быстро, approvals мало что предсказывают, а когда агент ошибается, audit trail показывает длинную историю approvals, которые пользователь не может вспомнить.

Паттерн 2026 года — propose-then-commit — переносит HITL на durable substrate, добавляет структурированные metadata и требует positive commit. Каждый managed agent SDK поставляет свою версию: LangGraph `interrupt()`, Microsoft Agent Framework `RequestInfoEvent`, Cloudflare `waitForApproval()`. Имена API различаются; форма — нет.

## Концепция

### State machine propose-then-commit

1. **Propose.** Агент создает предлагаемое действие. Оно сохраняется в durable store (PostgreSQL, Redis, Durable Object). Включает:
   - intent (зачем агент это делает)
   - data lineage (какой источник привел к этому proposal)
   - permissions touched (какие scopes / files / endpoints)
   - blast radius (худший сценарий)
   - rollback plan (если committed, как это отменить)
   - idempotency key (уникальный для proposal; повторная отправка возвращает ту же запись)
2. **Surface.** Reviewer видит proposal со всеми metadata. Reviewer — человек (не агент, проверяющий сам себя).
3. **Commit.** Positive acknowledgement. Действие выполняется.
4. **Verify.** После выполнения side effect считывается обратно и подтверждается. Если шаг verify не проходит, система находится в известном плохом состоянии и включается alerting.

### Idempotency key

Без idempotency key retry после transient failure может дважды выполнить одобренное действие. Конкретный пример: пользователь approve "transfer $100 from A to B." Network blips. Workflow retries. Пользователь одобрил один раз, но transfer выполняется дважды. Idempotency key привязывает approval к одному уникальному side effect; второе выполнение — no-op.

Это тот же паттерн idempotency, который используют Stripe и AWS APIs. Его повторное использование для agent approvals явно указано в документации Microsoft Agent Framework.

### Durability: почему approvals переживают процессы

Approval waiting room — это часть состояния, которой агент не владеет. Workflow приостановлен (Урок 12). Когда approval приходит, workflow возобновляется ровно с этой точки. Поэтому LangGraph сочетает `interrupt()` с PostgreSQL checkpointing, а не просто с in-memory state — approval через два дня все еще находит workflow целым.

### Rubber-stamp approvals и mitigation challenge-and-response

UI по умолчанию для HITL (кнопки "Approve" / "Reject") производит быстрые approvals без настоящего review. Задокументированная mitigation: checklist challenge-and-response, требующий положительных ответов на конкретные вопросы перед тем, как кнопка Approve станет доступной. Конкретная форма:

- "Вы понимаете, какой ресурс затрагивает это действие? [ ]"
- "Вы проверили, что blast radius приемлем? [ ]"
- "У вас есть rollback plan на случай сбоя? [ ]"

Это не бюрократия ради бюрократии, а forcing function. Reviewer, который не может поставить галочки, либо просит clarification (escalation), либо отклоняет действие (safe default). Исследования Anthropic по безопасности агентов прямо цитируют checklist-driven HITL как mitigation для rubber-stamp approval patterns.

### Что считается consequential

Не каждое действие требует propose-then-commit. Guidance 2026 года:

- **Consequential actions** (всегда HITL): необратимые записи, финансовые транзакции, исходящие коммуникации, изменения production-базы данных, разрушительные операции с файловой системой.
- **Reversible actions** (иногда HITL): правки локальных файлов, изменения staging-окружения, обратимые записи с ясным rollback.
- **Reads and inspections** (никогда HITL): чтение файла, перечисление ресурсов, вызов read-only API.

### Post-action verification

"Commit ran" не то же самое, что "side effect happened." Network partition и race conditions могут породить workflow, который считает, что завершился успешно, хотя backend ничего не сохранил. Шаг verify заново читает целевой ресурс после commit для подтверждения. Это тот же паттерн, что database transactions с `RETURNING` clauses или AWS `GetObject` после `PutObject`.

### EU AI Act Article 14

Article 14 требует effective human oversight для high-risk AI systems в ЕС. "Effective" — не декоративное слово. Язык регулирования прямо исключает rubber-stamp patterns. Propose-then-commit с challenge-and-response — форма, которая выдерживает проверку Article 14 в compliance docs Microsoft Agent Governance Toolkit.

## Используйте это

`code/main.py` реализует state machine propose-then-commit на stdlib Python. Durable store — JSON-файл. Idempotency key — hash от (thread_id, action_signature). Driver симулирует три случая: чистый approval flow, retry после transient failure (который не должен double-execute), и rubber-stamp default versus challenge-and-response flow.

## Отгрузите это

`outputs/skill-hitl-design.md` проверяет предлагаемый HITL workflow на форму propose-then-commit и отмечает отсутствующие metadata, idempotency, verification или слои challenge-and-response.

## Упражнения

1. Запустите `code/main.py`. Убедитесь, что retry одобренного proposal использует durable record и не re-execute. Теперь измените idempotency key так, чтобы он включал timestamp, и покажите, что retry выполняется дважды.

2. Расширьте proposal record полем `rollback`. Симулируйте выполнение, у которого verify step fails. Покажите, что rollback fires automatically.

3. Прочитайте документацию `RequestInfoEvent` в Microsoft Agent Framework. Найдите одно metadata field, которое API включает, а toy engine пропускает. Добавьте его и объясните, от чего оно защищает.

4. Спроектируйте checklist challenge-and-response для конкретного действия (например, "post to a public Twitter account"). На какие три вопроса reviewer должен ответить? Почему именно эти три?

5. Выберите один случай, где синхронного prompt "Approve?" было бы достаточно (durable store не нужен). Объясните почему и назовите risk class, который вы принимаете.

## Ключевые термины

| Термин | Как обычно говорят | Что это означает на самом деле |
|---|---|---|
| Propose-then-commit | "Двухфазное одобрение" | Сохраненный proposal + positive commit + verify |
| Idempotency key | "Retry-safe token" | Уникален для proposal; второе выполнение становится no-op |
| Data lineage | "Откуда это пришло" | Конкретный source content, который привел к proposal |
| Blast radius | "Худший сценарий" | Область эффекта, если действие пойдет не так |
| Rubber-stamp | "Быстрое одобрение" | Нажатие "Approve" без настоящего review |
| Challenge-and-response | "Принудительный checklist" | Reviewer должен явно подтвердить конкретные вопросы |
| RequestInfoEvent | "Примитив MS Agent Framework" | Durable HITL request со структурированными metadata |
| `interrupt()` / `waitForApproval()` | "Примитивы фреймворков" | Эквиваленты той же формы в LangGraph / Cloudflare |

## Дополнительное чтение

- [Microsoft Agent Framework — Human in the loop](https://learn.microsoft.com/en-us/agent-framework/workflows/human-in-the-loop) — `RequestInfoEvent`, durable approvals.
- [Cloudflare Agents — Human in the loop](https://developers.cloudflare.com/agents/concepts/human-in-the-loop/) — `waitForApproval()` и Durable Objects.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — HITL как mitigation для long-horizon risk.
- [EU AI Act — Article 14: Human oversight](https://artificialintelligenceact.eu/article/14/) — regulatory baseline для high-risk systems.
- [Anthropic — Claude's Constitution (January 2026)](https://www.anthropic.com/news/claudes-constitution) — constitutional framing вокруг надзора.
