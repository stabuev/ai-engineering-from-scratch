# Group Chat и Speaker Selection

> AutoGen GroupChat и AG2 GroupChat используют один общий conversation для N agents; selector function (LLM, round-robin или custom) выбирает, кто говорит следующим. Это архетип emergent multi-agent conversation - agents не знают своей роли в static graph, они просто реагируют на shared pool. Семантика GroupChat в AutoGen v0.2 была сохранена в форке AG2; AutoGen v0.4 переписал ее как event-driven actor model. Microsoft перевела AutoGen в maintenance mode в феврале 2026 года и объединила его с Semantic Kernel в Microsoft Agent Framework (RC February 2026). Примитив GroupChat сохраняется и в AG2, и в Microsoft Agent Framework - изучите один раз, используйте везде.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib)
**Пререквизиты:** Phase 16 · 04 (Primitive Model)
**Время:** ~60 минут

## Проблема

Static graphs (LangGraph) отличны, когда workflow известен. Реальные conversations не static: иногда coder спрашивает reviewer, иногда researcher, иногда writer. Жестко прописывать every possible handoff создает edge explosion. Вам нужны *agents reacting to a shared pool*, с некоторой function, которая решает, кто говорит следующим.

Именно это делает AutoGen GroupChat.

## Концепция

### Форма

```
              ┌─── shared pool ────┐
              │   m1  m2  m3  ...  │
              └─────────┬──────────┘
                        │ (everyone reads all)
      ┌───────┬─────────┼─────────┬───────┐
      ▼       ▼         ▼         ▼       ▼
    Agent A  Agent B  Agent C  Agent D  Selector
                                           │
                                           ▼
                                  "next speaker = C"
```

Каждый agent видит каждое message. Selector function вызывается на каждом turn, чтобы выбрать, кто говорит следующим.

### Три варианта selector

**Round-robin.** Fixed cycle. Deterministic. Масштабируется линейно по N, но игнорирует context - coder получает ход даже тогда, когда тема legal review.

**LLM-selected.** Вызов LLM, которая читает recent pool и возвращает лучшего next speaker. Context-aware, но slow: каждый turn добавляет LLM call. Default в AutoGen.

**Custom.** Python function с любой нужной логикой. Типично: LLM-selected with fallback rules (например, "always give the verifier the turn after the coder").

### ConversableAgent API

```
agent = ConversableAgent(
    name="coder",
    system_message="You write Python.",
    llm_config={...},
)
chat = GroupChat(agents=[coder, reviewer, tester], messages=[])
manager = GroupChatManager(groupchat=chat, llm_config={...})
```

`GroupChatManager` содержит selector. Когда agent завершает turn, manager вызывает selector, который возвращает next agent. Loop продолжается до termination condition.

### Termination

Три распространенных паттерна:

- **Max rounds.** Hard cap на total turns.
- **"TERMINATE" token.** Agents могут выдать sentinel message; manager останавливается, когда оно появляется.
- **Goal-reached check.** Lightweight verifier запускается на каждом turn и останавливает chat, когда задача выполнена.

### Разделение AutoGen → AG2 и слияние с Microsoft Agent Framework

В начале 2025 года Microsoft начала крупную переработку AutoGen (v0.4) вокруг event-driven actor model. Сообщество форкнуло семантику GroupChat из AutoGen v0.2 как AG2, сохранив API, который early adopters уже интегрировали.

В феврале 2026 года Microsoft объявила, что AutoGen перейдет в maintenance mode, а event-driven actor model войдет в **Microsoft Agent Framework** (RC February 2026, now merged with Semantic Kernel). Концепция GroupChat сохраняется в обеих ветках; детали реализации отличаются. AG2 - предпочтительный upstream для v0.2-compatible code.

### Когда GroupChat подходит

- **Emergent conversations.** Вы не хотите заранее wiring every possible next-speaker.
- **Role-mixing tasks.** Coder спрашивает researcher, researcher спрашивает archivist, archivist спрашивает coder в ответ. Поток не является DAG.
- **Exploratory problem-solving.** Думайте "brainstorm meeting", а не "assembly line."

### Когда это проваливается

- **Строгий determinism.** LLM selector может быть inconsistent. Тот же prompt, разные runs, разные next speakers.
- **Sycophancy cascades.** Agents уступают тому, кто говорил наиболее уверенно. Counter-prompt explicitly.
- **Context bloat.** Каждый agent читает каждое message; после 10 turns context огромен. Используйте projections (Lesson 15), чтобы scope views.
- **Hot speakers.** Один agent доминирует conversation, потому что selector предпочитает его specialties. Введите speaker balance как feature selector.

### Group chat vs supervisor

Те же primitives, другие defaults:

- Supervisor: один agent планирует, остальные execute. Selector - "ask the planner what to do."
- Group chat: все agents - peers; selector - function over the shared pool.

Оба используют четыре primitives из Lesson 04. Group chat по умолчанию использует LLM-selected orchestration и full-pool shared state.

## Соберите это

`code/main.py` реализует GroupChat from scratch на stdlib. Три agents (coder, reviewer, manager), варианты round-robin и LLM-selected, и termination на `TERMINATE` token.

Демо печатает conversation transcript плюс decision trace selector для обоих вариантов.

Запуск:

```
python3 code/main.py
```

## Используйте это

`outputs/skill-groupchat-selector.md` настраивает GroupChat selector для заданной задачи - round-robin vs LLM-selected vs custom, и какие selector inputs (recent messages, agent specialties, turn counts) использовать.

## Доведите до production

Чеклист:

- **Max rounds cap.** Всегда. 10-20 для типичных задач.
- **Speaker-balance metric.** Отслеживайте turns per agent; alert, когда imbalance exceeds a threshold.
- **Termination token.** `TERMINATE` или dedicated verifier agent.
- **Projection or scoped memory.** После ~10 messages подумайте о том, чтобы давать каждому agent только scoped view для предотвращения context bloat.
- **Selector logging.** Для LLM-selected variants логируйте и input selector, и его choice. Иначе debugging невозможен.

## Упражнения

1. Запустите `code/main.py`. Сравните conversation при round-robin vs LLM-selected. Какой agent доминирует в каждом случае?
2. Добавьте правило "max-speaks-per-agent" в selector. Как оно влияет на transcript?
3. Реализуйте goal-reached termination: остановить, когда reviewer возвращает "approved." Как часто это срабатывает до round cap?
4. Прочитайте stable docs AutoGen по GroupChat (https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html). Определите default selector, используемый `GroupChatManager`.
5. Прочитайте repo AG2 (https://github.com/ag2ai/ag2) и сравните его v0.2 GroupChat с event-driven version v0.4. Какое конкретное свойство (throughput, fault-tolerance, composability) добавляет v0.4?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| GroupChat | "Agents in one chat room" | Shared message pool + selector function. Примитив AutoGen / AG2. |
| Speaker selection | "Кто говорит следующим" | Function, выбирающая next agent. Round-robin, LLM-selected или custom. |
| GroupChatManager | "Ведущий встречи" | Компонент AutoGen, который owns selector и loops over turns. |
| ConversableAgent | "Базовый agent" | Базовый class AutoGen; agent, который может send and receive messages. |
| Termination token | "Слово 'stop'" | Sentinel string (обычно `TERMINATE`), завершающая chat. |
| Hot speaker | "Один agent доминирует" | Failure mode, где selector продолжает выбирать одного и того же agent. |
| Context bloat | "Pool grows unbounded" | Каждый agent читает каждое prior message; context grows with turns. |
| Projection | "Scoped view" | Role-specific view into the shared pool для предотвращения context bloat. |

## Дополнительное чтение

- [AutoGen group chat docs](https://microsoft.github.io/autogen/stable/user-guide/core-user-guide/design-patterns/group-chat.html) — reference implementation
- [AG2 repo](https://github.com/ag2ai/ag2) — community AutoGen v0.2 continuation
- [Microsoft Agent Framework docs](https://microsoft.github.io/agent-framework/) — merged successor, RC February 2026
- [AutoGen v0.4 release notes](https://microsoft.github.io/autogen/stable/) — детали event-driven actor model rewrite
