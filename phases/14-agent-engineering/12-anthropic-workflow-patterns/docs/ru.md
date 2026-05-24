# Workflow-паттерны Anthropic: простое важнее сложного

> Schluntz and Zhang (Anthropic, Dec 2024) различают workflows (предопределенные пути) и agents (динамическое использование tools). Пять workflow-паттернов покрывают большинство случаев. Начинайте с прямых API calls. Добавляйте agents только тогда, когда шаги нельзя предсказать.

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop)
**Время:** ~60 минут

## Цели обучения

- Назвать пять workflow-паттернов Anthropic: prompt chaining, routing, parallelization, orchestrator-workers, evaluator-optimizer.
- Объяснить различие agent-vs-workflow и инженерную стоимость каждого.
- Определить, когда выбирать workflow вместо agent (и наоборот).
- Реализовать все пять паттернов на stdlib поверх scripted LLM.

## Проблема

Команды тянутся к multi-agent frameworks для задач, которым нужен один function call. Стоимость реальна: frameworks добавляют слои, которые затуманивают prompts, прячут control flow и провоцируют преждевременную сложность. Пост Schluntz and Zhang от Dec 2024 — самый цитируемый отраслевой контраргумент: начинайте просто, добавляйте сложность только когда она окупает свою стоимость.

## Концепция

### Workflows vs agents

- **Workflow.** LLMs и tools оркестрируются через предопределенные code paths. Engineers владеют graph.
- **Agent.** LLMs динамически направляют собственные tools и делают собственные steps. Model владеет graph.

У обоих есть место. Workflows дешевле, быстрее и проще отлаживать. Agents открывают open-ended problems, но усложняют reasoning о failure modes.

### Augmented LLM

Основа всех пяти паттернов: одна LLM с тремя подключенными capabilities — search (retrieval), tools (actions), memory (persistence). Любой API call может их использовать.

### Пять паттернов

1. **Prompt chaining.** Output call 1 становится input call 2. Используйте, когда задача имеет чистую линейную декомпозицию. Между steps можно ставить optional programmatic gates.

2. **Routing.** Classifier LLM выбирает, какую downstream LLM или tool вызвать. Используйте, когда категориально разные inputs требуют разной обработки (tier-1 support vs refund vs bug vs sales).

3. **Parallelization.** Запустить N LLM calls конкурентно, агрегировать results. Две формы: sectioning (разные chunks) и voting (один prompt, N runs, majority/synthesis).

4. **Orchestrator-workers.** Orchestrator LLM динамически решает, каких workers (тоже LLMs) запускать, и синтезирует их output. Похоже на agent loops, но orchestrator не зацикливается бесконечно.

5. **Evaluator-optimizer.** Одна LLM предлагает answer, другая LLM его оценивает. Итерировать, пока evaluator не пропустит. Это обобщенный Self-Refine (Lesson 05).

### Где workflows лучше agents

- **Predictable tasks.** Если вы можете перечислить steps, стоит это сделать.
- **Cost-bound tasks.** У workflows ограниченное число steps; agents могут раскрутиться.
- **Compliance-bound tasks.** Auditors хотят читать graph, а не выводить его из trajectories.

### Где agents лучше workflows

- **Open-ended research.** Когда следующий step зависит от того, что вернул предыдущий.
- **Variable-length tasks.** Минуты или часы работы, где число steps неизвестно.
- **Novel domains.** Когда вы еще не знаете правильный workflow — сначала exploration, потом codify.

### Сопутствующая context engineering

"Effective context engineering for AI agents" (Anthropic 2025) формализует соседнюю дисциплину: окно 200k — это бюджет, а не контейнер. Что включать, когда compact, когда позволять context расти. Подробно разбирается в уроке Phase 14 о context compression (более ранний lesson 06 в этой curriculum до перенумерации).

## Соберите это

`code/main.py` реализует все пять workflow-паттернов поверх `ScriptedLLM`:

- `prompt_chain(input, steps)` — sequential.
- `route(input, classifier, handlers)` — classification + dispatch.
- `parallel_vote(prompt, n, aggregator)` — N runs, aggregate.
- `orchestrator_workers(task, workers)` — orchestrator picks workers.
- `evaluator_optimizer(task, proposer, evaluator, max_iter)` — loop until pass.

Запустите:

```
python3 code/main.py
```

Каждый паттерн печатает свою трассу. Общее число строк кода на паттерн — ~10-15; стоимость framework измеряется тысячами.

## Используйте это

- Прямые API calls для большинства задач.
- Framework только когда паттерну действительно нужны durable state (LangGraph), actor-model concurrency (AutoGen v0.4) или role templating (CrewAI).
- Беритесь за Claude Agent SDK, когда нужна форма harness Claude Code без rebuilding it.

## Доведите до продакшена

`outputs/skill-workflow-picker.md` выбирает правильный паттерн для заданного task description, включая rationale решения и refactor path к agent, если workflows недостаточны.

## Упражнения

1. Реализуйте routing с confidence threshold. Ниже threshold -> escalate to human. Где окажется threshold для tier-1 support use case?
2. Добавьте timeout в `parallel_vote`. Что происходит, когда один call зависает? Как aggregate with missing votes?
3. Превратите `evaluator_optimizer` в bandit: сохраняйте top-2 outputs между iterations, чтобы поздний хороший результат не был overwritten by a late bad one.
4. Скомбинируйте prompt chaining с routing: router выбирает одну из трех chains. Измерьте token cost против альтернативы с одним big-prompt.
5. Выберите одну production feature. Нарисуйте workflow graph. Посчитайте steps. Был бы agent здесь реально лучше?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Workflow | "Предопределенный flow" | Engineer-owned graph из LLM and tool calls |
| Agent | "Автономный AI" | Model-owned graph; dynamic tool direction |
| Augmented LLM | "LLM with tools" | LLM + search + tools + memory; atomic unit |
| Prompt chaining | "Sequential calls" | Output call N является input call N+1 |
| Routing | "Classifier dispatch" | Выбрать, какая chain/model обработает input |
| Parallelization | "Fan out" | N concurrent calls; aggregate by sectioning or voting |
| Orchestrator-workers | "Dispatcher agent" | Orchestrator LLM динамически выбирает specialist LLMs |
| Evaluator-optimizer | "Proposer + judge" | Итерировать, пока evaluator passes; обобщенный Self-Refine |

## Дополнительное чтение

- [Anthropic, Building Effective Agents (Dec 2024)](https://www.anthropic.com/research/building-effective-agents) — пять workflow-паттернов
- [Anthropic, Effective context engineering for AI agents](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — сопутствующая дисциплина
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — когда stateful graphs окупают свою стоимость
- [OpenAI Agents SDK](https://openai.github.io/openai-agents-python/) — productized паттерн orchestrator-workers
