# Паттерн Supervisor / Orchestrator-Worker

> Один lead agent планирует и делегирует; специализированные workers выполняют работу в параллельных контекстах и возвращают отчеты. Это паттерн за Research system Anthropic (Claude Opus 4 как lead, Sonnet 4 как subagents), где на внутренних research evals измерено +90.2% относительно single-agent Opus 4. В инженерном посте Anthropic сообщается, что 80% дисперсии на BrowseComp объясняется одной лишь token usage — multi-agent выигрывает в основном потому, что каждый subagent получает свежий context window. В этом уроке мы построим supervisor pattern из примитивов и разберем инженерные уроки 2026 года из production deployments.

**Тип:** Изучение + Build
**Языки:** Python (stdlib, `threading`)
**Предварительные требования:** Фаза 16 · 04 (Primitive Model)
**Время:** ~75 минут

## Проблема

Research — прототипическая задача, на которой single-agent системы проваливаются. Вы спрашиваете: "что изменилось в multi-agent systems между 2023 и 2026?" Single agent читает пять papers последовательно, забивает половину context их текстом, а затем должен рассуждать обо всех сразу. К моменту пятой статьи он забывает первую. Параллелить он не может.

Supervisor pattern исправляет это: один lead agent планирует поиск, делегирует каждый sub-question worker-у и синтезирует результат. Каждый worker получает собственное окно на 200k tokens для узкого вопроса. Lead никогда не видит raw papers — только summaries от workers.

Production Research system Anthropic сообщает +90.2% на внутренних research evals против single Opus 4. В том же посте отмечено, что 80% дисперсии BrowseComp объясняется *одной лишь token usage*. Fresh context на subagent — главный механизм.

## Концепция

### Паттерн

```
                 ┌──────────────┐
                 │   Lead       │  plans, decomposes,
                 │  (Opus 4)    │  synthesizes
                 └──┬────┬───┬──┘
                    │    │   │
            ┌───────┘    │   └───────┐
            ▼            ▼           ▼
      ┌─────────┐  ┌─────────┐  ┌─────────┐
      │ Worker1 │  │ Worker2 │  │ Worker3 │
      │(Sonnet) │  │(Sonnet) │  │(Sonnet) │
      └─────────┘  └─────────┘  └─────────┘
         fresh       fresh        fresh
         context     context      context
```

Lead никогда не читает raw materials. Workers не видят работу друг друга до synthesis у lead. Каждая стрелка — handoff с узким artifact.

### Почему он выигрывает

Три механизма:

1. **Fresh context per subagent.** Worker, исследующий "FIPA-ACL heritage", не несет 40k tokens, которые lead потратил на planning. Он получает окно 200k для одного вопроса.
2. **Specialization via prompt.** Prompt lead — "decompose and synthesize", а не "research". Prompt каждого worker узкий: "find what changed in X". Фокусированные prompts дают фокусированные outputs.
3. **Parallelism.** Workers запускаются concurrently. Wall-clock time примерно `max(worker_times) + plan + synthesis`, а не `sum(worker_times)`.

### Инженерные уроки (Anthropic 2025)

Пост Anthropic перечисляет несколько production lessons, актуальных и в 2026:

- **Scale effort to query complexity.** Простые queries: один agent, 3-10 tool calls. Сложные queries: 10+ agents. Оценивать это должен lead, а не caller.
- **Broad then narrow.** Сначала decomposе на широкие sub-questions, затем spawn больше workers по sub-question, если ответ требует глубины.
- **Rainbow deployments.** Agents long-running и stateful. Traditional blue-green не работает. Anthropic использует rainbow: постепенный rollout новых versions, пока старые drain.
- **Token usage dominates.** Multi-agent стоит примерно в 15× tokens от single-agent. Запускайте его только когда value задачи оправдывает cost.

### Поворот LangGraph

LangGraph изначально поставлял библиотеку `langgraph-supervisor` с high-level helper `create_supervisor`. В 2025 LangChain перенес рекомендацию на реализацию supervisor pattern напрямую через tool-calling, потому что tool calls дают больше контроля над тем, *что видит supervisor* (context engineering). Библиотека все еще работает; docs теперь рекомендуют форму tool-calling.

### Failure modes

- **Lead hallucinates the plan.** Если lead генерирует sub-questions, которые не декомпозируют реальный вопрос, workers проводят точное исследование не той цели.
- **Workers over-explore.** Без явных scope boundaries workers уходят за пределы assigned sub-question и загрязняют synthesis step.
- **Synthesis conflicts.** Два workers возвращают противоречащие факты. Lead должен либо переспросить (добавить round), либо явно отметить disagreement. Молчаливо выбрать одну сторону — худший failure: user никогда не узнает, что disagreement был.

### Когда supervisor не подходит

- **Sequential tasks.** Если step 2 буквально требует output step 1, parallelism ничего не дает. Используйте pipeline (CrewAI Sequential, LangGraph linear graph).
- **Simple queries.** Single-agent обработает их быстрее и дешевле. Используйте проверку lead "scale effort" перед spawning workers.
- **Strict determinism.** Supervisor использует LLM-selected delegation. Static graphs лучше, когда audit/replay важнее adaptability.

## Соберите это

`code/main.py` реализует supervisor из трех parallel workers с помощью `threading`. Lead decomposes query на sub-questions, workers concurrently работают над каждым sub-question, а lead synthesizes. Реальных LLM нет — workers scripted и имитируют fetch-and-summarize.

Ключевая структура:

- `Lead.plan(query)` делит query на 3 sub-questions.
- `Worker.run(sub_q)` возвращает fake summary (в production это мог бы быть любой tool-using agent).
- `Lead.run(query)` запускает workers в threads, делает joins и synthesizes.

Запустите:

```
python3 code/main.py
```

Вывод показывает plan, parallel worker traces с start/end timestamps и final synthesis. Видна победа по wall-clock: три worker по 0.3 seconds выполняются примерно за 0.35 seconds, а не за 0.9.

## Используйте это

`outputs/skill-supervisor-designer.md` принимает user query и производит supervisor-pattern design: lead system prompt, worker roles, правила decomposition на sub-questions и synthesis template. Используйте это перед построением новой research-style agent system.

## Доведите до production

Чеклист перед deployment supervisor pattern:

- **Model pairing.** Lead на reasoning-tier model (Opus class, `o3` class). Workers на более быстрой и дешевой model (Sonnet, `o4-mini`).
- **Worker timeout.** Любой worker, превышающий 2× median runtime, убивается; lead либо re-spawns его с narrower scope, либо продолжает без него.
- **Token cap per worker.** Hard limit (скажем, 10× expected synthesis input) не дает runaway worker взорвать budget.
- **Observability.** Trace plan lead, tool calls каждого worker и synthesis. Это основа любого post-hoc debugging.
- **Rainbow rollout.** Stateful long-running agents требуют gradual version transition, а не hot swap.

## Упражнения

1. Запустите `code/main.py`, затем измените lead так, чтобы он spawn-ил 5 workers вместо 3. Понаблюдайте wall-clock effect. При каком worker count spawn overhead превышает parallel savings в этом demo?
2. Реализуйте worker timeout: убивайте любого worker, который работает дольше 0.5 seconds, и пусть lead synthesizes remaining results. Какая observability нужна, чтобы знать, что worker был cut?
3. Добавьте conflict-detection step в synthesis lead: если два workers возвращают contradictory answers, lead отмечает disagreement, а не выбирает один. Как обнаружить contradiction без вызова LLM?
4. Прочитайте engineering post Anthropic о Research system. Перечислите три practices, которые этому toy demo нужно было бы принять для production.
5. Сравните LangGraph `create_supervisor` (legacy) и новую рекомендацию tool-calling. Что дает лучший контроль над тем, что видит supervisor? Почему Anthropic явно передает в synthesis только sub-answers, а не raw worker context?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Supervisor | "Lead agent" | Orchestrator agent, который планирует, делегирует и synthesizes. Сам работу не делает. |
| Worker | "Subagent" | Фокусированный agent, вызываемый supervisor с narrow scope и собственным context window. |
| Orchestrator-worker | "Supervisor pattern" | То же самое, другое имя. Литература 2026 использует оба. |
| Fresh context | "Clean window" | Context worker начинается с его system prompt и assigned question, а не с history lead. |
| Rainbow deployment | "Gradual rollout" | Long-running stateful agents требуют versioned drain-and-replace, а не blue-green. |
| Token dominance | "Context is the variable" | По Anthropic, 80% дисперсии research-eval приходит из total tokens used, а не из выбора model. |
| Scale effort | "Match agent count to complexity" | Lead оценивает difficulty query и соответственно spawn-ит 1 vs 10+ workers. |
| Synthesis conflict | "Workers disagree" | Два workers возвращают contradictory facts; lead должен показать disagreement, а не молча выбрать один. |

## Дополнительное чтение

- [Anthropic engineering — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — production reference для supervisor pattern
- [LangGraph workflows and agents](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — tool-calling supervisor теперь recommended form
- [LangGraph supervisor reference](https://reference.langchain.com/python/langgraph-supervisor) — legacy helper, все еще используется в production 2026
- [OpenAI cookbook — Orchestrating Agents: Routines and Handoffs](https://developers.openai.com/cookbook/examples/orchestrating_agents) — handoff-based supervisor variant
