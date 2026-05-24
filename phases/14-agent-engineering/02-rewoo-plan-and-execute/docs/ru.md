# ReWOO и Plan-and-Execute: разделённое планирование

> ReAct чередует мысль и действие в одном потоке. ReWOO разделяет их: сначала один большой план, затем выполнение. В 5 раз меньше токенов, +4% accuracy на HotpotQA, и planner можно дистиллировать в 7B model. Plan-and-Execute обобщил этот подход; Plan-and-Act масштабировал его на web navigation.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop)
**Время:** ~60 минут

## Цели обучения

- Объяснить, почему разделение Planner / Worker / Solver в ReWOO экономит токены и повышает robustness относительно interleaved loop ReAct.
- Реализовать plan DAG, executor по порядку зависимостей и solver, который композирует worker outputs, — всё на stdlib.
- Решать, когда задачу нужно выполнять как plan-then-execute, а когда как interleaved ReAct, используя framing 2026 года "five workflow patterns" (Anthropic).
- Распознавать, когда synthetic plan data из Plan-and-Act нужна для long-horizon web или mobile tasks.

## Проблема

Interleaved thought-action-observation loop ReAct прост и гибок, но каждый tool call должен нести весь предыдущий контекст — включая каждую предыдущую мысль. Использование токенов растёт квадратично с глубиной. Хуже того: когда tool fails посреди цикла, модель должна заново вывести весь план из error observation.

ReWOO (Xu et al., arXiv:2305.18323, May 2023) заметил это и сделал ставку: спланировать всё заранее, получить evidence параллельно, скомпоновать ответ в конце. Один LLM call для планирования, N tool calls для evidence (можно parallel), один LLM call для решения. Обмен: меньше гибкости (план статичен) на намного лучшую token efficiency и более ясные failure modes.

## Концепция

### Три роли

```
Planner:  user_question -> [plan_dag]
Workers:  [plan_dag]     -> [evidence]        (tool calls, possibly parallel)
Solver:   user_question, plan_dag, evidence -> final_answer
```

Planner создаёт DAG. Каждый node указывает tool, его arguments и зависимости от более ранних nodes (ссылки вроде `#E1`, `#E2`). Workers выполняют nodes в topological order. Solver сшивает всё вместе.

### Почему токенов в 5 раз меньше

В ReAct длина prompt растёт линейно с числом шагов. На шаге 10 prompt содержит thought 1 плюс action 1 плюс observation 1 плюс thought 2 плюс action 2 плюс observation 2 и так далее. Каждый промежуточный шаг также избыточно включает исходный prompt.

ReWOO платит за один planner prompt (большой), N маленьких worker prompts (каждый только tool call, без chain) и один solver prompt. На HotpotQA статья измеряет примерно 5x fewer tokens при +4 absolute accuracy.

### Почему он более robust

Если worker 3 падает в ReAct, циклу нужно рассуждать из ошибки прямо mid-stream. В ReWOO worker 3 возвращает error string; solver видит её в контексте исходного плана и может gracefully degrade. Failure localization — per-node, а не per-step.

### Дистилляция planner

Второй результат статьи: поскольку planner не видит observations, можно fine-tune 7B model на planner outputs от 175B teacher. Маленькая модель справляется с planning; большая модель не нужна на inference. Теперь это стандарт — многие production agents 2026 года используют small planner и big executor или наоборот.

### Plan-and-Execute (LangChain, 2023)

Пост команды LangChain за август 2023 обобщил ReWOO в имя паттерна: Plan-and-Execute. Up-front planner выдаёт список шагов, executor выполняет каждый шаг, optional replanner может пересмотреть план после наблюдения результатов. Это ближе к ReAct, чем ReWOO (replanner возвращает observations обратно в planning), но сохраняет token savings.

### Plan-and-Act (Erdogan et al., arXiv:2503.09572, ICML 2025)

Plan-and-Act масштабирует паттерн на long-horizon web и mobile agents. Ключевой вклад — synthetic plan data: labeled trajectory generator создаёт training data, где plan явный. Это используется для fine-tune planner models, которые продолжают работать после 30–50 шагов на WebArena-like tasks, где single ReAct trajectory теряет coherence.

### Что выбирать

| Паттерн | Когда использовать |
|---------|------|
| ReAct | Короткие задачи, неизвестная среда, нужна reactive exception handling |
| ReWOO | Структурированные задачи с known tools, token-sensitive, parallelizable evidence |
| Plan-and-Execute | Как ReWOO, но с replanning после partial execution |
| Plan-and-Act | Длинный горизонт (>30 шагов), web/mobile/computer-use |
| Tree of Thoughts | Search стоит своей цены (Lesson 04) |

Рекомендация Anthropic Dec 2024: начинайте с самого простого. Если задача — один tool call плюс summary, не стройте ReWOO. Если задача — research assignment на 40 шагов, не используйте один ReAct.

## Соберите это

`code/main.py` реализует игрушечный ReWOO:

- `Planner` — scripted policy, которая выдаёт plan DAG из prompt.
- `Worker` — dispatches tool call каждого node через registry.
- `Solver` — scripted composition, которая читает evidence и создаёт final answer.
- Dependency resolution — ссылки вроде `#E1` подставляются ранними worker outputs.

Демо отвечает на "What is the population of the capital of France, rounded to millions?" через двухшаговый план: (1) найти capital, (2) найти population, затем solve.

Запустите:

```
python3 code/main.py
```

Trace сначала показывает полный plan, затем worker results, затем solver composition. Сравните token count (мы печатаем rough character count) с ReAct-style interleaved run — ReWOO выигрывает на таком типе structured task.

## Используйте это

LangGraph поставляет Plan-and-Execute как recipe (`create_react_agent` для ReAct, custom graphs для plan-execute). CrewAI Flows кодируют паттерн напрямую: вы заранее определяете tasks, и Flow DAG выполняет их. Synthetic data approach из Plan-and-Act пока в основном research; runtime pattern (explicit plan DAG) поставляется в production через LangGraph и CrewAI Flows.

## Отгрузите это

`outputs/skill-rewoo-planner.md` генерирует ReWOO plan DAG из user request при заданном tool catalog. Он валидирует план (acyclic, every reference resolved, every tool exists) перед handoff к executor.

## Упражнения

1. Распараллельте worker execution для независимых plan nodes. Что это даёт на 6-node DAG с 2 parallel groups?
2. Добавьте replanner node, который срабатывает, если любой worker возвращает error. Какое минимальное изменение в ReWOO делает его Plan-and-Execute?
3. Замените `Planner` на small model (7B class) и оставьте `Solver` на frontier model. Сравните end-to-end quality — где split ломается?
4. Прочитайте Section 4 статьи ReWOO о planner distillation. Концептуально воспроизведите результат 175B -> 7B: какие training data нужны и как оценивать plan quality?
5. Перенесите игрушку в trajectory shape Plan-and-Act: plan — sequence, а не DAG. Какие tradeoffs меняются?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| ReWOO | "Reasoning without observations" | Сначала plan, затем fetch evidence parallel, затем solve — no observations in the planning prompt |
| Plan-and-Execute | "Паттерн plan-execute в LangChain" | ReWOO с optional replanner node после execution |
| Plan-and-Act | "Scaled plan-execute" | Явное разделение planner/executor с synthetic plan training data для long-horizon tasks |
| Evidence reference | "#E1, #E2, ..." | Placeholder plan-node, который при dispatch заменяется previous worker output |
| Planner distillation | "Маленький planner, большой executor" | Fine-tune small model на planner traces от large teacher |
| Token efficiency | "Fewer round trips" | В статье 5x fewer tokens на HotpotQA vs ReAct |
| DAG executor | "Topological dispatcher" | Запускает plan nodes в dependency order; parallel на каждом level |

## Дополнительное чтение

- [Xu et al., ReWOO: Decoupling Reasoning from Observations (arXiv:2305.18323)](https://arxiv.org/abs/2305.18323) — каноническая статья
- [Erdogan et al., Plan-and-Act (arXiv:2503.09572)](https://arxiv.org/abs/2503.09572) — scaled planner-executor with synthetic plans
- [LangGraph Plan-and-Execute tutorial](https://docs.langchain.com/oss/python/langgraph/overview) — framework recipe
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — выбирайте самый простой pattern, который работает
