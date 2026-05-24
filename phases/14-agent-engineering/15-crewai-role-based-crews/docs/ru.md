# CrewAI: ролевые команды и Flows

> CrewAI — ролевая мультиагентная платформа 2026 года: Agents, Tasks, Crews, Processes как четыре примитива. Рекомендация из документации для production: "for any production-ready application, start with a Flow."

**Тип:** Изучение + практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 12 (Workflow Patterns), Фаза 14 · 14 (Actor Model)
**Время:** ~60 минут

## Цели обучения

- Назвать четыре примитива CrewAI — Agent, Task, Crew, Process — и роль каждого.
- Отличать Crews (автономная ролевая совместная работа) от Flows (событийные детерминированные workflow).
- Объяснить, почему документация рекомендует начинать с Flows для production и использовать Crews для исследования.
- Реализовать Crew runner на stdlib и Flow runner на stdlib; показать, где каждый подход раскрывается лучше.

## Проблема

Команды, внедряющие мультиагентные фреймворки, упираются в одну и ту же стену: "автономная совместная работа" звучит отлично, но когда клиент заводит баг, нужен детерминированный replay. CrewAI явно разделяет это: Crews для творческой совместной работы, Flows для событийных, аудируемых workflow, пригодных для production.

## Концепция

### Четыре примитива

- **Agent.** Роль + цель + предыстория + инструменты. Предыстория имеет реальный вес: она формирует тон и суждение.
- **Task.** Описание + expected_output + назначенный agent. Переиспользуемая единица работы.
- **Crew.** Контейнер, который упорядочивает Agents и Tasks. Владеет Process выполнения.
- **Process.** Sequential или Hierarchical (с manager Agent) или Consensual.

### Crews vs Flows

- **Crew.** Автономный, управляемый LLM. Хорош для открытых задач: исследования, брейншторминг, первые черновики. Фреймворк выбирает форму выполнения во время runtime.
- **Flow.** Событийный граф, которым владеет код. Каждый шаг запускается по триггеру (декоратор функции, совпадение события). Хорош для production: наблюдаемый, тестируемый, детерминированный.

Документация CrewAI 2026 говорит: начинайте production-приложения с Flows; добавляйте Crews как подшаги, когда автономность оправдывает свою стоимость.

### Система памяти

CrewAI поставляет четыре типа памяти из коробки: short-term (внутри run), long-term (между runs), entity (факты по отдельным сущностям), contextual (сборка во время retrieval). Интеграции с vector stores являются first-party.

### Интеграция AWS Bedrock

У CrewAI есть документированная интеграция AWS Bedrock с observability hooks для CloudWatch, AgentOps и Langfuse. Документация AWS приводит ускорение 5.76x по сравнению с LangGraph на QA-задачах в их benchmarks — воспринимайте framework-specific числа как ориентировочные, а не абсолютные.

### Форма зависимостей

Независим от LangChain. Python 3.10–3.13. Использует `uv` для управления зависимостями. 30k+ GitHub stars в начале 2026 года.

### Где этот паттерн ломается

- **Crew-as-prod.** Использование свободной Crew в production без Flow-обертки. Вариативность output высокая; отладка болезненна.
- **Раздутая backstory.** Предыстории на 2000 слов вытесняют context budget. Держите их компактными.
- **Путаница с Process.** Hierarchical process добавляет manager Agent, который маршрутизирует; используйте только когда есть 4+ специалиста.

## Соберите это

`code/main.py` реализует stdlib-версии обоих подходов:

- `Agent`, `Task`, `Crew`, `SequentialCrew` (по одной task за раз), `HierarchicalCrew` (manager маршрутизирует).
- `Flow` с декораторами `@start()` и `@listen()` (замены на обычных функциях), которые срабатывают по именованным событиям.
- Одна и та же трехшаговая task (research, outline, draft), реализованная обоими способами.

Запустите:

```
python3 code/main.py
```

Трасса Crew текучая и вариативная; трасса Flow фиксированная и наблюдаемая. В этом и состоит выбор.

## Используйте это

- **CrewAI Flow** для production.
- **CrewAI Crew** для исследования, парной работы, первых черновиков.
- **LangGraph** (Урок 13), если нужна более явная state machine.
- **AutoGen v0.4** (Урок 14), если нужна concurrency в стиле actor model.

## Отправьте в работу

`outputs/skill-crew-or-flow.md` выбирает Crew vs Flow для task и создает scaffold минимальной реализации.

## Упражнения

1. Преобразуйте Crew-based demo в Flow. Посчитайте точки, где уменьшается вариативность.
2. Добавьте entity memory в Crew: факты о customer сохраняются между tasks.
3. Реализуйте Hierarchical process: manager Agent выбирает, какой specialist запускается следующим, на основе предыдущего output.
4. Прочитайте вводную часть документации CrewAI. Перенесите свой toy в настоящий API `crewai`. Что меняется в testability?
5. Подключите AgentOps или Langfuse к одному из runs. Каких traces вам не хватало в stdlib-версии?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Agent | "Persona" | Роль + цель + backstory + tools |
| Task | "Unit of work" | Описание + expected output + assignee |
| Crew | "Agent team" | Контейнер для Agents + Tasks + Process |
| Process | "Execution strategy" | Sequential / Hierarchical / Consensual |
| Flow | "Deterministic workflow" | Событийный, принадлежащий коду, тестируемый |
| Backstory | "Persona prompt" | Формирователь тона и суждения для Agent |
| Entity memory | "Per-entity facts" | Память, ограниченная customer/account/issue |

## Дополнительное чтение

- [CrewAI docs introduction](https://docs.crewai.com/en/introduction) — концепции и рекомендуемый путь к production
- [Anthropic, Building Effective Agents](https://www.anthropic.com/research/building-effective-agents) — когда multi-agent помогает, а когда нет
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — альтернатива на основе state machine
