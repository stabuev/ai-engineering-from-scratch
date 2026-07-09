# Генеративные агенты и эмерджентная симуляция

> Park et al. 2023 (UIST '23, arXiv:2304.03442) заселили **Smallville**, песочницу из 25 агентов, трехкомпонентной архитектурой: **memory stream** (естественно-языковый журнал), **reflection** (высокоуровневые обобщения, которые агент генерирует о собственном потоке) и **plan** (поведение на уровне дня, затем под-планы). Знаковым результатом стало эмерджентное появление вечеринки на День святого Валентина: один агент, инициализированный фразой "wants to throw a Valentine's Day party," без дальнейшего скриптинга породил распространение приглашений по популяции, согласование дат, и вечеринка состоялась — от 24 агентов, которые изначально ничего о ней не знали. Ablations показывают, что все три компонента нужны для правдоподобия. Задокументированные сбои — ошибки пространственных норм (вход в закрытые магазины, совместное использование одноместных ванных комнат). Это эталонная архитектура для агентных симуляций и социальной оценки многоагентных систем в 2026 году.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib)
**Требования:** Фаза 16 · 04 (Primitive Model), Фаза 16 · 13 (Shared Memory)
**Время:** ~75 минут

## Цели обучения

- Строить трехчастную архитектуру генеративного агента (поток памяти, рефлексия, планирование).
- Объяснять через абляцию, почему важны все три компонента.
- Распознавать эмерджентное поведение и задокументированные режимы отказа.

## Задача

Большинство многоагентных систем — жестко scripted команды: planner планирует, coder пишет код, reviewer проверяет. Это работает для хорошо определенных задач. Это не отражает эмерджентное, несценарное поведение, которое возникает, когда у агентов есть память, приоритеты и открытый мир. Исследования, социальная симуляция и все чаще game AI нуждаются во втором типе.

Архитектура Smallville — бенчмарк для этого. До Park 2023 лучшие агентные симуляции были поверхностными script-followers; после нее паттерн стал дефолтным для генеративных агентов в открытых мирах. Если вы строите агентную симуляцию в 2026 году, вы либо используете три компонента Smallville, либо явно обосновываете, почему не используете.

## Концепция

### Три компонента

**Memory stream.** Append-only журнал наблюдений, действий, reflections и plans. У каждой записи есть timestamp, type, description (естественный язык) и производные метаданные: **recency**, **importance** (самооценка агента 1-10) и **relevance** (cosine similarity к текущему запросу).

```
[2026-02-14 09:12:03] observation: Isabella Rodriguez asked me if I like jazz
[2026-02-14 09:14:22] reflection:   I enjoy long conversations about music
[2026-02-14 10:05:00] plan:         Attend Isabella's Valentine's Day party tonight
```

Извлечение из памяти объединяет три оценки: `score = w_recency * e^(-decay * age) + w_importance * importance + w_relevance * cos_sim`. Top-k записей попадают в текущий prompt.

**Reflection.** Периодически (каждые N memories или при важных событиях) агент генерирует higher-order обобщения из недавних memories. Записи reflection возвращаются в stream и извлекаются как любая другая memory. Так агенты строят "понимания" — архитектурный аналог долгосрочных убеждений.

**Plan.** Декомпозиция сверху вниз. Сначала план уровня дня широкими мазками ("go to work, have dinner with Klaus"). Затем планы уровня часа. Затем планы уровня действия. Plans пересматриваемы: когда observation противоречит plan, агент перепланирует затронутый сегмент.

### Почему важны все три (ablation)

Park et al. проводили ablations, удаляя observation, reflection и plan. Каждая ablation ухудшает правдоподобие:

- Без **observation** агент упускает контекст и действует по устаревшим убеждениям.
- Без **reflection** агент не может формировать higher-order beliefs; взаимодействия остаются поверхностными.
- Без **plan** поведение становится реактивным шумом; цели рассеиваются.

Оценки правдоподобия от human raters максимальны при наличии всех трех; удаление любого одного компонента дает измеримую регрессию.

### Эмерджентность Дня святого Валентина

Один агент, Isabella Rodriguez, инициализируется целью "wants to throw a Valentine's Day party at Hobbs Cafe on Feb 14 at 5pm." Остальные 24 агента не получают такой seed. За симулированные дни:

1. Plan Isabella включает приглашение людей.
2. Каждое приглашение становится observation в memory stream соседа.
3. Reflection соседа генерирует beliefs: "Isabella is throwing a party."
4. Plan соседа включает "attend party on Feb 14."
5. Соседи рассказывают другим соседям. Приглашение распространяется без центральной координации.
6. В 5pm on Feb 14 несколько агентов сходятся в Hobbs Cafe.

Это эмерджентность в техническом смысле: поведение уровня системы (вечеринка) возникло из локальных взаимодействий (двусторонние приглашения + индивидуальное планирование) без центрального orchestrator.

### Задокументированные режимы отказа

Park et al. явно документируют:

- **Ошибки пространственных норм.** Агенты заходят в закрытые магазины. Агенты пытаются пользоваться одной и той же одноместной ванной. Агенты едят в комнатах, не предназначенных для еды. Модель не выводит социально-физические нормы только из окружения.
- **Переполнение памяти.** Долгие simulation runs увеличивают стоимость memory-retrieval. Практическое средство: периодическая memory compaction (summarize-and-prune) и decay для low-importance entries.
- **Галлюцинация reflection.** Reflections могут изобретать отношения, которых нет в memory stream. Смягчение: включайте source memory ids в reflection prompts и проверяйте при retrieval.

Это production-значимые режимы отказа: любая агентная симуляция 2026 года наследует их.

### Правила реализации трех компонентов

1. **Memory является append-only.** Никогда не изменяйте memory entry. Исправления — это новые entries.
2. **Importance scores дешевы.** Вызывайте LLM для оценки importance 1-10 во время записи. Кэшируйте score.
3. **Retrieval ранжируется, а не фильтруется.** Top-k по combined score; не используйте жесткие filters (они теряют контекст).
4. **Reflection запускается периодически.** Триггер, когда сумма importance необработанных memories превышает threshold (например, 150).
5. **Plans пересматриваемы.** Когда новая observation противоречит plan, регенерируйте только затронутый сегмент, а не весь plan.

### Генеративные агенты за пределами Smallville

Литература 2024-2026 годов расширяет архитектуру:

- **Multi-agent social simulation for policy / market research.** Популяции в стиле Smallville симулируют поведение пользователей в ответ на features. Быстрее A/B tests; точность спорная.
- **NPC AI for games.** RPG с агентами Smallville порождают эмерджентные сюжетные линии вместо scripted quests.
- **Generative-agent evaluation benchmarks.** Вместо task accuracy метрикой становится believability + coherence поведения на длинных runs.

Архитектура является эталонной. Расширения заменяют компоненты (vector store для memory, retrieval-augmented reflection, neurosymbolic plan), но сохраняют трехчастную структуру.

### Почему это важно для multi-agent engineering

Smallville — proof of concept, что многоагентная эмерджентность дешева, когда компоненты правильные. Архитектура уже воспроизведена на open-source models (меньшие LLM теряют правдоподобие плавно, не резко). Любая production-система, которой нужно **emergent social behavior**, использует эту форму. Любая система, которой нужно **tight task execution**, использует паттерны supervisor / roles / primitives из более ранних уроков этой фазы.

## Сборка

`code/main.py` реализует три компонента на stdlib Python со scripted agent policies (без настоящей LLM). Демо воспроизводит эмерджентность Valentine's-party в миниатюре:

- `MemoryStream` — append-only журнал с retrieval по recency/importance/relevance.
- `reflect(stream)` — scripted reflection по недавним high-importance memories.
- `plan(agent_state)` — планы уровня дня и часа на основе текущих beliefs.
- Сценарий: 5 агентов. Agent 1 начинает с "throw party at 5pm." За симулированные ticks приглашение распространяется, и агенты сходятся.

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод: пошаговая trace по ticks. К финальному tick как минимум 3 из 5 агентов показывают вечеринку в своем plan и сходятся в party location. Единственный seed породил скоординированный приход без orchestrator.

## Использование

`outputs/skill-simulation-designer.md` проектирует generative-agent simulation: число агентов, memory schema, reflection cadence, plan horizon и evaluation metric.

## Доставка

Правила для production simulations:

- **Memory — это database.** В масштабе выбирайте реальное хранилище (vector DB, Postgres). In-memory stdlib — для прототипов.
- **Логируйте retrieval trace.** Для каждого действия логируйте top-k memories, которые его обусловили. Это ваша возможность debugging.
- **Бюджетируйте tokens на агента.** Retrieve + reflect + plan каждого агента на tick — это O(k) LLM calls. N agents × T ticks × calls-per-tick могут превысить ваш бюджет.
- **Периодически compact memory.** Summarize-and-prune low-importance entries. Retention policy — проектное решение, а не деталь.
- **Явно обнаруживайте spatial / social norm violations.** Архитектура не учится им сама.

## Упражнения

1. Запустите `code/main.py`. Убедитесь, что 3+ agents сходятся на вечеринке. Увеличьте число agents до 10 — сохраняется ли emergence?
2. Уберите шаг reflection. Как выглядит поведение? Сопоставьте с ablation finding в Park 2023.
3. Введите конкурирующую seeded goal ("Klaus wants to give a research talk at 5pm"). Агенты разделяются или одна цель доминирует? Что это определяет?
4. Добавьте spatial constraints: Hobbs Cafe вмещает максимум 4 agents. Обрабатывает ли simulation overflow корректно или попадает в паттерн отказа "single-person bathroom"?
5. Прочитайте Park et al. (arXiv:2304.03442) Section 6 (emergent behavior experiments). Определите одно поведение, невоспроизводимое в вашей миниатюре. Какой компонент архитектуры нужно усилить?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Memory stream | "Дневник агента" | Append-only журнал observations, actions, reflections, plans. |
| Recency | "Насколько новая memory" | Оценка с экспоненциальным decay по возрасту. |
| Importance | "Насколько агенту важно" | Самооценка 1-10 во время записи. Кэшируется. |
| Relevance | "Насколько связано с текущим запросом" | Cosine similarity (embedding-based). |
| Reflection | "Higher-order belief" | Обобщение, сгенерированное из недавних memories и заново внесенное как новая memory. |
| Plan | "Декомпозиция день/час/действие" | Дерево плана сверху вниз. Пересматривается, когда observations противоречат. |
| Smallville | "Песочница Park 2023" | Симуляция из 25 агентов, породившая Valentine's Day emergence. |
| Believability | "Метрика качества" | Оценка human-rater, кажется ли поведение правдоподобным агентом. |

## Дополнительное чтение

- [Park et al. — Generative Agents: Interactive Simulacra of Human Behavior](https://arxiv.org/abs/2304.03442) — эталонная архитектура
- [UIST '23 paper page](https://dl.acm.org/doi/10.1145/3586183.3606763) — место публикации
- [Smallville code release](https://github.com/joonspk-research/generative_agents) — эталонная Python-реализация
- [Hayes-Roth 1985 — A Blackboard Architecture for Control](https://www.sciencedirect.com/science/article/abs/pii/0004370285900639) — prior art для агентов со структурированной памятью
