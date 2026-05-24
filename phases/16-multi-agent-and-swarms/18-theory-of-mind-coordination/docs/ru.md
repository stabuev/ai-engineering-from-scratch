# Theory of Mind и эмерджентная координация

> Li et al. (arXiv:2310.10701) показали, что LLM-агенты в кооперативной текстовой игре проявляют **эмерджентную Theory of Mind высокого порядка** (ToM) — рассуждение о том, что один агент думает об убеждениях третьего агента, — но проваливаются на долгосрочном планировании из-за управления контекстом и галлюцинаций. Riedl (arXiv:2510.05174) измерил higher-order synergy в популяции и обнаружил, что **только** условие с ToM-prompt порождает identity-linked differentiation и goal-directed complementarity; LLM меньшей емкости показывают только ложную эмерджентность. То есть эмерджентность координации зависит от prompt и модели, а не дается бесплатно. В этом уроке реализуется минимальный ToM-aware agent, запускается кооперативная задача с ToM prompting и без него, и измеряется coordination delta по протоколу Riedl 2025.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib)
**Требования:** Фаза 16 · 07 (Society of Mind and Debate), Фаза 16 · 17 (Generative Agents)
**Время:** ~75 минут

## Задача

Многоагентная координация часто выглядит магически: агенты делят труд, предвосхищают друг друга, избегают дублирования. Обычно эта "эмерджентность" — артефакт prompt engineering: кто-то сказал агентам "coordinate." Уберите prompt — исчезнет координация.

Вывод Riedl 2025 строже: в контролируемых условиях координация возникает только когда agents получают prompt рассуждать о **сознаниях других агентов** (ToM). Без ToM prompt даже сильные модели показывают паттерны координации, которые не выдерживают статистических контролей. Это важно для production: команды выпускают features "multi-agent coordination", которые зависят от prompt и хрупки.

В этом уроке ToM рассматривается как конкретная способность (рассуждать об убеждениях об убеждениях), строится минимальный ToM-aware agent и измеряется, как выглядит реальная координация по сравнению с prompt dressing.

## Концепция

### Что означает ToM

Психология развития: трехлетний ребенок думает, что внутренний мир каждого совпадает с его собственным. Пятилетний понимает, что у других есть отличные убеждения. Семилетний рассуждает об убеждениях об убеждениях ("она думает, что я думаю, что шарик под чашкой"). Это нулевой, первый и второй порядок ToM.

Для LLM agents порядки ToM соответствуют:

- **Zeroth-order:** нет модели других. Агент действует только на основе собственных наблюдений.
- **First-order:** у агента есть модель убеждений каждого другого агента. "Alice believes X."
- **Second-order:** агент моделирует рекурсивные убеждения. "Alice believes that Bob believes X."

Li et al. 2023 обнаружили, что first- and second-order ToM появляются у LLM agents в кооперативных играх, но деградируют при длинном горизонте и ненадежной коммуникации.

### Тест Sally-Anne вкратце

False-belief test 1985 года: Sally кладет мраморный шарик в корзину A и уходит. Anne перекладывает его в корзину B. Где Sally будет искать, когда вернется? Ребенок с first-order ToM говорит: корзина A (убеждение Sally отличается от реальности). Ребенок без него говорит: корзина B.

LLM эпохи GPT-4 проходят Sally-Anne-style tests, когда они сформулированы прямо. Они проваливаются, когда повествование длинное, сцена меняется несколько раз или вопрос задан косвенно. Это практическое состояние ToM в production LLMs в 2026 году.

### Измерение координации у Riedl

Riedl (arXiv:2510.05174) построил population-scale test: N agents, кооперативная objective, переменные prompt conditions. Измеряются:

1. **Identity-linked differentiation.** Развивают ли agents устойчивые ролевые различия со временем?
2. **Goal-directed complementarity.** Дополняют ли действия agents друг друга (разные subtasks), а не дублируют?
3. **Higher-order synergy.** Статистическая мера того, достигает ли группа того, чего не могла бы ни одна подгруппа.

Результат: только при условии ToM prompt все три метрики дают сигнал выше baseline. Без ToM prompting метрики держатся около случайного уровня для моделей средней емкости. Крупные модели показывают некоторую координацию без явного ToM prompting, но эффект меньше, чем с явным prompting.

### Иллюзия координации

Без статистических контролей "emergent coordination" в демо часто отражает:

- Prompt engineering, который заранее встраивает координацию (system prompts, где сказано "work together").
- Observer bias (мы видим паттерны, которых ожидаем).
- Post-hoc selection успешных runs.

Production-системы, которые продвигают "emergent coordination" без измеримого сигнала, стоит считать маркетингом. Сначала измеряйте, потом заявляйте.

### Минимальный ToM-aware agent

Структура:

```
agent state:
  own_beliefs:    {facts the agent believes}
  other_models:   {other_agent_id -> {beliefs_the_agent_attributes_to_them}}
  actions_last_N: [history of others' actions]

observation update:
  - update own_beliefs from direct observation
  - update other_models[agent_id] from their action + prior beliefs

action selection:
  - enumerate candidate actions
  - for each, predict what each other agent will do next given their modeled beliefs
  - pick action that maximizes joint outcome under those predictions
```

Атрибут `other_models` — это ToM state. First-order ToM хранит один уровень. Second-order добавляет `other_models[i][other_models_of_j]` — что, по моему мнению, агент i думает об убеждениях агента j.

### Почему длинный горизонт вредит

Li et al. документируют: ограничения context заставляют агентов забывать, какое убеждение кому принадлежит. Галлюцинация добавляет ложные beliefs в модели других агентов. Оба эффекта порождают ошибки "I thought he thought X", которые накапливаются со временем.

Меры смягчения, задокументированные в статье и последующих работах 2024-2026:

- **Явное ToM state в prompt.** Структурированный формат: `{agent_id: belief_list}`. Принуждает retrieval сохранять связку identity-belief.
- **Более короткие reasoning chains.** Меньше ToM updates за turn снижает накопление галлюцинаций.
- **Внешнее ToM store.** Поддерживайте модель вне контекста LLM; вставляйте только релевантные части на turn.

### Где ToM отказывает в production

- **Adversarial settings.** Агентов с хорошей ToM проще манипулировать (вы можете моделировать, что они моделируют о вас, затем эксплуатировать это).
- **Heterogeneous teams.** Когда модели разные, ToM model, работающая для одного оппонента, не обобщается.
- **Ground-truth-dependent tasks.** ToM про beliefs; если корректность зависит от фактов, ToM может отвлекать.

### Координация, которую можно реально измерить

Три практических сигнала того, что координация команды реальна, а не prompt-dressed:

1. **Complementarity over time.** В многоходовой задаче покрывают ли действия agents непересекающиеся sub-tasks?
2. **Anticipation.** Зависит ли действие agent A на turn T+1 от предсказания действия B на turn T+2, которое оказалось верным?
3. **Correction.** Когда A неверно читает belief B на turn T, исправляет ли A это к turn T+2?

Это измеримо в логируемой многоагентной системе. Это содержательная версия нарратива "coordination".

## Сборка

`code/main.py` реализует:

- `ToMAgent` — отслеживает собственные beliefs и belief models по другим агентам.
- Кооперативная задача: три agents должны собрать три tokens из трех boxes; каждая box может содержать один token. Агенты не могут общаться; они выводят намерения из действий друг друга.
- Две конфигурации: `zeroth_order` (без ToM) и `first_order` (ToM с одноуровневой belief model).
- Измерение на 200 randomized trials: completion rate, duplication rate (два agents нацеливаются на одну box), average turns to completion.

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод: zeroth-order agents дублируют усилия примерно в 35% случаев и завершают около 60% trials за 10 turns. First-order ToM agents дублируют около 5% и завершают около 95%. Разница — измеримый coordination effect.

## Использование

`outputs/skill-tom-auditor.md` — skill, который аудирует заявление многоагентной системы об "emergent coordination." Проверяет prompt dressing, статистическую значимость относительно control и измеренную complementarity.

## Доставка

Чеклист заявлений о координации:

- **Control condition.** Версия вашей системы без coordination prompt. Измеряйте обе.
- **Statistical test.** Значима ли разница между system и control при `p < 0.05` на вашей метрике?
- **Complementarity measure.** Action-disjointness во времени, а не только финальный успех.
- **Failure-case log.** Когда agents miscoordinate, как выглядит ToM state?
- **Model-capacity disclosure.** Если эффект исчезает на меньших моделях, скажите об этом.

## Упражнения

1. Запустите `code/main.py`. Убедитесь, что first-order ToM снижает duplication rate примерно в 7x. Сохраняется ли разрыв при масштабировании до 5 agents и 5 boxes?
2. Реализуйте second-order ToM (agent A моделирует, что B думает о C). Улучшает ли это first-order? На каких задачах?
3. Внесите **hallucination** в ToM state: случайно переворачивайте одно belief на turn. Насколько это ухудшает first-order performance?
4. Прочитайте Li et al. (arXiv:2310.10701). Воспроизведите вывод "long-horizon degradation": как меняется first-order ToM performance при росте turns с 10 до 30?
5. Прочитайте Riedl 2025 (arXiv:2510.05174). Реализуйте higher-order synergy statistic на ваших simulation logs. Присутствует ли эффект без ToM prompt condition?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Theory of Mind | "Понимание сознания других" | Способность моделировать убеждения другого агента. Градуируется по порядку (0, 1, 2+). |
| Sally-Anne test | "Тест ложного убеждения" | Психология развития 1985 года; LLM проходят простые версии и проваливают сложные. |
| First-order ToM | "A believes X" | Моделирование убеждений одного другого агента о фактах. |
| Second-order ToM | "A believes B believes X" | Рекурсивное моделирование на один уровень глубже. |
| Identity-linked differentiation | "Стабильные роли во времени" | Метрика Riedl: роли сохраняются, а не случайны. |
| Goal-directed complementarity | "Непересекающиеся действия" | Agents нацеливаются на разные subtasks, а не на один и тот же. |
| Higher-order synergy | "Группа превосходит любую подгруппу" | Статистическая мера Riedl для реальной координации. |
| Coordination illusion | "Выглядит скоординированно" | Prompt-dressed видимость координации без измеримого сигнала. |

## Дополнительное чтение

- [Li et al. — Theory of Mind for Multi-Agent Collaboration via Large Language Models](https://arxiv.org/abs/2310.10701) — эмерджентная ToM в кооперативных играх; long-horizon failure modes
- [Riedl — Emergent Coordination in Multi-Agent Language Models](https://arxiv.org/abs/2510.05174) — population-scale measurement; ToM prompting является load-bearing condition
- [Premack & Woodruff — Does the chimpanzee have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-chimpanzee-have-a-theory-of-mind/1E96B02CD9850E69AF20F81FA7EB3595) — происхождение концепции ToM в 1978 году
- [Baron-Cohen, Leslie, Frith — Does the autistic child have a theory of mind?](https://www.cambridge.org/core/journals/behavioral-and-brain-sciences/article/does-the-autistic-child-have-a-theory-of-mind/) — статья Sally-Anne (1985)
