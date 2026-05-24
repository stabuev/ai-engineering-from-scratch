# Tree of Thoughts и LATS: намеренный поиск

> У одной траектории chain-of-thought нет пространства для отката. ToT (Yao et al., 2023) превращает рассуждение в дерево с самооценкой на каждом узле. LATS (Zhou et al., 2024) объединяет ToT с ReAct и Reflexion под Monte Carlo Tree Search. Game of 24 растёт с 4% (CoT) до 74% (ToT); LATS достигает 92.7% pass@1 на HumanEval.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 03 (Reflexion)
**Время:** ~75 минут

## Цели обучения

- Представлять рассуждение как поиск: узлы — это "thoughts", рёбра — "expansions", value — "how promising".
- Реализовать поиск по дереву в стиле ToT BFS на stdlib с оцениванием через самооценку.
- Расширить его до игрушечного цикла LATS MCTS с select / expand / simulate / backpropagate.
- Решать, когда поиск стоит множителя по токенам (Game of 24, генерация кода), а когда достаточно одной траектории (простые Q&A).

## Проблема

Chain-of-thought — это линейная прогулка. Если первый шаг неверен, каждый следующий шаг работает с плохой предпосылкой. На Game of 24 (использовать четыре цифры с + − × ÷, чтобы получить 24) GPT-4 CoT достигает точности 4%. Модель рано выбирает неверное подвыражение и не может восстановиться.

Рассуждению нужна способность предлагать несколько кандидатов, оценивать их, выбирать перспективные и откатываться, когда появляются тупики. Это поиск. Tree of Thoughts и LATS — две канонические формулировки.

## Концепция

### Tree of Thoughts (Yao et al., NeurIPS 2023)

Каждый узел — связный промежуточный шаг ("a thought"). Каждый узел может расширяться в K дочерних мыслей. LLM самооценивает каждый узел через оценочный промпт. Поиск исследует дерево — BFS, DFS или beam.

```
                     (root: "find 24 from 4 6 4 1")
                    /               |            \
           ("6 - 4 = 2")    ("4 + 1 = 5")    ("4 * 6 = 24")  <- Score: HIGH
              /   \              |                  |
          ...    ...          ...                finish
```

Самооценка — несущая часть. Статья показывает три варианта: классификация `sure / likely / impossible`, числовой балл `1..10` и голосование между кандидатами. Все три существенно превосходят CoT на Game of 24 (4% -> 74% с GPT-4).

### LATS (Zhou et al., ICML 2024)

LATS объединяет ToT, ReAct и Reflexion под MCTS. LLM играет три роли:

- **Policy**: предлагает возможные следующие действия (в стиле ReAct).
- **Value function**: оценивает частичную траекторию (самооценка в стиле ToT).
- **Self-reflector**: при сбое пишет reflection на естественном языке (в стиле Reflexion) и использует её для повторного запуска будущих rollouts.

Обратная связь от среды (observations) смешивается в value function, поэтому поиск опирается на реальные результаты инструментов, а не только на мнения модели. Результаты на момент статьи: HumanEval pass@1 92.7% с GPT-4 (SOTA), средний WebShop 75.9 с GPT-3.5 (приближаясь к fine-tuning на градиентах).

### MCTS, минимально

Четыре фазы на итерацию:

1. **Select** — пройти от root до leaf, используя UCT (upper confidence bound for trees).
2. **Expand** — сгенерировать K дочерних узлов через policy.
3. **Simulate** — сделать rollout от дочернего узла через policy, оценить лист через value function (или reward среды).
4. **Backpropagate** — обновить visit counts и оценки value вверх по пути.

Формула UCT: `Q(s, a) + c * sqrt(ln N(s) / N(s, a))`. Первый член — exploitation; второй — exploration. Настраивайте `c` под задачу.

### Реальность стоимости

Поиск взрывает расход токенов. ToT на Game of 24 использует в 100–1000 раз больше токенов, чем CoT. LATS похож. Это не бесплатно; резервируйте поиск для:

- Задач, где одной траектории явно недостаточно (Game of 24, сложный код).
- Задач, где wall-clock менее важен, чем корректность.
- Задач с дешёвой и надёжной value function (unit tests для кода, явная цель для математики).

Если у задачи один правильный ответ и шумный evaluator, поиск часто ухудшает результат — он находит неверный ответ с хорошей оценкой.

### Позиционирование в 2026 году

Большинство production agents не запускают LATS. Они запускают ReAct с верификацией, заземлённой на инструменты (CRITIC, Lesson 05). Поиск появляется в специализированных нишах:

- Coding agents, которые запускают тесты как value function (в стиле HumanEval).
- Deep-research agents, которые исследуют несколько путей запросов.
- Workflows с тяжёлым планированием внутри subgraphs LangGraph.

AlphaEvolve (Lesson 11) — экстремум 2025 года: эволюционный поиск по коду, машинно проверяемая fitness-функция, frontier gains (первое улучшение 4x4 matmul за 56 лет).

## Соберите это

`code/main.py` реализует:

- Миниатюрный ToT BFS на стилизованной задаче "pick arithmetic ops".
- Игрушечный LATS MCTS loop на той же задаче (Select / Expand / Simulate / Backpropagate) с выбором через UCT.
- Value function, которая объединяет символический балл и self-eval score.

Запустите:

```
python3 code/main.py
```

Трасса показывает ToT, который расширяет по три кандидата на узел через BFS, в сравнении с LATS, который сходится к лучшему rollout через MCTS. Количество токенов печатается для обоих.

## Используйте это

LangGraph поставляет исследование в стиле ToT как шаблоны subgraph; блог команды LangChain о LATS (May 2024) — эталонный туториал. LlamaIndex поставляет agent `TreeOfThoughts`. Для большинства production agents 2026 года этот паттерн живёт за gate `if task_complexity > threshold: use_search()` — см. паттерн evaluator-optimizer в Lesson 05.

## Отгрузите это

`outputs/skill-search-policy.md` выбирает между линейным ReAct, ToT, LATS и эволюционным поиском по форме задачи, бюджету и надёжности evaluator.

## Упражнения

1. Запустите игрушечный LATS с UCT c=0.1 vs c=2.0. Что меняется в трассе?
2. Замените value function на более шумный scorer (добавьте random jitter). MCTS всё ещё находит лучший leaf? Какой минимальный signal-to-noise он выдерживает?
3. Реализуйте beam-search ToT (keep top-k at each level) и сравните с BFS. Что лучше при жёстком бюджете токенов?
4. Прочитайте LATS Section 5.1. Воспроизведите количество траекторий HumanEval: сколько rollouts нужно, чтобы достичь заявленного pass@1?
5. Прочитайте discussion статьи LATS о "when LATS helps less." Напишите правило принятия решения в один абзац, связывающее форму задачи со стратегией поиска.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Tree of Thoughts | "Branching CoT" | Yao et al. — дерево thought nodes с самооценкой |
| LATS | "MCTS for LLMs" | Zhou et al. — объединяет ToT + ReAct + Reflexion под MCTS |
| UCT | "Upper confidence bound" | Формула Select, балансирующая exploitation (Q) и exploration (ln N / n) |
| Value function | "How good is this state" | Prompted LLM score или reward среды; питает backprop |
| Policy | "Action proposer" | Генератор в стиле ReAct; выдаёт кандидаты следующих thoughts/actions |
| Rollout | "Simulated trajectory" | Проход от node до leaf через policy, оценивание через value |
| Backpropagate | "Update ancestors" | Протолкнуть reward leaf вверх по пути, обновляя visit counts и Q |
| Search cost | "Token explosion" | 100-1000x CoT на Game of 24; заложите бюджет до внедрения |

## Дополнительное чтение

- [Yao et al., Tree of Thoughts (arXiv:2305.10601)](https://arxiv.org/abs/2305.10601) — каноническая статья
- [Zhou et al., LATS (arXiv:2310.04406)](https://arxiv.org/abs/2310.04406) — MCTS с обратной связью Reflexion
- [LangGraph overview](https://docs.langchain.com/oss/python/langgraph/overview) — шаблоны subgraph для поиска
- [AlphaEvolve (arXiv:2506.13131)](https://arxiv.org/abs/2506.13131) — эволюционный поиск с программными evaluators
