# Роевая оптимизация для LLM (PSO, ACO)

> Биоинспирированная оптимизация возвращается в эпоху LLM. **LMPSO** (arXiv:2504.09247) использует PSO, где velocity каждой particle — это prompt, а LLM генерирует следующий candidate; хорошо работает на structured-sequence outputs (математические выражения, программы). **Model Swarms** (arXiv:2410.11163) рассматривает каждого LLM expert как PSO particle на многообразии весов модели и сообщает о **13.3% average gain** относительно 12 baselines на 9 datasets всего с 200 instances. **SwarmPrompt** (ICAART 2025) гибридизирует PSO + Grey Wolf для prompt optimization. **AMRO-S** (arXiv:2603.12933) — ACO-inspired pheromone specialists для multi-agent LLM routing: **4.7x speedup**, интерпретируемые routing evidence, quality-gated asynchronous update, который отделяет inference от learning. В этом уроке реализуется PSO в prompt parameter space и ACO в agent routing, измеряется, почему эти классические алгоритмы подходят эпохе LLM и когда не подходят.

**Тип:** Изучение + сборка
**Языки:** Python (stdlib)
**Требования:** Фаза 16 · 09 (Parallel Swarm Networks), Фаза 16 · 14 (Consensus and BFT)
**Время:** ~75 минут

## Задача

У вас есть prompt, который набирает 62% на вашей task eval. Вы хотите его улучшить. Наивный ход — gradient-free ручная правка, которая плохо масштабируется. Reinforcement learning требует reward signals и достаточного числа rollouts для обучения. Backprop через prompts фактически невозможен — prompt является дискретной строкой, а не дифференцируемым параметром.

Классическая биоинспирированная оптимизация — PSO для continuous search spaces, ACO для path selection — была спроектирована именно для этого режима: без градиентов, population-based, дешево на evaluation. Соедините ее с LLM для gradient-free search step, и получится неожиданно практичный optimizer.

Те же паттерны применимы к agent *routing* в многоагентных системах. ACO-style pheromone trail записывает, какой agent лучше всего справлялся с каким task-type, позволяет router использовать trail и испаряет pheromones, чтобы маршруты могли переоткрываться.

## Концепция

### Напоминание о PSO (Kennedy & Eberhart 1995)

Particle Swarm Optimization: популяция particles в continuous search space. У каждой particle есть position `x_i` и velocity `v_i`. На каждой итерации:

```
v_i <- w * v_i + c1 * r1 * (p_best_i - x_i) + c2 * r2 * (g_best - x_i)
x_i <- x_i + v_i
evaluate fitness(x_i)
update p_best_i if improved
update g_best if global best
```

Где `p_best` — собственный лучший результат particle, `g_best` — лучший результат swarm, `w, c1, c2` — inertia + cognitive + social weights, `r1, r2` — random factors.

### PSO на LLM outputs — LMPSO

arXiv:2504.09247 адаптирует PSO для LLM-generated structured outputs (математические выражения, программы). Каждая particle — candidate output. Velocity — это *prompt*, который описывает, как изменить текущий output в сторону personal/global best. LLM генерирует новый output из velocity prompt. "Inertia" velocity — prompt вроде "make small incremental changes."

Это хорошо работает, когда:
- Output структурирован (parseable, evaluable).
- Fitness автоматическая (test runs, arithmetic evaluation).
- Population мала (~10-30 particles), чтобы общее число LLM calls оставалось управляемым.

Это плохо работает, когда fitness требует human review — per-iteration cost становится запретительным.

### Model Swarms

arXiv:2410.11163 переносит PSO с output layer на *model* layer. Каждая "particle" — expert LLM (parameters). Swarm двигает parameters к коллективному лучшему через gradient-free update. Заявлено: 13.3% average gain относительно 12 baselines на 9 datasets, всего с 200 instances per iteration.

Ключевая идея: LLM expert models уже находятся рядом в общем parameter manifold (adapter weights, LoRA deltas). PSO на этом low-dimensional subspace дешев и эффективен.

### Напоминание об ACO (Dorigo 1992)

Ant Colony Optimization: ants проходят по graph; у каждого path есть pheromone trail. Вероятности хода ants взвешиваются силой pheromone. Ants, завершившие task, откладывают pheromone пропорционально quality решения. Pheromone со временем decay.

### AMRO-S — ACO для agent routing

arXiv:2603.12933 использует ACO для multi-agent routing. Каждый task-type — "destination"; каждый agent — возможный route. Pheromones усиливают routes, которые дают хорошие outputs. Ключевые вклады:

- **Interpretable routing evidence.** Сила pheromone — человекочитаемый сигнал.
- **Quality-gated asynchronous update.** Pheromones обновляются только после прохождения quality checks, отделяя inference от learning.
- **4.7x speedup** на multi-agent routing benchmark.

Quality gate важен: без него быстрые, но ошибочные agents накапливают pheromone, и система фиксируется на плохих routes.

### Когда использовать PSO / ACO для LLM

**Используйте PSO, когда:**
- Search space непрерывен или отображается в continuous parameters (prompt embeddings, LoRA weights, numeric generation parameters).
- Fitness дешева и автоматическая.
- Population может быть малой (10-30).

**Используйте ACO, когда:**
- У вас routing или path-selection problem.
- Decisions усиливаются со временем (те же task types возвращаются).
- Вам нужны интерпретируемые evidence для routing decisions.

**Не используйте ни то ни другое, когда:**
- Fitness требует human review (слишком дорого на iteration).
- Search space дискретное и комбинаторное так, что PSO его не покрывает (используйте genetic algorithms).
- Real-time decisions требуют строгой latency (PSO/ACO сходятся медленно относительно single-pass heuristics).

### Почему bio-inspired все еще выигрывает

Gradient-based methods требуют дифференцируемых signals. LLM outputs и routing decisions не являются тривиально дифференцируемыми. Pseudo-gradient methods (reinforcement-learned routers, DPO-style prompt tuners) работают, но требуют дорогого training.

PSO и ACO требуют только *evaluator* function. Если вы можете оценить candidate output или routing decision, вы можете оптимизировать пространство. Это резко снижает порог применимости.

### Практические ограничения

- **Population budget.** N particles × T iterations × per-eval cost. Для LLM evals по ~$0.02 / call, PSO с 20 particles на 50 iterations стоит ~$20. Планируйте соответственно.
- **Exploration vs exploitation.** Pheromone decay rate и PSO inertia — компромисс; слишком быстрый decay → забывание решений; слишком медленный → застревание в ранних local optima.
- **Catastrophic drift.** Оба алгоритма могут сойтись, а затем разойтись, если fitness landscape сдвигается (новое data distribution). Мониторьте стабильность best-fitness.

## Сборка

`code/main.py` реализует:

- `LMPSO` — PSO по numeric prompt parameters (temperature, top_k weights). "LLM generation" каждой particle симулируется scripted fitness function. Запускает алгоритм на 30 iterations и показывает сходимость g_best.
- `AMRO_S` — ACO-style routing. 3 agents, 4 task types, pheromone matrix, 100 routed tasks. Печатает распределение (task_type → agent choices) во времени, чтобы показать формирование trail.
- Сравнение: random routing vs ACO routing на том же task stream. Измеряет quality и latency.

Запуск:

```
python3 code/main.py
```

Ожидаемый вывод:
- LMPSO: g_best fitness улучшается от случайного до почти оптимального за 30 iterations.
- AMRO-S: pheromone table стабилизируется на правильном agent для каждого task-type; ACO routing превосходит random примерно на 30-40% по quality и также снижает latency (меньше retries).

## Использование

`outputs/skill-swarm-optimizer.md` помогает выбирать между PSO, ACO, genetic algorithms и gradient-based optimizers для LLM / agent optimization problems.

## Доставка

- **Начинайте с малого.** 10-20 particles, 20-50 iterations. Масштабируйте только если convergence curve показывает явный gain.
- **Логируйте pheromones или g_best на каждой iteration.** Отлаживать swarm optimizers без trail болезненно.
- **Quality-gate updates.** Особенно для ACO routing: быстрые-и-ошибочные agents не должны накапливать pheromone.
- **Reset decay при distribution shift.** Когда eval distribution меняется, старые pheromones устаревают; сбросьте или временно удвойте decay rate.
- **Ограничивайте per-iteration cost.** Выводите cost-per-iteration metric. PSO, который стоит $500 / iteration и дает 0.5%, не готов к ship.

## Упражнения

1. Запустите `code/main.py`. Наблюдайте сходимость LMPSO. Меняйте population size 5, 10, 20, 50. При каком размере time-to-converge насыщается?
2. Реализуйте эксперимент "catastrophic drift": после iteration 30 измените fitness function. Как быстро PSO адаптируется? Помогает ли сброс `p_best`?
3. Добавьте quality gate в AMRO-S: pheromone deposit только на runs с eval score > 0.7. Как это меняет convergence по сравнению с un-gated версией?
4. Прочитайте LMPSO (arXiv:2504.09247). Сопоставьте "velocity as a prompt" из статьи с вашей numeric velocity. Что теряется в simulation и что сохраняется?
5. Прочитайте AMRO-S (arXiv:2603.12933). Реализуйте decoupled "inference fast-path" с asynchronous pheromone update. Как это меняет system latency под sustained load?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| PSO | "Particle Swarm Optimization" | Kennedy-Eberhart 1995. Population-based gradient-free optimizer. |
| ACO | "Ant Colony Optimization" | Dorigo 1992. Оптимизация paths/routes через pheromone trails. |
| LMPSO | "PSO with LLM generation" | arXiv:2504.09247. Velocity — prompt; LLM производит candidates. |
| Model Swarms | "PSO on expert weights" | arXiv:2410.11163. Gradient-free update на подпространстве model parameters. |
| AMRO-S | "ACO for agent routing" | arXiv:2603.12933. Pheromone matrix по task-type × agent. |
| p_best / g_best | "Personal / global best" | Лучшие найденные решения на уровне particle и всего swarm. |
| Pheromone | "Routing memory" | Сила на edge; decay со временем; deposit по quality. |
| Quality-gated update | "Учиться только на хороших runs" | Pheromone deposit, обусловленный quality check. |
| Catastrophic drift | "Distribution shift" | Fitness landscape меняется; старые p_best и pheromones устаревают. |

## Дополнительное чтение

- [Kennedy & Eberhart — Particle Swarm Optimization](https://ieeexplore.ieee.org/document/488968) — статья PSO 1995 года
- [Dorigo — Ant Colony Optimization](https://www.aco-metaheuristic.org/about.html) — основы ACO 1992 года
- [LMPSO — Language Model Particle Swarm Optimization](https://arxiv.org/abs/2504.09247) — PSO для structured LLM outputs
- [Model Swarms — gradient-free LLM expert optimization](https://arxiv.org/abs/2410.11163) — PSO на подпространстве model weights
- [AMRO-S — ant-colony multi-agent routing](https://arxiv.org/abs/2603.12933) — pheromone-driven routing с quality gate
