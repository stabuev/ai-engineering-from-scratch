# MDPs, States, Actions & Rewards

> Markov Decision Process состоит из пяти вещей: состояний, действий, переходов, наград и discount. Все в RL — Q-learning, PPO, DPO, GRPO — оптимизируется поверх этой формы. Разберитесь с ней один раз, и остальная часть reinforcement learning станет намного проще.

**Type:** Learn
**Languages:** Python
**Prerequisites:** Phase 1 · 06 (Probability & Distributions), Phase 2 · 01 (ML Taxonomy)
**Time:** ~45 minutes

## Цели обучения

- Определять марковский процесс принятия решений через пять компонентов — состояния, действия, переходы, награды и дисконт — и объяснять вклад каждого.
- Прокатывать политику на маленьком MDP и точно вычислять её функцию ценности по уравнению Беллмана.
- Объяснять, что задаёт коэффициент дисконтирования γ и почему у него есть физический смысл (эффективный горизонт).

## Проблема

Вы пишете шахматного бота. Или планировщик запасов. Или торгового агента. Или PPO loop, который обучает reasoning model. Четыре разные области, один неожиданный факт: все четыре сводятся к одному и тому же математическому объекту.

Supervised learning дает вам пары `(x, y)` и просит подобрать функцию. Reinforcement learning не дает меток — только поток состояний, действий, которые вы выбрали, и скалярную награду. Привел ли ход к победе? Сэкономило ли решение о пополнении запасов деньги? Дала ли сделка прибыль? Привел ли только что сгенерированный LLM токен к более высокой награде от judge?

Вы не сможете учиться по этому потоку, пока не формализуете его. "Что я видел", "что я сделал", "что произошло дальше", "насколько это было хорошо" — каждое из этого должно стать объектом, о котором можно рассуждать. Такая формализация называется Markov Decision Process. Каждый RL-алгоритм в этой фазе, включая RLHF и GRPO loops в конце, оптимизируется поверх этой формы.

## Концепция

![Markov decision process: states, actions, transitions, rewards, discount](../assets/mdp.svg)

**Пять объектов.**

- **States** `S`. Все, что нужно агенту для принятия решения. В GridWorld — клетка. В шахматах — доска. В LLM — context window плюс любая память.
- **Actions** `A`. Варианты выбора. Двигаться вверх/вниз/влево/вправо. Сделать ход. Выдать токен.
- **Transitions** `P(s' | s, a)`. Для состояния `s` и действия `a` это распределение следующего состояния. Детерминированное в шахматах, стохастическое в inventory, почти детерминированное при LLM decoding.
- **Rewards** `R(s, a, s')`. Скалярный сигнал. Победа = +1, поражение = -1. Выручка минус стоимость. Log-likelihood ratio term в GRPO.
- **Discount** `γ ∈ [0, 1)`. Насколько будущая награда важна по сравнению с текущей. `γ = 0.99` дает horizon примерно в 100 шагов; `γ = 0.9` дает примерно 10.

**Markov property** `P(s_{t+1} | s_t, a_t) = P(s_{t+1} | s_0, a_0, …, s_t, a_t)`. Будущее зависит только от текущего состояния. Если это не так, представление состояния неполно — это не провал метода, а провал определения состояния.

**Policies and returns.** Policy `π(a | s)` отображает состояния в распределения действий. Return `G_t = r_t + γ r_{t+1} + γ² r_{t+2} + …` — это дисконтированная сумма будущих наград. Value `V^π(s) = E[G_t | s_t = s]` — ожидаемый return при старте из `s` под policy `π`. Q-value `Q^π(s, a) = E[G_t | s_t = s, a_t = a]` — ожидаемый return при старте с конкретного действия. Каждый RL-алгоритм оценивает одно из этих двух значений, а затем соответственно улучшает `π`.

**Bellman equations.** Уравнения неподвижной точки, которые используются во всей этой фазе:

`V^π(s) = Σ_a π(a|s) Σ_{s', r} P(s', r | s, a) [r + γ V^π(s')]`
`Q^π(s, a) = Σ_{s', r} P(s', r | s, a) [r + γ Σ_{a'} π(a'|s') Q^π(s', a')]`

Они раскладывают ожидаемый return на "награду этого шага" плюс "дисконтированную ценность состояния, в которое вы попадете". Рекурсивно. Каждый алгоритм в Phase 9 либо итерирует это уравнение до сходимости (dynamic programming), либо семплирует из него (Monte Carlo), либо bootstraps его на один шаг (temporal difference).

## Практика

### Step 1: a tiny deterministic MDP

GridWorld 4×4. Агент стартует в верхнем левом углу, terminal находится в нижнем правом, награда -1 за шаг, действия `{up, down, left, right}`. См. `code/main.py`.

```python
GRID = 4
TERMINAL = (3, 3)
ACTIONS = {"up": (-1, 0), "down": (1, 0), "left": (0, -1), "right": (0, 1)}

def step(state, action):
    if state == TERMINAL:
        return state, 0.0, True
    dr, dc = ACTIONS[action]
    r, c = state
    nr = min(max(r + dr, 0), GRID - 1)
    nc = min(max(c + dc, 0), GRID - 1)
    return (nr, nc), -1.0, (nr, nc) == TERMINAL
```

Пять строк. Это вся среда. Детерминированные переходы, постоянный штраф за шаг, absorbing terminal state.

### Step 2: roll out a policy

Policy — это функция из состояния в распределение действий. Самая простая: равномерная случайная.

```python
def uniform_policy(state):
    return {a: 0.25 for a in ACTIONS}

def rollout(policy, max_steps=200):
    s, total, steps = (0, 0), 0.0, 0
    for _ in range(max_steps):
        a = sample(policy(s))
        s, r, done = step(s, a)
        total += r
        steps += 1
        if done:
            break
    return total, steps
```

Запустите random policy 1000 раз. Средний return для этой доски 4×4 будет примерно от -60 до -80. Оптимальный return равен -6 (прямой путь вниз-вправо). Закрыть этот разрыв — вся суть Phase 9.

### Step 3: compute `V^π` exactly via the Bellman equation

Для маленьких MDP Bellman equation является линейной системой. Перечислите состояния, примените expectation, итерируйте, пока значения не перестанут меняться.

```python
def policy_evaluation(policy, gamma=0.99, tol=1e-6):
    V = {s: 0.0 for s in all_states()}
    while True:
        delta = 0.0
        for s in all_states():
            if s == TERMINAL:
                continue
            v = 0.0
            for a, pi_a in policy(s).items():
                s_next, r, _ = step(s, a)
                v += pi_a * (r + gamma * V[s_next])
            delta = max(delta, abs(v - V[s]))
            V[s] = v
        if delta < tol:
            return V
```

Это iterative policy evaluation. Это первый алгоритм в Sutton & Barto и теоретический фундамент каждого следующего RL-метода.

### Step 4: `γ` is a hyperparameter with physical meaning

Effective horizon примерно равен `1 / (1 - γ)`. `γ = 0.9` → 10 шагов. `γ = 0.99` → 100 шагов. `γ = 0.999` → 1000 шагов.

Слишком низкое значение делает агента близоруким. Слишком высокое делает credit assignment шумным, потому что многие ранние шаги делят ответственность за далекую будущую награду. LLM RLHF обычно использует `γ = 1`, потому что episodes короткие и ограниченные. Control tasks используют `0.95–0.99`. Длинные strategy games используют `0.999`.

## Pitfalls

- **Non-Markovian state.** Если для решения нужны последние три наблюдения, "state" — это не только текущее наблюдение. Исправление: stack frames (DQN на Atari складывает 4) или recurrent state (LSTM/GRU поверх observations).
- **Sparse rewards.** Награды только за победу делают обучение почти невозможным в больших пространствах состояний. Shape rewards (intermediate signal) или bootstrap with imitation (Phase 9 · 09).
- **Reward hacking.** Оптимизация proxy reward часто порождает патологическое поведение. Boat-racing agent от OpenAI бесконечно крутился кругами, собирая powerups, вместо завершения гонки. Всегда определяйте reward через целевой outcome, а не proxy.
- **Discount mis-spec.** `γ = 1` на infinite-horizon task делает каждую value бесконечной. Всегда ограничивайте либо finite horizon, либо `γ < 1`.
- **Reward scale.** Награды {+100, -100} и {+1, -1} дают одинаковые оптимальные policies, но радикально разные magnitude градиентов. Нормализуйте примерно к `[-1, 1]` перед подачей в PPO/DQN.

## Использование

Стек 2026 года сводит каждый RL pipeline к MDP до написания кода:

| Situation | State | Action | Reward | γ |
|-----------|-------|--------|--------|---|
| Control (locomotion, manipulation) | Joint angles + velocities | Continuous torques | Task-specific shaped | 0.99 |
| Games (chess, Go, poker) | Board + history | Legal move | Win=+1 / loss=-1 | 1.0 (finite) |
| Inventory / pricing | Stock + demand | Order qty | Revenue - cost | 0.95 |
| RLHF for LLMs | Context tokens | Next token | Reward-model score at end | 1.0 (episode ~200 tokens) |
| GRPO for reasoning | Prompt + partial response | Next token | Verifier 0/1 at end | 1.0 |

Запишите пять элементов tuple до написания любого training loop. Большинство баг-репортов "RL does not work" восходят к MDP formulation, которая была сломана уже на бумаге.

## Результат

Сохраните как `outputs/skill-mdp-modeler.md`:

```markdown
---
name: mdp-modeler
description: Given a task description, produce a Markov Decision Process spec and flag formulation risks before training.
version: 1.0.0
phase: 9
lesson: 1
tags: [rl, mdp, modeling]
---

Given a task (control / game / recommendation / LLM fine-tuning), output:

1. State. Exact feature vector or tensor spec. Justify Markov property.
2. Action. Discrete set or continuous range. Dimensionality.
3. Transition. Deterministic, stochastic-with-known-model, or sample-only.
4. Reward. Function and source. Sparse vs shaped. Terminal vs per-step.
5. Discount. Value and horizon justification.

Refuse to ship any MDP where the state is non-Markovian without explicit mention of frame-stacking or recurrent state. Refuse any reward that was not defined in terms of the target outcome. Flag any `γ ≥ 1.0` on an infinite-horizon task. Flag any reward range >100x the typical step reward as a likely gradient-explosion source.
```

## Упражнения

1. **Легко.** Реализуйте 4×4 GridWorld и random-policy rollout в `code/main.py`. Запустите 10,000 episodes. Сообщите mean и std return. Сравните с оптимальным return (-6).
2. **Средне.** Запустите `policy_evaluation` с `γ ∈ {0.5, 0.9, 0.99}` для uniform-random policy. Выведите `V` как сетку 4×4 для каждого значения. Объясните, почему state values рядом с terminal растут быстрее при большем `γ`.
3. **Сложно.** Сделайте GridWorld стохастическим: каждое действие с вероятностью `p = 0.1` соскальзывает в соседнее направление. Переоцените uniform policy. `V[start]` становится лучше или хуже? Почему?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| MDP | "Reinforcement learning setup" | Tuple `(S, A, P, R, γ)`, удовлетворяющий Markov property. |
| State | "What the agent sees" | Достаточная статистика для будущей dynamics при выбранном классе policies. |
| Policy | "Agent's behavior" | Conditional distribution `π(a | s)` или deterministic map `s → a`. |
| Return | "Total reward" | Discounted sum `Σ γ^t r_t` от текущего шага. |
| Value | "How good a state is" | Expected return under `π` starting from `s`. |
| Q-value | "How good an action is" | Expected return under `π` starting from `s` with first action `a`. |
| Bellman equation | "Dynamic programming recursion" | Fixed-point decomposition of value / Q into one-step reward plus discounted successor value. |
| Discount `γ` | "Future vs present" | Geometric weight on far-future reward; effective horizon `~1/(1-γ)`. |

## Дополнительное чтение

- [Sutton & Barto (2018). Reinforcement Learning: An Introduction, 2nd ed.](http://incompleteideas.net/book/RLbook2020.pdf) — основной учебник. Ch. 3 рассматривает MDPs and Bellman equations; Ch. 1 мотивирует reward hypothesis, лежащую в основе каждого следующего урока.
- [Bellman (1957). Dynamic Programming](https://press.princeton.edu/books/paperback/9780691146683/dynamic-programming) — источник Bellman equation.
- [OpenAI Spinning Up — Part 1: Key Concepts](https://spinningup.openai.com/en/latest/spinningup/rl_intro.html) — краткий MDP primer с позиции deep-RL.
- [Puterman (2005). Markov Decision Processes](https://onlinelibrary.wiley.com/doi/book/10.1002/9780470316887) — operations-research reference по MDPs и exact solution methods.
- [Littman (1996). Algorithms for Sequential Decision Making (PhD thesis)](https://www.cs.rutgers.edu/~mlittman/papers/thesis-main.pdf) — самое чистое выведение MDPs как специализации dynamic programming.
