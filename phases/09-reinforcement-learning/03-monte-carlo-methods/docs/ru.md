# Monte Carlo Methods — Learning from Complete Episodes

> Dynamic programming требует модель. Monte Carlo не требует ничего, кроме episodes. Запустите policy, посмотрите returns, усредните их. Самая простая идея в RL — и та, которая открывает все последующие методы.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 01 (MDPs), Phase 9 · 02 (Dynamic Programming)
**Time:** ~75 minutes

## Проблема

Dynamic programming элегантен, но предполагает, что можно запросить `P(s' | s, a)` для каждого состояния и действия. Почти ничто в реальном мире так не работает. Робот не может аналитически вычислить распределение пикселей камеры после torque в суставе. Pricing algorithm не может проинтегрировать все возможные реакции клиентов. LLM не может перечислить все возможные продолжения после токена.

Нужен метод, которому достаточно способности *sample* из среды. Запустите policy. Получите trajectory `s_0, a_0, r_1, s_1, a_1, r_2, …, s_T`. Используйте ее для оценки values. Это Monte Carlo.

Переход от DP к MC важен концептуально: мы переходим от *known model + exact backup* к *sampled rollouts + averaged return*. Variance растет, но применимость резко расширяется. Каждый RL-алгоритм после этого урока — TD, Q-learning, REINFORCE, PPO, GRPO — в основе является Monte Carlo estimator, иногда с bootstrapping поверх.

## Концепция

![Monte Carlo: rollout, compute returns, average; first-visit vs every-visit](../assets/monte-carlo.svg)

**Основная идея в одну строку:** `V^π(s) = E_π[G_t | s_t = s] ≈ (1/N) Σ_i G^{(i)}(s)`, где `G^{(i)}(s)` — observed returns после visits to `s` under policy `π`.

**First-visit vs every-visit MC.** Если episode посещает state `s` несколько раз, first-visit MC учитывает return только с первого посещения; every-visit MC учитывает все посещения. Оба несмещены в пределе. First-visit проще анализировать (iid samples). Every-visit использует больше данных из episode и обычно быстрее сходится на практике.

**Incremental mean.** Вместо хранения всех returns обновляйте running average:

`V_n(s) = V_{n-1}(s) + (1/n) [G_n - V_{n-1}(s)]`

Перепишите: `V_new = V_old + α · (target - V_old)` с `α = 1/n`. Замените `1/n` на постоянный step-size `α ∈ (0, 1)`, и получится non-stationary MC estimator, который отслеживает изменения в `π`. Этот шаг — весь прыжок от MC к TD и далее к современным RL algorithms.

**Exploration is now a problem.** DP касался каждого состояния перечислением. MC видит только states, которые посещает policy. Если `π` детерминирована, целые области state space никогда не sampled, и их value estimates навсегда остаются нулевыми. Три исправления в историческом порядке:

1. **Exploring starts.** Начинать каждый episode из случайной пары (s, a). Гарантирует coverage; нереалистично на практике (вы не можете "reset" robot в произвольное состояние).
2. **ε-greedy.** Действовать greedy относительно текущего Q, но с вероятностью `ε` выбирать случайное действие. Все state-action pairs asymptotically sampled.
3. **Off-policy MC.** Collect data под behavior policy `μ`, learn about target policy `π` через importance sampling. High variance, но это мост к replay-buffer methods вроде DQN.

**Monte Carlo Control.** Evaluate → improve → evaluate, как policy iteration, но evaluation основан на sampling:

1. Запустить `π`, получить episode.
2. Обновить `Q(s, a)` по observed returns.
3. Сделать `π` ε-greedy относительно `Q`.
4. Повторять.

Сходится к `Q*` и `π*` с вероятностью 1 при мягких условиях (каждая пара посещается бесконечно часто, `α` удовлетворяет Robbins-Monro).

## Практика

### Step 1: rollout → list of (s, a, r)

```python
def rollout(env, policy, max_steps=200):
    trajectory = []
    s = env.reset()
    for _ in range(max_steps):
        a = policy(s)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r))
        s = s_next
        if done:
            break
    return trajectory
```

Нет модели, только `env.reset()` и `env.step(s, a)`. Тот же interface, что у gym environment, но stripped down.

### Step 2: compute returns (reverse sweep)

```python
def returns_from(trajectory, gamma):
    returns = []
    G = 0.0
    for _, _, r in reversed(trajectory):
        G = r + gamma * G
        returns.append(G)
    return list(reversed(returns))
```

Один проход, `O(T)`. Backward recurrence `G_t = r_{t+1} + γ G_{t+1}` избегает повторного суммирования.

### Step 3: first-visit MC evaluation

```python
def mc_policy_evaluation(env, policy, episodes, gamma=0.99):
    V = defaultdict(float)
    counts = defaultdict(int)
    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for t, ((s, _, _), G) in enumerate(zip(trajectory, returns)):
            if s in seen:
                continue
            seen.add(s)
            counts[s] += 1
            V[s] += (G - V[s]) / counts[s]
    return V
```

Три строки делают работу: пометить state как seen при первом посещении, увеличить count, обновить running mean.

### Step 4: ε-greedy MC control (on-policy)

```python
def mc_control(env, episodes, gamma=0.99, epsilon=0.1):
    Q = defaultdict(lambda: {a: 0.0 for a in ACTIONS})
    counts = defaultdict(lambda: {a: 0 for a in ACTIONS})

    def policy(s):
        if random() < epsilon:
            return choice(ACTIONS)
        return max(Q[s], key=Q[s].get)

    for _ in range(episodes):
        trajectory = rollout(env, policy)
        returns = returns_from(trajectory, gamma)
        seen = set()
        for (s, a, _), G in zip(trajectory, returns):
            if (s, a) in seen:
                continue
            seen.add((s, a))
            counts[s][a] += 1
            Q[s][a] += (G - Q[s][a]) / counts[s][a]
    return Q, policy
```

### Step 5: compare to DP gold standard

Ваша MC estimate of `V^π` должна совпадать с DP result из Lesson 02 при episodes → ∞. На практике: 50,000 episodes на 4×4 GridWorld дают точность в пределах `~0.1` от DP answer.

## Pitfalls

- **Infinite episodes.** MC требует, чтобы episodes *terminate*. Если policy может зациклиться навсегда, ограничьте `max_steps` и трактуйте cap как implicit failure. GridWorld со случайной policy регулярно times out — это нормально, просто учитывайте это правильно.
- **Variance.** MC использует full returns. На длинных episodes variance огромна — одна неудачная награда в конце сдвигает `V(s_0)` на ту же величину. TD methods (Lesson 04) уменьшают это через bootstrapping.
- **State coverage.** Greedy MC на свежем Q с ties будет пробовать только одно действие. Нужно exploration (ε-greedy, exploring starts, UCB).
- **Non-stationary policies.** Если `π` меняется (как в MC control), старые returns пришли от другой policy. Constant-α MC справляется; sample-average MC — нет.
- **Off-policy importance sampling.** Веса `π(a|s)/μ(a|s)` перемножаются вдоль trajectory. Variance взрывается с horizon. Ограничивайте per-decision weighted IS или переходите к TD.

## Использование

Роль Monte Carlo methods в 2026:

| Use case | Why MC |
|----------|--------|
| Short-horizon games (blackjack, poker) | Episodes terminate naturally; returns are clean. |
| Offline evaluation of a logged policy | Average discounted returns over stored trajectories. |
| Monte Carlo Tree Search (AlphaZero) | MC rollouts from tree leaves guide selection. |
| LLM RL evaluation | Compute average reward over sampled completions for a given policy. |
| Baseline estimation in PPO | The advantage target `A_t = G_t - V(s_t)` uses an MC `G_t`. |
| Teaching RL | Simplest algorithm that actually works — strip bootstrapping to see the core. |

Современные deep-RL algorithms (PPO, SAC) интерполируют между pure MC (full returns) и pure TD (one-step bootstrap) через `n`-step returns или GAE. Оба конца — экземпляры одного estimator.

## Результат

Сохраните как `outputs/skill-mc-evaluator.md`:

```markdown
---
name: mc-evaluator
description: Evaluate a policy via Monte Carlo rollouts and produce a convergence report with DP-comparison if available.
version: 1.0.0
phase: 9
lesson: 3
tags: [rl, monte-carlo, evaluation]
---

Given an environment (episodic, with reset+step API) and a policy, output:

1. Method. First-visit vs every-visit MC. Reason.
2. Episode budget. Target number, variance diagnostic, expected standard error.
3. Exploration plan. ε schedule (if needed) or exploring starts.
4. Gold-standard comparison. DP-optimal V* if tabular; otherwise a bound from a Q-learning / PPO baseline.
5. Termination check. Max-step cap, timeouts, handling of non-terminating trajectories.

Refuse to run MC on non-episodic tasks without a finite horizon cap. Refuse to report V^π estimates from fewer than 100 episodes per state for tabular tasks. Flag any policy with zero-variance actions as an exploration risk.
```

## Упражнения

1. **Легко.** Реализуйте first-visit MC evaluation для uniform-random policy на 4×4 GridWorld. Запустите 10,000 episodes. Постройте `V(0,0)` как функцию episode count против DP answer.
2. **Средне.** Реализуйте ε-greedy MC control с `ε ∈ {0.01, 0.1, 0.3}`. Сравните mean return после 20,000 episodes. Как выглядит кривая? Где находится bias-variance tradeoff?
3. **Сложно.** Реализуйте *off-policy* MC с importance sampling: collect data under uniform-random policy `μ`, estimate `V^π` для deterministic optimal policy `π`. Сравните plain IS vs per-decision IS vs weighted IS. У какого variance меньше?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Monte Carlo | "Random sampling" | Оценка expectations усреднением iid samples из distribution. |
| Return `G_t` | "Future reward" | Сумма discounted rewards от шага `t` до конца episode: `Σ_{k≥0} γ^k r_{t+k+1}`. |
| First-visit MC | "Count each state once" | Только первое посещение в episode вносит вклад в value estimate. |
| Every-visit MC | "Use all visits" | Вклад вносит каждое посещение; slightly biased, но sample-efficient. |
| ε-greedy | "Exploration noise" | Выбрать greedy action с prob `1-ε`; random action с prob `ε`. |
| Importance sampling | "Correcting for sampling from the wrong distribution" | Reweight returns произведениями `π(a|s)/μ(a|s)`, чтобы оценить `V^π` по данным `μ`. |
| On-policy | "Learn from my own data" | Target policy = behavior policy. Vanilla MC, PPO, SARSA. |
| Off-policy | "Learn from someone else's data" | Target policy ≠ behavior policy. Importance-sampled MC, Q-learning, DQN. |

## Дополнительное чтение

- [Sutton & Barto (2018). Ch. 5 — Monte Carlo Methods](http://incompleteideas.net/book/RLbook2020.pdf) — каноническое изложение.
- [Singh & Sutton (1996). Reinforcement Learning with Replacing Eligibility Traces](https://link.springer.com/article/10.1007/BF00114726) — анализ first-visit vs every-visit.
- [Precup, Sutton, Singh (2000). Eligibility Traces for Off-Policy Policy Evaluation](http://incompleteideas.net/papers/PSS-00.pdf) — off-policy MC and variance control.
- [Mahmood et al. (2014). Weighted Importance Sampling for Off-Policy Learning](https://arxiv.org/abs/1404.6362) — современные low-variance IS estimators.
- [Tesauro (1995). TD-Gammon, A Self-Teaching Backgammon Program](https://dl.acm.org/doi/10.1145/203330.203343) — первая крупная empirical demonstration того, как MC/TD self-play сходится к superhuman play; концептуальный предшественник второй половины этой фазы.
