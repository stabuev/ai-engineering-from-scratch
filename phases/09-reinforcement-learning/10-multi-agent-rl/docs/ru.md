# Multi-Agent RL

> Single-agent RL предполагает, что среда стационарна. Поместите двух обучающихся агентов в один и тот же мир, и это предположение ломается: каждый агент становится частью среды другого, и оба меняются. Multi-agent RL — это набор приемов, позволяющих обучению сходиться, когда Markov assumption больше не выполняется.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 04 (Q-learning), Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~45 minutes

## Проблема

Робот, обучающийся перемещаться по комнате, — это single-agent RL задача. Футбольная команда — нет. AlphaStar против соперников в StarCraft — нет. Маркетплейс bidding agents — нет. Две машины, договаривающиеся на перекрестке с четырехсторонним stop-знаком, — нет. Многие реальные задачи many-on-many — нет.

В любой multi-agent setting, с точки зрения любого одного агента, другие агенты *являются* частью среды. По мере того как они обучаются и меняют свое поведение, среда становится non-stationary. Markov property — "следующее состояние зависит только от текущего состояния и моего действия" — нарушается, потому что следующее состояние также зависит от того, что выбрали *другие* агенты, а их политики являются moving targets.

Это ломает доказательства сходимости для табличных методов (гарантия Q-learning предполагает стационарную среду). Это ломает и наивный deep RL: агенты гоняются друг за другом по циклам и никогда не сходятся к стабильной политике. Нужны техники, специфичные для multi-agent: centralized training / decentralized execution, counterfactual baselines, league play, self-play.

Приложения 2026 года: robot swarms, traffic routing, флоты autonomous vehicles, market simulators, multi-agent LLM systems (Phase 16) и любая игра с более чем одним интеллектуальным игроком.

## Концепция

![Четыре режима MARL: indep, centralized critic, self-play, league](../assets/marl.svg)

**Формализм: Markov Game.** Обобщение MDP: состояния `S`, joint action `a = (a_1, …, a_n)`, переход `P(s' | s, a)` и награды по агентам `R_i(s, a, s')`. Каждый агент `i` максимизирует собственный return в рамках собственной политики `π_i`. Если награды идентичны, это **fully cooperative**. Если zero-sum, это **adversarial**. Если смешанный случай, это **general-sum**.

**Ключевые сложности:**

- **Non-stationarity.** `P(s' | s, a_i)` с точки зрения агента `i` зависит от `π_{-i}`, которая меняется.
- **Credit assignment.** При общей награде какой агент ее вызвал?
- **Exploration coordination.** Агенты должны исследовать взаимодополняющие стратегии, а не избыточно исследовать одно и то же состояние.
- **Scalability.** Joint action space растет экспоненциально по `n`.
- **Partial observability.** Каждый агент видит только свое наблюдение; глобальное состояние скрыто.

**Четыре доминирующих режима:**

**1. Independent Q-learning / independent PPO (IQL, IPPO).** Каждый агент обучает свой Q или policy, рассматривая остальных как часть среды. Просто, иногда работает (особенно когда experience replay действует как сглаживающий прием agent modeling). Теоретическая сходимость: отсутствует. На практике: нормально для слабо связанных задач, плохо для tightly-coupled задач.

**2. Centralized training, decentralized execution (CTDE).** Самая распространенная современная парадигма. У каждого агента есть собственная *policy* `π_i`, которая зависит от локального наблюдения `o_i` — стандартное decentralized execution при deployment. Во время *training* centralized critic `Q(s, a_1, …, a_n)` зависит от полного глобального состояния и joint action. Примеры:
- **MADDPG** (Lowe et al. 2017): DDPG с centralized critic для каждого агента.
- **COMA** (Foerster et al. 2017): counterfactual baseline — вопрос "какой была бы моя награда, если бы я вместо этого выбрал действие `a'`?" — изолирует мой вклад.
- **MAPPO** / **IPPO** with shared critic (Yu et al. 2022): PPO с centralized value function. Доминирует в 2026 году для cooperative MARL.
- **QMIX** (Rashid et al. 2018): value decomposition — `Q_tot(s, a) = f(Q_1(s, a_1), …, Q_n(s, a_n))` с monotonic mixing.

**3. Self-play.** Две копии одного и того же агента играют друг против друга. Policy оппонента *является* моей policy из прошлого snapshot. AlphaGo / AlphaZero / MuZero. OpenAI Five. Лучше всего работает для zero-sum games; training signal симметричен.

**4. League play.** Расширение self-play на general-sum / adversarial environments: держать population прошлых и текущих policies, выбирать оппонента из league, обучаться против него. Добавляет exploiters (специализируются на победе над текущим лучшим) и main exploiters (специализируются на победе над exploiters). AlphaStar (StarCraft II). Нужно, когда игра допускает циклы стратегий "rock-paper-scissors".

**Communication.** Разрешите агентам отправлять друг другу выученные сообщения `m_i`. Работает в cooperative settings. Foerster et al. (2016) показали, что differentiable inter-agent communication можно обучать end-to-end. Сегодняшние LLM-based multi-agent systems (Phase 16) по сути коммуницируют на естественном языке.

## Соберите это

В этом уроке используется 6×6 GridWorld с двумя cooperative agents. Они стартуют в противоположных углах и должны достичь общей цели. Shared reward: `-1` за шаг, пока хотя бы один агент еще движется, `+10`, когда оба пришли. См. `code/main.py`.

### Шаг 1: multi-agent env

```python
class CoopGridWorld:
    def __init__(self):
        self.size = 6
        self.goal = (5, 5)

    def reset(self):
        return ((0, 0), (5, 0))  # two agents

    def step(self, state, actions):
        a1, a2 = state
        new1 = move(a1, actions[0])
        new2 = move(a2, actions[1])
        done = (new1 == self.goal) and (new2 == self.goal)
        reward = 10.0 if done else -1.0
        return (new1, new2), reward, done
```

*Joint* action space равен `|A|² = 16`. Глобальное состояние — это две позиции.

### Шаг 2: independent Q-learning

Каждый агент ведет собственную Q-table, индексированную по joint state. На каждом шаге: оба выбирают ε-greedy actions, получают joint transition, каждый обновляет свой Q с общей наградой.

```python
def independent_q(env, episodes, alpha, gamma, epsilon):
    Q1, Q2 = defaultdict(default_q), defaultdict(default_q)
    for _ in range(episodes):
        s = env.reset()
        while not done:
            a1 = epsilon_greedy(Q1, s, epsilon)
            a2 = epsilon_greedy(Q2, s, epsilon)
            s_next, r, done = env.step(s, (a1, a2))
            target1 = r + gamma * max(Q1[s_next].values())
            target2 = r + gamma * max(Q2[s_next].values())
            Q1[s][a1] += alpha * (target1 - Q1[s][a1])
            Q2[s][a2] += alpha * (target2 - Q2[s][a2])
            s = s_next
```

Работает на этой задаче, потому что награды dense и aligned. Проваливается на tightly-coupled задачах (например, там, где одному агенту нужно *ждать* другого).

### Шаг 3: centralized Q с decomposed-value update

Используйте один Q по joint actions `Q(s, a_1, a_2)`. Обновляйте его по shared reward. Децентрализуйте при execution через маргинализацию: `π_i(s) = argmax_{a_i} max_{a_{-i}} Q(s, a_1, a_2)`. Это обменивает экспоненциальный joint action space на *корректный* глобальный взгляд.

### Шаг 4: простой self-play (adversarial 2-agent)

Один и тот же агент, две роли. Обучайте агента A против агента B; после `K` эпизодов скопируйте веса A в B. Симметричное обучение, последовательный прогресс. Рецепт AlphaZero в миниатюре.

## Подводные камни

- **Non-stationary replay.** Experience replay с independent agents хуже, чем в single-agent случае, потому что старые transitions были сгенерированы уже устаревшими opponents. Исправление: relabel или взвешивание по recency.
- **Credit assignment ambiguity.** Общая награда после длинного эпизода; нет ясного способа сказать, какой агент внес вклад. Исправление: counterfactual baselines (COMA) или reward shaping по агентам.
- **Policy drift / chasing.** Best response каждого агента меняется с каждым обновлением другого. Исправление: centralized critic, медленные learning rates или freeze-one-at-a-time.
- **Reward hacking via coordination.** Агенты находят coordinated exploits, которых дизайнер не ожидал. Auction agents сходятся к bid zero. Исправление: аккуратный reward design, behavioral constraints.
- **Exploration redundancy.** Оба агента исследуют одни и те же state-action pairs. Исправление: entropy bonuses per-agent или role-conditioning.
- **League cycles.** Чистый self-play может застрять в dominance cycle. Исправление: league play с разнообразными opponents.
- **Sample explosion.** `n` agents × state space × joint actions. Аппроксимируйте с помощью function approximation; factored action spaces (одна policy output head на агента).

## Используйте это

Карта приложений MARL в 2026 году:

| Domain | Method | Notes |
|--------|--------|-------|
| Cooperative navigation / manipulation | MAPPO / QMIX | CTDE; shared critic + decentralized actors. |
| Two-player games (chess, Go, poker) | Self-play with MCTS (AlphaZero) | Zero-sum; symmetric training. |
| Complex multiplayer (Dota, StarCraft) | League play + imitation pretraining | OpenAI Five, AlphaStar. |
| Autonomous-vehicle fleets | CTDE MAPPO / PPO with attention | Partial obs; variable team sizes. |
| Auction markets | Game-theoretic equilibrium + RL | Mean-field RL when `n` → ∞. |
| LLM multi-agent systems (Phase 16) | Natural-language comm + role conditioning | RL loop at the agent-planning layer. |

В 2026 году самая большая зона роста MARL — LLM-based: рои language-model agents, которые ведут переговоры, спорят и строят software. RL появляется как preference optimization на *trajectory-level* outputs, а не token-level (Phase 16 · 03).

## Отгрузите это

Сохраните как `outputs/skill-marl-architect.md`:

```markdown
---
name: marl-architect
description: Pick the right multi-agent RL regime (IPPO, CTDE, self-play, league) for a given task.
version: 1.0.0
phase: 9
lesson: 10
tags: [rl, multi-agent, marl, self-play]
---

Given a task with `n` agents, output:

1. Regime classification. Cooperative / adversarial / general-sum. Justify.
2. Algorithm. IPPO / MAPPO / QMIX / self-play / league. Reason tied to coupling tightness and reward structure.
3. Information access. Centralized training (what global info goes to the critic)? Decentralized execution?
4. Credit assignment. Counterfactual baseline, value decomposition, or reward shaping.
5. Exploration plan. Per-agent entropy, population-based training, or league.

Refuse independent Q-learning on tightly-coupled cooperative tasks. Refuse to recommend self-play for general-sum with cycle risks. Flag any MARL pipeline without a fixed-opponent eval (cherry-picked self-play numbers are common).
```

## Упражнения

1. **Easy.** Обучите independent Q-learning на 2-agent cooperative GridWorld. Сколько эпизодов нужно, пока mean return > 0? Постройте joint learning curve.
2. **Medium.** Добавьте задачу "coordination": цель считается достигнутой только когда оба агента наступают на нее в один и тот же ход. Independent Q все еще сходится? Что ломается?
3. **Hard.** Реализуйте centralized critic для MAPPO-style training и сравните скорость сходимости с independent PPO на coordination task.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Markov game | "Multi-agent MDP" | `(S, A_1, …, A_n, P, R_1, …, R_n)`; у каждого агента своя награда. |
| CTDE | "Centralized training, decentralized execution" | Joint critic во время training; policy каждого агента использует только local obs. |
| IPPO | "Independent PPO" | Каждый агент запускает PPO отдельно. Простой baseline; часто недооценен. |
| MAPPO | "Multi-agent PPO" | PPO с centralized value function, зависящей от global state. |
| QMIX | "Monotonic value decomposition" | `Q_tot = f_monotone(Q_1, …, Q_n)` позволяет decentralized argmax. |
| COMA | "Counterfactual multi-agent" | Advantage = мой Q минус expected Q с маргинализацией по моему действию. |
| Self-play | "Agent vs past self" | Один агент, две роли; стандарт для zero-sum games. |
| League play | "Population training" | Кэшируйте прошлые policies, выбирайте opponents из pool; обрабатывает strategy cycles. |

## Дополнительное чтение

- [Lowe et al. (2017). Multi-Agent Actor-Critic for Mixed Cooperative-Competitive Environments (MADDPG)](https://arxiv.org/abs/1706.02275) — CTDE с centralized critic.
- [Foerster et al. (2017). Counterfactual Multi-Agent Policy Gradients (COMA)](https://arxiv.org/abs/1705.08926) — counterfactual baselines для credit assignment.
- [Rashid et al. (2018). QMIX: Monotonic Value Function Factorisation](https://arxiv.org/abs/1803.11485) — value decomposition с monotonicity.
- [Yu et al. (2022). The Surprising Effectiveness of PPO in Cooperative Multi-Agent Games (MAPPO)](https://arxiv.org/abs/2103.01955) — PPO неожиданно силен для MARL.
- [Vinyals et al. (2019). Grandmaster level in StarCraft II using multi-agent reinforcement learning (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z) — league play в масштабе.
- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270) — чистый self-play в zero-sum games.
- [Sutton & Barto (2018). Ch. 15 — Neuroscience & Ch. 17 — Frontiers](http://incompleteideas.net/book/RLbook2020.pdf) — включает краткое изложение multi-agent settings и проблемы non-stationarity, для решения которой разработан CTDE.
- [Zhang, Yang & Başar (2021). Multi-Agent Reinforcement Learning: A Selective Overview](https://arxiv.org/abs/1911.10635) — обзор cooperative, competitive и mixed MARL с результатами сходимости.
