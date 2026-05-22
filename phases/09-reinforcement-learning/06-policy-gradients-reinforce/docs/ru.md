# Policy Gradient — REINFORCE from Scratch

> Перестаньте оценивать value. Параметризуйте policy напрямую, вычислите gradient expected return и сделайте шаг вверх. Williams (1992) записал это одной теоремой. Поэтому существуют PPO, GRPO и каждый LLM RL loop.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 03 (Monte Carlo), Phase 9 · 04 (TD Learning)
**Time:** ~75 minutes

## The Problem

Q-learning и DQN параметризуют *value* function. Actions выбираются через `argmax Q`. Это нормально для discrete actions и discrete states. Но ломается, когда actions continuous (какой `argmax` по 10-dimensional torque?) или когда нужна stochastic policy (`argmax` по конструкции deterministic).

Policy gradients параметризуют *policy* вместо value. `π_θ(a | s)` — neural net, которая выдает distribution over actions. Семплируйте из нее, чтобы действовать. Вычисляйте gradient expected return относительно `θ`. Делайте шаг вверх. Нет `argmax`. Нет Bellman recursion. Только gradient ascent по `J(θ) = E_{π_θ}[G]`.

Теорема REINFORCE (Williams 1992) говорит, что этот gradient вычислим: `∇J(θ) = E_π[ G · ∇_θ log π_θ(a | s) ]`. Запустите episode. Посчитайте return. Умножьте на `∇ log π_θ(a | s)` на каждом шаге. Усредните. Gradient-ascent. Готово.

Каждый LLM-RL algorithm в 2026 — PPO, DPO, GRPO — refinement of REINFORCE. Понять его руками — prerequisite для остальной части фазы, а также для Phase 10 · 07 (RLHF implementation) и Phase 10 · 08 (DPO).

## The Concept

![Policy gradient: softmax policy, log-π gradient, return-weighted update](../assets/policy-gradient.svg)

**The policy gradient theorem.** Для любой policy `π_θ`, parameterized by `θ`:

`∇J(θ) = E_{τ ~ π_θ}[ Σ_{t=0}^{T} G_t · ∇_θ log π_θ(a_t | s_t) ]`

где `G_t = Σ_{k=t}^{T} γ^{k-t} r_{k+1}` — discounted return from step `t`. Expectation берется по full trajectories `τ`, sampled from `π_θ`.

**The proof is short.** Дифференцируйте `J(θ) = Σ_τ P(τ; θ) G(τ)` под expectation. Используйте `∇P(τ; θ) = P(τ; θ) ∇ log P(τ; θ)` (log-derivative trick). Разложите `log P(τ; θ) = Σ log π_θ(a_t | s_t) + environment terms that do not depend on θ`. Environment terms исчезают. Две строки алгебры дают theorem.

**Variance reduction tricks.** Vanilla REINFORCE имеет убийственную variance — returns noisy, `∇ log π` noisy, их произведение очень noisy. Два стандартных исправления:

1. **Baseline subtraction.** Замените `G_t` на `G_t - b(s_t)` для любого baseline `b(s_t)`, не зависящего от `a_t`. Несмещено, потому что `E[b(s_t) · ∇ log π(a_t | s_t)] = 0`. Typical choice: `b(s_t) = V̂(s_t)`, learned by critic → actor-critic (Lesson 07).
2. **Reward-to-go.** Замените `Σ_t G_t · ∇ log π_θ(a_t | s_t)` на `Σ_t G_t^{from t} · ∇ log π_θ(a_t | s_t)`. Для данного action важны только future returns — past rewards дают zero-mean noise.

Вместе получается:

`∇J ≈ (1/N) Σ_{i=1}^{N} Σ_{t=0}^{T_i} [ G_t^{(i)} - V̂(s_t^{(i)}) ] · ∇_θ log π_θ(a_t^{(i)} | s_t^{(i)})`

Это REINFORCE with a baseline — прямой предок A2C (Lesson 07) и PPO (Lesson 08).

**Softmax policy parameterization.** Для discrete actions стандартный выбор:

`π_θ(a | s) = exp(f_θ(s, a)) / Σ_{a'} exp(f_θ(s, a'))`

где `f_θ` — любая neural net, которая выдает score per action. Gradient имеет чистую форму:

`∇_θ log π_θ(a | s) = ∇_θ f_θ(s, a) - Σ_{a'} π_θ(a' | s) ∇_θ f_θ(s, a')`

то есть score of taken action minus its expected value under policy.

**Gaussian policy for continuous actions.** `π_θ(a | s) = N(μ_θ(s), σ_θ(s))`. `∇ log N(a; μ, σ)` имеет closed form. Этого достаточно для SAC в Phase 9 · 07.

## Build It

### Step 1: softmax policy network

```python
def policy_logits(theta, state_features):
    return [dot(theta[a], state_features) for a in range(N_ACTIONS)]

def softmax(logits):
    m = max(logits)
    exps = [exp(l - m) for l in logits]
    Z = sum(exps)
    return [e / Z for e in exps]
```

Используйте linear policy (один weight vector per action) для tabular env. Для Atari замените на CNN и оставьте softmax head.

### Step 2: sampling and log-probability

```python
def sample_action(probs, rng):
    x = rng.random()
    cum = 0
    for a, p in enumerate(probs):
        cum += p
        if x <= cum:
            return a
    return len(probs) - 1

def log_prob(probs, a):
    return log(probs[a] + 1e-12)
```

### Step 3: rollout with log-probs captured

```python
def rollout(theta, env, rng, gamma):
    trajectory = []
    s = env.reset()
    while not done:
        logits = policy_logits(theta, s)
        probs = softmax(logits)
        a = sample_action(probs, rng)
        s_next, r, done = env.step(s, a)
        trajectory.append((s, a, r, probs))
        s = s_next
    return trajectory
```

### Step 4: REINFORCE update

```python
def reinforce_step(theta, trajectory, gamma, lr, baseline=0.0):
    returns = compute_returns(trajectory, gamma)
    for (s, a, _, probs), G in zip(trajectory, returns):
        advantage = G - baseline
        grad_log_pi_a = [-p for p in probs]
        grad_log_pi_a[a] += 1.0
        for i in range(N_ACTIONS):
            for j in range(len(s)):
                theta[i][j] += lr * advantage * grad_log_pi_a[i] * s[j]
```

Gradient `∇ log π(a|s) = e_a - π(·|s)` (onehot of `a` minus probabilities) — сердце softmax policy gradients. Доведите это до автоматизма.

### Step 5: baselines

Running mean of `G` по recent episodes уже достаточно снижает variance, чтобы запустить 4×4 GridWorld; сходимость занимает ~500 episodes. Замените baseline на learned `V̂(s)`, и получите actor-critic.

## Pitfalls

- **Exploding gradients.** Returns могут быть huge. Всегда normalize `G` к `~N(0, 1)` по batch перед умножением на `∇ log π`.
- **Entropy collapse.** Policy слишком рано становится near-deterministic, перестает exploring и застревает. Fix: add entropy bonus `β · H(π(·|s))` to objective.
- **High variance.** Vanilla REINFORCE требует тысячи episodes. Critic baseline (Lesson 07) или trust region TRPO/PPO (Lesson 08) — стандартное исправление.
- **Sample inefficiency.** On-policy означает, что каждый transition выбрасывается после одного update. Off-policy corrections через importance sampling возвращают данные ценой variance (PPO ratio — clipped IS weight).
- **Non-stationary gradients.** Gradient из 100 episodes назад использует old `π`. Поэтому on-policy methods обновляются каждые несколько rollouts.
- **Credit assignment.** Без reward-to-go past rewards добавляют noise. Всегда используйте reward-to-go.

## Use It

В 2026 REINFORCE редко запускают напрямую, но его gradient formula везде:

| Use case | Derived method |
|----------|---------------|
| Continuous control | PPO / SAC with Gaussian policy |
| LLM RLHF | PPO with KL penalty, running on token-level policy |
| LLM reasoning (DeepSeek) | GRPO — REINFORCE with group-relative baseline, no critic |
| Multi-agent | Centralized-critic REINFORCE (MADDPG, COMA) |
| Discrete action robotics | A2C, A3C, PPO |
| Preference-only settings | DPO — REINFORCE rewritten as a preference-likelihood loss, no sampling |

Когда в training script 2026 года видите `loss = -advantage * log_prob`, это REINFORCE with a baseline. Целые papers (DPO, GRPO, RLOO) — variance-reduction tricks поверх одной строки.

## Ship It

Сохраните как `outputs/skill-policy-gradient-trainer.md`:

```markdown
---
name: policy-gradient-trainer
description: Produce a REINFORCE / actor-critic / PPO training config for a given task and diagnose variance issues.
version: 1.0.0
phase: 9
lesson: 6
tags: [rl, policy-gradient, reinforce]
---

Given an environment (discrete / continuous actions, horizon, reward stats), output:

1. Policy head. Softmax (discrete) or Gaussian (continuous) with parameter counts.
2. Baseline. None (vanilla), running mean, learned `V̂(s)`, or A2C critic.
3. Variance controls. Reward-to-go on by default, return normalization, gradient clip value.
4. Entropy bonus. Coefficient β and decay schedule.
5. Batch size. Episodes per update; on-policy data freshness contract.

Refuse REINFORCE-no-baseline on horizons > 500 steps. Refuse continuous-action control with a softmax head. Flag any run with `β = 0` and observed policy entropy < 0.1 as entropy-collapsed.
```

## Exercises

1. **Easy.** Реализуйте REINFORCE на 4×4 GridWorld с linear softmax policy. Train for 1,000 episodes без baseline. Постройте learning curve; измерьте variance (std of returns).
2. **Medium.** Добавьте running-mean baseline. Train again. Сравните sample efficiency и variance с vanilla run. Насколько baseline уменьшает steps to convergence?
3. **Hard.** Добавьте entropy bonus `β · H(π)`. Sweep `β ∈ {0, 0.01, 0.1, 1.0}`. Постройте final return and policy entropy. Где sweet spot on this task?

## Key Terms

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Policy gradient | "Train the policy directly" | `∇J(θ) = E[G · ∇ log π_θ(a|s)]`; derived from log-derivative trick. |
| REINFORCE | "The original PG algorithm" | Williams (1992); Monte Carlo returns multiplied by log-policy gradient. |
| Log-derivative trick | "Score function estimator" | `∇P(τ;θ) = P(τ;θ) · ∇ log P(τ;θ)`; makes gradients of expectations tractable. |
| Baseline | "Variance reduction" | Любой `b(s)`, вычитаемый из `G`; unbiased because `E[b · ∇ log π] = 0`. |
| Reward-to-go | "Only future returns count" | `G_t^{from t}` вместо полного `G_0`; correct and lower-variance. |
| Entropy bonus | "Encourage exploration" | `+β · H(π(·|s))` term keeps policy from collapsing. |
| On-policy | "Train on what you just saw" | Gradient expectation is w.r.t. current policy — old data cannot be reused directly. |
| Advantage | "How much better than average" | `A(s, a) = G(s, a) - V(s)`; signed quantity multiplied by REINFORCE-with-baseline. |

## Further Reading

- [Williams (1992). Simple Statistical Gradient-Following Algorithms for Connectionist Reinforcement Learning](https://link.springer.com/article/10.1007/BF00992696) — original REINFORCE paper.
- [Sutton et al. (2000). Policy Gradient Methods for Reinforcement Learning with Function Approximation](https://papers.nips.cc/paper_files/paper/1999/hash/464d828b85b0bed98e80ade0a5c43b0f-Abstract.html) — modern policy-gradient theorem with function approximation.
- [Sutton & Barto (2018). Ch. 13 — Policy Gradient Methods](http://incompleteideas.net/book/RLbook2020.pdf) — textbook presentation.
- [OpenAI Spinning Up — VPG / REINFORCE](https://spinningup.openai.com/en/latest/algorithms/vpg.html) — clear pedagogical exposition with PyTorch code.
- [Peters & Schaal (2008). Reinforcement Learning of Motor Skills with Policy Gradients](https://homes.cs.washington.edu/~todorov/courses/amath579/reading/PolicyGradient.pdf) — variance-reduction and natural-gradient view, связывающий REINFORCE с trust-region family (TRPO, PPO).
