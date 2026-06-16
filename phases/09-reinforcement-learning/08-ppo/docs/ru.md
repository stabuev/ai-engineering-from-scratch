# Proximal Policy Optimization (PPO)

> A2C отбрасывает каждый rollout после одного update. PPO оборачивает policy gradient в clipped importance ratio, чтобы можно было делать 10+ epochs на тех же данных без взрыва policy. Schulman et al. (2017). В 2026 году все еще default policy-gradient algorithm.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 06 (REINFORCE), Phase 9 · 07 (Actor-Critic)
**Time:** ~75 minutes

## Цели обучения

- Реализовывать clipped surrogate-цель PPO, чтобы один rollout выдерживал много эпох обновления без коллапса политики.
- Считать GAE-advantage и объединять потери политики, ценности и энтропии.
- Читать диагностики PPO (clip fraction, KL), отличая здоровый прогон от расходящегося.

## Проблема

A2C (Lesson 07) является on-policy: gradient `E_{π_θ}[A · ∇ log π_θ]` требует данные, sampled from текущей `π_θ`. Сделайте один update, и `π_θ` изменится; данные, которые вы использовали, теперь off-policy. Повторное использование даст biased gradient.

Rollouts стоят дорого. На Atari один rollout по 8 envs × 128 steps = 1024 transitions и десяток секунд environment time. Выбрасывать это после одного gradient step расточительно.

Trust Region Policy Optimization (TRPO, Schulman 2015) был первым исправлением: ограничивать каждый update так, чтобы KL divergence между old и new policy оставалась ниже `δ`. Теоретически чисто, но требует conjugate-gradient solve на каждый update. В 2026 году TRPO почти никто не запускает.

PPO (Schulman et al. 2017) заменяет жесткое trust-region constraint простым clipped objective. Одна дополнительная строка кода. Десять epochs на rollout. Без conjugate gradients. Достаточно хорошие theoretical guarantees. Девять лет спустя это все еще default policy-gradient algorithm для всего, от MuJoCo до RLHF.

## Концепция

![PPO clipped surrogate objective: ratio clipping at 1 ± ε](../assets/ppo.svg)

**The importance ratio.**

`r_t(θ) = π_θ(a_t | s_t) / π_{θ_old}(a_t | s_t)`

Это likelihood ratio новой policy относительно policy, которая собрала данные. `r_t = 1` означает, что изменений нет. `r_t = 2` означает, что новая policy вдвое вероятнее выберет `a_t`, чем старая.

**The clipped surrogate.**

`L^{CLIP}(θ) = E_t [ min( r_t(θ) A_t, clip(r_t(θ), 1-ε, 1+ε) A_t ) ]`

Два terms:

- Если advantage `A_t > 0` и ratio пытается вырасти выше `1 + ε`, clip выравнивает gradient — не продвигайте good action дальше, чем на `+ε` выше old probability.
- Если advantage `A_t < 0` и ratio пытается вырасти выше `1 - ε` (то есть мы сделали бы bad action более likely по сравнению с его clipped reduction), clip ограничивает gradient — не опускайте bad action ниже `-ε`.

`min` обрабатывает другое направление: если ratio сдвинулся в *beneficial* direction, вы все еще получаете gradient (без clipping на стороне, которая навредила бы вам).

Типичное `ε = 0.2`. Если построить objective как функцию от `r_t`, получится piecewise-linear function с плоской крышей на "good side" и плоским полом на "bad side."

**The full PPO loss.**

`L(θ, φ) = L^{CLIP}(θ) - c_v · (V_φ(s_t) - V_t^{target})² + c_e · H(π_θ(·|s_t))`

Та же actor-critic structure, что и в A2C. Три coefficients, обычно `c_v = 0.5`, `c_e = 0.01`, `ε = 0.2`.

**The training loop.**

1. Соберите `N × T` transitions по `N` parallel envs, по `T` steps в каждом.
2. Вычислите advantages (GAE), зафиксируйте их как constants.
3. Заморозьте `π_{θ_old}` как snapshot текущей `π_θ`.
4. Для `K` epochs, для каждого minibatch из `(s, a, A, V_target, log π_old(a|s))`:
   - Вычислите `r_t(θ) = exp(log π_θ(a|s) - log π_old(a|s))`.
   - Примените `L^{CLIP}` + value loss + entropy.
   - Gradient step.
5. Отбросьте rollout. Вернитесь к step 1.

`K = 10` и minibatches по 64 — стандартный набор hyperparameters. PPO robust: точные числа редко имеют значение в пределах ±50%.

**KL-penalty variant.** В оригинальной paper была предложена альтернатива с adaptive KL penalty: `L = L^{PG} - β · KL(π_θ || π_old)` с `β`, adjusted на основе observed KL. Clipping version стала доминирующей; KL variant сохранился в RLHF (где KL к reference policy — отдельное constraint, которое вам все равно всегда нужно).

## Практика

### Step 1: capture `log π_old(a | s)` at rollout time

```python
for step in range(T):
    probs = softmax(logits(theta, state_features(s)))
    a = sample(probs, rng)
    s_next, r, done = env.step(s, a)
    buffer.append({
        "s": s, "a": a, "r": r, "done": done,
        "v_old": value(w, state_features(s)),
        "log_pi_old": log(probs[a] + 1e-12),
    })
    s = s_next
```

Snapshot снимается один раз, во время rollout. Он не меняется во время update epochs.

### Step 2: compute GAE advantages (Lesson 07)

То же, что в A2C. Normalize across the batch.

### Step 3: clipped surrogate update

```python
for _ in range(K_EPOCHS):
    for mb in minibatches(buffer, size=64):
        for rec in mb:
            x = state_features(rec["s"])
            probs = softmax(logits(theta, x))
            logp = log(probs[rec["a"]] + 1e-12)
            ratio = exp(logp - rec["log_pi_old"])
            adv = rec["advantage"]
            surrogate = min(
                ratio * adv,
                clamp(ratio, 1 - EPS, 1 + EPS) * adv,
            )
            # backprop -surrogate, add value loss, subtract entropy
            grad_logpi = onehot(rec["a"]) - probs
            if (adv > 0 and ratio >= 1 + EPS) or (adv < 0 and ratio <= 1 - EPS):
                pg_grad = 0.0  # clipped
            else:
                pg_grad = ratio * adv
            for i in range(N_ACTIONS):
                for j in range(N_FEAT):
                    theta[i][j] += LR * pg_grad * grad_logpi[i] * x[j]
```

Паттерн "clipped → zero gradient" — сердце PPO. Если new policy уже слишком далеко ушла в beneficial direction, update останавливается.

### Step 4: value and entropy

Добавьте стандартный MSE к critic target и entropy bonus на actor, как в A2C.

### Step 5: diagnostics

Три вещи, за которыми нужно следить на каждом update:

- **Mean KL** `E[log π_old - log π_θ]`. Должен оставаться в `[0, 0.02]`. Если он улетает выше `0.1`, уменьшите `K_EPOCHS` или `LR`.
- **Clip fraction** — доля samples, у которых ratio лежит вне `[1-ε, 1+ε]`. Должна быть `~0.1-0.3`. Если `~0`, clip никогда не срабатывает → увеличьте `LR` или `K_EPOCHS`. Если `~0.5+`, вы over-fitting the rollout → уменьшите их.
- **Explained variance** `1 - Var(V_target - V_pred) / Var(V_target)`. Critic quality metric. Должна расти к 1 по мере обучения critic.

## Pitfalls

- **Clip coefficient mistuned.** `ε = 0.2` — de-facto standard. Переход к `0.1` делает updates слишком робкими; `0.3+` провоцирует instability.
- **Too many epochs.** `K > 20` регулярно destabilizes, потому что policy далеко уходит от `π_old`. Ограничивайте epochs, особенно для large networks.
- **No reward normalization.** Большие reward scales съедают clip range. Normalize rewards (running std) перед вычислением advantages.
- **Forgetting advantage normalization.** Per-batch zero-mean/unit-std normalization — стандарт. Пропуск этого ломает PPO на большинстве benchmarks.
- **Learning rate not decayed.** PPO выигрывает от linear LR decay to zero. Constant LR часто хуже.
- **Importance ratio math errors.** Всегда `exp(log_new - log_old)` для numerical stability, не `new / old`.
- **Wrong gradient sign.** Maximize surrogate = *minimize* `-L^{CLIP}`. Перевернутый sign — самый частый PPO bug.

## Использование

PPO — default RL algorithm 2026 года в удивительно большом числе domains:

| Use case | PPO variant |
|----------|-------------|
| MuJoCo / robotics control | PPO with Gaussian policy, GAE(0.95) |
| Atari / discrete games | PPO with categorical policy, rolling 128-step rollouts |
| RLHF for LLMs | PPO with KL penalty to reference model, reward from RM at end of response |
| Large-scale game agents | IMPALA + PPO (AlphaStar, OpenAI Five) |
| Reasoning LLMs | GRPO (Lesson 12) — PPO variant without critic |
| Preference-only data | DPO — closed-form collapsing of PPO+KL, no online sampling |

*Loss shape* PPO — clipped surrogate + value + entropy — это scaffolding для DPO, GRPO и почти каждого RLHF pipeline.

## Результат

Save as `outputs/skill-ppo-trainer.md`:

```markdown
---
name: ppo-trainer
description: Produce a PPO training config and a diagnostic plan for a given environment.
version: 1.0.0
phase: 9
lesson: 8
tags: [rl, ppo, policy-gradient]
---

Given an environment and training budget, output:

1. Rollout size. `N` envs × `T` steps.
2. Update schedule. `K` epochs, minibatch size, LR schedule.
3. Surrogate params. `ε` (clip), `c_v`, `c_e`, advantage normalization on.
4. Advantage. GAE(`λ`) with explicit `γ` and `λ`.
5. Diagnostics plan. KL, clip fraction, explained variance thresholds with alerts.

Refuse `K > 30` or `ε > 0.3` (unsafe trust region). Refuse any PPO run without advantage normalization or KL/clip monitoring. Flag clip fraction sustained above 0.4 as drift.
```

## Упражнения

1. **Легко.** Запустите PPO на 4×4 GridWorld с `ε=0.2, K=4`. Сравните sample efficiency с A2C (one epoch per rollout) при одинаковом числе env steps.
2. **Средне.** Переберите `K ∈ {1, 4, 10, 30}`. Постройте return vs env steps и отслеживайте mean KL на update. При каком `K` KL взрывается на этой задаче?
3. **Сложно.** Замените clipped surrogate на adaptive KL penalty (`β` doubled if `KL > 2·target`, halved if `KL < target/2`). Сравните final return, stability и clip-free-ness.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Importance ratio | "r_t(θ)" | `π_θ(a|s) / π_old(a|s)`; отклонение от policy, которая собрала данные. |
| Clipped surrogate | "PPO's main trick" | `min(r·A, clip(r, 1-ε, 1+ε)·A)`; flat gradient beyond the clip on beneficial side. |
| Trust region | "TRPO / PPO intent" | Ограничить KL каждого update, чтобы гарантировать monotone improvement. |
| KL penalty | "Soft trust region" | Alternative PPO: `L - β · KL(π_θ || π_old)`. Adaptive `β`. |
| Clip fraction | "How often clipping triggers" | Diagnostic — должно быть 0.1-0.3; вне диапазона означает mistuned. |
| Multi-epoch training | "Data reuse" | K epochs на каждом rollout; variance cost обменивается на sample efficiency. |
| On-policy-ish | "Mostly on-policy" | PPO номинально on-policy, но K>1 epochs uses slightly-off-policy data safely. |
| PPO-KL | "The other PPO" | KL-penalty variant; используется в RLHF, где KL-to-reference уже является constraint. |

## Дополнительное чтение

- [Schulman et al. (2017). Proximal Policy Optimization Algorithms](https://arxiv.org/abs/1707.06347) — paper.
- [Schulman et al. (2015). Trust Region Policy Optimization](https://arxiv.org/abs/1502.05477) — TRPO, predecessor PPO.
- [Andrychowicz et al. (2021). What Matters In On-Policy RL? A Large-Scale Empirical Study](https://arxiv.org/abs/2006.05990) — ablation каждого PPO hyperparameter.
- [Ouyang et al. (2022). Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — InstructGPT; PPO-in-RLHF recipe.
- [OpenAI Spinning Up — PPO](https://spinningup.openai.com/en/latest/algorithms/ppo.html) — чистое modern exposition with PyTorch.
- [CleanRL PPO implementation](https://github.com/vwxyzjn/cleanrl) — reference single-file PPO, используемый во многих papers.
- [Hugging Face TRL — PPOTrainer](https://huggingface.co/docs/trl/main/en/ppo_trainer) — production recipe для PPO on language models; читайте вместе с Lesson 09 (RLHF).
- [Engstrom et al. (2020). Implementation Matters in Deep Policy Gradients](https://arxiv.org/abs/2005.12729) — paper про "37 code-level optimizations"; какие PPO tricks are load-bearing, а какие folklore.
