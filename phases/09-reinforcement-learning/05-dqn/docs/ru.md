# Deep Q-Networks (DQN)

> 2013: Mnih обучил одну Q-learning сеть на raw pixels и превзошел всех classical RL agents на семи Atari games. 2015: расширил до 49 games, опубликовал в Nature и запустил эпоху deep-RL. DQN — это Q-learning плюс три приема, которые стабилизируют function approximation.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 · 03 (Backpropagation), Phase 9 · 04 (Q-learning, SARSA)
**Time:** ~75 minutes

## Проблема

Tabular Q-learning требует отдельный Q-value для каждой пары (state, action). В шахматах примерно ~10⁴³ states. Atari frame имеет 210×160×3 = 100,800 features. Tabular RL умирает уже на тысячах states, не говоря о миллиардах.

Исправление задним числом очевидно: заменить Q-table neural network, `Q(s, a; θ)`. Но это "очевидно" заняло десятилетия. Naive function approximation с Q-learning расходится из-за "deadly triad" — function approximation + bootstrapping + off-policy learning. Mnih et al. (2013, 2015) выделили три engineering tricks, которые стабилизируют обучение:

1. **Experience replay** decorrelates transitions.
2. **Target network** freezes the bootstrap target.
3. **Reward clipping** normalizes gradient magnitudes.

DQN на Atari был первым случаем, когда одна architecture с одним набором hyperparameters решила десятки control problems from raw pixels. Все "deep-RL", построенное позже — DDQN, Rainbow, Dueling, Distributional, R2D2, Agent57 — стоит поверх этой базы из трех tricks.

## Концепция

![DQN training loop: env, replay buffer, online net, target net, Bellman TD loss](../assets/dqn.svg)

**The objective.** DQN минимизирует one-step TD loss на neural Q-function:

`L(θ) = E_{(s,a,r,s')~D} [ (r + γ max_{a'} Q(s', a'; θ^-) - Q(s, a; θ))² ]`

`θ` = online network, обновляемая каждым step через gradient descent. `θ^-` = target network, периодически копируемая из `θ` (примерно каждые ~10,000 steps). `D` = replay buffer прошлых transitions.

**Три приема, по важности:**

**Experience replay.** Ring buffer из `~10⁶` transitions. Каждый training step семплирует minibatch uniformly at random. Это ломает temporal correlation (successive frames почти одинаковы), позволяет сети много раз учиться на редких rewarding transitions и decorrelates consecutive gradient updates. Без этого on-policy TD с neural net расходится на Atari.

**Target network.** Использование одной сети `Q(·; θ)` по обе стороны Bellman equation заставляет target двигаться на каждом update — "chasing your own tail." Исправление: держать вторую сеть `Q(·; θ^-)` с frozen weights. Каждые `C` steps копировать `θ → θ^-`. Это стабилизирует regression target на тысячи gradient steps. Soft updates `θ^- ← τ θ + (1-τ) θ^-` (DDPG, SAC) — более плавный вариант.

**Reward clipping.** Atari rewards меняются от 1 до 1000+. Clipping к `{-1, 0, +1}` не дает одной игре доминировать gradient. Неверно, если reward magnitude важен; нормально для Atari, где важен знак.

**Double DQN.** Hasselt (2016) исправляет maximization bias: online net *выбирает* action, target net *оценивает* его.

`target = r + γ Q(s', argmax_{a'} Q(s', a'; θ); θ^-)`

Drop-in replacement, стабильно лучше. Используйте по умолчанию.

**Other improvements (Rainbow, 2017):** prioritized replay (чаще sample high-TD-error transitions), dueling architecture (отдельные `V(s)` и advantage heads), noisy networks (learned exploration), n-step returns, distributional Q (C51/QR-DQN), multi-step bootstrapping. Каждый добавляет несколько процентов; gains примерно additive.

## Практика

Код здесь stdlib-only и numpy-free — hand-rolled single-hidden-layer MLP на tiny continuous GridWorld, чтобы каждый training step выполнялся за микросекунды. Algorithm идентичен Atari DQN в масштабе.

### Step 1: replay buffer

```python
class ReplayBuffer:
    def __init__(self, capacity):
        self.buf = []
        self.capacity = capacity
    def push(self, s, a, r, s_next, done):
        if len(self.buf) == self.capacity:
            self.buf.pop(0)
        self.buf.append((s, a, r, s_next, done))
    def sample(self, batch, rng):
        return rng.sample(self.buf, batch)
```

~50,000 capacity для Atari; 5,000 достаточно для toy env.

### Step 2: a tiny Q-network (manual MLP)

```python
class QNet:
    def __init__(self, n_in, n_hidden, n_actions, rng):
        self.W1 = [[rng.gauss(0, 0.3) for _ in range(n_in)] for _ in range(n_hidden)]
        self.b1 = [0.0] * n_hidden
        self.W2 = [[rng.gauss(0, 0.3) for _ in range(n_hidden)] for _ in range(n_actions)]
        self.b2 = [0.0] * n_actions
    def forward(self, x):
        h = [max(0.0, sum(w * xi for w, xi in zip(row, x)) + b) for row, b in zip(self.W1, self.b1)]
        q = [sum(w * hi for w, hi in zip(row, h)) + b for row, b in zip(self.W2, self.b2)]
        return q, h
```

Forward pass: linear → ReLU → linear. Это вся сеть.

### Step 3: the DQN update

```python
def train_step(online, target, batch, gamma, lr):
    grads = zeros_like(online)
    for s, a, r, s_next, done in batch:
        q, h = online.forward(s)
        if done:
            y = r
        else:
            q_next, _ = target.forward(s_next)
            y = r + gamma * max(q_next)
        td_error = q[a] - y
        accumulate_grads(grads, online, s, h, a, td_error)
    apply_sgd(online, grads, lr / len(batch))
```

Форма та же, что Q-learning из Lesson 04, с двумя отличиями: (a) backprop через differentiable `Q(·; θ)` вместо indexing table, (b) target uses `Q(·; θ^-)`.

### Step 4: the outer loop

В каждом episode действуйте ε-greedy по `Q(·; θ)`, кладите transitions в buffer, sample minibatch, делайте gradient step, периодически sync `θ^- ← θ`. Pattern:

```python
for episode in range(N):
    s = env.reset()
    while not done:
        a = epsilon_greedy(online, s, epsilon)
        s_next, r, done = env.step(s, a)
        buffer.push(s, a, r, s_next, done)
        if len(buffer) >= batch:
            train_step(online, target, buffer.sample(batch), gamma, lr)
        if steps % sync_every == 0:
            target = copy(online)
        s = s_next
```

На tiny GridWorld с 16-dim one-hot state агент учит near-optimal policy примерно за ~500 episodes. На Atari масштабируйте до 200M frames и добавьте CNN feature extractor.

## Pitfalls

- **Deadly triad.** Function approximation + off-policy + bootstrapping может расходиться. DQN смягчает это target net + replay; не удаляйте ни одно.
- **Exploration.** `ε` должен decay, обычно от 1.0 до 0.01 за первые ~10% training. Без раннего exploration Q-net сходится в local basin.
- **Overestimation.** `max` over noisy Q biased upward. Всегда используйте Double DQN in production.
- **Reward scale.** Clip or normalize rewards; gradient magnitude пропорциональна reward magnitude.
- **Replay buffer coldstart.** Не обучайтесь, пока buffer не содержит несколько тысяч transitions. Early gradients на ~20 samples overfit.
- **Target sync frequency.** Слишком часто ≈ no target net; слишком редко ≈ stale targets. Atari DQN использует 10,000 env steps. Правило: sync каждые ~1/100 training horizon.
- **Observation preprocessing.** Atari DQN stacks 4 frames, чтобы state был Markov. Любая env с velocity info требует frame-stacking или recurrent state.

## Использование

В 2026 DQN редко state-of-the-art, но остается reference off-policy algorithm:

| Task | Method of choice | Why not DQN? |
|------|------------------|--------------|
| Discrete-action Atari-like | Rainbow DQN or Muesli | Same framework, more tricks. |
| Continuous control | SAC / TD3 (Phase 9 · 07) | DQN has no policy network. |
| On-policy / high-throughput | PPO (Phase 9 · 08) | No replay buffer; easier to scale. |
| Offline RL | CQL / IQL / Decision Transformer | Conservative Q targets, no bootstrapping blowups. |
| Large discrete action spaces (recommender) | DQN with action embedding, or IMPALA | Fine; decoration matters. |
| LLM RL | PPO / GRPO | Sequence-level, not step-level; different loss. |

Уроки сохраняются. Replay и target networks встречаются в SAC, TD3, DDPG, SAC-X, AlphaZero self-play buffer и каждом offline RL method. Reward clipping живет как advantage normalization в PPO. Architecture — это blueprint.

## Результат

Сохраните как `outputs/skill-dqn-trainer.md`:

```markdown
---
name: dqn-trainer
description: Produce a DQN training config (buffer, target sync, ε schedule, reward clipping) for a discrete-action RL task.
version: 1.0.0
phase: 9
lesson: 5
tags: [rl, dqn, deep-rl]
---

Given a discrete-action environment (observation shape, action count, horizon, reward scale), output:

1. Network. Architecture (MLP / CNN / Transformer), feature dim, depth.
2. Replay buffer. Capacity, minibatch size, warmup size.
3. Target network. Sync strategy (hard every C steps or soft τ).
4. Exploration. ε start / end / schedule length.
5. Loss. Huber vs MSE, gradient clip value, reward clipping rule.
6. Double DQN. On by default unless explicit reason to disable.

Refuse to ship a DQN with no target network, no replay buffer, or ε held at 1. Refuse continuous-action tasks (route to SAC / TD3). Flag any reward range > 10× per-step mean as needing clipping or scale normalization.
```

## Упражнения

1. **Легко.** Запустите `code/main.py`. Постройте per-episode return curve. Сколько episodes нужно, пока running mean превысит -10?
2. **Средне.** Отключите target network (используйте online net по обе стороны Bellman target). Измерьте training instability — return осциллирует или расходится?
3. **Сложно.** Добавьте Double DQN: online net выбирает `argmax a'`, target net оценивает. Сравните bias `Q(s_0, best_a)` vs true `V*(s_0)` after 1,000 episodes with vs without Double DQN на noisy-reward GridWorld.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| DQN | "Deep Q-learning" | Q-learning with a neural Q-function, replay buffer, and target network. |
| Experience replay | "Shuffled transitions" | Ring buffer, sampled uniformly each gradient step; decorrelates data. |
| Target network | "Frozen bootstrap" | Periodic copy of Q used in Bellman target; stabilizes training. |
| Deadly triad | "Why RL diverges" | Function approximation + bootstrapping + off-policy = no convergence guarantee. |
| Double DQN | "Fix for maximization bias" | Online net selects action, target net evaluates it. |
| Dueling DQN | "V and A heads" | Decompose Q = V + A - mean(A); same output, better gradient flow. |
| Rainbow | "All the tricks" | DDQN + PER + dueling + n-step + noisy + distributional in one. |
| PER | "Prioritized Replay" | Sample transitions proportional to TD-error magnitude. |

## Дополнительное чтение

- [Mnih et al. (2013). Playing Atari with Deep Reinforcement Learning](https://arxiv.org/abs/1312.5602) — 2013 NeurIPS workshop paper, с которой стартовал deep RL.
- [Mnih et al. (2015). Human-level control through deep reinforcement learning](https://www.nature.com/articles/nature14236) — Nature paper, 49-game DQN.
- [Hasselt, Guez, Silver (2016). Deep Reinforcement Learning with Double Q-learning](https://arxiv.org/abs/1509.06461) — DDQN.
- [Wang et al. (2016). Dueling Network Architectures](https://arxiv.org/abs/1511.06581) — dueling DQN.
- [Hessel et al. (2018). Rainbow: Combining Improvements in Deep RL](https://arxiv.org/abs/1710.02298) — stacked-tricks paper.
- [OpenAI Spinning Up — DQN](https://spinningup.openai.com/en/latest/algorithms/dqn.html) — ясное современное изложение.
- [Sutton & Barto (2018). Ch. 9 — On-policy Prediction with Approximation](http://incompleteideas.net/book/RLbook2020.pdf) — textbook treatment of "deadly triad", которую DQN target network и replay buffer должны укротить.
- [CleanRL DQN implementation](https://docs.cleanrl.dev/rl-algorithms/dqn/) — reference single-file DQN for ablation studies; полезно читать рядом с from-scratch version этого урока.
