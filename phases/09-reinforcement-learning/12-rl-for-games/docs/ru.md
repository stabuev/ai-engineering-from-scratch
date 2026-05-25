# RL for Games — AlphaZero, MuZero и эпоха LLM-Reasoning

> 1992: TD-Gammon победил чемпионов по backgammon с чистым TD. 2016: AlphaGo победил Lee Sedol. 2017: AlphaZero с нуля доминировал в chess, shogi и Go. 2024: DeepSeek-R1 доказал, что тот же рецепт, где GRPO заменяет PPO, работает для reasoning. Игры — benchmark, который двигает каждый прорыв в этой фазе.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 9 · 05 (DQN), Phase 9 · 08 (PPO), Phase 9 · 09 (RLHF), Phase 9 · 10 (MARL)
**Time:** ~120 minutes

## Проблема

В играх есть все, что нужно RL. Чистая reward (win/loss). Бесконечные episodes (self-play resets). Идеальная simulation (сама игра *и есть* simulator). Дискретные или небольшие continuous action spaces. Multi-agent структура, которая заставляет учиться adversarial robustness.

Именно в играх проверяли каждый крупный прорыв RL. TD-Gammon (backgammon, 1992). Atari-DQN (2013). AlphaGo (2016). AlphaZero (2017). OpenAI Five (Dota 2, 2019). AlphaStar (StarCraft II, 2019). MuZero (learned model, 2019). AlphaTensor (matrix multiplication, 2022). AlphaDev (sorting algorithms, 2023). DeepSeek-R1 (math reasoning, 2025) — новейшая демонстрация того, что game-RL техники работают с текстом.

Этот capstone рассматривает три знаковые архитектуры — AlphaZero, MuZero и GRPO — через единую призму: **self-play + search + policy improvement**. Каждая обобщает предыдущую; GRPO, в частности, — это рецепт AlphaZero, примененный к LLM reasoning, где tokens выступают actions, а математическая verification — сигналом win.

## Концепция

![AlphaZero ↔ MuZero ↔ GRPO: тот же цикл, разные среды](../assets/rl-games.svg)

**Объединяющий цикл.**

```
while True:
    trajectory = self_play(current_policy, search)     # play game against self
    policy_target = search.improved_policy(trajectory) # search improves raw policy
    policy_net.update(policy_target, value_target)     # supervised on search output
```

**AlphaZero (2017).** Silver et al. Дана игра (chess, shogi, Go) с известными правилами:

- Policy-value network: одна башня `f_θ(s) → (p, v)`. `p` — prior по legal moves. `v` — expected game outcome.
- Monte Carlo Tree Search (MCTS): на каждом ходе разворачивает дерево возможных продолжений. Использует `(p, v)` как prior + bootstrap. Выбирает nodes по UCB (PUCT): `a* = argmax Q(s, a) + c · p(a|s) · √N(s) / (1 + N(s, a))`.
- Self-play: игры agent-vs-agent. На ходе `t` distribution посещений MCTS `π_t` становится target для обучения policy.
- Loss: `L = (v - z)² - π · log p + c · ||θ||²`. `z` — game outcome (+1 / 0 / -1).

Ноль человеческих знаний. Ноль handcrafted heuristics. Один рецепт, который освоил chess, shogi и Go после нескольких десятков миллионов self-play партий в каждой игре.

**MuZero (2019).** Schrittwieser et al. Убирает требование, что правила известны.

- Вместо фиксированной environment учим *latent dynamics model* `(h, g, f)`:
  - `h(s)`: кодирует observation в latent state.
  - `g(s_latent, a)`: предсказывает следующий latent state + reward.
  - `f(s_latent)`: предсказывает policy prior + value.
- MCTS работает в *learned latent space*. Тот же search, тот же training loop.
- Работает на Go, chess, shogi *и* Atari — один algorithm, без знания правил.

**Stochastic MuZero (2022).** Добавляет stochastic dynamics и chance nodes; расширяет подход на игры класса backgammon.

**Muesli, Gumbel MuZero (2022-2024).** Улучшения sample efficiency и deterministic search.

**GRPO (2024-2025).** Рецепт DeepSeek-R1. Тот же цикл формы AlphaZero, примененный к language-model reasoning:

- "Game": ответить на math / coding / reasoning задачу. "Win" = verifier (test case passes, numerical answer matches) возвращает 1.
- Policy: LLM. Actions: tokens. State: prompt + response-so-far.
- Нет critic (PPO-style V_φ). Вместо этого для каждого prompt сэмплируем `G` completions из policy. Считаем reward для каждого. Используем **group-relative advantage** `A_i = (r_i - mean_r) / std_r` как сигнал для REINFORCE-style update.
- KL penalty к reference policy, чтобы предотвратить drift (как в RLHF).
- Полный loss:

  `L_GRPO(θ) = -E_{q, {o_i}} [ (1/G) Σ_i A_i · log π_θ(o_i | q) ] + β · KL(π_θ || π_ref)`

Нет reward model, нет critic, нет MCTS. Group-relative baseline заменяет все три. Качество на reasoning benchmarks соответствует PPO-RLHF или превосходит его при доле compute.

**Полный рецепт R1.** DeepSeek-R1 (DeepSeek 2025) — это две модели в одной статье:

- **R1-Zero.** Начать с base model DeepSeek-V3. Без SFT. Применить GRPO напрямую с двумя компонентами reward: *accuracy reward* (rule-based — распарсился ли финальный ответ в правильное число / прошел ли код unit tests) и *format reward* (обернул ли completion свой chain-of-thought в теги `<think>…</think>`). За тысячи steps средняя длина ответа растет с ~100 до ~10,000 tokens, а scores на math benchmarks поднимаются почти до уровня o1-preview. Модель учится рассуждать с нуля. Недостаток: ее chains of thought часто нечитаемы, смешивают языки и не имеют стилистической полировки.
- **R1.** Исправляет проблемы читаемости R1-Zero четырехэтапным pipeline:
  1. **Cold-start SFT.** Собрать несколько тысяч long-CoT demonstrations с чистым formatting. Supervised-finetune base model на них. Это дает читаемую стартовую точку.
  2. **Reasoning-oriented GRPO.** Применить GRPO с rewards accuracy+format плюс *language-consistency* reward, чтобы предотвращать code-switching.
  3. **Rejection sampling + SFT round 2.** Сэмплировать ~600K reasoning trajectories из RL checkpoint, оставить только те, где финальный ответ верный и CoT читаем, и объединить с ~200K non-reasoning SFT examples (writing, QA, self-cognition). Снова fine-tune base model.
  4. **Full-spectrum GRPO.** Еще один RL round, покрывающий и reasoning (rule-based rewards), и general alignment (helpfulness/harmlessness preference-based rewards).

Результат соответствует o1 на AIME и MATH-500 при open weights и достаточно мал для distillation. Та же статья также выпускает шесть distilled dense models (от Qwen-1.5B до Llama-70B), обученных SFT на reasoning traces R1 — без RL у student. Distillation сильного RL teacher стабильно превосходит RL с нуля в масштабе student.

**Почему GRPO вместо PPO для reasoning.** Три причины из статьи DeepSeekMath (Feb 2024): (1) нет value network для обучения, что вдвое снижает memory; (2) group baseline естественно работает со sparse end-of-trajectory reward, который дают reasoning tasks; (3) per-prompt normalization делает advantages сопоставимыми между задачами радикально разной сложности, чего не может один critic в PPO.

**Search-free vs search-based.** Игры разделились:

- *Perfect-information games with long horizons* (Go, chess): все еще search-based. Доминируют AlphaZero / MuZero.
- *LLM reasoning*: MCTS пока нет в production; GRPO на full rollouts, best-of-N для inference compute. Process reward models (PRMs) намекают, что step-level search будет добавлен обратно.

## Практика

Код в `code/main.py` реализует **GRPO в миниатюре** — bandit с несколькими группами samples. Algorithm тот же, что и на LLM; проще только policy и environment. Он показывает *loss* и *group-relative advantage*, то есть инновацию 2025 года.

### Step 1: крошечная verifier environment

```python
QUESTIONS = [
    {"prompt": "q1", "correct": 3},
    {"prompt": "q2", "correct": 1},
]

def verify(prompt_idx, answer_token):
    return 1.0 if answer_token == QUESTIONS[prompt_idx]["correct"] else 0.0
```

В настоящем GRPO verifier запускает unit tests или проверяет math equality.

### Step 2: policy: softmax по K answer tokens на prompt

```python
def policy_probs(theta, p_idx):
    return softmax(theta[p_idx])
```

Эквивалент final-layer output LLM, conditioned on a prompt.

### Step 3: group sampling и group-relative advantage

```python
def grpo_step(theta, p_idx, G=8, beta=0.01, lr=0.1, rng=None):
    probs = policy_probs(theta, p_idx)
    samples = [sample(probs, rng) for _ in range(G)]
    rewards = [verify(p_idx, s) for s in samples]
    mean_r = sum(rewards) / G
    std_r = stddev(rewards) + 1e-8
    advs = [(r - mean_r) / std_r for r in rewards]

    for a, A in zip(samples, advs):
        grad = onehot(a) - probs
        for i in range(len(probs)):
            theta[p_idx][i] += lr * A * grad[i]
    # KL penalty: pull theta toward reference
    for i in range(len(probs)):
        theta[p_idx][i] -= beta * (theta[p_idx][i] - reference[p_idx][i])
```

Group-relative advantage — прием DeepSeek 2024 года. Critic не нужен. "Baseline" — group mean, а normalization использует group std.

### Step 4: сравнить с REINFORCE baseline (value-free)

Та же setup, тот же compute, простой REINFORCE. GRPO сходится быстрее и стабильнее.

### Step 5: наблюдать entropy и KL

Те же diagnostics, что в RLHF: mean KL к reference, policy entropy, reward-over-time. Когда они стабилизируются, training завершен.

## Подводные камни

- **Reward hacking через verifier gaming.** GRPO наследует риск RLHF: если verifier ошибается или его можно exploit, LLM найдет exploit. Важны robust verifiers (multiple test cases, formal proofs).
- **Слишком маленький group size.** Variance group baseline убывает как `1/√G`. Ниже `G = 4` advantage signal шумный; стандартный выбор — `G = 8` to `64`.
- **Length bias.** LLM completions разной длины имеют разные log-probabilities. Нормализуйте по token count, используйте sequence-level log-prob или обрезайте до max length.
- **Чистые self-play cycles.** Обучение в стиле AlphaZero может застревать в dominance loops на general-sum games. Смягчается разнообразными opponent pools (league play, Lesson 10).
- **Search-policy mismatch.** AlphaZero учит policy имитировать search output. Если policy net слишком мала, чтобы представить distribution search, training останавливается.
- **Compute floor.** MuZero / AlphaZero требуют огромного compute. Одна ablation часто стоит сотни GPU-hours. Для обучения есть миниатюрные demos (например, AlphaZero на Connect Four).
- **Verifier coverage.** Unit tests, которые проходят для buggy solution, reinforce bug. Проектируйте verifiers, которые ловят edge cases.

## Использование

Ландшафт game-RL в 2026 году по domain:

| Domain | Dominant method |
|--------|-----------------|
| Two-player zero-sum board games (Go, chess, shogi) | AlphaZero / MuZero / KataGo |
| Imperfect info card games (poker) | CFR + deep learning (DeepStack, Libratus, Pluribus) |
| Atari / pixel games | Muesli / MuZero / IMPALA-PPO |
| Large multiplayer strategy (Dota, StarCraft) | PPO + self-play + league (OpenAI Five, AlphaStar) |
| LLM math/code reasoning | GRPO (DeepSeek-R1, Qwen-RL, open replications) |
| LLM alignment | DPO / RLHF-PPO (not GRPO; verifier is preference not verifiable) |
| Robotics | PPO + DR (not game-RL, but uses same policy-gradient tools) |
| Combinatorial problems | AlphaZero variants (AlphaTensor, AlphaDev) |

*Рецепт* — self-play, search-augmented improvement, policy distillation — охватывает текст, pixels и physical control. GRPO — самый молодой пример; будут и другие.

## Результат

Сохраните как `outputs/skill-game-rl-designer.md`:

```markdown
---
name: game-rl-designer
description: Design a game-RL or reasoning-RL training pipeline (AlphaZero / MuZero / GRPO) for a given domain.
version: 1.0.0
phase: 9
lesson: 12
tags: [rl, alphazero, muzero, grpo, self-play]
---

Given a target (perfect-info game / imperfect-info / Atari / LLM reasoning / combinatorial), output:

1. Environment fit. Known rules? Markov? Stochastic? Multi-agent? Informs AlphaZero vs MuZero vs GRPO.
2. Search strategy. MCTS (PUCT with learned prior), Gumbel-sampled, best-of-N, or none.
3. Self-play plan. Symmetric self-play / league / offline data / verifier-generated.
4. Target signal. Game outcome / verifier reward / preference / learned model. Include robustness plan.
5. Diagnostics. Win rate vs baseline, ELO curve, verifier pass rate, KL to reference.

Refuse AlphaZero on imperfect-info games (route to CFR). Refuse GRPO without a trusted verifier. Refuse any game-RL pipeline without a fixed baseline opponent set (self-play ELO is uncalibrated otherwise).
```

## Упражнения

1. **Легко.** Реализуйте GRPO bandit в `code/main.py`. Обучите на 2 prompts × 4 answer tokens каждый. Добейтесь сходимости за < 1,000 updates с `G=8`.
2. **Средне.** Подключите PPO (clipped) и vanilla REINFORCE. Сравните sample efficiency и reward variance с GRPO на том же bandit.
3. **Сложно.** Расширьте до length-2 "reasoning chain": agent emits two tokens, а verifier rewards the pair. Измерьте, как GRPO справляется с credit assignment across two-step sequences. (Hint: compute group advantage per *full sequence*, propagate to both token positions.)

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| MCTS | "Tree search with learned net" | Monte Carlo Tree Search; UCB1/PUCT selection with learned `(p, v)` priors. |
| AlphaZero | "Self-play + MCTS" | Policy-value net trained to match MCTS visits and game outcome. |
| MuZero | "Learned-model AlphaZero" | Same loop but in latent space via learned dynamics. |
| GRPO | "Critic-free PPO" | Group Relative Policy Optimization; REINFORCE with group-mean baseline + KL. |
| PUCT | "AlphaZero's UCB" | `Q + c · p · √N / (1 + N_a)` — balances value estimate with prior. |
| Self-play | "Agent vs past self" | Standard for zero-sum; symmetric training signal. |
| League play | "Population-based self-play" | Past + current + exploiters sampled as opponents. |
| Verifier reward | "Verifiable RL" | Reward comes from a deterministic checker (tests pass, answer matches). |
| Process reward | "PRM" | Scores each reasoning step, not just the final answer. |

## Дополнительное чтение

- [Silver et al. (2017). Mastering the game of Go without human knowledge (AlphaGo Zero)](https://www.nature.com/articles/nature24270).
- [Silver et al. (2018). A general reinforcement learning algorithm that masters chess, shogi, and Go through self-play (AlphaZero)](https://www.science.org/doi/10.1126/science.aar6404).
- [Schrittwieser et al. (2020). Mastering Atari, Go, chess and shogi by planning with a learned model (MuZero)](https://www.nature.com/articles/s41586-020-03051-4).
- [Vinyals et al. (2019). Grandmaster level in StarCraft II (AlphaStar)](https://www.nature.com/articles/s41586-019-1724-z).
- [DeepSeek-AI (2024). DeepSeekMath: Pushing the Limits of Mathematical Reasoning in Open Language Models (GRPO)](https://arxiv.org/abs/2402.03300) — статья, которая ввела GRPO и group-relative baseline.
- [DeepSeek-AI (2025). DeepSeek-R1: Incentivizing Reasoning Capability in LLMs via Reinforcement Learning](https://arxiv.org/abs/2501.12948) — полный четырехэтапный рецепт R1 плюс ablation R1-Zero.
- [Brown et al. (2019). Superhuman AI for multiplayer poker (Pluribus)](https://www.science.org/doi/10.1126/science.aay2400) — CFR + deep-learning в большом масштабе.
- [Tesauro (1995). Temporal Difference Learning and TD-Gammon](https://dl.acm.org/doi/10.1145/203330.203343) — статья, с которой все началось.
- [Hugging Face TRL — GRPOTrainer](https://huggingface.co/docs/trl/main/en/grpo_trainer) — production reference для применения GRPO с custom reward functions.
- [Qwen Team (2024). Qwen2.5-Math — GRPO replication](https://github.com/QwenLM/Qwen2.5-Math) — открытая replication рецепта R1 в нескольких масштабах.
- [Sutton & Barto (2018). Ch. 17 — Frontiers of Reinforcement Learning](http://incompleteideas.net/book/RLbook2020.pdf) — textbook framing для self-play, search и "designed reward", который R1 воплощает в масштабе LLM.
