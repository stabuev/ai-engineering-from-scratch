# Reward Modeling & RLHF

> Люди не могут написать reward function для "good assistant response", но могут сравнить два ответа и выбрать лучший. Обучите reward model на таких сравнениях, затем примените RL к language model относительно этой модели. Christiano 2017. InstructGPT 2022. Рецепт, который превратил GPT-3 в ChatGPT. В 2026 его в основном заменяет DPO, но ментальная модель сохраняется.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 05 (Sentiment), Phase 9 · 08 (PPO)
**Time:** ~45 minutes

## Цели обучения

- Обучать reward-модель Брэдли–Терри на парных предпочтениях человека.
- Запускать PPO-подобную политику против reward-модели, отслеживая KL-штраф к референсу.
- Объяснять reward hacking и место RLAIF и Constitutional AI.

## Проблема

Вы обучили language model на objective next-token prediction. Она пишет грамматический английский. Но она также лжет, растекается мыслью и отказывается отказываться. Это нельзя исправить дополнительным pretraining: web text — это проблема, а не лекарство.

Вам нужен *scalar reward*, который говорит: "response A is better than response B for instruction X." Написать такую reward function вручную невозможно. "Helpfulness" не является closed-form выражением над токенами. Но люди могут сравнить два выхода и отметить preference. Такие данные дешево собирать в масштабе.

RLHF (Christiano et al. 2017; Ouyang et al. 2022) превращает preferences в reward model, затем оптимизирует LM через PPO относительно этого reward. В три шага: SFT → RM → PPO. Это рецепт, с которым были выпущены ChatGPT, Claude, Gemini и все остальные aligned-LLM в 2023–2025.

В 2026 шаг PPO в основном заменяют DPO (Phase 10 · 08), потому что он дешевле и почти так же хорош для alignment tuning. Но часть с *reward model* по-прежнему лежит в основе каждого Best-of-N sampler, каждого RL-from-verifiable-rewards pipeline и каждой reasoning model, использующей process reward model. Поймите RLHF — и вы поймете весь alignment stack.

## Концепция

![Three-stage RLHF: SFT, RM training on pairwise prefs, PPO with KL penalty](../assets/rlhf.svg)

**Stage 1: Supervised Fine-Tuning (SFT).** Начните с pretrained base model. Fine-tune на написанных людьми demonstrations целевого поведения (instruction-following responses, helpful replies и т. д.). Результат: модель `π_SFT`, которая *смещена в сторону хорошего поведения*, но все еще имеет неограниченное action space.

**Stage 2: Reward Model training.**

- Соберите пары responses `(y_+, y_-)` для prompts `x`, размеченные людьми как "y_+ is preferred over y_-."
- Обучите reward model `R_φ(x, y)` присваивать более высокие scores для `y_+`.
- Loss: **Bradley-Terry pairwise logistic**:

  `L(φ) = -E[ log σ(R_φ(x, y_+) - R_φ(x, y_-)) ]`

  σ — это sigmoid. Разница в reward задает log-odds preference. BT является стандартом с 1952 года (Bradley-Terry) и доминирующим выбором в современном RLHF.

- `R_φ` обычно инициализируют из SFT model со scalar head сверху. Тот же transformer backbone; один linear layer выводит reward.

**Stage 3: PPO against the RM with KL penalty.**

- Инициализируйте обучаемую policy `π_θ` из `π_SFT`. Сохраните замороженную *reference* `π_ref = π_SFT`.
- Reward в конце response `y`:

  `r_total(x, y) = R_φ(x, y) - β · KL(π_θ(·|x) || π_ref(·|x))`

  KL penalty не дает `π_θ` произвольно удаляться от `π_SFT`: это *regularizer*, а не жесткий trust region. `β` обычно `0.01`-`0.05`.
- Запустите PPO (Lesson 08) с этим reward. Advantages вычисляются на token-level trajectory, но RM оценивает только полный response.

**Why the KL?** Без него PPO охотно найдет стратегии reward hacking: RM обучалась только на in-distribution completions. Out-of-distribution response может получить score выше любого написанного человеком. KL удерживает `π_θ` рядом с manifold, на котором обучалась RM. Это самая важная ручка в RLHF.

**2026 status:**

- **DPO** (Rafailov 2023): closed-form алгебра сворачивает Stage 2+3 в один supervised loss по preference data. Без RM, без PPO. То же качество на alignment benchmarks за долю compute. Рассматривается в Phase 10 · 08.
- **GRPO** (DeepSeek 2024–2025): PPO с group-relative baseline вместо critic, reward от *verifier* (код запускается / math answer совпадает) вместо обученной на людях RM. Доминирует для reasoning models. Рассматривается в Phase 9 · 12.
- **Process reward models (PRMs):** оценивают partial solutions (каждый reasoning step), используются и в RLHF, и в вариантах GRPO для reasoning.
- **Constitutional AI / RLAIF:** используют aligned LLM для генерации preferences вместо людей. Масштабирует preference budget.

## Практика

Этот урок использует маленькие синтетические "prompts" и "responses", представленные строками. RM — это linear scorer поверх bag-of-tokens представления. Никакой настоящей LLM: важна *форма* pipeline, а не масштаб. См. `code/main.py`.

### Step 1: synthetic preference data

```python
PROMPTS = ["help me", "answer me", "explain this"]
GOOD_WORDS = {"clear", "specific", "kind", "thorough"}
BAD_WORDS = {"vague", "rude", "wrong", "short"}

def make_pair(rng):
    x = rng.choice(PROMPTS)
    y_good = rng.choice(list(GOOD_WORDS)) + " " + rng.choice(list(GOOD_WORDS))
    y_bad = rng.choice(list(BAD_WORDS)) + " " + rng.choice(list(BAD_WORDS))
    return (x, y_good, y_bad)
```

В реальном RLHF это заменяется human labelers. Форма — `(prompt, preferred_response, rejected_response)` — идентична.

### Step 2: Bradley-Terry reward model

Linear score: `R(x, y) = w · bag(y)`. Обучайте, минимизируя BT pairwise log-loss:

```python
def rm_train_step(w, x, y_pos, y_neg, lr):
    r_pos = dot(w, bag(y_pos))
    r_neg = dot(w, bag(y_neg))
    p = sigmoid(r_pos - r_neg)
    for tok, cnt in bag(y_pos).items():
        w[tok] += lr * (1 - p) * cnt
    for tok, cnt in bag(y_neg).items():
        w[tok] -= lr * (1 - p) * cnt
```

После нескольких сотен updates `w` назначает positive weights токенам good-word и negative weights плохим токенам.

### Step 3: PPO-like policy on top of RM

Наша toy policy производит один token из vocabulary. Мы оцениваем token через RM, вычисляем `log π_θ(token | prompt)`, добавляем KL-to-reference penalty и применяем clipped PPO surrogate.

```python
def rlhf_step(theta, ref, w, prompt, rng, eps=0.2, beta=0.1, lr=0.05):
    logits_theta = policy_logits(theta, prompt)
    probs = softmax(logits_theta)
    token = sample(probs, rng)
    logits_ref = policy_logits(ref, prompt)
    probs_ref = softmax(logits_ref)
    reward = dot(w, bag([token])) - beta * kl(probs, probs_ref)
    # ppo-style update on theta, treating reward as the return
    ...
```

### Step 4: monitor the KL

Отслеживайте mean `KL(π_θ || π_ref)` на каждом update. Если он переползает за `~5-10`, policy сильно ушла от `π_SFT`: возможно, `β` слишком низкая или начинается reward hacking. Это главный diagnostic в реальном RLHF.

### Step 5: the production recipe with TRL

Когда вы поняли toy pipeline, вот тот же loop в виде, в котором его пишет пользователь реальной библиотеки. Hugging Face [TRL](https://huggingface.co/docs/trl) — reference implementation: `RewardTrainer` для Stage 2 и `PPOTrainer` (со встроенным KL-to-reference) для Stage 3.

```python
# Stage 2: reward model from pairwise preferences
from trl import RewardTrainer, RewardConfig
from transformers import AutoModelForSequenceClassification, AutoTokenizer

tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.1-8B-Instruct")
rm = AutoModelForSequenceClassification.from_pretrained(
    "meta-llama/Llama-3.1-8B-Instruct", num_labels=1
)

# dataset rows: {"prompt", "chosen", "rejected"} — Bradley-Terry format
trainer = RewardTrainer(
    model=rm,
    tokenizer=tok,
    train_dataset=preference_data,
    args=RewardConfig(output_dir="./rm", num_train_epochs=1, learning_rate=1e-5),
)
trainer.train()
```

```python
# Stage 3: PPO against the RM with KL penalty to the SFT reference
from trl import PPOTrainer, PPOConfig, AutoModelForCausalLMWithValueHead

policy = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")
ref    = AutoModelForCausalLMWithValueHead.from_pretrained("./sft-checkpoint")  # frozen

ppo = PPOTrainer(
    config=PPOConfig(learning_rate=1.41e-5, batch_size=64, init_kl_coef=0.05,
                     target_kl=6.0, adap_kl_ctrl=True),
    model=policy, ref_model=ref, tokenizer=tok,
)

for batch in dataloader:
    responses = ppo.generate(batch["query_ids"], max_new_tokens=128)
    rewards   = rm(torch.cat([batch["query_ids"], responses], dim=-1)).logits[:, 0]
    stats     = ppo.step(batch["query_ids"], responses, rewards)
    # stats includes: mean_kl, clip_frac, value_loss — the three PPO diagnostics
```

Три вещи библиотека делает за вас. `adap_kl_ctrl=True` реализует adaptive-β schedule: если observed KL превышает `target_kl`, β удваивается; если ниже половины, β уменьшается вдвое. Reference model заморожена по соглашению: нельзя случайно разделить параметры с `policy`. А value head живет на том же backbone, что и policy (`AutoModelForCausalLMWithValueHead` добавляет scalar MLP head), поэтому TRL отдельно сообщает `policy/kl` и `value/loss`.

## Pitfalls

- **Over-optimization / reward hacking.** RM несовершенна; `π_θ` находит adversarial completions, которые получают высокий score, но плохи. Симптомы: reward бесконечно растет, пока human eval score выходит на плато или падает. Исправление: остановиться раньше, поднять `β`, расширить RM training data.
- **Length hacking.** RMs, обученные на helpful responses, часто неявно вознаграждают длину. Policy учится дополнять responses лишним текстом. Исправление: length-normalized reward или RLAIF с length-aware RM.
- **Too-small RM.** RM должна быть как минимум такой же большой, как policy. Крошечная RM не может надежно оценивать outputs policy.
- **KL tuning.** Слишком низкая β → drift и reward hacking. Слишком высокая β → policy почти не меняется. Стандартный прием — *adaptive* β, которая нацеливается на фиксированный KL per step.
- **Preference-data noise.** ~30% human labels шумные или неоднозначные. Калибруйте, обучая RM на agreement-filtered data, или используйте temperature в BT.
- **Off-policy problems.** PPO data становится слегка off-policy после первой epoch. Следите за clip fraction, как в Lesson 08.

## Использование

RLHF в 2026 устроен слоями:

| Layer | Target | Method |
|-------|--------|--------|
| Instruction following, helpfulness, harmlessness | Alignment | DPO (Phase 10 · 08) preferred over RLHF-PPO. |
| Reasoning correctness (math, code) | Capability | GRPO with verifier reward (Phase 9 · 12). |
| Long-horizon multi-step tasks | Agentic | PPO / GRPO with process reward models over steps. |
| Safety / refusal behavior | Safety | RLHF-PPO with separate safety RM, or Constitutional AI. |
| Best-of-N at inference | Fast alignment | Use RM at decode time; no policy training needed. |
| Reward distillation | Inference compute | Train a small "reward head" on top of a frozen LM. |

RLHF был *тем самым* методом в 2022–2024. В 2026 production alignment pipelines в первую очередь используют DPO, а PPO оставляют только для RM-intensive или safety-critical шагов.

## Результат

Сохраните как `outputs/skill-rlhf-architect.md`:

```markdown
---
name: rlhf-architect
description: Design an RLHF / DPO / GRPO alignment pipeline for a language model, including RM, KL, and data strategy.
version: 1.0.0
phase: 9
lesson: 9
tags: [rl, rlhf, alignment, llm]
---

Given a base LM, a target behavior (alignment / reasoning / refusal / agent), and a preference or verifier budget, output:

1. Stage. SFT? RM? DPO? GRPO? With justification.
2. Preference or verifier source. Humans, AI feedback, rule-based, unit-test-pass, or reward distillation.
3. KL strategy. Fixed β, adaptive β, or DPO (implicit KL).
4. Diagnostics. Mean KL, reward stability, over-optimization guard (holdout human eval).
5. Safety gate. Red-team set, refusal rate, safety RM separate from helpfulness RM.

Refuse to ship RLHF-PPO without a KL monitor. Refuse to use an RM smaller than the target policy. Refuse length-only rewards. Flag any pipeline that does not hold back a blind human-eval set as lacking over-optimization protection.
```

## Упражнения

1. **Легко.** Обучите Bradley-Terry reward model в `code/main.py` на 500 synthetic preference pairs. Измерьте pairwise accuracy на held-out 100 pairs. Должно быть выше 90%.
2. **Средне.** Запустите toy PPO-RLHF loop с `β ∈ {0.0, 0.1, 1.0}`. Для каждого варианта постройте график RM score vs KL-to-reference по updates. Какие runs дают reward-hack?
3. **Сложно.** Реализуйте DPO (closed-form preference-likelihood loss) на тех же preference data и сравните с RLHF-PPO pipeline по использованному compute и достигнутому final RM score.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| RLHF | "Alignment RL" | Трехэтапный pipeline SFT + RM + PPO (Christiano 2017, Ouyang 2022). |
| Reward Model (RM) | "The scoring net" | Learned scalar function, обученная на pairwise preferences через Bradley-Terry. |
| Bradley-Terry | "Pairwise logistic loss" | `P(y_+ ≻ y_-) = σ(R(y_+) - R(y_-))`; стандартная RM objective. |
| KL penalty | "Stay near the reference" | `β · KL(π_θ || π_ref)` в reward; anti-reward-hacking regularizer. |
| Reward hacking | "Goodhart's law" | Policy эксплуатирует flaws RM; симптомы: reward растет, human eval плоский. |
| RLAIF | "AI-labeled preferences" | RLHF, где labels приходят от другой LM, а не от людей. |
| PRM | "Process Reward Model" | Оценивает partial reasoning steps; используется в reasoning pipelines. |
| Constitutional AI | "Anthropic's method" | AI-generated preferences, направляемые явными правилами. |

## Дополнительное чтение

- [Christiano et al. (2017). Deep Reinforcement Learning from Human Preferences](https://arxiv.org/abs/1706.03741) — статья, с которой начался RLHF.
- [Ouyang et al. (2022). InstructGPT — Training language models to follow instructions with human feedback](https://arxiv.org/abs/2203.02155) — рецепт, лежащий за ChatGPT.
- [Stiennon et al. (2020). Learning to summarize with human feedback](https://arxiv.org/abs/2009.01325) — более ранний RLHF для summarization.
- [Rafailov et al. (2023). Direct Preference Optimization](https://arxiv.org/abs/2305.18290) — DPO; post-RLHF default в 2026.
- [Bai et al. (2022). Constitutional AI: Harmlessness from AI Feedback](https://arxiv.org/abs/2212.08073) — RLAIF и self-critique loop.
- [Anthropic RLHF paper (Bai et al. 2022). Training a Helpful and Harmless Assistant](https://arxiv.org/abs/2204.05862) — HH paper.
- [Hugging Face TRL library](https://huggingface.co/docs/trl) — production `RewardTrainer` и `PPOTrainer`. Прочитайте trainer source, чтобы понять adaptive-KL и value-head details.
- [Hugging Face — Illustrating Reinforcement Learning from Human Feedback](https://huggingface.co/blog/rlhf) by Lambert, Castricato, von Werra, Havrilla — canonical walk-through трехэтапного pipeline с diagrams.
- [von Werra et al. (2020). TRL: Transformer Reinforcement Learning](https://github.com/huggingface/trl) — библиотека; в `examples/` есть end-to-end RLHF scripts для Llama, Mistral и Qwen.
- [Sutton & Barto (2018). Ch. 17.4 — Designing Reward Signals](http://incompleteideas.net/book/RLbook2020.pdf) — reward-hypothesis view; essential prerequisite для размышлений о reward hacking.
