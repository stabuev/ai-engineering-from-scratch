# Speculative Decoding — черновик, проверка, повтор

> Autoregressive decoding последовательно. Каждый token ждет предыдущий. Speculative decoding разрывает цепочку: дешевая модель drafts N tokens, дорогая модель verifies все N за один forward pass. Когда draft прав, вы заплатили один большой forward за N generations.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 07 (GPT causal LM), Фаза 7 · 12 (KV cache и Flash Attention)
**Время:** ~60 минут

## Проблема

70B LLM sampling one token занимает ~30 ms на H100. 3B draft model занимает ~3 ms. Если позволить 3B draft заглянуть на 5 tokens вперед, а затем запустить 70B *один раз*, чтобы проверить все 5, итог `5×3 + 30 = 45 ms` за до 5 accepted tokens — против `5×30 = 150 ms` для straight-line generation. Это вся идея speculative decoding: обменять небольшой объем дополнительной GPU memory (draft model) на 2–4× меньшую decode latency.

Трюк должен сохранять distribution. Speculative sampling, введенный Leviathan et al. (2023) и параллельно Chen et al., гарантирует, что output sequence **identically distributed** относительно того, что большая модель выдала бы сама. Без quality tradeoff. Просто быстрее.

В inference 2026 года доминируют четыре семейства draft-verifier pairs:

1. **Vanilla speculative (Leviathan 2023).** Отдельная draft model (например, Llama 3 1B) + verifier (например, Llama 3 70B).
2. **Medusa (Cai 2024).** Multiple decoding heads на verifier предсказывают positions `t+1..t+k` параллельно. Без отдельной draft model.
3. **EAGLE family (Li 2024, 2025).** Lightweight draft, который переиспользует hidden states verifier; acceptance rate ближе к vanilla; типично 3–4×.
4. **Lookahead decoding (Fu 2024).** Jacobi iteration; draft model вообще не требуется. Self-speculation. Нишево, но dependency-free.

Каждый production inference stack в 2026 году поставляет speculative decoding по умолчанию. vLLM, TensorRT-LLM, SGLang и llama.cpp все поддерживают как минимум vanilla + EAGLE-2.

## Концепция

### Базовый алгоритм

Даны verifier `M_q` и более дешевый draft `M_p`:

1. Пусть `x_1..x_k` — уже decoded prefix.
2. **Draft**: используйте `M_p`, чтобы autoregressively предложить `d_{k+1}, d_{k+2}, ..., d_{k+N}` с draft probabilities `p_1..p_N`.
3. **Verify in parallel**: запустите `M_q` один раз на `x_1..x_k, d_{k+1}, ..., d_{k+N}`, получив verifier probabilities `q_1..q_{N+1}` для positions `k+1..k+N+1`.
4. **Accept/reject each draft token left to right**: для каждого `i` принять с вероятностью `min(1, q_i(d_i) / p_i(d_i))`.
5. При первом rejection at position `j`: sample `t_j` из "residual" distribution `(q_j - p_j)_+` normalized. Все drafts после `j` discarded.
6. Если приняты все `N`: sample one extra token `t_{N+1}` из `q_{N+1}` (free bonus token).

Residual distribution trick — математическая идея, которая сохраняет output distributed exactly as if `M_q` sampled from scratch.

### Что определяет speedup

Пусть `α` = expected acceptance rate per draft token. Пусть `c` = draft-to-verifier cost ratio. На step:

- Naive generation делает 1 big-model call per token.
- Speculative делает 1 big-model call per `(1 - α^{N+1}) / (1 - α) ≈ 1/(1-α)` tokens, когда `α` высока.

Typical rule of thumb при `α = 0.75` и `N = 5`: в 3× меньше big-model calls. Draft cost — 5× дешевых. Total wall-clock падает ~2.5×.

**α зависит от:**

- Насколько хорошо draft приближает verifier. Same family / same training data значительно повышает α.
- Decoding strategy. Greedy draft против greedy verifier: высокая α. Temperature sampling: труднее совпасть; acceptance drops.
- Task type. Code и structured output принимаются чаще (predictable); free-form creative writing — реже.

### Medusa — drafts без draft model

Medusa заменяет draft model дополнительными output heads на verifier. At position `t`:

```
shared trunk → hidden h_t
    ├── head_0: predict token at t+1  (standard LM head)
    ├── head_1: predict token at t+2
    ├── head_2: predict token at t+3
    ├── head_3: predict token at t+4
```

Каждая head выдает свои logits. На inference вы sample from each head, чтобы получить candidate sequence, затем verify одним forward pass через tree-attention scheme, который рассматривает все candidate continuations сразу.

Плюсы: нет второй модели. Минусы: добавляет trainable parameters; нужен supervised fine-tuning stage (~1B tokens); acceptance rate немного ниже, чем у vanilla speculative с хорошим draft.

### EAGLE — лучший draft через переиспользование hidden states

EAGLE-1/2/3 (Li et al., 2024–2025) делает draft model крошечным transformer (обычно 1 layer), который принимает last-layer hidden states verifier. Поскольку draft видит feature representation verifier, его predictions strongly correlate with verifier output distribution. Acceptance rates растут с ~0.6 (vanilla) до 0.85+.

EAGLE-3 (2025) добавил tree search over candidate continuations. vLLM и SGLang поставляют EAGLE-2/3 как default spec pathway для Llama 3/4 и Qwen 3.

### Танец KV cache

Verification подает `N` draft tokens в verifier за один forward pass. Это расширяет KV cache verifier на `N` entries. Если часть drafts rejected, нужно откатить cache до accepted prefix length.

Production implementations (vLLM `--speculative-model`, TensorRT-LLM LookaheadDecoder) делают это через scratch KV buffers. Сначала write, commit on acceptance. Концептуально несложно, но fiddly.

## Соберите это

См. `code/main.py`. Мы реализуем core speculative-sampling algorithm (rejection step + residual distribution) с:

- "Big model", которая является deterministic-softmax over hand-coded distribution (чтобы можно было analytically verify acceptance math).
- "Draft model", которая является perturbation of the big model.
- Acceptance / rejection loop, который производит ту же marginal distribution, что direct sampling.

### Шаг 1: rejection step

```python
def accept_or_reject(q_prob, p_prob, draft_token, u):
    ratio = q_prob / p_prob if p_prob > 0 else float("inf")
    return u < min(1.0, ratio)
```

`u` — uniform random number. `q_prob` — probability verifier для drafted token. `p_prob` — probability draft model. Теорема Leviathan: это Bernoulli decision, за которым следует sampling from residual on rejection, exactly preserves verifier distribution.

### Шаг 2: residual distribution

```python
def residual_dist(q, p):
    raw = [max(0.0, qi - pi) for qi, pi in zip(q, p)]
    s = sum(raw)
    return [r / s for r in raw]
```

Вычесть `p` из `q` element-wise, clamp negative values to zero, renormalize. Sample from this on any rejection.

### Шаг 3: один speculative step

```python
def spec_step(prefix, q_model, p_model, N, rng):
    drafts = []
    p_probs = []
    ctx = list(prefix)
    for _ in range(N):
        p_dist = p_model(ctx)
        d = sample(p_dist, rng)
        drafts.append(d)
        p_probs.append(p_dist[d])
        ctx.append(d)

    q_dists = [q_model(prefix + drafts[:i]) for i in range(N + 1)]

    for i, d in enumerate(drafts):
        u = rng.random()
        q_prob = q_dists[i][d]
        p_prob = p_probs[i]
        if u < min(1.0, q_prob / p_prob if p_prob > 0 else float("inf")):
            prefix = prefix + [d]
        else:
            res = residual_dist(q_dists[i], p_model(prefix))
            prefix = prefix + [sample(res, rng)]
            return prefix
    prefix = prefix + [sample(q_dists[N], rng)]
    return prefix
```

Пять принятых → один bonus → шесть tokens produced за один verifier pass.

### Шаг 4: измерьте acceptance rate

Запустите 10 000 speculative steps при разных draft-quality levels. Plot acceptance rate vs. KL divergence между draft и verifier distributions. Должна быть чистая monotone relationship.

### Шаг 5: проверьте distribution equivalence

Empirically: histogram tokens, produced by speculative loop, должен совпадать с histogram, produced by sampling directly from verifier. Это theorem Leviathan на практике. Chi-square test подтверждает within sampling error.

## Используйте это

Production:

```bash
# vLLM with EAGLE
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model /models/llama-3.1-eagle-70b \
    --speculative-draft-tensor-parallel-size 1 \
    --num-speculative-tokens 5

# vLLM with vanilla draft model
vllm serve meta-llama/Llama-3.1-70B-Instruct \
    --speculative-model meta-llama/Llama-3.2-1B-Instruct \
    --num-speculative-tokens 5
```

TensorRT-LLM имеет самый быстрый Medusa path на mid-2026. `faster-whisper` wraps speculative decoding для Whisper-large с small draft.

**Выбор draft:**

| Strategy | Когда выбирать | Speedup |
|----------|--------------|---------|
| Vanilla draft (семейство 1B/3B Llama) | Быстрый prototype, без training | 1.8–2.3× |
| Medusa heads | Можно fine-tune verifier | 2–3× |
| EAGLE-2 / 3 | Production, максимальная скорость | 3–4× |
| Lookahead | Без draft, без training, без extra params | 1.3–1.6× |

**Когда НЕ использовать spec-decode:**

- Single-sequence generation на 1–5 tokens. Overhead доминирует.
- Сильно creative / high-temperature sampling (α падает).
- Memory-constrained deployments (draft model добавляет VRAM).

## Доведите до поставки

См. `outputs/skill-spec-decode-picker.md`. Skill выбирает speculative decoding strategy (vanilla / Medusa / EAGLE / lookahead) и tuning parameters (N, draft temperature) для новой inference workload.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Подтвердите, что speculative token distribution совпадает с verifier direct-sample distribution на 50 000 tokens в пределах chi-square p > 0.05.
2. **Средне.** Постройте speedup (tokens per big-model forward) как function of `N` для `α = 0.5, 0.7, 0.85`. Определите optimal `N` для каждого α. (Hint: expected tokens per verify call = `(1 - α^{N+1}) / (1 - α)`.)
3. **Сложно.** Реализуйте tiny Medusa: возьмите capstone GPT из Урок 14, добавьте 3 extra LM heads, которые предсказывают positions t+2, t+3, t+4. Обучите на tinyshakespeare с joint multi-head loss. Сравните acceptance rates с vanilla draft, полученным truncating the same model.
4. **Сложно.** Реализуйте rollback: начните с 10-token prefix KV cache, подайте 5 draft tokens, simulate rejection at position 3. Проверьте, что cache reads correctly match "prefix + first 2 accepted drafts" на следующей iteration.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|------------|-------------------------------|
| Draft model | "Дешевая модель" | Меньшая model, которая предлагает candidate tokens; обычно в 10–50× дешевле verifier. |
| Verifier | "Большая модель" | Target model, distribution которой мы сохраняем; runs once per speculative step. |
| Acceptance rate (α) | "Как часто draft прав" | Per-token probability, что verifier принимает draft. Типично 0.7–0.9. |
| Residual distribution | "Fallback при rejection" | `(q - p)_+` normalized; sampling from this on rejection preserves verifier distribution. |
| Bonus token | "Бесплатный токен" | Когда все N drafts accepted, sample one more from verifier next-step distribution. |
| Medusa | "Speculative без draft model" | Multiple LM heads on verifier predict positions t+1..t+k in parallel. |
| EAGLE | "Draft по hidden states" | Tiny transformer draft conditioned on verifier last-layer hidden states. |
| Lookahead decoding | "Итерация Якоби" | Self-speculation using fixed-point iteration; no draft model. |
| Tree attention | "Проверить много кандидатов сразу" | Branching verification, который рассматривает несколько draft continuations simultaneously. |
| KV rollback | "Откатить rejected drafts" | Scratch KV buffer; commit on acceptance, discard on reject. |

## Дополнительное чтение

- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — core algorithm и equivalence theorem.
- [Chen et al. (2023). Accelerating Large Language Model Decoding with Speculative Sampling](https://arxiv.org/abs/2302.01318) — concurrent introduction; clean Bernoulli-rejection proof.
- [Cai et al. (2024). Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads](https://arxiv.org/abs/2401.10774) — статья Medusa; tree-attention verification.
- [Li et al. (2024). EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty](https://arxiv.org/abs/2401.15077) — EAGLE-1; hidden-state-conditioned draft.
- [Li et al. (2024). EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees](https://arxiv.org/abs/2406.16858) — EAGLE-2; dynamic tree depth.
- [Li et al. (2025). EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test](https://arxiv.org/abs/2503.01840) — EAGLE-3.
- [Fu et al. (2024). Break the Sequential Dependency of LLM Inference Using Lookahead Decoding](https://arxiv.org/abs/2402.02057) — lookahead, no-draft approach.
- [vLLM docs — Speculative Decoding](https://docs.vllm.ai/en/latest/features/spec_decode.html) — canonical production reference со всеми четырьмя подключенными strategies.
- [SafeAILab / EAGLE reference implementation](https://github.com/SafeAILab/EAGLE) — reference code для EAGLE-1/2/3.
