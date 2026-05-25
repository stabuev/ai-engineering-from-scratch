# Speculative Decoding and EAGLE

> Frontier LLM, генерирующая один token, требует полного forward pass по миллиардам parameters. Этот forward pass сильно over-provisioned: большую часть времени намного меньшая model может правильно угадать следующие 3-5 tokens, а big model должна только *verify* guess. Когда guess верный, вы получили 5 tokens по цене одного. Speculative decoding (Leviathan et al. 2023) сделало это точным, а EAGLE-3 (2025) поднял acceptance rates до ~4.5 tokens per verify — speedup 4-5x при совпадающем output distribution.

**Type:** Build
**Languages:** Python (with numpy)
**Prerequisites:** Phase 10 Lesson 12 (Inference Optimization), Phase 10 Lesson 04 (Pre-training Mini-GPT)
**Time:** ~75 minutes

## Проблема

Decode throughput для модели класса 70B на H100 обычно 40-80 tokens/second. Каждый token требует полного forward pass, читающего все model weights из HBM. Нельзя сделать model меньше без изменения output. Нельзя увеличить batch size beyond memory. Вы застряли — если только не позволить model output more than one token per forward pass.

Autoregressive generation выглядит intrinsically serial: `x_{t+1} = sample(p(· | x_{1:t}))`. Но есть concurrency opportunity. Если бы у вас был cheap predictor, который сказал "следующие 4 tokens probably [a, b, c, d]", вы могли бы verify all 5 positions in a **single forward pass of the big model** и accept longest matching prefix.

Leviathan, Kalai, Matias (2023, "Fast Inference from Transformers via Speculative Decoding") сделали это точным через clever accept/reject rule, сохраняющее target model sampling distribution. Та же output distribution, 2-4× быстрее.

## Концепция

### The Two-Model Setup

- **Target model** `M_p`: большая, медленная, high-quality model, из которой вы реально хотите samples. Distribution: `p(x)`.
- **Draft model** `M_q`: маленькая, быстрая, lower-quality model. Distribution: `q(x)`. В 5-30× меньше.

Per step:

1. Draft model предлагает `K` tokens autoregressively: `x_1, x_2, ..., x_K ~ q`.
2. Target model запускает ONE forward pass по всем `K+1` positions in parallel, producing `p(x_k)` для каждого proposed token.
3. Accept/reject каждый token left-to-right по modified rejection-sampling rule ниже. Accept longest matching prefix.
4. Если какой-то token rejected, sample replacement from corrected distribution и stop. Иначе sample one bonus token from `p(· | x_1...x_K)`.

Если draft идеально совпадает с target, вы получаете K+1 tokens на target-forward. Если draft ошибается на position 1, вы получаете только 1 token.

### The Exactness Rule

Speculative decoding **provably equivalent in distribution to sampling from p**. Rejection rule:

```
For each drafted token x_t:
    r ~ Uniform(0, 1)
    if r < p(x_t) / q(x_t):
        accept x_t
    else:
        sample replacement from residual: (p - q)+ / ||(p - q)+||_1
        stop
```

where `(p - q)+` denotes the positive part of the pointwise difference. Когда draft и target agree (`p ≈ q`), acceptance почти 1. Когда they disagree, residual distribution constructed so that overall sample is still exactly `p`.

**Greedy case.** Для temperature=0 sampling просто проверяйте `argmax(p) == x_t`. Если да, accept; если нет, output `argmax(p)` and stop.

### Expected Speedup

Если token-level acceptance rate draft model равен `α`, expected tokens produced per target-forward pass:

```
E[tokens] = (1 - α^{K+1}) / (1 - α)        # K = draft length, α in [0, 1]
```

При `α = 0.8, K = 4`: `(1 - 0.8^5)/(1 - 0.8) = 3.36` tokens per forward. Single target forward стоит roughly `cost_q * K + cost_p` (K draft steps plus one target verify). Если `cost_p >> cost_q * K`, speedup ratio равен `3.36× / 1 = 3.36×` по throughput.

Единственный реальный parameter — `α`, полностью зависящий от draft-target alignment. Хороший draft — все.

### Training the Draft: Distillation

Random small model — poor draft. Стандартный recipe: distill from target:

1. Выбрать small architecture (~1B для 70B target, ~500M для 7B target).
2. Запустить target model на large text corpus; сохранить ее next-token distributions.
3. Обучить draft с KL divergence against target's distribution (не against ground-truth tokens).

Результат: `α` обычно 0.6-0.8 на coding, 0.7-0.85 на natural-language chat. Speedups 2-3× in production.

### EAGLE: Tree Drafting + Feature Reuse

Li, Wei, Zhang, Zhang (2024, "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty") заметили две inefficiencies в standard speculative decoding:

1. Draft делает K serial steps, каждый full-stack. Но draft мог бы reuse target's features (hidden states) из most recent verify — target уже посчитала rich representations, которые draft заново выводит from scratch.
2. Draft outputs linear chain. Если draft мог бы output a *tree* of candidates (each node multiple guesses), target's single forward pass мог бы verify multiple candidate paths in parallel через tree attention mask и выбрать longest accepted branch.

EAGLE-1 changes:
- Draft input = target's final hidden state at position t, not raw tokens.
- Draft architecture = 1 transformer decoder layer (not a separate small model).
- Output = tree of K = 4-8 candidates per depth, depth 4-6.

EAGLE-2 (2024) добавляет dynamic tree topology: tree grows wider where draft uncertain and stays narrow where confident. Raises `α_effective` without increasing verify cost.

EAGLE-3 (Li et al. 2025, "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test") removes fixed top-layer feature dependency and trains the draft with a new "test-time simulation" loss — draft is trained on outputs that match target's test-time distribution rather than teacher-forced training distribution. Acceptance rate rises from 0.75 (EAGLE-2) to 0.82 (EAGLE-3), and mean tokens/verify from 3.0 to 4.5.

### Tree Attention Verification

Когда draft outputs a tree, target model verifies it in a single forward pass using a **tree attention mask** — causal mask, encoding tree topology rather than pure line. Each token attends only to its ancestors in the tree. Verify pass still one forward, one matmul; topological mask costs only a few extra KV entries.

```
        root
       /    \
      a      b
     / \    / \
    c  d   e   f
```

Если `a, b` — competing first-token candidates, а `c, d, e, f` — second-token candidates, все шесть positions verified in one forward pass. Output — longest prefix along any accepted path.

### When It Wins, When It Doesn't

**Wins:**
- Chat / completion с predictable text (code, common English, structured output). `α` high.
- Settings with unused GPU compute during decode (memory-bound phase). Tree drafting uses available FLOPs.

**Loses / no win:**
- Highly stochastic outputs (creative writing at high temperature). `α` drops toward `1/|vocab|`.
- Batch serving with very high concurrency — batching already fills FLOPs, little room for tree verification.
- Very small target models, где draft не намного меньше.

Production shops обычно report 2-3× wall-clock speedup на chat, 3-5× на code generation и near-zero на creative writing.

## Практика

`code/main.py`:

- Reference `speculative_decode(target, draft, prompt, K, temperature)`, который реализует exact rejection rule и verifies it preserves target's distribution (empirical KL < 0.01 vs plain target sampling).
- EAGLE-style tree drafter, который строит depth-K tree with top-p branching.
- Tree attention mask builder, producing the right causal pattern for verifier.
- Acceptance-rate harness, который запускает оба на tiny LM (distill one GPT-2-small from a GPT-2-medium target).

```python
def speculative_step(p_target, q_draft, K, temperature=1.0):
    """One round of speculative decoding. Returns list of accepted tokens."""
    # 1. Draft K tokens
    draft_tokens = []
    q_probs = []
    state = draft_state_init()
    for _ in range(K):
        probs = softmax(q_draft(state) / temperature)
        t = np.random.choice(len(probs), p=probs)
        draft_tokens.append(t)
        q_probs.append(probs[t])
        state = draft_step(state, t)

    # 2. Target computes p at every drafted position + 1 extra
    p_probs_all = target_forward_batched(p_target, draft_tokens, temperature)

    # 3. Accept/reject left-to-right
    accepted = []
    for k, tok in enumerate(draft_tokens):
        r = np.random.uniform()
        if r < p_probs_all[k][tok] / q_probs[k]:
            accepted.append(tok)
        else:
            residual = np.maximum(p_probs_all[k] - q_probs[k], 0)
            residual /= residual.sum()
            accepted.append(np.random.choice(len(residual), p=residual))
            return accepted
    # 4. All K accepted → sample bonus token from target
    accepted.append(np.random.choice(len(p_probs_all[-1]), p=p_probs_all[-1]))
    return accepted
```

## Использование

- **vLLM** и **SGLang** ship first-class speculative decoding. Flags: `--speculative_model`, `--num_speculative_tokens`. EAGLE-2/3 support через flag `--spec_decoding_algorithm eagle`.
- **NVIDIA TensorRT-LLM** поддерживает Medusa and EAGLE trees natively.
- **Reference draft models**: `Qwen/Qwen3-0.6B-spec` (drafts for Qwen3-32B), `meta-llama/Llama-3.2-1B-Instruct-spec` (drafts for 70B).
- **Medusa heads** (Cai et al. 2024, "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"): вместо draft model добавляет K parallel prediction heads к самому target. Simpler to deploy, slightly lower acceptance than EAGLE.

## Результат

Этот урок создает `outputs/skill-speculative-tuning.md` — skill, который profiles target model's workload и выбирает: draft model, K (draft length), tree width, temperature и when to fall back to plain decode.

## Упражнения

1. Реализуйте exact rejection rule и empirically verify it. Запустите 10K samples через `speculative_decode` и через plain target sampling; посчитайте TV distance между двумя output distributions. Должно быть < 0.01.

2. Посчитайте speedup formula. При fixed `α` and `K` постройте expected tokens per target-forward. Найдите optimal K для α ∈ {0.5, 0.7, 0.9}.

3. Обучите tiny draft. Возьмите 124M GPT-2 target и distill 30M GPT-2 draft на 100M tokens с KL loss. Измерьте `α` на held-out text. Expected: 0.6-0.7.

4. Реализуйте EAGLE-style tree drafting. Вместо chain пусть draft outputs top-3 branches at each depth. Постройте tree attention mask. Проверьте, что target accepts longest correct branch.

5. Измерьте failure modes. Запустите speculative decode при temperature=1.5 (high stochasticity). Покажите, что α collapses и algorithm slower than plain decode из-за draft overhead.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Target model | "The big model" | Slow, high-quality model, из которой нужны samples (p distribution) |
| Draft model | "The speculator" | Small, fast predictor (q distribution); в 5-30x меньше |
| K / draft length | "Look-ahead" | Число speculated tokens per verify pass |
| α / acceptance rate | "Hit rate" | Per-token probability, что proposal draft будет accepted |
| Exact rejection rule | "The accept test" | Compare r < p/q, preserving target's distribution |
| Residual distribution | "Corrected p-q" | (p - q)+ / ||(p - q)+||_1, distribution to sample from on rejection |
| Tree drafting | "Branching speculation" | Draft outputs tree of candidates, verified in one pass with tree-structured attention mask |
| Tree attention mask | "Topological mask" | Causal mask encoding tree topology so each node attends only to its ancestors |
| Medusa heads | "Parallel heads" | K extra prediction heads on target itself; no separate draft model |
| EAGLE feature reuse | "Hidden-state draft" | Draft input is target's last hidden state, not raw tokens, shrinking the draft |
| Test-time simulation loss | "EAGLE-3 training" | Train draft on outputs matching target's test-time distribution, not teacher forcing |

## Дополнительное чтение

- [Leviathan, Kalai, Matias, 2023 — "Fast Inference from Transformers via Speculative Decoding"](https://arxiv.org/abs/2211.17192) — exact rejection rule и theoretical speedup analysis
- [Chen, Borgeaud, Irving et al., 2023 — "Accelerating Large Language Model Decoding with Speculative Sampling"](https://arxiv.org/abs/2302.01318) — concurrent speculative-sampling paper at DeepMind
- [Cai, Li, Geng, Wang, Wang, Zhu, Dao, 2024 — "Medusa: Simple LLM Inference Acceleration Framework with Multiple Decoding Heads"](https://arxiv.org/abs/2401.10774) — parallel-heads alternative to draft model
- [Li, Wei, Zhang, Zhang, 2024 — "EAGLE: Speculative Sampling Requires Rethinking Feature Uncertainty"](https://arxiv.org/abs/2401.15077) — feature reuse and tree drafting
- [Li et al., 2024 — "EAGLE-2: Faster Inference of Language Models with Dynamic Draft Trees"](https://arxiv.org/abs/2406.16858) — dynamic tree topology
- [Li et al., 2025 — "EAGLE-3: Scaling up Inference Acceleration of Large Language Models via Training-Time Test"](https://arxiv.org/abs/2503.01840) — train-time test-time matching
- [Fu, Haotian, Peng et al., 2024 — "Break the Sequential Dependency of LLM Inference Using Lookahead Decoding"](https://arxiv.org/abs/2402.02057) — Jacobi/lookahead decoding, speculator-free alternative
