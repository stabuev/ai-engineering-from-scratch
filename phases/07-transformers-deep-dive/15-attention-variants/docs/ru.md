# Варианты Attention — Sliding Window, Sparse, Differential

> Full attention — это круг. Каждый токен видит каждый токен, и memory платит цену. Четыре варианта изгибают форму круга и возвращают половину стоимости.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 02 (Self-Attention), Фаза 7 · 03 (Multi-Head), Фаза 7 · 12 (KV Cache / Flash Attention)
**Время:** ~60 минут

## Проблема

Full attention стоит `O(N²)` memory и `O(N²)` compute по длине sequence. Для 128K-context Llama 3 70B это 16 billion attention entries per layer, умножить на 80 layers. Flash Attention (Lesson 12) скрывает `O(N²)` activation memory, но не меняет arithmetic cost — каждый token все еще attends to every other token.

Три класса вариантов меняют саму topology attention matrix:

1. **Sliding window attention (SWA).** Каждый token attends to fixed window of neighbors, а не full prefix. Memory и compute падают до `O(N · W)`, где `W` — window. Gemma 2/3, первые layers Mistral 7B, Phi-3-Long.
2. **Sparse / block attention.** Оцениваются только выбранные пары `(i, j)`; остальные принудительно получают zero weight. Longformer, BigBird, OpenAI sparse transformer.
3. **Differential attention.** Вычислить две attention maps с отдельными Q/K projections, вычесть одну из другой. Убирает "attention sink", который уводит weight в первые несколько tokens. Microsoft DIFF Transformer (2024).

Они сосуществуют. Frontier model 2026 года часто смешивает их: большинство layers — SWA-1024, каждый пятый — global full attention, и несколько differential heads, которые очищают retrieval. Ratio Gemma 3 5:1 SWA-to-global — текущий textbook default.

## Концепция

### Sliding Window Attention (SWA)

Каждый query at position `i` attends only to positions in `[i - W, i]` (causal SWA) или `[i - W/2, i + W/2]` (bidirectional). Tokens вне window получают `-inf` в score matrix.

```
full causal:           sliding window (W=4):
positions 0-7          positions 0-7, W=4
    0 1 2 3 4 5 6 7        0 1 2 3 4 5 6 7
0 | x                0 |  x
1 | x x              1 |  x x
2 | x x x            2 |  x x x
3 | x x x x          3 |  x x x x
4 | x x x x x        4 |    x x x x
5 | x x x x x x      5 |      x x x x
6 | x x x x x x x    6 |        x x x x
7 | x x x x x x x x  7 |          x x x x
```

Для `N = 8192` и `W = 1024` score matrix имеет 1024 × 8192 non-zero rows in expectation — reduction 8×.

**KV cache shrinks with SWA.** Нужно хранить только последние `W` tokens K и V на layer. Для Gemma-3-ish config (1024 window, 128K context) KV cache падает в 128×.

**Quality cost.** SWA-only transformers struggle with long-range retrieval. Исправление: interleave SWA layers with full-attention layers. Gemma 3 использует 5:1 SWA:global. Mistral 7B использовал causal-SWA stack, где information "flows forward" through overlapping windows — каждый layer расширяет effective receptive field на `W`, и после `L` layers модель может attend на `L × W` tokens назад.

### Sparse / Block Attention

Заранее выберите `N × N` sparsity pattern. Три canonical shapes:

- **Local + strided (OpenAI sparse transformer).** Attend to last `W` tokens плюс каждый `stride`-th token до этого. Захватывает local и long-range при `O(N · sqrt(N))` compute.
- **Longformer / BigBird.** Local window + небольшой набор global tokens (например, `[CLS]`), которые attend to everyone и attended by everyone + random-sparse links. Empirical 2× context at matched quality.
- **Native Sparse Attention (DeepSeek, 2025).** Learn which blocks of `(Q, K)` matter; skip zero blocks на kernel level. FlashAttention-compatible.

Sparse attention — это история kernel engineering. Math простой (mask score matrix); выигрыш приходит от того, что zero entries никогда не загружаются в SRAM. FlashAttention-3 и API FlexAttention 2026 года делают custom sparse patterns first-class в PyTorch.

### Differential Attention (DIFF Transformer, 2024)

Regular attention имеет проблему "attention sink": softmax заставляет каждую row sum to 1, поэтому tokens, которые не хотят attend to anything particular, сбрасывают weight на first token (или первые несколько). Это крадет capacity, которая должна была уйти на real content.

Differential attention исправляет это, вычисляя **две** attention maps и вычитая:

```
A1 = softmax(Q1 K1^T / √d)
A2 = softmax(Q2 K2^T / √d)
DiffAttn = (A1 - λ · A2) V
```

где `λ` — learned scalar (обычно 0.5–0.8). A1 захватывает real content weights; A2 захватывает sink. Вычитание отменяет sink и перераспределяет weight к relevant tokens.

Reported results (Microsoft 2024): 5–10% lower perplexity, 1.5–2× longer effective context at same trained length, sharper needle-in-haystack retrieval.

### Сравнение вариантов

| Variant | Compute | KV cache | Quality vs full | Production use |
|---------|---------|----------|-----------------|----------------|
| Full attention | O(N²) | O(N) per layer | baseline | every model's default layer |
| SWA (window 1024) | O(N·W) | O(W) per layer | -0.1 ppl, good with global layers | Gemma 2/3, Phi-3-Long |
| Local + strided sparse | O(N·√N) | mixed | similar to SWA | OpenAI sparse transformer, Longformer |
| BigBird (local + global + random) | O(N) approx | mixed | matches full at 2× context | early long-context BERT |
| Native Sparse (DeepSeek-V3.2) | O(N · active fraction) | O(N) | within 0.05 ppl | DeepSeek-V3.2, 2025 |
| Differential | O(2·N²) | O(2N) | -5 to -10% ppl | DIFF Transformer, early 2026 models |

## Соберите это

См. `code/main.py`. Мы реализуем causal mask comparator, который показывает full, SWA, local+strided и differential attention side by side на toy sequence.

### Шаг 1: full causal mask (baseline)

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Baseline из Lesson 07. Lower triangular; zero weight above the diagonal.

### Шаг 2: sliding window causal mask

```python
def swa_mask(n, window):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
    return M
```

Один parameter — `window`. При `window >= n` вы восстанавливаете full causal attention. При `window = 1` каждый token attends only to itself.

### Шаг 3: local + strided sparse mask

```python
def strided_mask(n, window, stride):
    M = [[float("-inf")] * n for _ in range(n)]
    for i in range(n):
        lo = max(0, i - window + 1)
        for j in range(lo, i + 1):
            M[i][j] = 0.0
        for j in range(0, i + 1, stride):
            M[i][j] = 0.0
    return M
```

Dense local window плюс каждый `stride`-th token back to the start of the sequence. Receptive field растет log steps с дополнительными layers.

### Шаг 4: differential attention

```python
def diff_attention(Q1, K1, Q2, K2, V, lam):
    A1 = softmax_causal(Q1 @ K1.T / sqrt_d)
    A2 = softmax_causal(Q2 @ K2.T / sqrt_d)
    return (A1 - lam * A2) @ V
```

Два attention passes, вычитание с learned mixing coefficient. В коде мы сравниваем attention-sink heatmap single vs differential и наблюдаем, как sink collapse.

### Шаг 5: размеры KV cache

Напечатайте cache size per layer при `N = 131072` для каждого variant. SWA и sparse variants падают в 10–100×. Differential удваивает. Платите memory bill осознанно.

## Используйте это

Production patterns 2026:

```python
from transformers import AutoModelForCausalLM
# Gemma 3 mixes SWA (window=1024) and global layers at 5:1.
model = AutoModelForCausalLM.from_pretrained("google/gemma-3-27b-it")
# print(model.config.sliding_window, model.config.layer_types)
```

FlexAttention в PyTorch 2.5+ принимает mask function:

```python
from torch.nn.attention.flex_attention import flex_attention, create_block_mask

def swa_pattern(b, h, q_idx, kv_idx):
    return (q_idx - kv_idx < 1024) & (q_idx >= kv_idx)

mask = create_block_mask(swa_pattern, B=batch, H=heads, Q_LEN=n, KV_LEN=n)
out = flex_attention(q, k, v, block_mask=mask)
```

Это компилируется в custom Triton kernel. В пределах 10% скорости FlashAttention-3 для common patterns, а mask function — Python callable.

**Когда что выбирать:**

- **Pure full attention** — каждый layer до ~16K context или когда retrieval quality первична.
- **SWA + global mix** — long context (>32K), training and inference memory-bound. Default 2026 года выше 32K.
- **Sparse block attention** — custom kernel, custom pattern. Зарезервировано для specialized workloads (retrieval, audio).
- **Differential attention** — любая workload, где attention-sink contamination вредит (long-context RAG, needle-in-haystack).

## Доведите до поставки

См. `outputs/skill-attention-variant-picker.md`. Skill выбирает attention topology для новой модели по target context length, retrieval demands и training/inference compute profile.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Проверьте, что SWA при `window=4` zeroes everything outside the last 4 tokens per row. Проверьте, что `window=n` воспроизводит full causal attention bit-identically.
2. **Средне.** Реализуйте causal SWA с `window=1024` поверх capstone Lesson 07. Обучите 1 000 steps на tinyshakespeare. Насколько val loss регрессирует относительно full attention? Насколько падает peak memory?
3. **Сложно.** Реализуйте Gemma-3-style 5:1 layer mix (5 SWA, 1 global) в capstone model. Сравните loss, memory и generation quality с pure-SWA и pure-global baselines при matched parameters.
4. **Сложно.** Реализуйте differential attention с learned `λ` per head. Обучите на synthetic retrieval task (one needle, 2 000 distractors). Измерьте retrieval accuracy vs single-attention baseline при matched parameters.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Sliding window attention (SWA) | "Local attention" | Каждый query attends to its last `W` tokens; KV cache shrinks to `O(W)`. |
| Effective receptive field | "How far back the model sees" | В `L`-layer SWA stack с window `W`, до `L × W` tokens. |
| Longformer / BigBird | "Local + global + random" | Sparse patterns с несколькими always-attending global tokens; ранний long-context approach. |
| Native Sparse Attention | "DeepSeek's kernel trick" | Learn block-level sparsity; skip zero blocks at kernel level while keeping quality. |
| Differential attention | "Two maps, one subtracts" | DIFF Transformer: вычитает learned `λ` times a second attention map из первой, чтобы отменить attention sinks. |
| Attention sink | "Weight bleeds to token 0" | Softmax normalization forces rows to sum to 1; uninformative queries dump weight on position 0. |
| FlexAttention | "Mask-as-Python" | PyTorch 2.5+ API, который compiles arbitrary mask functions into FlashAttention-shape kernels. |
| Layer type mix | "5:1 SWA-to-global" | Interleave sparse and full attention layers in a stack, чтобы удержать quality при lower memory. |

## Дополнительное чтение

- [Beltagy, Peters, Cohan (2020). Longformer: The Long-Document Transformer](https://arxiv.org/abs/2004.05150) — canonical paper о sliding-window + global-token.
- [Zaheer et al. (2020). Big Bird: Transformers for Longer Sequences](https://arxiv.org/abs/2007.14062) — local + global + random.
- [Child et al. (2019). Generating Long Sequences with Sparse Transformers](https://arxiv.org/abs/1904.10509) — OpenAI local+strided pattern.
- [Gemma Team (2024). Gemma 2: Improving Open Language Models at a Practical Size](https://arxiv.org/abs/2408.00118) — mix 1:1 SWA:global.
- [Gemma Team (2025). Gemma 3 technical report](https://arxiv.org/abs/2503.19786) — mix 5:1 с window=1024, теперь textbook default.
- [Ye et al. (2024). Differential Transformer](https://arxiv.org/abs/2410.05258) — статья DIFF Transformer.
- [Yuan et al. (2025). Native Sparse Attention](https://arxiv.org/abs/2502.11089) — learned-sparsity attention DeepSeek-V3.2.
- [PyTorch — FlexAttention blog and docs](https://pytorch.org/blog/flexattention/) — API reference для mask-as-callable pattern в Use It.
