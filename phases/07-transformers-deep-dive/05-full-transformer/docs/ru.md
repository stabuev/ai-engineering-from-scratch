# The Full Transformer — Encoder + Decoder

> Attention — главный элемент. Все остальное — residuals, normalization, feed-forward, cross-attention — это каркас, который позволяет складывать его глубоко.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 03 (Multi-Head Attention), Phase 7 · 04 (Positional Encoding)
**Time:** ~75 minutes

## Проблема

Один attention layer — это feature extractor, а не полноценная модель. Одного matmul на слой недостаточно для языка. Нужна глубина — а глубина ломается без правильной обвязки.

Статья Vaswani 2017 года собрала шесть проектных решений, которые превратили один attention layer в блок, который можно складывать в стек. Каждый transformer после этого — encoder-only (BERT), decoder-only (GPT), encoder-decoder (T5) — наследует тот же скелет. В 2026 году блоки уточнены (RMSNorm, SwiGLU, pre-norm, RoPE), но скелет тот же.

Этот урок — про скелет. Следующие уроки специализируют его: 06 для encoders, 07 для decoders, 08 для encoder-decoder.

## Концепция

![Encoder and decoder block internals, wired](../assets/full-transformer.svg)

### Шесть частей

1. **Embedding + positional signal.** Tokens → vectors. Позиция внедряется через RoPE (современно) или sinusoidal (классически).
2. **Self-attention.** Каждая позиция смотрит на каждую другую. В decoders применяется masking.
3. **Feed-forward network (FFN).** Position-wise двухслойный MLP: `W_2 · activation(W_1 · x)`. Expansion ratio по умолчанию 4×.
4. **Residual connection.** `x + sublayer(x)`. Без этого gradients исчезают после ~6 layers.
5. **Layer normalization.** `LayerNorm` или `RMSNorm` (современно). Стабилизирует residual stream.
6. **Cross-attention (только decoder).** Queries приходят из decoder, keys и values — из выхода encoder.

### Encoder block (used by BERT, T5 encoder)

```
x → LN → MHA(self) → + → LN → FFN → + → out
                     ^              ^
                     |              |
                     └── residual ──┘
```

Encoder двунаправленный. Без masking. Все позиции видят все позиции.

### Decoder block (used by GPT, T5 decoder)

```
x → LN → MHA(masked self) → + → LN → MHA(cross to encoder) → + → LN → FFN → + → out
```

Decoder имеет три sublayers на block. Средний — cross-attention — единственное место, где информация течет из encoder в decoder. В чистой decoder-only архитектуре (GPT) cross-attention опускается, остаются masked self-attention + FFN.

### Pre-norm vs post-norm

Оригинальная статья: `x + sublayer(LN(x))` против `LN(x + sublayer(x))`. Post-norm вышел из моды примерно в 2019 году — глубокие модели с ним сложнее обучать без аккуратного warmup. Pre-norm (`LN` *до* sublayer) — default 2026 года: Llama, Qwen, GPT-3+, Mistral используют его.

### Modernized block 2026 года

Vaswani 2017 использовал LayerNorm + ReLU. Современные стеки заменили оба. Вот как реально выглядят production blocks:

| Component | 2017 | 2026 |
|-----------|------|------|
| Normalization | LayerNorm | RMSNorm |
| FFN activation | ReLU | SwiGLU |
| FFN expansion | 4× | 2.6× (SwiGLU uses three matrices, total params match) |
| Position | Sinusoidal absolute | RoPE |
| Attention | Full MHA | GQA (or MLA) |
| Bias terms | Yes | No |

RMSNorm убирает mean-centering из LayerNorm (на одно вычитание меньше), что экономит compute и эмпирически не хуже по stability. SwiGLU (`Swish(W1 x) ⊙ W3 x`) стабильно превосходит ReLU/GELU FFN примерно на ~0.5 point ppl в статьях Llama, PaLM и Qwen.

### Число параметров

Для одного block с `d_model = d` и FFN expansion `r`:

- MHA: `4 · d²` (Q, K, V, O projections)
- FFN (SwiGLU): `3 · d · (r · d)` ≈ `3rd²`
- Norms: пренебрежимо мало

При `d = 4096, r = 2.6, layers = 32` (примерно Llama 3 8B), total: `32 · (4·4096² + 3·2.6·4096²) ≈ 32 · (16 + 32) M = ~1.5B parameters per layer × 32 ≈ 7B` (плюс embeddings и head). Это совпадает с опубликованными числами.

## Build It

### Step 1: building blocks

Используя маленький класс `Matrix` из Lesson 03 (скопирован сюда для независимости):

- `layer_norm(x, eps=1e-5)` — вычесть mean, поделить на std.
- `rms_norm(x, eps=1e-6)` — поделить на RMS. Без вычитания mean.
- `gelu(x)` и `silu(x) * W3 x` (SwiGLU).
- `ffn_swiglu(x, W1, W2, W3)`.
- `encoder_block(x, params)` и `decoder_block(x, enc_out, params)`.

Полную проводку см. в `code/main.py`.

### Step 2: соедините 2-layer encoder и 2-layer decoder

Сложите их в стек. Передайте выход encoder в каждый decoder cross-attention. Добавьте final LN перед output projection.

```python
def encode(tokens, params):
    x = embed(tokens, params.emb) + sinusoidal(len(tokens), params.d)
    for block in params.encoder_blocks:
        x = encoder_block(x, block)
    return x

def decode(target_tokens, encoder_out, params):
    x = embed(target_tokens, params.emb) + sinusoidal(len(target_tokens), params.d)
    for block in params.decoder_blocks:
        x = decoder_block(x, encoder_out, block)
    return x
```

### Step 3: run forward on a toy example

Пропустите 6-token source и 5-token target через модель. Проверьте, что форма выхода равна `(5, vocab)`. Без обучения — этот урок про архитектуру, а не про loss.

### Step 4: замените на RMSNorm + SwiGLU

Замените LayerNorm и ReLU-FFN на RMSNorm и SwiGLU. Подтвердите, что shapes все еще совпадают. Это modernization 2026 года через замену одной функции.

## Use It

Референсные реализации PyTorch/TF: `nn.TransformerEncoderLayer`, `nn.TransformerDecoderLayer`. Но большинство production-кода 2026 года пишет свой block, потому что:

- Flash Attention вызывается внутри attention, а не через `nn.MultiheadAttention`.
- GQA / MLA отсутствуют в stdlib reference.
- RoPE, RMSNorm, SwiGLU не являются defaults в PyTorch.

У HF `transformers` есть чистые reference blocks, которые стоит прочитать: `modeling_llama.py` — canonical decoder-only block 2026 года. Это ~500 строк, и их стоит один раз пройти.

**Encoder vs decoder vs encoder-decoder — когда что выбирать:**

| Need | Pick | Example |
|------|------|---------|
| Classification, embeddings, QA over text | Encoder-only | BERT, DeBERTa, ModernBERT |
| Text generation, chat, code, reasoning | Decoder-only | GPT, Llama, Claude, Qwen |
| Structured input → structured output (translation, summarization) | Encoder-decoder | T5, BART, Whisper |

Decoder-only победил в языке, потому что масштабируется чище всего и обрабатывает и понимание, и генерацию. Encoder-decoder все еще лучше, когда вход имеет явную идентичность "source sequence" (translation, speech recognition, structured tasks).

## Ship It

См. `outputs/skill-transformer-block-reviewer.md`. Skill проверяет новую реализацию transformer block против defaults 2026 года и отмечает пропущенные части (pre-norm, RoPE, RMSNorm, GQA, FFN expansion ratio).

## Упражнения

1. **Easy.** Посчитайте параметры в вашем `encoder_block` при `d_model=512, n_heads=8, ffn_expansion=4, swiglu=True`. Проверьте, реализовав block и используя `sum(p.numel() for p in block.parameters())`.
2. **Medium.** Переключитесь с post-norm на pre-norm. Инициализируйте оба варианта и измерьте activation norm после 12 stacked layers на random input. У post-norm activations должны взрываться; у pre-norm оставаться bounded.
3. **Hard.** Реализуйте 4-layer encoder-decoder на toy copy task (копировать `x` в обратном порядке). Обучите 100 steps. Сообщите loss. Замените на RMSNorm + SwiGLU + RoPE — падает ли loss?

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Block | "One transformer layer" | Стек norm + attention + norm + FFN, обернутый residual connections. |
| Residual | "Skip connection" | Выход `x + f(x)`; обеспечивает gradient flow через глубокие стеки. |
| Pre-norm | "Normalize before, not after" | Современный вариант: `x + sublayer(LN(x))`. Обучает более глубокие модели без gymnastics с warmup. |
| RMSNorm | "LayerNorm without the mean" | Деление на RMS; на одну операцию меньше, та же эмпирическая stability. |
| SwiGLU | "The FFN everyone switched to" | `Swish(W1 x) ⊙ W3 x → W2`. Лучше ReLU/GELU по LM ppl. |
| Cross-attention | "How the decoder sees the encoder" | MHA с Q из decoder, K/V из encoder outputs. |
| FFN expansion | "How wide the middle MLP is" | Отношение hidden-size к d_model, обычно 4 (LayerNorm) или 2.6 (SwiGLU). |
| Bias-free | "Drop the +b terms" | Современные стеки опускают biases в linear layers; небольшое улучшение ppl, меньшая модель. |

## Дополнительное чтение

- [Vaswani et al. (2017). Attention Is All You Need](https://arxiv.org/abs/1706.03762) — оригинальная спецификация block.
- [Xiong et al. (2020). On Layer Normalization in the Transformer Architecture](https://arxiv.org/abs/2002.04745) — почему pre-norm лучше post-norm в глубине.
- [Zhang, Sennrich (2019). Root Mean Square Layer Normalization](https://arxiv.org/abs/1910.07467) — RMSNorm.
- [Shazeer (2020). GLU Variants Improve Transformer](https://arxiv.org/abs/2002.05202) — статья SwiGLU.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical decoder-only block 2026 года.
