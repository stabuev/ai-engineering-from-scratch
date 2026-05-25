# Multi-Head Attention

> Одна attention head за раз учит одно отношение. Восемь heads учат восемь. Heads дешевы. Берите больше.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 02 (self-attention с нуля)
**Время:** ~75 минут

## Проблема

Одна self-attention head вычисляет одну матрицу attention. Эта матрица захватывает один тип отношения — обычно тот, который минимизирует loss для текущего training signal. Если в ваших данных subject-verb agreement, co-reference, long-range discourse и syntactic chunking переплетены вместе, одна head размазывает их в одно softmax-распределение и теряет половину сигнала.

Исправление из статьи Vaswani 2017 года: запустить несколько attention-функций параллельно, каждую со своими проекциями Q, K, V, и конкатенировать выходы. Каждая head работает в меньшем подпространстве размерности `d_model / n_heads`. Общее число параметров остается тем же. Выразительная мощность растет.

Multi-head attention — default, с которым в 2026 году поставляется каждый transformer. Спорят только о том, *сколько* heads брать и должны ли keys и values разделять проекции (Grouped-Query Attention, Multi-Query Attention, Multi-head Latent Attention).

## Концепция

![Multi-head attention splits, attends, concatenates](../assets/multi-head-attention.svg)

**Разделение.** Возьмите `X` формы `(N, d_model)`. Спроецируйте в Q, K, V, каждую формы `(N, d_model)`. Измените форму на `(N, n_heads, d_head)`, где `d_head = d_model / n_heads`. Транспонируйте в `(n_heads, N, d_head)`.

**Параллельное attention.** Запустите scaled dot-product attention внутри каждой head. Каждая head производит `(N, d_head)`. Heads работают с разными подпространствами embedding и не взаимодействуют во время самого вычисления attention.

**Конкатенация и проекция.** Соберите heads обратно в `(N, d_model)` и умножьте на обучаемую выходную матрицу `W_o` формы `(d_model, d_model)`. `W_o` — место, где heads смешиваются.

**Почему это работает.** Каждая head может специализироваться, не конкурируя с другими за representational budget. Probing studies 2019–2024 показывают разные роли heads: positional heads, head, которая смотрит на предыдущий токен, copy heads, named-entity heads, induction heads (лежащие в основе in-context learning).

**Линия вариантов к 2026 году:**

| Вариант | Q-heads | K/V-heads | Где используется |
|---------|---------|-----------|------------------|
| Multi-head (MHA) | N | N | GPT-2, BERT, T5 |
| Multi-query (MQA) | N | 1 | PaLM, Falcon |
| Grouped-query (GQA) | N | G (e.g. N/8) | Llama 2 70B, Llama 3+, Qwen 2+, Mistral |
| Multi-head latent (MLA) | N | сжаты до low-rank | DeepSeek-V2, V3 |

GQA — современный default, потому что он сокращает память KV-cache в `N/G` раз, сохраняя почти полное качество. MLA идет дальше: сжимает K/V в latent space, затем проецирует обратно во время вычислений — тратит FLOPs, но экономит намного больше памяти.

## Соберите это

### Шаг 1: разделите heads из single-head attention, который у нас уже есть

Возьмите `SelfAttention` из Урок 02 и оберните его парой split/concat. См. `code/main.py` для numpy-реализации; логика такая:

```python
def split_heads(X, n_heads):
    n, d = X.shape
    d_head = d // n_heads
    return X.reshape(n, n_heads, d_head).transpose(1, 0, 2)  # (heads, n, d_head)

def combine_heads(H):
    h, n, d_head = H.shape
    return H.transpose(1, 0, 2).reshape(n, h * d_head)
```

Один reshape и один transpose. Без цикла. Именно это PyTorch делает внутри `nn.MultiheadAttention`.

### Шаг 2: запустите scaled-dot-product attention для каждой head

Каждая head получает свой срез Q, K, V. Attention становится batched matmul:

```python
def mha_forward(X, W_q, W_k, W_v, W_o, n_heads):
    Q = X @ W_q
    K = X @ W_k
    V = X @ W_v
    Qh = split_heads(Q, n_heads)         # (heads, n, d_head)
    Kh = split_heads(K, n_heads)
    Vh = split_heads(V, n_heads)
    scores = Qh @ Kh.transpose(0, 2, 1) / np.sqrt(Qh.shape[-1])
    weights = softmax(scores, axis=-1)
    out = weights @ Vh                    # (heads, n, d_head)
    concat = combine_heads(out)
    return concat @ W_o, weights
```

На реальном железе `Qh @ Kh.transpose(...)` — это один `bmm`. GPU видит один batched matmul формы `(heads, N, d_head) × (heads, d_head, N) -> (heads, N, N)`. Добавлять heads почти бесплатно.

### Шаг 3: вариант Grouped-Query Attention

Меняются только проекции key и value. Q получает `n_heads` групп; K и V получают `n_kv_heads < n_heads` групп и повторяются для совпадения:

```python
def gqa_project(X, W, n_kv_heads, n_heads):
    kv = split_heads(X @ W, n_kv_heads)       # (kv_heads, n, d_head)
    repeat = n_heads // n_kv_heads
    return np.repeat(kv, repeat, axis=0)      # (n_heads, n, d_head)
```

На inference это экономит память, потому что в KV cache живут только `n_kv_heads` копий, а не `n_heads`. Llama 3 70B использует 64 query heads с 8 KV heads — сжатие cache в 8×.

### Шаг 4: исследуйте, что выучила каждая head

Запустите MHA на коротком предложении с 4 heads. Для каждой head напечатайте матрицу attention `(N, N)`. Вы увидите, что разные heads выбирают разную структуру даже при random initialization — это частично сигнал, частично rotational symmetry в подпространствах.

## Используйте это

В PyTorch однострочная версия:

```python
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=512, num_heads=8, batch_first=True)
```

GQA начиная с PyTorch 2.5+:

```python
from torch.nn.functional import scaled_dot_product_attention

# scaled_dot_product_attention auto-dispatches Flash Attention on CUDA.
# For GQA, pass Q of shape (B, n_heads, N, d_head) and K,V of shape
# (B, n_kv_heads, N, d_head). PyTorch handles the repeat.
out = scaled_dot_product_attention(q, k, v, is_causal=True, enable_gqa=True)
```

**Сколько heads?** Практические правила из production-моделей в 2026 году:

| Размер модели | d_model | n_heads | d_head |
|---------------|---------|---------|--------|
| Small (~125M) | 768 | 12 | 64 |
| Base (~350M) | 1024 | 16 | 64 |
| Large (~1B) | 2048 | 16 | 128 |
| Frontier (~70B) | 8192 | 64 | 128 |

`d_head` почти всегда равен 64 или 128. Это единица того, сколько одна head может "видеть". Ниже 32 heads начинают бороться с scaling factor `sqrt(d_head)`; выше 256 вы теряете преимущество "множества маленьких специалистов".

## Доведите до поставки

См. `outputs/skill-mha-configurator.md`. Skill рекомендует число heads, число kv-heads и стратегию projection для нового transformer с учетом parameter budget, длины последовательности и deployment target.

## Упражнения

1. **Легко.** Возьмите MHA из `code/main.py` и измените `n_heads` с 1 на 16 при фиксированном `d_model=64`. Постройте loss маленькой one-layer модели на синтетической copy task. Больше heads помогают, выходят на плато или вредят?
2. **Средне.** Реализуйте MQA (одна KV head, общая для всех query heads). Измерьте, насколько падает число параметров по сравнению с full MHA. Посчитайте, насколько уменьшается KV-cache на inference для N=2048.
3. **Сложно.** Реализуйте маленькую версию Multi-head Latent Attention: сожмите K,V в rank-`r` latent, храните latent в KV cache, распаковывайте во время attention. При каком `r` память cache падает ниже 1/8 от full MHA, пока качество остается в пределах 1 bit от validation ppl?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|------------|-------------------------------|
| Head | "Один attention circuit" | Одна Q/K/V projection размерности `d_head = d_model / n_heads` со своей матрицей attention. |
| d_head | "Размерность head" | Скрытая ширина на одну head; в production почти всегда 64 или 128. |
| Split / combine | "Трюки с reshape" | `(N, d_model) ↔ (n_heads, N, d_head)` reshape+transpose вокруг attention. |
| W_o | "Выходная проекция" | Матрица `(d_model, d_model)`, применяемая после конкатенации heads; место, где heads смешиваются. |
| MQA | "Одна KV-head" | Multi-Query Attention: одна общая K/V projection. Минимальный KV cache, некоторая потеря качества. |
| GQA | "Default со времен Llama 2" | Grouped-Query Attention с `n_kv_heads < n_heads`; повторяет их для соответствия Q. |
| MLA | "Трюк DeepSeek" | Multi-head Latent Attention: K,V сжимаются в low-rank latent и распаковываются во время attend. |
| Induction head | "Circuit за in-context learning" | Пара heads, которые обнаруживают предыдущие вхождения и копируют то, что следовало за ними. |

## Дополнительное чтение

- [Vaswani et al. (2017). Attention Is All You Need §3.2.2](https://arxiv.org/abs/1706.03762) — оригинальная спецификация multi-head.
- [Shazeer (2019). Fast Transformer Decoding: One Write-Head is All You Need](https://arxiv.org/abs/1911.02150) — статья про MQA.
- [Ainslie et al. (2023). GQA: Training Generalized Multi-Query Transformer Models from Multi-Head Checkpoints](https://arxiv.org/abs/2305.13245) — как преобразовать MHA в GQA после обучения.
- [DeepSeek-AI (2024). DeepSeek-V2 Technical Report](https://arxiv.org/abs/2405.04434) — MLA и почему он превосходит MHA/GQA по памяти cache.
- [Olsson et al. (2022). In-context Learning and Induction Heads](https://transformer-circuits.pub/2022/in-context-learning-and-induction-heads/index.html) — mechanistic взгляд на то, что heads действительно делают.
