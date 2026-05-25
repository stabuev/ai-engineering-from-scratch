# Attention Mechanism — Прорыв

> Декодер перестает вглядываться в сжатое резюме и начинает смотреть на весь источник. Все после этого - attention плюс инженерия.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Phase 5 · 09 (Sequence-to-Sequence Models)
**Время:** ~45 минут

## Проблема

Урок 09 закончился измеримой неудачей. GRU encoder-decoder, обученный на игрушечной задаче копирования, падает с 89% accuracy при длине 5 до почти случайного уровня при длине 80. Причина структурная, а не баг обучения: каждый бит информации, который извлек encoder, должен уместиться в одно hidden state фиксированного размера, и decoder больше ничего не видит.

Bahdanau, Cho и Bengio в 2014 году опубликовали исправление в три строки. Вместо того чтобы давать decoder только финальное состояние encoder, сохраняйте каждое состояние encoder. На каждом шаге decoder вычисляйте взвешенное среднее состояний encoder, где веса отвечают на вопрос: "насколько decoder должен сейчас смотреть на позицию encoder `i`?" Это взвешенное среднее и есть context, и оно меняется на каждом шаге decoder.

В этом вся идея. Transformers ее расширили. Self-attention применил ее к одной последовательности. Multi-head attention запустил ее параллельно. Но версия 2014 года уже сломала bottleneck, и когда она у вас есть, переход к transformers - это инженерия, а не концептуальный скачок.

## Концепция

![Attention Бахданау: decoder запрашивает все состояния encoder](../assets/attention.svg)

На каждом шаге decoder `t`:

1. Используйте предыдущее hidden state decoder `s_{t-1}` как **query**.
2. Оцените его относительно каждого hidden state encoder `h_1, ..., h_T`. Один скаляр на позицию encoder.
3. Примените softmax к scores, чтобы получить attention weights `α_{t,1}, ..., α_{t,T}`, сумма которых равна 1.
4. Context vector `c_t = Σ α_{t,i} * h_i`. Взвешенное среднее состояний encoder.
5. Decoder берет `c_t` плюс предыдущий output token и производит следующий token.

Смысл именно во взвешенном среднем. Когда decoder должен перевести "Je" как "I", он дает высокий вес состоянию encoder над "Je" и низкие веса остальным. Когда ему нужно "not", он дает высокий вес "pas". Context vector перестраивается на каждом шаге.

## Shapes (то, на чем все спотыкаются)

Именно здесь каждая первая реализация attention ломается. Читайте медленно.

| Объект | Shape | Примечания |
|-------|-------|-------|
| Encoder hidden states `H` | `(T_enc, d_h)` | Если BiLSTM, `d_h = 2 * d_hidden` |
| Decoder hidden state `s_{t-1}` | `(d_s,)` | Один вектор |
| Attention score `e_{t,i}` | scalar | Один на позицию encoder |
| Attention weight `α_{t,i}` | scalar | После softmax по всем `i` |
| Context vector `c_t` | `(d_h,)` | Такая же shape, как у состояния encoder |

**Оценка Bahdanau (additive).** `e_{t,i} = v_α^T * tanh(W_a * s_{t-1} + U_a * h_i)`.

- `s_{t-1}` имеет shape `(d_s,)`, `h_i` имеет shape `(d_h,)`.
- `W_a` имеет shape `(d_attn, d_s)`. `U_a` имеет shape `(d_attn, d_h)`.
- Их сумма внутри tanh имеет shape `(d_attn,)`.
- `v_α` имеет shape `(d_attn,)`. Скалярное произведение с `v_α` схлопывается в scalar. **Вот что делает `v_α`.** Это не магия. Это проекция, которая превращает attention-dim vector в scalar score.

**Luong (multiplicative) score.** Три варианта:

- `dot`: `e_{t,i} = s_t^T * h_i`. Требует `d_s == d_h`. Жесткое ограничение. Пропустите, если ваш encoder bidirectional.
- `general`: `e_{t,i} = s_t^T * W * h_i`, где `W` имеет shape `(d_s, d_h)`. Убирает ограничение равных размерностей.
- `concat`: по сути форма Bahdanau. Используется редко, потому что первые две дешевле.

**Один важный нюанс Bahdanau / Luong.** Bahdanau использует `s_{t-1}` (состояние decoder *до* генерации текущего слова). Luong использует `s_t` (состояние *после*). Если их перепутать, получаются тонко неправильные gradients, которые крайне трудно отлаживать. Выберите одну статью и держитесь ее convention.

## Соберите это

### Шаг 1: additive (Bahdanau) attention

```python
import numpy as np


def additive_attention(decoder_state, encoder_states, W_a, U_a, v_a):
    projected_dec = W_a @ decoder_state
    projected_enc = encoder_states @ U_a.T
    combined = np.tanh(projected_enc + projected_dec)
    scores = combined @ v_a
    weights = softmax(scores)
    context = weights @ encoder_states
    return context, weights


def softmax(x):
    x = x - np.max(x)
    e = np.exp(x)
    return e / e.sum()
```

Проверьте свои shapes по таблице выше. `encoder_states` имеет shape `(T_enc, d_h)`. `projected_enc` имеет shape `(T_enc, d_attn)`. `projected_dec` имеет shape `(d_attn,)` и broadcast-ится. `combined` имеет shape `(T_enc, d_attn)`. `scores` имеет shape `(T_enc,)`. `weights` имеет shape `(T_enc,)`. `context` имеет shape `(d_h,)`. Можно отправлять.

### Шаг 2: Luong dot и general

```python
def dot_attention(decoder_state, encoder_states):
    scores = encoder_states @ decoder_state
    weights = softmax(scores)
    return weights @ encoder_states, weights


def general_attention(decoder_state, encoder_states, W):
    projected = W.T @ decoder_state
    scores = encoder_states @ projected
    weights = softmax(scores)
    return weights @ encoder_states, weights
```

По три строки на каждый. Поэтому статья Luong и сработала. Та же accuracy на большинстве задач, гораздо меньше кода.

### Шаг 3: разобранный численный пример

Даны три состояния encoder (примерно "cat", "sat", "mat") и состояние decoder, которое больше всего совпадает с первым; attention distribution концентрируется на позиции 0. Если состояние decoder сдвигается ближе к последнему состоянию encoder, attention переходит на позицию 2. Context vector отслеживает это.

```python
H = np.array([
    [1.0, 0.0, 0.2],
    [0.5, 0.5, 0.1],
    [0.1, 0.9, 0.3],
])

s_close_to_cat = np.array([0.9, 0.1, 0.2])
ctx, w = dot_attention(s_close_to_cat, H)
print("weights:", w.round(3))
```

```
weights: [0.464 0.305 0.231]
```

Побеждает первая строка. Затем сдвиньте состояние decoder ближе к третьему состоянию encoder и посмотрите, как смещаются веса. Вот и все. Attention - это явное alignment.

### Шаг 4: почему это мост к transformers

Переведите язык выше в Q/K/V:

- **Query** = состояние decoder `s_{t-1}`
- **Key** = состояния encoder (то, относительно чего мы считаем score)
- **Value** = состояния encoder (то, что мы взвешиваем и суммируем)

В classical attention keys и values - одно и то же. Self-attention разделяет их: можно query-ить последовательность относительно самой себя, с разными learned projections для K и V. Multi-head attention запускает это параллельно с разными learned projections. Transformers много раз складывают весь этот stage в стек и отбрасывают RNNs.

Математика та же. Shapes те же. Педагогический прыжок от Bahdanau attention к scaled dot-product attention - в основном notation.

## Используйте это

PyTorch и TensorFlow поставляют attention напрямую.

```python
import torch
import torch.nn as nn

mha = nn.MultiheadAttention(embed_dim=128, num_heads=8, batch_first=True)
query = torch.randn(2, 5, 128)
key = torch.randn(2, 10, 128)
value = torch.randn(2, 10, 128)

output, weights = mha(query, key, value)
print(output.shape, weights.shape)
```

```
torch.Size([2, 5, 128]) torch.Size([2, 5, 10])
```

Это transformer attention layer. Batch query из 5 позиций, batch key/value из 10 позиций, каждая 128-dimensional, 8 heads. `output` - новые queries, дополненные context. `weights` - матрица alignment 5x10, которую можно визуализировать.

### Когда classical attention все еще важен

- Педагогика. Single-head, single-layer, RNN-based версия делает каждую концепцию видимой.
- On-device sequence tasks, где transformers не помещаются.
- Любая статья 2014-2017 годов. Вы неверно ее прочитаете, если не знаете convention Bahdanau.
- Тонкий alignment analysis в MT. Raw attention weights - инструмент interpretability даже в transformer models, и чтобы их читать, нужно понимать, что они такое.

### Ловушка attention-weight-as-explanation

Attention weights выглядят интерпретируемыми. Это веса, которые суммируются в единицу по позициям; их можно построить на графике; высокий вес означает "смотрел сюда". Reviewers их любят.

Они не настолько интерпретируемы, насколько выглядят. Jain and Wallace (2019) показали, что attention distributions можно переставлять и заменять произвольными альтернативами без изменения predictions модели для некоторых задач. Никогда не подавайте attention weights как доказательство reasoning без ablation или counterfactual check.

## Отгрузите это

Сохраните как `outputs/prompt-attention-shapes.md`:

```markdown
---
name: attention-shapes
description: Debug shape bugs in attention implementations.
phase: 5
lesson: 10
---

Given a broken attention implementation, you identify the shape mismatch. Output:

1. Which matrix has the wrong shape. Name the tensor.
2. What its shape should be, derived from (d_s, d_h, d_attn, T_enc, T_dec, batch_size).
3. One-line fix. Transpose, reshape, or project.
4. A test to catch regressions. Typically: assert `output.shape == (batch, T_dec, d_h)` and `weights.shape == (batch, T_dec, T_enc)` and `weights.sum(dim=-1) close to 1`.

Refuse to recommend fixes that silently broadcast. Broadcast-hiding bugs surface later as silent accuracy degradation, the worst kind of attention bug.

For Bahdanau confusion, insist the decoder input is `s_{t-1}` (pre-step state). For Luong, `s_t` (post-step state). For dot-product, flag dimension mismatch between query and key as the most common first-time error.
```

## Упражнения

1. **Легко.** Реализуйте masking для `softmax`, чтобы padding tokens в encoder получали attention weight zero. Проверьте на batch с последовательностями переменной длины.
2. **Средне.** Добавьте multi-head attention к форме Luong `general`. Разбейте `d_h` на `n_heads` groups, выполните attention по каждой head, concatenate. Проверьте, что single-head case совпадает с вашей предыдущей реализацией.
3. **Сложно.** Обучите GRU encoder-decoder с Bahdanau attention на игрушечной задаче копирования из урока 09. Постройте accuracy vs sequence length. Сравните с baseline без attention. Вы должны увидеть, как gap растет с увеличением length, подтверждая, что attention снимает bottleneck.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| Attention | Смотреть на вещи | Взвешенное среднее последовательности values; веса вычисляются по similarity между query и key. |
| Query, Key, Value | QKV | Три проекции: Q спрашивает, K - то, с чем сопоставлять, V - то, что возвращать. |
| Additive attention | Bahdanau | Feed-forward оценка: `v^T tanh(W q + U k)`. |
| Multiplicative attention | Luong dot / general | Score - это `q^T k` или `q^T W k`. Дешевле, та же accuracy на большинстве задач. |
| Alignment matrix | Красивая картинка | Attention weights как grid `(T_dec, T_enc)`. Читайте ее, чтобы увидеть, на что модель обращала внимание. |

## Дополнительное чтение

- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — статья.
- [Luong, Pham, Manning (2015). Effective Approaches to Attention-based Neural Machine Translation](https://arxiv.org/abs/1508.04025) — три варианта score и их сравнение.
- [Jain and Wallace (2019). Attention is not Explanation](https://arxiv.org/abs/1902.10186) — оговорка об interpretability.
- [Dive into Deep Learning — Bahdanau Attention](https://d2l.ai/chapter_attention-mechanisms-and-transformers/bahdanau-attention.html) — запускаемый walkthrough с PyTorch.
