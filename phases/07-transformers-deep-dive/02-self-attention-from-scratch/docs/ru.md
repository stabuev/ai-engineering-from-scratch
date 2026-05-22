# Self-Attention from Scratch

> Attention — это lookup table, где каждое слово спрашивает "кто для меня важен?" — и учится ответу.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 3 (Deep Learning Core), Phase 5 Lesson 10 (Sequence-to-Sequence)
**Time:** ~90 minutes

## Цели обучения

- Реализовать scaled dot-product self-attention с нуля только на NumPy, включая query/key/value projections и softmax-weighted sum
- Построить multi-head attention layer, который разделяет heads, вычисляет parallel attention и конкатенирует результаты
- Проследить, как attention matrix захватывает отношения между токенами, и объяснить, почему scaling by sqrt(d_k) предотвращает saturation softmax
- Применить causal masking, чтобы превратить bidirectional attention в autoregressive (decoder-style) attention

## Проблема

RNN обрабатывают sequences по одному токену. К моменту, когда вы доходите до token 50, информация из token 1 уже прошла через 50 шагов сжатия. Long-range dependencies сминаются в hidden state фиксированного размера — bottleneck, который никакие LSTM gates полностью не устраняют.

Статья Bahdanau attention 2014 года показала исправление: дать decoder возможность смотреть назад на каждую encoder position и решать, какие важны для текущего step. Но это все еще было прикручено к RNN. Статья 2017 года "Attention Is All You Need" задала более острый вопрос: что если attention — *единственный* механизм? Без recurrence. Без convolution. Только attention.

Self-attention позволяет каждой позиции sequence attend к каждой другой позиции за один parallel step. Именно это делает transformers быстрыми, масштабируемыми и доминирующими.

## Концепция

### The Database Lookup Analogy

Думайте об attention как о мягком database lookup:

```
Traditional database:
  Query: "capital of France"  -->  exact match  -->  "Paris"

Attention:
  Query: "capital of France"  -->  similarity to ALL keys  -->  weighted blend of ALL values
```

Каждый token создает три вектора:
- **Query (Q)**: "Что я ищу?"
- **Key (K)**: "Что я содержу?"
- **Value (V)**: "Какую информацию я даю, если меня выбрали?"

Dot product между query и всеми keys дает attention scores. Высокий score означает "этот key подходит к моему query". Эти scores взвешивают values. Output — weighted sum of values.

### Q, K, V Computation

Каждый token embedding проецируется через три обучаемые weight matrices:

```
Input embeddings (sequence of n tokens, each d-dimensional):

  X = [x1, x2, x3, ..., xn]       shape: (n, d)

Three weight matrices:

  Wq  shape: (d, dk)
  Wk  shape: (d, dk)
  Wv  shape: (d, dv)

Projections:

  Q = X @ Wq    shape: (n, dk)      each token's query
  K = X @ Wk    shape: (n, dk)      each token's key
  V = X @ Wv    shape: (n, dv)      each token's value
```

Визуально для одного token:

```
             Wq
  x_i ------[*]------> q_i    "What am I looking for?"
       |
       |     Wk
       +----[*]------> k_i    "What do I contain?"
       |
       |     Wv
       +----[*]------> v_i    "What do I offer?"
```

### The Attention Matrix

Когда Q, K, V получены для всех tokens, attention scores образуют matrix:

```
Scores = Q @ K^T    shape: (n, n)

              k1    k2    k3    k4    k5
        +-----+-----+-----+-----+-----+
   q1   | 2.1 | 0.3 | 0.1 | 0.8 | 0.2 |   <- how much q1 attends to each key
        +-----+-----+-----+-----+-----+
   q2   | 0.4 | 1.9 | 0.7 | 0.1 | 0.3 |
        +-----+-----+-----+-----+-----+
   q3   | 0.2 | 0.6 | 2.3 | 0.5 | 0.1 |
        +-----+-----+-----+-----+-----+
   q4   | 0.9 | 0.1 | 0.4 | 1.7 | 0.6 |
        +-----+-----+-----+-----+-----+
   q5   | 0.1 | 0.3 | 0.2 | 0.5 | 2.0 |
        +-----+-----+-----+-----+-----+

Each row: one token's attention over the entire sequence
```

### Why Scale?

Dot products растут с размерностью dk. Если dk = 64, dot products могут быть десятками, загоняя softmax в области исчезающих gradients. Исправление: делить на sqrt(dk).

```
Scaled scores = (Q @ K^T) / sqrt(dk)
```

Так values остаются в диапазоне, где softmax дает полезные gradients.

### Softmax Turns Scores into Weights

Softmax превращает raw scores в probability distribution по каждой row:

```
Raw scores for q1:   [2.1, 0.3, 0.1, 0.8, 0.2]
                            |
                         softmax
                            |
Attention weights:   [0.52, 0.09, 0.07, 0.14, 0.08]   (sums to ~1.0)
```

Теперь у каждого token есть набор weights, показывающий, насколько сильно attend к каждому другому token.

### Weighted Sum of Values

Итоговый output для каждого token — weighted sum всех value vectors:

```
output_i = sum( attention_weight[i][j] * v_j  for all j )

For token 1:
  output_1 = 0.52 * v1 + 0.09 * v2 + 0.07 * v3 + 0.14 * v4 + 0.08 * v5
```

### Full Pipeline

```
                    +-------+
  X (input)  ----->|  @ Wq  |-----> Q
                    +-------+
                    +-------+
  X (input)  ----->|  @ Wk  |-----> K
                    +-------+                     +----------+
                    +-------+                     |          |
  X (input)  ----->|  @ Wv  |-----> V ---------->| weighted |----> output
                    +-------+          ^          |   sum    |
                                       |          +----------+
                              +--------+--------+
                              |    softmax      |
                              +---------+-------+
                                        ^
                              +---------+-------+
                              | Q @ K^T / sqrt  |
                              +-----------------+
```

Formula in one line:

```
Attention(Q, K, V) = softmax( Q @ K^T / sqrt(dk) ) @ V
```

## Build It

### Step 1: Softmax from scratch

Softmax converts raw logits into probabilities. Subtract the max for numerical stability.

```python
import numpy as np

def softmax(x):
    shifted = x - np.max(x, axis=-1, keepdims=True)
    exp_x = np.exp(shifted)
    return exp_x / np.sum(exp_x, axis=-1, keepdims=True)

logits = np.array([2.0, 1.0, 0.1])
print(f"logits:  {logits}")
print(f"softmax: {softmax(logits)}")
print(f"sum:     {softmax(logits).sum():.4f}")
```

### Step 2: Scaled dot-product attention

Core function. Принимает matrices Q, K, V и возвращает attention output плюс weight matrix.

```python
def scaled_dot_product_attention(Q, K, V):
    dk = Q.shape[-1]
    scores = Q @ K.T / np.sqrt(dk)
    weights = softmax(scores)
    output = weights @ V
    return output, weights
```

### Step 3: Self-attention class with learned projections

Полный self-attention module с weight matrices Wq, Wk, Wv, инициализированными Xavier-like scaling.

```python
class SelfAttention:
    def __init__(self, d_model, dk, dv, seed=42):
        rng = np.random.default_rng(seed)
        scale = np.sqrt(2.0 / (d_model + dk))
        self.Wq = rng.normal(0, scale, (d_model, dk))
        self.Wk = rng.normal(0, scale, (d_model, dk))
        scale_v = np.sqrt(2.0 / (d_model + dv))
        self.Wv = rng.normal(0, scale_v, (d_model, dv))
        self.dk = dk

    def forward(self, X):
        Q = X @ self.Wq
        K = X @ self.Wk
        V = X @ self.Wv
        output, weights = scaled_dot_product_attention(Q, K, V)
        return output, weights
```

### Step 4: Run it on a sentence

Создайте fake embeddings для sentence и посмотрите attention weights.

```python
sentence = ["The", "cat", "sat", "on", "the", "mat"]
n_tokens = len(sentence)
d_model = 8
dk = 4
dv = 4

rng = np.random.default_rng(42)
X = rng.normal(0, 1, (n_tokens, d_model))

attn = SelfAttention(d_model, dk, dv, seed=42)
output, weights = attn.forward(X)

print("Attention weights (each row: where that token looks):\n")
print(f"{'':>6}", end="")
for token in sentence:
    print(f"{token:>6}", end="")
print()

for i, token in enumerate(sentence):
    print(f"{token:>6}", end="")
    for j in range(n_tokens):
        w = weights[i][j]
        print(f"{w:6.3f}", end="")
    print()
```

### Step 5: Visualize attention with ASCII heatmap

Отобразите attention weights в символы для быстрого visual.

```python
def ascii_heatmap(weights, tokens, chars=" ░▒▓█"):
    n = len(tokens)
    print(f"\n{'':>6}", end="")
    for t in tokens:
        print(f"{t:>6}", end="")
    print()

    for i in range(n):
        print(f"{tokens[i]:>6}", end="")
        for j in range(n):
            level = int(weights[i][j] * (len(chars) - 1) / weights.max())
            level = min(level, len(chars) - 1)
            print(f"{'  ' + chars[level] + '   '}", end="")
        print()

ascii_heatmap(weights, sentence)
```

## Use It

PyTorch `nn.MultiheadAttention` делает то же, что мы построили, плюс multi-head splitting и output projection:

```python
import torch
import torch.nn as nn

d_model = 8
n_heads = 2
seq_len = 6

mha = nn.MultiheadAttention(embed_dim=d_model, num_heads=n_heads, batch_first=True)

X_torch = torch.randn(1, seq_len, d_model)

output, attn_weights = mha(X_torch, X_torch, X_torch)

print(f"Input shape:            {X_torch.shape}")
print(f"Output shape:           {output.shape}")
print(f"Attention weight shape: {attn_weights.shape}")
print(f"\nAttn weights (averaged over heads):")
print(attn_weights[0].detach().numpy().round(3))
```

Ключевое отличие: multi-head attention запускает несколько attention functions параллельно, каждую со своими Q, K, V projections размера dk = d_model / n_heads, затем concatenates results. Это позволяет модели одновременно attend к разным типам отношений.

## Ship It

Этот урок производит:
- `outputs/prompt-attention-explainer.md` - prompt для объяснения attention через database lookup analogy

## Упражнения

1. Измените `scaled_dot_product_attention`, чтобы она принимала optional mask matrix, задающую отдельные positions как negative infinity перед softmax (так работает causal/decoder masking)
2. Реализуйте multi-head attention с нуля: split Q, K, V на `n_heads` chunks, запустите attention на каждом, concatenate и project через final weight matrix Wo
3. Возьмите два разных sentences одинаковой длины, пропустите их через один и тот же SelfAttention instance и сравните attention patterns. Что меняется? Что остается тем же?

## Ключевые термины

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| Query (Q) | "The question vector" | Обучаемая projection input, представляющая, какую информацию ищет token |
| Key (K) | "The label vector" | Обучаемая projection, представляющая, какую информацию содержит token, matched against queries |
| Value (V) | "The content vector" | Обучаемая projection с фактической информацией, агрегируемой по attention scores |
| Scaled dot-product attention | "The attention formula" | softmax(QK^T / sqrt(dk)) @ V - scaling предотвращает softmax saturation в high dimensions |
| Self-attention | "The token looks at itself and others" | Attention, где Q, K, V приходят из одной sequence, позволяя каждой position attend к каждой другой |
| Attention weights | "How much focus" | Probability distribution over positions, produced by softmax over scaled dot products |
| Multi-head attention | "Parallel attention" | Запуск нескольких attention functions с разными projections и concatenation результатов для richer representations |

## Дополнительное чтение

- [Attention Is All You Need (Vaswani et al., 2017)](https://arxiv.org/abs/1706.03762) - оригинальная статья transformer.
- [The Illustrated Transformer (Jay Alammar)](https://jalammar.github.io/illustrated-transformer/) - лучшая визуальная walkthrough всей architecture.
- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) - построчная PyTorch implementation с explanations.
