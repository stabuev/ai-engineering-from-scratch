# Соберите Transformer с нуля — capstone

> Тринадцать уроков. Одна модель. Без shortcuts.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 01 through 13. Не пропускайте.
**Время:** ~120 минут

## Проблема

Вы прочитали каждую статью. Вы реализовали attention, multi-head splits, positional encodings, encoder and decoder blocks, BERT and GPT losses, MoE, KV cache. Теперь заставьте их работать вместе на реальной задаче.

Capstone: обучить небольшой decoder-only transformer end-to-end на character-level language modeling task. Он читает Shakespeare. Он генерирует нового Shakespeare. Он достаточно мал, чтобы обучиться на laptop меньше чем за 10 минут. Он достаточно корректен, чтобы замена dataset на больший и longer training дали реальный LM.

Это "nanoGPT" курса. Он не оригинален — tutorial nanoGPT Karpathy 2023 года является reference implementation, который каждый студент пишет хотя бы один раз. Мы берем форму и перестраиваем ее вокруг того, что уже прошли.

## Концепция

![Transformer-from-scratch block diagram](../assets/capstone.svg)

Архитектура с аннотациями:

```
input tokens (B, N)
   │
   ▼
token embedding + positional embedding  ◀── Lesson 04 (RoPE option)
   │
   ▼
┌──── block × L ────────────────────┐
│  RMSNorm                          │  ◀── Lesson 05
│  MultiHeadAttention (causal)      │  ◀── Lesson 03 + 07 (causal mask)
│  residual                         │
│  RMSNorm                          │
│  SwiGLU FFN                       │  ◀── Lesson 05
│  residual                         │
└────────────────────────────────── ┘
   │
   ▼
final RMSNorm
   │
   ▼
lm_head (tied to token embedding)
   │
   ▼
logits (B, N, V)
   │
   ▼
shift-by-one cross-entropy            ◀── Lesson 07
```

### Что мы поставляем

- `GPTConfig` — одно место для настройки всех hyperparameters.
- `MultiHeadAttention` — causal, batched, с optional Flash-style pathway (PyTorch `scaled_dot_product_attention`).
- `SwiGLUFFN` — современный FFN.
- `Block` — pre-norm, residual-wrapped attention + FFN.
- `GPT` — embeddings, stacked blocks, LM head, generate().
- Training loop с AdamW, cosine LR, gradient clipping.
- Char-level tokenizer на тексте Shakespeare.

### Что мы не поставляем

- RoPE — реализован концептуально в Lesson 04. Здесь для простоты используем learned positional embeddings. В упражнениях вы замените их на RoPE.
- KV cache during generation — каждый generation step пересчитывает attention по всему prefix. Медленнее, но проще. В упражнениях вы добавите KV cache.
- Flash Attention — PyTorch 2.0+ auto-dispatches, если inputs подходят; мы используем `F.scaled_dot_product_attention`.
- MoE — один FFN на block. Вы видели MoE в Lesson 11.

### Target metrics

На laptop Mac M2, 4-layer, 4-head, d_model=128 GPT, обученный 2 000 steps на `tinyshakespeare.txt`:

- Training loss сходится от ~4.2 (random) до ~1.5 примерно за 6 минут.
- Sampled output выглядит Shakespeare-shaped: появляются архаичные слова, line breaks, proper names вроде "ROMEO:".
- Val loss (held-out final 10% текста) идет рядом с training loss; overfitting при таком size/budget нет.

## Соберите это

Этот урок использует PyTorch. Установите `torch` (CPU build подходит). См. `code/main.py`. Скрипт делает:

- Downloading `tinyshakespeare.txt`, если его нет (или чтение local copy).
- Byte-level char tokenizer.
- Train/val split at 90/10.
- Training loop с bf16 autocast на supported hardware.
- Sampling после завершения training.

### Шаг 1: data

```python
text = open("tinyshakespeare.txt").read()
chars = sorted(set(text))
stoi = {c: i for i, c in enumerate(chars)}
itos = {i: c for c, i in stoi.items()}
encode = lambda s: [stoi[c] for c in s]
decode = lambda xs: "".join(itos[x] for x in xs)
```

65 unique characters. Tiny vocabulary. Помещается в 4-byte vocab_size. Без BPE, без tokenizer drama.

### Шаг 2: model

См. `code/main.py`. Block — textbook из Lesson 05: pre-norm, RMSNorm, SwiGLU, causal MHA. Parameter count для 4/4/128: ~800K.

### Шаг 3: training loop

Берем random batch из length-256 token windows. Forward. Shift-by-one cross-entropy. Backward. AdamW step. Log. Repeat.

```python
for step in range(max_steps):
    x, y = get_batch("train")
    logits = model(x)
    loss = F.cross_entropy(logits.view(-1, vocab_size), y.view(-1))
    loss.backward()
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    opt.step()
    opt.zero_grad()
```

### Шаг 4: sample

По prompt повторно делаем forward, sample from top-p logits, append и продолжаем. Остановиться после 500 tokens.

### Шаг 5: прочитать output

После 2 000 steps:

```
ROMEO:
Away and mild will not thy friend, that thou shalt wit:
The chief that well shame and hath been his friends,
...
```

Не Shakespeare. Но Shakespeare-shaped. Явная победа для ~800K parameters и 6 minutes на laptop.

## Используйте это

Этот capstone — reference architecture. Три расширения, чтобы довести его до чего-то реального:

1. **Замените tokenizer.** Используйте BPE (например, `tiktoken.get_encoding("cl100k_base")`). Vocab size прыгает с 65 до ~50 000. Model capacity нужно масштабировать, чтобы компенсировать.
2. **Обучите на большем corpus.** Используйте `OpenWebText` или `fineweb-edu` (HuggingFace). 10B tokens на одном A100 занимает ~24 hours для 125M-param GPT.
3. **Добавьте RoPE + KV cache + Flash Attention.** Упражнения ниже проведут вас через каждое.

В итоге получается 125M-parameter GPT, который генерирует fluent English. Не frontier model. Но тот же code path — просто больше — используют Karpathy, EleutherAI и Allen Institute для обучения research checkpoints в 2026 году.

## Доведите до поставки

См. `outputs/skill-transformer-review.md`. Skill проверяет transformer-from-scratch implementation на корректность по всем 13 предыдущим lessons.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Проверьте, что final-step validation loss обученной модели ниже 2.0. Измените `max_steps` с 2 000 до 5 000 — продолжает ли val loss улучшаться?
2. **Средне.** Замените learned positional embeddings на RoPE. Примените rotation к Q и K внутри `MultiHeadAttention`. Обучите и проверьте, что val loss как минимум не выше.
3. **Средне.** Реализуйте KV cache в sampling loop. Сгенерируйте 500 tokens с cache и без. Wall-clock должен улучшиться в 5–20× на laptop.
4. **Сложно.** Добавьте вторую head к модели, которая предсказывает next-plus-one token (MTP — Multi-Token Prediction from DeepSeek-V3). Обучайте jointly. Помогает ли это?
5. **Сложно.** Замените один FFN на block на 4-expert MoE. Router + top-2 routing. Посмотрите, как меняется val loss при matched active parameters.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| nanoGPT | "Karpathy's tutorial repo" | Минимальный decoder-only transformer training code, ~300 LOC; canonical reference. |
| tinyshakespeare | "The standard toy corpus" | ~1.1 MB текста; каждый character-LM tutorial с 2015 года использует его. |
| Tied embeddings | "Share input/output matrix" | LM head weight = transpose of token embedding matrix; экономит parameters, улучшает quality. |
| bf16 autocast | "Training precision trick" | Run forward/back in bf16, держать optimizer state in fp32; standard since 2021. |
| Gradient clipping | "Stops spikes" | Ограничить global grad norm at 1.0; предотвращает training blowups. |
| Cosine LR schedule | "The 2020+ default" | LR ramps up linearly (warmup), затем decays cosine-shaped до 10% peak. |
| MFU | "Model FLOP Utilization" | Achieved FLOPs / theoretical peak; 40% dense, 30% MoE — сильный результат в 2026. |
| Val loss | "Held-out loss" | Cross-entropy на данных, которые модель не видела; overfit detector. |

## Дополнительное чтение

- [The Annotated Transformer (Harvard NLP)](https://nlp.seas.harvard.edu/annotated-transformer/) — классическая annotated implementation.
