# Модели Sequence-to-Sequence

> Две RNN притворяются переводчиком. Bottleneck, в который они упираются, - причина существования attention.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Phase 5 · 08 (CNNs + RNNs for Text), Phase 3 · 11 (PyTorch Intro)
**Время:** ~75 минут

## Проблема

Classification отображает sequence переменной длины в одну label. Translation отображает sequence переменной длины в другую sequence переменной длины. Input и output живут в разных vocabularies, возможно на разных языках, без гарантии равенства длин.

Архитектура seq2seq (Sutskever, Vinyals, Le, 2014) решила это намеренно простым рецептом. Две RNN. Одна читает source sentence и создает fixed-size context vector. Другая читает этот vector и генерирует target sentence token by token. Тот же код, который вы писали для урока 08, просто склеенный иначе.

Это стоит изучить по двум причинам. Во-первых, context-vector bottleneck - самый педагогически полезный провал в NLP. Он мотивирует все, в чем хороши attention и transformers. Во-вторых, training recipe (teacher forcing, scheduled sampling, beam search at inference) все еще применим к каждой современной generation system, включая LLMs.

## Концепция

![Encoder-decoder с узким местом в виде вектора контекста](./assets/seq2seq.svg)

**Encoder.** RNN, читающая source sentence. Ее final hidden state - **context vector**, fixed-size summary всего input. Предположительно, ничего не теряет, кроме source.

**Decoder.** Другая RNN, инициализированная context vector. На каждом шаге она принимает ранее сгенерированный token как input и создает distribution по target vocabulary. Sample или argmax выбирает следующий token. Он подается обратно. Повторять, пока не будет создан token `<EOS>` или не достигнут max length.

**Training:** Cross-entropy loss на каждом decoder step, суммируется по sequence. Стандартный backprop through time через обе сети.

**Teacher forcing.** Во время training input decoder на step `t` - это *ground-truth* token в позиции `t-1`, а не собственное предыдущее предсказание decoder. Это стабилизирует обучение; без этого ранние ошибки каскадируют, и модель не учится. Во время inference приходится использовать собственные predictions модели, поэтому всегда есть train/inference distribution gap. Этот разрыв называется **exposure bias**.

**The bottleneck.** Все, что encoder узнал о source, должно быть сжато в один context vector. Длинные предложения теряют детали. Редкие слова размываются. Reordering (chat noir vs. black cat) приходится запоминать, а не вычислять.

Attention (урок 10) исправляет это, позволяя decoder смотреть на *каждый* encoder hidden state, а не только на последний. В этом весь pitch.

## Сборка

### Шаг 1: encoder

```python
import torch
import torch.nn as nn


class Encoder(nn.Module):
    def __init__(self, src_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(src_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)

    def forward(self, src):
        e = self.embed(src)
        outputs, hidden = self.gru(e)
        return outputs, hidden
```

`outputs` имеет shape `[batch, seq_len, hidden_dim]` - один hidden state на каждую input position. `hidden` имеет shape `[1, batch, hidden_dim]` - final step. Урок 08 говорил "pool over outputs for classification." Здесь мы сохраняем last hidden state как context vector и игнорируем per-step outputs.

### Шаг 2: decoder

```python
class Decoder(nn.Module):
    def __init__(self, tgt_vocab_size, embed_dim, hidden_dim):
        super().__init__()
        self.embed = nn.Embedding(tgt_vocab_size, embed_dim, padding_idx=0)
        self.gru = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, tgt_vocab_size)

    def forward(self, token, hidden):
        e = self.embed(token)
        out, hidden = self.gru(e, hidden)
        logits = self.fc(out)
        return logits, hidden
```

Decoder вызывается по одному step за раз. Input: batch одиночных tokens и текущий hidden state. Output: vocabulary logits для следующего token и updated hidden state.

### Шаг 3: training loop with teacher forcing

```python
def train_batch(encoder, decoder, src, tgt, bos_id, optimizer, teacher_forcing_ratio=0.9):
    optimizer.zero_grad()
    _, hidden = encoder(src)
    batch_size, tgt_len = tgt.shape
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    loss = 0.0
    loss_fn = nn.CrossEntropyLoss(ignore_index=0)

    for t in range(tgt_len):
        logits, hidden = decoder(input_token, hidden)
        step_loss = loss_fn(logits.squeeze(1), tgt[:, t])
        loss += step_loss
        use_teacher = torch.rand(1).item() < teacher_forcing_ratio
        if use_teacher:
            input_token = tgt[:, t].unsqueeze(1)
        else:
            input_token = logits.argmax(dim=-1)

    loss.backward()
    optimizer.step()
    return loss.item() / tgt_len
```

Два knobs, которые стоит назвать. `ignore_index=0` пропускает loss на padding tokens. `teacher_forcing_ratio` - вероятность использовать true token вместо prediction модели на каждом step. Начинайте с 1.0 (full teacher forcing) и anneal down до ~0.5 по ходу training, чтобы сократить exposure-bias gap.

### Шаг 4: inference loop (greedy)

```python
@torch.no_grad()
def greedy_decode(encoder, decoder, src, bos_id, eos_id, max_len=50):
    _, hidden = encoder(src)
    batch_size = src.shape[0]
    input_token = torch.full((batch_size, 1), bos_id, dtype=torch.long)
    output_ids = []
    for _ in range(max_len):
        logits, hidden = decoder(input_token, hidden)
        next_token = logits.argmax(dim=-1)
        output_ids.append(next_token)
        input_token = next_token
        if (next_token == eos_id).all():
            break
    return torch.cat(output_ids, dim=1)
```

Greedy decoding выбирает самый вероятный token на каждом step. Он может увести в сторону: как только вы committed to a token, отменить его нельзя. **Beam search** держит top-`k` partial sequences живыми и в конце выбирает highest-scoring complete one. Beam width 3-5 - стандарт.

### Шаг 5: bottleneck, demonstrated

Обучите модель на toy copy task: source `[a, b, c, d, e]`, target `[a, b, c, d, e]`. Увеличивайте sequence length. Наблюдайте accuracy.

```
seq_len=5   copy accuracy: 98%
seq_len=10  copy accuracy: 91%
seq_len=20  copy accuracy: 62%
seq_len=40  copy accuracy: 23%
```

Один GRU hidden state не может без потерь запомнить 40-token input. Информация есть на каждом encoder step, но decoder видит только last state. Attention исправляет это напрямую.

## Использование

PyTorch имеет `nn.Transformer` и `nn.LSTM`-based seq2seq templates. Библиотека Hugging Face `transformers` поставляет полные encoder-decoder models (BART, T5, mBART, NLLB), обученные на миллиардах tokens.

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

tok = AutoTokenizer.from_pretrained("facebook/bart-base")
model = AutoModelForSeq2SeqLM.from_pretrained("facebook/bart-base")

src = tok("Translate this to French: Hello, how are you?", return_tensors="pt")
out = model.generate(**src, max_new_tokens=50, num_beams=4)
print(tok.decode(out[0], skip_special_tokens=True))
```

Современные encoder-decoders заменили RNNs на transformers. Высокоуровневая форма (encoder, decoder, generate-token-by-token) идентична seq2seq paper 2014 года. Механизм внутри каждого block другой.

### Когда все еще стоит брать RNN-based seq2seq

Почти никогда для новых проектов. Конкретные исключения:

- Streaming translation, где input потребляется по одному token с bounded memory.
- On-device text generation, где memory cost transformer слишком высок.
- Pedagogy. Понимание encoder-decoder bottleneck - самый быстрый путь к пониманию, почему transformers победили.

### Exposure bias и способы его смягчения

- **Scheduled sampling.** Anneal teacher forcing ratio во время training, чтобы модель училась восстанавливаться после собственных ошибок.
- **Minimum risk training.** Обучайте на sentence-level BLEU score вместо token-level cross-entropy. Ближе к тому, что вам действительно нужно.
- **Reinforcement learning fine-tuning.** Reward sequence generator через metric. Используется в современных LLM RLHF.

Все три по-прежнему применимы к transformer-based generation.

## Доставка

Сохраните как `outputs/prompt-seq2seq-design.md`:

```markdown
---
name: seq2seq-design
description: Design a sequence-to-sequence pipeline for a given task.
phase: 5
lesson: 09
---

Given a task (translation, summarization, paraphrase, question rewrite), output:

1. Architecture. Pretrained transformer encoder-decoder (BART, T5, mBART, NLLB) is the default. RNN-based seq2seq only for specific constraints.
2. Starting checkpoint. Name it (`facebook/bart-base`, `google/flan-t5-base`, `facebook/nllb-200-distilled-600M`). Match the checkpoint to task and language coverage.
3. Decoding strategy. Greedy for deterministic output, beam search (width 4-5) for quality, sampling with temperature for diversity. One sentence justification.
4. One failure mode to verify before shipping. Exposure bias manifests as generation drift on longer outputs; sample 20 outputs at the 90th-percentile length and eyeball.

Refuse to recommend training a seq2seq from scratch for under a million parallel examples. Flag any pipeline that uses greedy decoding for user-facing content as fragile (greedy repeats and loops).
```

## Упражнения

1. **Легко.** Реализуйте toy copy task. Обучите GRU seq2seq на input-output pairs, где target равен source. Измерьте accuracy на длинах 5, 10, 20. Воспроизведите bottleneck.
2. **Средне.** Добавьте beam search decoding с beam width 3. Измерьте BLEU на небольшом parallel corpus против greedy. Задокументируйте, где beam search выигрывает (обычно last tokens) и где не дает разницы.
3. **Сложно.** Fine-tune `facebook/bart-base` на 10k-pair paraphrase dataset. Сравните beam-4 output fine-tuned model с output base model на held-out inputs. Сообщите BLEU и выберите 10 qualitative examples.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| Encoder | Input RNN | Читает source. Создает per-step hidden states и final context vector. |
| Decoder | Output RNN | Инициализируется context vector. Генерирует target tokens по одному. |
| Context vector | Summary | Final encoder hidden state. Fixed size. Bottleneck, который решает attention. |
| Teacher forcing | Использовать true tokens | Подавать ground-truth previous token во время training. Стабилизирует learning. |
| Exposure bias | Train/test gap | Модель, обученная на true tokens, не тренировалась восстанавливаться после собственных mistakes. |
| Beam search | Лучшее decoding | Держит top-k partial sequences живыми на каждом step вместо greedy commitment. |

## Дополнительное чтение

- [Sutskever, Vinyals, Le (2014). Sequence to Sequence Learning with Neural Networks](https://arxiv.org/abs/1409.3215) — оригинальная seq2seq paper. Четыре страницы.
- [Cho et al. (2014). Learning Phrase Representations using RNN Encoder-Decoder for Statistical Machine Translation](https://arxiv.org/abs/1406.1078) — ввела GRU и encoder-decoder framing.
- [Bahdanau, Cho, Bengio (2014). Neural Machine Translation by Jointly Learning to Align and Translate](https://arxiv.org/abs/1409.0473) — статья об attention. Читайте сразу после этого урока.
- [PyTorch NLP from Scratch tutorial](https://pytorch.org/tutorials/intermediate/seq2seq_translation_tutorial.html) — готовый к запуску код seq2seq + attention.
