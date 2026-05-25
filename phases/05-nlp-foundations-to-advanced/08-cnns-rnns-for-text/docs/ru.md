# CNN и RNN для текста

> Convolutions учат n-grams. Recurrences запоминают. Оба подхода вытеснены attention. Оба все еще важны на ограниченном hardware.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Phase 3 · 11 (PyTorch Intro), Phase 5 · 03 (Word Embeddings), Phase 4 · 02 (Convolutions from Scratch)
**Время:** ~75 минут

## Проблема

TF-IDF и Word2Vec создавали плоские векторы, игнорирующие порядок слов. Классификатор на их основе не мог отличить `dog bites man` от `man bites dog`. Порядок слов иногда несет сигнал.

Два семейства архитектур закрыли этот пробел до появления transformers.

**Convolutional nets for text (TextCNN).** Применяют 1D convolutions к последовательностям word embeddings. Фильтр ширины 3 - learnable trigram detector: он охватывает три слова и выдает score. Набор разных ширин (2, 3, 4, 5) обнаруживает multi-scale patterns. Max-pool дает fixed-size representation. Плоско, параллельно, быстро.

**Recurrent nets (RNN, LSTM, GRU).** Обрабатывают токены по одному, поддерживая hidden state, который переносит информацию вперед. Последовательные, с памятью, гибкие по длине входа. Доминировали в sequence modeling с 2014 по 2017 год, затем появился attention.

Этот урок строит оба подхода, а затем называет сбой, который мотивировал attention.

## Концепция

![Фильтры TextCNN и разворачивание скрытого состояния RNN](./assets/cnn-rnn.svg)

**TextCNN** (Kim, 2014). Токены эмбеддятся. 1D convolution ширины `k` скользит фильтром по последовательным `k`-grams embeddings, создавая feature map. Global max-pooling по этой карте выбирает сильнейшую активацию. Max-pooled outputs от нескольких ширин фильтра конкатенируются. Затем идут в classifier head.

Почему это работает. Фильтр - learnable n-gram. Max-pooling position-invariant, поэтому "not good" активирует тот же feature в начале или середине review. Три ширины фильтра по 100 фильтров каждая дают 300 learned n-gram detectors. Обучение параллельное; последовательной зависимости нет.

**RNN.** На каждом time step `t` hidden state `h_t = f(W * x_t + U * h_{t-1} + b)`. `W`, `U`, `b` общие во времени. Hidden state на time `T` - summary всего prefix. Для classification применяют pooling по `h_1 ... h_T` (max, mean или last).

Plain RNNs страдают от vanishing gradients. **LSTM** добавляет gates, решающие, что забыть, что сохранить и что вывести, стабилизируя gradients через длинные последовательности. **GRU** упрощает LSTM до двух gates; работает сопоставимо с меньшим числом параметров.

**Bidirectional RNNs** запускают одну RNN вперед и другую назад, конкатенируя hidden states. Представление каждого токена видит левый и правый контекст. Это критично для tagging tasks.

## Сборка

### Шаг 1: TextCNN в PyTorch

```python
import torch
import torch.nn as nn
import torch.nn.functional as F


class TextCNN(nn.Module):
    def __init__(self, vocab_size, embed_dim, n_classes, filter_widths=(2, 3, 4), n_filters=64, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.convs = nn.ModuleList([
            nn.Conv1d(embed_dim, n_filters, kernel_size=k)
            for k in filter_widths
        ])
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids).transpose(1, 2)
        pooled = []
        for conv in self.convs:
            c = F.relu(conv(x))
            p = F.max_pool1d(c, c.size(2)).squeeze(2)
            pooled.append(p)
        h = torch.cat(pooled, dim=1)
        return self.fc(self.dropout(h))
```

`transpose(1, 2)` меняет форму `[batch, seq_len, embed_dim]` на `[batch, embed_dim, seq_len]`, потому что `nn.Conv1d` трактует среднюю ось как channels. Pooled output имеет fixed size независимо от длины входа.

### Шаг 2: LSTM classifier

```python
class LSTMClassifier(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, n_classes, bidirectional=True, dropout=0.3):
        super().__init__()
        self.embed = nn.Embedding(vocab_size, embed_dim, padding_idx=0)
        self.lstm = nn.LSTM(embed_dim, hidden_dim, batch_first=True, bidirectional=bidirectional)
        factor = 2 if bidirectional else 1
        self.dropout = nn.Dropout(dropout)
        self.fc = nn.Linear(hidden_dim * factor, n_classes)

    def forward(self, token_ids):
        x = self.embed(token_ids)
        out, _ = self.lstm(x)
        pooled = out.max(dim=1).values
        return self.fc(self.dropout(pooled))
```

Max-pool по sequence, не last-state pool. Для classification max-pooling обычно лучше взятия последнего hidden state, потому что информация в конце длинной последовательности склонна доминировать в last state.

### Шаг 3: демонстрация vanishing gradient (интуиция)

Plain RNN без gating не может выучить long-range dependencies. Рассмотрим toy task: предсказать, встречался ли token `A` где-либо в последовательности. Если `A` стоит в позиции 1, а последовательность имеет длину 100, gradient от loss должен пройти назад через 99 умножений recurrent weight. Если weight меньше 1, gradient исчезает. Если больше 1, взрывается.

```python
def vanishing_gradient_sim(seq_len, recurrent_weight=0.9):
    import math
    return math.pow(recurrent_weight, seq_len)


# At weight=0.9 over 100 steps:
#   0.9 ^ 100 ≈ 2.7e-5
# The gradient from step 100 to step 1 is effectively zero.
```

LSTMs исправляют это с помощью **cell state**, который проходит через сеть только с аддитивными взаимодействиями (forget gate масштабирует его мультипликативно, но gradients все равно текут по "highway"). GRUs делают похожее с меньшим числом параметров. Оба дают стабильное обучение на последовательностях 100+ steps.

### Шаг 4: почему этого все равно было недостаточно

Даже с LSTMs сохранялись три проблемы.

1. **Sequential bottleneck.** Обучение RNN на sequence длины 1000 требует 1000 последовательных forward/backward steps. Нельзя распараллелить по времени.
2. **Fixed-size context vector in encoder-decoder setups.** Decoder видит только final hidden state encoder, сжатый по всему input. Длинные входы теряют детали. Урок 09 разбирает это напрямую.
3. **Distant-dependency accuracy ceiling.** LSTMs лучше plain RNNs, но все еще с трудом передают конкретную информацию через 200+ steps.

Attention решил все три. Transformers полностью отказались от recurrence. Урок 10 - поворотная точка.

## Использование

`nn.LSTM`, `nn.GRU` и `nn.Conv1d` в PyTorch готовы для production. Training code стандартный.

Hugging Face поставляет pretrained embeddings, которые можно подключить как input layer:

```python
from transformers import AutoModel

encoder = AutoModel.from_pretrained("bert-base-uncased")
for param in encoder.parameters():
    param.requires_grad = False


class BertCNN(nn.Module):
    def __init__(self, n_classes, filter_widths=(2, 3, 4), n_filters=64):
        super().__init__()
        self.encoder = encoder
        self.convs = nn.ModuleList([nn.Conv1d(768, n_filters, kernel_size=k) for k in filter_widths])
        self.fc = nn.Linear(n_filters * len(filter_widths), n_classes)

    def forward(self, input_ids, attention_mask):
        with torch.no_grad():
            out = self.encoder(input_ids=input_ids, attention_mask=attention_mask).last_hidden_state
        x = out.transpose(1, 2)
        pooled = [F.max_pool1d(F.relu(conv(x)), kernel_size=conv(x).size(2)).squeeze(2) for conv in self.convs]
        return self.fc(torch.cat(pooled, dim=1))
```

Checklist для случаев, где это подходит под constraints.

- **Edge / on-device inference.** TextCNN с GloVe embeddings в 10-100x меньше transformer. Если deploy target - телефон, это подходящий stack.
- **Streaming / online classification.** RNN обрабатывает по одному токену; transformers требуют полную sequence. Для real-time incoming text LSTMs все еще выигрывают.
- **Tiny models for baselines.** Быстрая итерация на новой задаче. TextCNN можно обучить за 5 минут на CPU.
- **Sequence labeling with limited data.** BiLSTM-CRF (урок 06) все еще production-grade NER architecture для 1k-10k размеченных предложений.

Все остальное идет к transformer.

## Доставка

Сохраните как `outputs/prompt-text-encoder-picker.md`:

```markdown
---
name: text-encoder-picker
description: Pick a text encoder architecture for a given constraint set.
phase: 5
lesson: 08
---

Given constraints (task, data volume, latency budget, deploy target, compute budget), output:

1. Encoder architecture: TextCNN, BiLSTM, BiLSTM-CRF, transformer fine-tune, or "use a pretrained transformer as a frozen encoder + small head".
2. Embedding input: random init, GloVe / fastText frozen, or contextualized transformer embeddings.
3. Training recipe in 5 lines: optimizer, learning rate, batch size, epochs, regularization.
4. One monitoring signal. For RNN/CNN models: attention mechanism absence means they miss long-range deps; check per-length accuracy. For transformers: fine-tuning collapse if LR too high; check train loss.

Refuse to recommend fine-tuning a transformer when data is under ~500 labeled examples without showing that a TextCNN / BiLSTM baseline has plateaued. Flag edge deployment as needing architecture-before-everything.
```

## Упражнения

1. **Легко.** Обучите TextCNN на toy dataset с 3 классами (данные придумайте сами). Проверьте, что filter widths (2, 3, 4) превосходят single width (3) по average F1.
2. **Средне.** Реализуйте max-pool, mean-pool и last-state pooling для LSTM classifier. Сравните на небольшом dataset; задокументируйте, какой pooling выигрывает, и предположите почему.
3. **Сложно.** Постройте BiLSTM-CRF NER tagger (объедините урок 06 и этот урок). Обучите на CoNLL-2003. Сравните с CRF-alone baseline из урока 06 и с BERT fine-tune. Сообщите training time, memory и F1.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| TextCNN | CNN для текста | Стек 1D convolutions поверх word embeddings с global max-pool. Kim (2014). |
| RNN | Recurrent net | Hidden state обновляется на каждом time step: `h_t = f(W x_t + U h_{t-1})`. |
| LSTM | Gated RNN | Добавляет input / forget / output gates + cell state. Стабильно обучается через длинные sequences. |
| GRU | Более простой LSTM | Два gates вместо трех. Схожая accuracy, меньше параметров. |
| Bidirectional | Оба направления | Forward + backward RNN конкатенируются. Каждый token видит обе стороны своего context. |
| Vanishing gradient | Training signal исчезает | Повторное умножение на weights <1 в plain RNNs делает gradients ранних steps фактически нулевыми. |

## Дополнительное чтение

- [Kim, Y. (2014). Convolutional Neural Networks for Sentence Classification](https://arxiv.org/abs/1408.5882) — статья о TextCNN. Восемь страниц. Читается легко.
- [Hochreiter, S. and Schmidhuber, J. (1997). Long Short-Term Memory](https://www.bioinf.jku.at/publications/older/2604.pdf) — статья о LSTM. Неожиданно ясная.
- [Olah, C. (2015). Understanding LSTM Networks](https://colah.github.io/posts/2015-08-Understanding-LSTMs/) — диаграммы, которые сделали LSTM понятными для всех.
