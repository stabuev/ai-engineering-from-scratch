# Word Embeddings — Word2Vec с нуля

> Слово определяется окружением, в котором оно встречается. Обучите неглубокую сеть на этой идее, и геометрия появится сама.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Фаза 5 · 02 (BoW + TF-IDF), Фаза 3 · 03 (Backpropagation from Scratch)
**Время:** ~75 минут

## Проблема

TF-IDF знает, что `dog` и `puppy` — разные слова. Он не знает, что они означают почти одно и то же. Классификатор, обученный на `dog`, не сможет обобщить сигнал на отзыв про `puppy`. Это можно замаскировать списками синонимов, но такой подход ломается на редких терминах, domain jargon и любом языке, который вы не предусмотрели.

Нужно представление, где `dog` и `puppy` оказываются рядом в пространстве. Где `king - man + woman` попадает около `queen`. Где модель, обученная на `dog`, бесплатно переносит часть сигнала на `puppy`.

Word2Vec дал нам это пространство. Двухслойная neural network, training runs на триллионах токенов, публикация в 2013 году. Архитектура почти неловко проста. Результаты изменили NLP на десятилетие.

## Концепция

![Skip-gram window and embedding space](./assets/word2vec.svg)

**Distributional hypothesis** (Firth, 1957): "You shall know a word by the company it keeps." Если два слова встречаются в похожих контекстах, они, вероятно, имеют похожий смысл.

Word2Vec существует в двух вариантах, оба используют эту идею.

- **Skip-gram.** По center word предсказывать surrounding words. `cat -> (the, sat, on)` с window size 2.
- **CBOW (continuous bag of words).** По surrounding words предсказывать center. `(the, sat, on) -> cat`.

Skip-gram обучается медленнее, но лучше работает с редкими словами. Он стал default.

Сеть имеет один hidden layer без нелинейности. Вход — one-hot vector по словарю. Выход — softmax по словарю. После обучения output layer выбрасывают. Веса hidden layer и есть embeddings.

```
one-hot(center) ── W ──▶ hidden (d-dim) ── W' ──▶ softmax(vocab)
                          ^
                          this is the embedding
```

Трюк: softmax по 100k словам запретительно дорог. Word2Vec использует **negative sampling**, чтобы превратить задачу в binary classification. Предсказать "появлялось ли это context word рядом с этим center word, да или нет". Вместо вычисления softmax по всему словарю для каждой training pair выбирается несколько negative (non-co-occurring) words.

## Построение

### Шаг 1: training pairs из корпуса

```python
def skipgram_pairs(docs, window=2):
    pairs = []
    for doc in docs:
        for i, center in enumerate(doc):
            for j in range(max(0, i - window), min(len(doc), i + window + 1)):
                if i == j:
                    continue
                pairs.append((center, doc[j]))
    return pairs
```

```python
>>> skipgram_pairs([["the", "cat", "sat", "on", "mat"]], window=2)
[('the', 'cat'), ('the', 'sat'),
 ('cat', 'the'), ('cat', 'sat'), ('cat', 'on'),
 ('sat', 'the'), ('sat', 'cat'), ('sat', 'on'), ('sat', 'mat'),
 ...]
```

Каждая пара (center, context) в окне — positive training example.

### Шаг 2: embedding tables

Две матрицы. `W` — embedding table для center-word (та, которую оставляют). `W'` — table для context-word (часто отбрасывают, иногда усредняют с `W`).

```python
import numpy as np


def init_embeddings(vocab_size, dim, seed=0):
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(vocab_size, dim))
    W_prime = rng.normal(0, 0.1, size=(vocab_size, dim))
    return W, W_prime
```

Небольшая случайная инициализация. Vocab size 10k и dim 100 — реалистично; для обучения достаточно 50 vocab x 16 dim, чтобы увидеть геометрию.

### Шаг 3: negative sampling objective

Для каждой positive pair `(center, context)` выберите `k` случайных слов из словаря как negatives. Обучайте модель так, чтобы dot product `W[center] · W'[context]` был высоким для positives и низким для negatives.

```python
def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_pair(W, W_prime, center_idx, context_idx, negative_indices, lr):
    v_c = W[center_idx]
    u_pos = W_prime[context_idx]
    u_negs = W_prime[negative_indices]

    pos_score = sigmoid(v_c @ u_pos)
    neg_scores = sigmoid(u_negs @ v_c)

    grad_center = (pos_score - 1) * u_pos
    for i, u in enumerate(u_negs):
        grad_center += neg_scores[i] * u

    W[context_idx] = W[context_idx]
    W_prime[context_idx] -= lr * (pos_score - 1) * v_c
    for i, neg_idx in enumerate(negative_indices):
        W_prime[neg_idx] -= lr * neg_scores[i] * v_c
    W[center_idx] -= lr * grad_center
```

Магическая формула: logistic loss на positive pair (хотим sigmoid около 1) плюс logistic loss на negative pairs (хотим sigmoid около 0). Gradients проходят в обе таблицы. Полный derivation есть в оригинальной статье; один раз пройдите его карандашом на бумаге, если хотите, чтобы он закрепился.

### Шаг 4: обучение на toy corpus

```python
def train(docs, dim=16, window=2, k_neg=5, epochs=100, lr=0.05, seed=0):
    vocab = build_vocab(docs)
    vocab_size = len(vocab)
    rng = np.random.default_rng(seed)
    W, W_prime = init_embeddings(vocab_size, dim, seed=seed)
    pairs = skipgram_pairs(docs, window=window)

    for epoch in range(epochs):
        rng.shuffle(pairs)
        for center, context in pairs:
            c_idx = vocab[center]
            ctx_idx = vocab[context]
            negs = rng.integers(0, vocab_size, size=k_neg)
            negs = [n for n in negs if n != ctx_idx and n != c_idx]
            train_pair(W, W_prime, c_idx, ctx_idx, negs, lr)
    return vocab, W
```

После достаточного числа epochs на большом корпусе слова с общими контекстами имеют похожие center embeddings. На toy corpus эффект виден слабо. На миллиардах токенов — очень ясно.

### Шаг 5: analogy trick

```python
def nearest(vocab, W, target_vec, topk=5, exclude=None):
    exclude = exclude or set()
    inv_vocab = {i: w for w, i in vocab.items()}
    norms = np.linalg.norm(W, axis=1, keepdims=True) + 1e-9
    W_norm = W / norms
    target = target_vec / (np.linalg.norm(target_vec) + 1e-9)
    sims = W_norm @ target
    order = np.argsort(-sims)
    out = []
    for i in order:
        if i in exclude:
            continue
        out.append((inv_vocab[i], float(sims[i])))
        if len(out) == topk:
            break
    return out


def analogy(vocab, W, a, b, c, topk=5):
    v = W[vocab[b]] - W[vocab[a]] + W[vocab[c]]
    return nearest(vocab, W, v, topk=topk, exclude={vocab[a], vocab[b], vocab[c]})
```

На pre-trained 300d Google News vectors:

```python
>>> analogy(vocab, W, "man", "king", "woman")
[('queen', 0.71), ('monarch', 0.62), ('princess', 0.59), ...]
```

`king - man + woman = queen`. Не потому, что модель знает, что такое королевская власть. А потому, что vector `(king - man)` захватывает что-то вроде "royal", и добавление его к `woman` переносит точку в область royal-female.

## Использование

Писать Word2Vec с нуля полезно для обучения. Production NLP использует `gensim`.

```python
from gensim.models import Word2Vec

sentences = [
    ["the", "cat", "sat", "on", "the", "mat"],
    ["the", "dog", "ran", "across", "the", "room"],
]

model = Word2Vec(
    sentences,
    vector_size=100,
    window=5,
    min_count=1,
    sg=1,
    negative=5,
    workers=4,
    epochs=30,
)

print(model.wv["cat"])
print(model.wv.most_similar("cat", topn=3))
```

В реальной работе вы почти никогда не обучаете Word2Vec самостоятельно. Вы скачиваете pre-trained vectors.

- **GloVe** — подход Stanford на основе factorization of co-occurrence matrix. Checkpoints 50d, 100d, 200d, 300d. Хорошее общее покрытие. Урок 04 отдельно рассматривает GloVe.
- **fastText** — расширение Word2Vec от Facebook, которое embedded character n-grams. Обрабатывает out-of-vocabulary words через composition of subwords. Урок 04.
- **Pretrained Word2Vec on Google News** — 300d, словарь 3M words, опубликован в 2013. Его все еще скачивают каждый день.

### Когда Word2Vec все еще выигрывает в 2026 году

- Lightweight domain-specific retrieval. Обучите на медицинских abstracts за час на laptop и получите специализированные vectors, которые не захватывает general model.
- Analogy-style feature engineering. `gender_vector = mean(man - woman pairs)`. Вычтите его из других слов, чтобы получить gender-neutral axis. Все еще используется в fairness research.
- Interpretability. 100d достаточно мал, чтобы отобразить через PCA или t-SNE и действительно увидеть clusters.
- Везде, где inference должен работать on-device без GPU. Word2Vec lookup — это один row fetch.

### Где Word2Vec ломается

Стена polysemy. У `bank` один vector. `river bank` и `financial bank` используют его совместно. `table` (spreadsheet vs. furniture) тоже. Последующий классификатор не может различить значения по этому vector.

Contextual embeddings (ELMo, BERT, каждый transformer после них) решили это, создавая отдельный vector для каждого occurrence слова на основе surrounding context. Это скачок от Word2Vec к BERT: от static к contextual. Фаза 7 разбирает transformer-часть.

Проблема out-of-vocabulary — второй failure. Word2Vec никогда не видел `Zoomer-approved`, если этого не было в training data. Fallback отсутствует. fastText исправляет это через subword composition (урок 04).

## Доставка

Сохраните как `outputs/skill-embedding-probe.md`:

```markdown
---
name: embedding-probe
description: Inspect a word2vec model. Run analogies, find neighbors, diagnose quality.
version: 1.0.0
phase: 5
lesson: 03
tags: [nlp, embeddings, debugging]
---

You probe trained word embeddings to verify they are working. Given a `gensim.models.KeyedVectors` object and a vocabulary, you run:

1. Three canonical analogy tests. `king : man :: queen : woman`. `paris : france :: tokyo : japan`. `walking : walked :: swimming : ?`. Report the top-1 result and its cosine.
2. Five nearest-neighbor tests on domain-specific words the user supplies. Print top-5 neighbors with cosines.
3. One symmetry check. `similarity(a, b) == similarity(b, a)` to within float precision.
4. One degenerate check. If any embedding has a norm below 0.01 or above 100, the model has a training bug. Flag it.

Refuse to declare a model good on analogy accuracy alone. Analogy benchmarks are gameable and do not transfer to downstream tasks. Recommend intrinsic + downstream evaluation together.
```

## Упражнения

1. **Easy.** Запустите training loop на tiny corpus (20 предложений про cats and dogs). После 200 epochs проверьте, что `nearest(vocab, W, W[vocab["cat"]])` возвращает `dog` в top 3. Если нет, увеличьте epochs или vocabulary.
2. **Medium.** Добавьте subsampling of frequent words. Words with frequency above `10^-5` are dropped from training pairs with probability proportional to their frequency. Измерьте влияние на rare-word similarity.
3. **Hard.** Обучите модель на корпусе 20 Newsgroups. Вычислите две bias axes: `he - she` и `doctor - nurse`. Спроецируйте occupation words на обе axes. Сообщите, у каких occupations самый большой bias gap. Это тип probe, который используют fairness researchers.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Word embedding | Word as a vector | Плотное low-dim (обычно 100-300) представление, выученное из контекста. |
| Skip-gram | Word2Vec trick | Предсказывать context words по center word. Медленнее CBOW, лучше для rare words. |
| Negative sampling | Training shortcut | Заменяет softmax по полному vocab на binary classification против `k` random words. |
| Static embedding | One vector per word | Один и тот же vector независимо от контекста. Ломается на polysemy. |
| Contextual embedding | Context-sensitive vector | Разный vector для каждого occurrence на основе surrounding words. То, что создают transformers. |
| OOV | Out of vocabulary | Слово, не встречавшееся при обучении. Word2Vec не может создать для него vector. |

## Дополнительное чтение

- [Mikolov et al. (2013). Distributed Representations of Words and Phrases and their Compositionality](https://arxiv.org/abs/1310.4546) — статья о negative-sampling. Короткая и читаемая.
- [Rong, X. (2014). word2vec Parameter Learning Explained](https://arxiv.org/abs/1411.2738) — самое ясное derivation of the gradients, если математика оригинальной статьи кажется плотной.
- [gensim Word2Vec tutorial](https://radimrehurek.com/gensim/models/word2vec.html) — production training settings, которые действительно работают.
