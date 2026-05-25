# GloVe, FastText и Subword Embeddings

> Word2Vec обучал один embedding на слово. GloVe факторизовал co-occurrence matrix. FastText embedded части слова. BPE стал мостом к transformers.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Фаза 5 · 03 (Word2Vec from Scratch)
**Время:** ~45 минут

## Проблема

Word2Vec оставил два открытых вопроса.

Во-первых, существовала параллельная линия исследований, которая напрямую факторизовала co-occurrence matrix (LSA, HAL), а не делала online skip-gram updates. Был ли iterative approach Word2Vec фундаментально лучше, или различие было артефактом того, как два метода обрабатывали counts? **GloVe** ответил на это: matrix factorization с продуманно выбранной loss не уступает Word2Vec или превосходит его, и стоит дешевле в обучении.

Во-вторых, ни у одного метода не было решения для слов, которых он никогда не видел. `Zoomer-approved`, `dogecoin`, любое proper noun, придуманное на прошлой неделе, каждая inflected form редкого корня. **FastText** исправил это, embedding character n-grams: слово — это сумма его частей, включая morphemes, поэтому даже out-of-vocabulary words получают разумный vector.

В-третьих, после появления transformers вопрос снова изменился. Word-level vocabularies упираются примерно в миллион entries; настоящий язык более открыт. **Byte-pair encoding (BPE)** и родственные методы решили это, обучая vocabulary из частых subword units, покрывающий все. Каждый современный tokenizer для каждого современного LLM — subword tokenizer.

Этот урок проходит по всем трем подходам, затем объясняет, какой выбирать и когда.

## Концепция

![Три подхода к эмбеддингам: совместные появления GloVe, подслова FastText, слияния BPE](./assets/embeddings.svg)

**GloVe (Global Vectors).** Постройте word-word co-occurrence matrix `X`, где `X[i][j]` — как часто слово `j` встречается в контексте слова `i`. Обучите vectors так, чтобы `v_i · v_j + b_i + b_j ≈ log(X[i][j])`. Взвесьте loss так, чтобы frequent pairs не доминировали. Готово.

**FastText.** Слово — это сумма его character n-grams плюс само слово. `where` превращается в `<wh, whe, her, ere, re>, <where>`. Word vector — сумма этих component vectors. Обучается как Word2Vec. Преимущество: unseen words (`whereupon`) собираются из известных n-grams.

**BPE (Byte-Pair Encoding).** Начните с vocabulary из отдельных bytes (или characters). Посчитайте каждую adjacent pair в корпусе. Объедините самую частую пару в новый token. Повторите `k` iterations. Результат: vocabulary из `k + 256` tokens, где frequent sequences (`ing`, `tion`, `the`) являются single tokens, а редкие слова разбиваются на знакомые части. Любое предложение токенизируется во что-то.

## Построение

### GloVe: факторизовать co-occurrence matrix

```python
import numpy as np
from collections import Counter


def build_cooccurrence(docs, window=5):
    pair_counts = Counter()
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    for doc in docs:
        indexed = [vocab[t] for t in doc]
        for i, center in enumerate(indexed):
            for j in range(max(0, i - window), min(len(indexed), i + window + 1)):
                if i != j:
                    distance = abs(i - j)
                    pair_counts[(center, indexed[j])] += 1.0 / distance
    return vocab, pair_counts


def glove_train(vocab, pair_counts, dim=16, epochs=100, lr=0.05, x_max=100, alpha=0.75, seed=0):
    n = len(vocab)
    rng = np.random.default_rng(seed)
    W = rng.normal(0, 0.1, size=(n, dim))
    W_tilde = rng.normal(0, 0.1, size=(n, dim))
    b = np.zeros(n)
    b_tilde = np.zeros(n)

    for epoch in range(epochs):
        for (i, j), x_ij in pair_counts.items():
            weight = (x_ij / x_max) ** alpha if x_ij < x_max else 1.0
            diff = W[i] @ W_tilde[j] + b[i] + b_tilde[j] - np.log(x_ij)
            coef = weight * diff

            grad_W_i = coef * W_tilde[j]
            grad_W_tilde_j = coef * W[i]
            W[i] -= lr * grad_W_i
            W_tilde[j] -= lr * grad_W_tilde_j
            b[i] -= lr * coef
            b_tilde[j] -= lr * coef

    return W + W_tilde
```

Здесь стоит назвать две движущие части. Weighting function `f(x) = (x/x_max)^alpha` уменьшает вес очень frequent pairs (например `(the, and)`), чтобы они не доминировали в loss. Final embedding — сумма таблиц `W` (center) и `W_tilde` (context). Суммирование обеих — опубликованный трюк, который обычно превосходит использование только одной.

### FastText: эмбеддинги с учетом подслов

```python
def char_ngrams(word, n_min=3, n_max=6):
    wrapped = f"<{word}>"
    grams = {wrapped}
    for n in range(n_min, n_max + 1):
        for i in range(len(wrapped) - n + 1):
            grams.add(wrapped[i:i + n])
    return grams
```

```python
>>> char_ngrams("where")
{'<where>', '<wh', 'whe', 'her', 'ere', 're>', '<whe', 'wher', 'here', 'ere>', '<wher', 'where', 'here>'}
```

Каждое слово представляется своим набором n-grams (обычно от 3 до 6 символов). Word embedding — сумма его n-gram embeddings. Для skip-gram training подставьте это туда, где Word2Vec использовал один vector.

```python
def fasttext_vector(word, ngram_table):
    grams = char_ngrams(word)
    vecs = [ngram_table[g] for g in grams if g in ngram_table]
    if not vecs:
        return None
    return np.sum(vecs, axis=0)
```

Для unseen word вы все равно получите vector, пока известна хотя бы часть его n-grams. `whereupon` делит `<wh`, `her`, `ere` и `<where` с `where`, поэтому они оказываются рядом.

### BPE: выученный подсловный словарь

```python
def learn_bpe(corpus, k_merges):
    vocab = Counter()
    for word, freq in corpus.items():
        tokens = tuple(word) + ("</w>",)
        vocab[tokens] = freq

    merges = []
    for _ in range(k_merges):
        pair_freq = Counter()
        for tokens, freq in vocab.items():
            for a, b in zip(tokens, tokens[1:]):
                pair_freq[(a, b)] += freq
        if not pair_freq:
            break
        best = pair_freq.most_common(1)[0][0]
        merges.append(best)

        new_vocab = Counter()
        for tokens, freq in vocab.items():
            new_tokens = []
            i = 0
            while i < len(tokens):
                if i + 1 < len(tokens) and (tokens[i], tokens[i + 1]) == best:
                    new_tokens.append(tokens[i] + tokens[i + 1])
                    i += 2
                else:
                    new_tokens.append(tokens[i])
                    i += 1
            new_vocab[tuple(new_tokens)] = freq
        vocab = new_vocab
    return merges


def apply_bpe(word, merges):
    tokens = list(word) + ["</w>"]
    for a, b in merges:
        new_tokens = []
        i = 0
        while i < len(tokens):
            if i + 1 < len(tokens) and tokens[i] == a and tokens[i + 1] == b:
                new_tokens.append(a + b)
                i += 2
            else:
                new_tokens.append(tokens[i])
                i += 1
        tokens = new_tokens
    return tokens
```

```python
>>> corpus = Counter({"low": 5, "lower": 2, "newest": 6, "widest": 3})
>>> merges = learn_bpe(corpus, k_merges=10)
>>> apply_bpe("lowest", merges)
['low', 'est</w>']
```

Первая iteration объединяет самую частую adjacent pair. После достаточного числа iterations frequent substrings (`low`, `est`, `tion`) становятся single tokens, а rare words разбиваются чисто.

Настоящие GPT / BERT / T5 tokenizers обучают 30k-100k merges. Результат: любой text токенизируется в sequence of known IDs ограниченной длины, OOV никогда не возникает.

## Использование

На практике вы редко обучаете что-либо из этого самостоятельно. Вы загружаете pre-trained checkpoints.

```python
import fasttext.util
fasttext.util.download_model("en", if_exists="ignore")
ft = fasttext.load_model("cc.en.300.bin")
print(ft.get_word_vector("whereupon").shape)
print(ft.get_word_vector("zoomerapproved").shape)
```

Для BPE-style subword tokenization в эпоху transformers:

```python
from transformers import AutoTokenizer

tok = AutoTokenizer.from_pretrained("gpt2")
print(tok.tokenize("unbelievably tokenized"))
```

```
['un', 'bel', 'iev', 'ably', 'Ġtoken', 'ized']
```

Префикс `Ġ` отмечает word boundaries (конвенция GPT-2). Каждый современный tokenizer — это вариант BPE, WordPiece (BERT) или SentencePiece (T5, LLaMA).

### Что выбирать и когда

| Ситуация | Выбор |
|-----------|------|
| Предобученные векторы слов общего назначения, устойчивость к OOV не нужна | GloVe 300d |
| Предобученные векторы слов общего назначения, нужно обрабатывать опечатки / неологизмы / морфологически богатые языки | FastText |
| Все, что идет в transformer (обучение или инференс) | Тот токенизатор, с которым поставляется модель. Никогда не заменяйте. |
| Обучение собственной языковой модели с нуля | Сначала обучите BPE- или SentencePiece-токенизатор на своем корпусе |
| Production-классификация текста с линейной моделью | Все еще TF-IDF. Урок 02. |

## Доставка

Сохраните как `outputs/skill-tokenizer-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick a tokenization approach for a new language model or text pipeline.
version: 1.0.0
phase: 5
lesson: 04
tags: [nlp, tokenization, embeddings]
---

Given a task and dataset description, you output:

1. Tokenization strategy (word-level, BPE, WordPiece, SentencePiece, byte-level). One-sentence reason.
2. Vocabulary size target (e.g., 32k for an English-only LM, 64k-100k for multilingual).
3. Library call with the exact training command. Name the library. Quote the arguments.
4. One reproducibility pitfall. Tokenizer-model mismatch is the single most common silent production bug; call out which pair must be used together.

Refuse to recommend training a custom tokenizer when the user is fine-tuning a pretrained LLM. Refuse to recommend word-level tokenization for any model targeting production inference. Flag non-English / multi-script corpora as needing SentencePiece with byte fallback.
```

## Упражнения

1. **Легко.** Запустите `char_ngrams("playing")` и `char_ngrams("played")`. Вычислите Jaccard overlap двух n-gram sets. Вы должны увидеть много общих частей (`pla`, `lay`, `play`), поэтому FastText хорошо переносится между morphological variants.
2. **Средне.** Расширьте `learn_bpe`, чтобы отслеживать vocabulary growth. Постройте график tokens-per-corpus-character как функцию number of merges. Вы должны увидеть rapid compression сначала и asymptoting около ~2-3 chars per token.
3. **Сложно.** Обучите 1k-merge BPE на complete works of Shakespeare. Сравните tokenization common words и rare proper nouns. Измерьте average tokens per word before and after. Опишите, что вас удивило.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| Co-occurrence matrix | Word-word frequency table | `X[i][j]` = как часто слово `j` встречается в окне вокруг слова `i`. |
| Subword | Piece of a word | Character n-gram (FastText) или learned token (BPE/WordPiece/SentencePiece). |
| BPE | Byte-pair encoding | Итеративное объединение most-frequent adjacent pairs, пока vocabulary не достигнет target size. |
| OOV | Out of vocabulary | Слово, которое модель никогда не видела. Word2Vec/GloVe ломаются. FastText и BPE справляются. |
| Byte-level BPE | BPE on raw bytes | Схема GPT-2. Vocabulary начинается с 256 bytes, поэтому OOV не бывает. |

## Дополнительное чтение

- [Pennington, Socher, Manning (2014). GloVe: Global Vectors for Word Representation](https://nlp.stanford.edu/pubs/glove.pdf) — статья GloVe, семь страниц, все еще лучший derivation of the loss.
- [Bojanowski et al. (2017). Enriching Word Vectors with Subword Information](https://arxiv.org/abs/1607.04606) — FastText.
- [Sennrich, Haddow, Birch (2016). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — статья, которая ввела BPE в modern NLP.
- [Hugging Face tokenizer summary](https://huggingface.co/docs/transformers/tokenizer_summary) — как BPE, WordPiece и SentencePiece реально различаются на практике.
