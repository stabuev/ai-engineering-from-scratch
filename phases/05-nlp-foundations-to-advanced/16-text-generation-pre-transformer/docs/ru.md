# Генерация текста до трансформеров — N-граммные языковые модели

> Если слово неожиданно, модель плоха. Perplexity превращает неожиданность в число. Сглаживание сохраняет его конечным.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 5 · 01 (Text Processing), Phase 2 · 14 (Naive Bayes)
**Time:** ~45 minutes

## Проблема

До трансформеров, до RNN, до эмбеддингов слов языковая модель предсказывала следующее слово, считая, как часто оно следовало за предыдущими `n-1` словами. Посчитать "the cat" → "sat" 47 раз, "the cat" → "jumped" 12 раз, "the cat" → "refrigerator" 0 раз. Нормализовать, чтобы получить распределение вероятностей.

Это и есть n-граммная языковая модель. Она работала в каждом распознавателе речи, каждой проверке орфографии и каждой фразовой системе машинного перевода с 1980 по 2015 год. Она все еще используется, когда нужно дешевое языковое моделирование на устройстве.

Интересная проблема состоит в том, что делать с невиденными n-граммами. Сырая модель на основе счетчиков назначает нулевую вероятность всему, чего не видела, а это катастрофично, потому что предложения длинные и почти каждое длинное предложение содержит хотя бы одну невиданную последовательность. Пятьдесят лет исследований сглаживания исправили это. Сглаживание Кнезера-Нея — результат этой работы, а современное глубокое обучение унаследовало ее эмпирическую традицию.

## Концепция

![N-gram model: count, smooth, generate](../assets/ngram.svg)

**N-граммная вероятность:** `P(w_i | w_{i-n+1}, ..., w_{i-1})`. Зафиксируйте `n` (обычно 3 для триграмм, 4 для 4-грамм). Вычислите по счетчикам:

```text
P(w | context) = count(context, w) / count(context)
```

**Проблема нулевых счетчиков.** Любая n-грамма, не встречавшаяся при обучении, получает нулевую вероятность. Исследование 2007 года на корпусе Brown показало, что даже у 4-граммной модели 30% 4-грамм из отложенной выборки не встречались при обучении. Без сглаживания невозможно оценивать модель на любом реальном тексте.

**Подходы к сглаживанию, в порядке усложнения:**

1. **Лапласовское (add-one).** Добавить 1 к каждому счетчику. Просто, ужасно для редких событий.
2. **Good-Turing.** Перераспределить вероятностную массу от более частотных событий к невиданным на основе frequency-of-frequencies.
3. **Интерполяция.** Объединить оценки n-грамм, (n-1)-грамм и т. д. с настраиваемыми весами.
4. **Backoff.** Если счетчик n-граммы равен нулю, откатиться к (n-1)-грамме. Katz backoff нормализует это.
5. **Absolute discounting.** Вычесть фиксированный дисконт `D` из всех счетчиков, перераспределить его к невиданным.
6. **Kneser-Ney.** Absolute discounting плюс умный выбор модели меньшего порядка: использовать *continuation probability* (в скольких контекстах появляется слово) вместо сырой частоты.

Идея Kneser-Ney глубока. "San Francisco" — частая биграмма. Униграмма "Francisco" появляется в основном после "San." Наивный absolute discounting дает "Francisco" высокую униграммную вероятность (потому что счетчик высок). Kneser-Ney замечает, что "Francisco" появляется только в одном контексте, и соответственно снижает его continuation probability. Результат: новая биграмма, заканчивающаяся на "Francisco", получает подходящую низкую вероятность.

**Оценка: perplexity.** Экспонента средней отрицательной log-likelihood на слово на отложенном тестовом наборе. Ниже — лучше. Perplexity 100 означает, что модель так же растеряна, как если бы выбирала равномерно среди 100 слов.

```text
perplexity = exp(- (1/N) * Σ log P(w_i | context_i))
```

## Соберите это

### Шаг 1: счетчики триграмм

```python
from collections import Counter, defaultdict


def train_ngram(corpus_tokens, n=3):
    ngrams = Counter()
    contexts = Counter()
    for sentence in corpus_tokens:
        padded = ["<s>"] * (n - 1) + sentence + ["</s>"]
        for i in range(len(padded) - n + 1):
            ctx = tuple(padded[i:i + n - 1])
            word = padded[i + n - 1]
            ngrams[ctx + (word,)] += 1
            contexts[ctx] += 1
    return ngrams, contexts


def raw_probability(ngrams, contexts, context, word):
    ctx = tuple(context)
    if contexts.get(ctx, 0) == 0:
        return 0.0
    return ngrams.get(ctx + (word,), 0) / contexts[ctx]
```

Вход — список токенизированных предложений. Выход — счетчики n-грамм и счетчики контекстов. `<s>` и `</s>` — границы предложения.

### Шаг 2: лапласовское сглаживание

```python
def laplace_probability(ngrams, contexts, vocab_size, context, word):
    ctx = tuple(context)
    numerator = ngrams.get(ctx + (word,), 0) + 1
    denominator = contexts.get(ctx, 0) + vocab_size
    return numerator / denominator
```

Добавьте 1 к каждому счетчику. Это сглаживает, но чрезмерно выделяет массу невиданным событиям, ухудшая и редкие известные события.

### Шаг 3: Kneser-Ney (биграммы, интерполированный)

```python
def kneser_ney_bigram_model(corpus_tokens, discount=0.75):
    unigrams = Counter()
    bigrams = Counter()
    unigram_contexts = defaultdict(set)

    for sentence in corpus_tokens:
        padded = ["<s>"] + sentence + ["</s>"]
        for i, w in enumerate(padded):
            unigrams[w] += 1
            if i > 0:
                prev = padded[i - 1]
                bigrams[(prev, w)] += 1
                unigram_contexts[w].add(prev)

    total_unique_bigrams = sum(len(ctx_set) for ctx_set in unigram_contexts.values())
    continuation_prob = {
        w: len(ctx_set) / total_unique_bigrams for w, ctx_set in unigram_contexts.items()
    }

    context_totals = Counter()
    for (prev, w), count in bigrams.items():
        context_totals[prev] += count

    unique_follow = defaultdict(set)
    for (prev, w) in bigrams:
        unique_follow[prev].add(w)

    def prob(prev, w):
        count = bigrams.get((prev, w), 0)
        denom = context_totals.get(prev, 0)
        if denom == 0:
            return continuation_prob.get(w, 1e-9)
        first_term = max(count - discount, 0) / denom
        lambda_prev = discount * len(unique_follow[prev]) / denom
        return first_term + lambda_prev * continuation_prob.get(w, 1e-9)

    return prob
```

Три движущиеся части. `continuation_prob` фиксирует "в скольких разных контекстах появляется это слово?" (инновация Kneser-Ney). `lambda_prev` — масса, высвобожденная дисконтом, используется как вес backoff. Итоговая вероятность — это дисконтированный основной член плюс взвешенный continuation term.

### Шаг 4: генерация текста с семплированием

```python
import random


def generate(prob_fn, vocab, prefix, max_len=30, seed=0):
    rng = random.Random(seed)
    tokens = list(prefix)
    for _ in range(max_len):
        candidates = [(w, prob_fn(tokens[-1], w)) for w in vocab]
        total = sum(p for _, p in candidates)
        r = rng.random() * total
        acc = 0.0
        for w, p in candidates:
            acc += p
            if r <= acc:
                tokens.append(w)
                break
        if tokens[-1] == "</s>":
            break
    return tokens
```

Семплирование пропорционально вероятности. Всегда дает разный вывод для разных seed. Для вывода в стиле beam search выбирайте argmax на каждом шаге (greedy) и добавьте небольшую ручку случайности (temperature).

### Шаг 5: perplexity

```python
import math


def perplexity(prob_fn, sentences):
    total_log_prob = 0.0
    total_tokens = 0
    for sentence in sentences:
        padded = ["<s>"] + sentence + ["</s>"]
        for i in range(1, len(padded)):
            p = prob_fn(padded[i - 1], padded[i])
            total_log_prob += math.log(max(p, 1e-12))
            total_tokens += 1
    return math.exp(-total_log_prob / total_tokens)
```

Ниже — лучше. Для корпуса Brown хорошо настроенная 4-граммная KN-модель достигает perplexity около 140. Transformer LM достигает 15-30 на том же тестовом наборе. Разрыв примерно 10x. Именно из-за этого разрыва область пошла дальше.

## Используйте это

- **Классическое обучение NLP.** Самое ясное знакомство со сглаживанием, MLE и perplexity, которое можно получить.
- **KenLM.** Production-библиотека n-грамм. Используется как rescoring-модель в речевых и MT-системах, где важна низкая задержка.
- **Автодополнение на устройстве.** Триграммные модели в клавиатурах. До сих пор.
- **Бейзлайны.** Всегда считайте perplexity n-граммной LM, прежде чем объявлять вашу нейронную LM хорошей. Если ваш transformer не превосходит KN с большим отрывом, что-то не так.

## Доведите до поставки

Сохраните как `outputs/prompt-lm-baseline.md`:

```markdown
---
name: lm-baseline
description: Build a reproducible n-gram language model baseline before training a neural LM.
phase: 5
lesson: 16
---

Given a corpus and target use (next-word prediction, rescoring, perplexity baseline), output:

1. N-gram order. Trigram for general English, 4-gram if corpus is large, 5-gram for speech rescoring.
2. Smoothing. Modified Kneser-Ney is the default; Laplace only for teaching.
3. Library. `kenlm` for production, `nltk.lm` for teaching, roll your own only to learn.
4. Evaluation. Held-out perplexity with consistent tokenization between train and test sets.

Refuse to report perplexity computed with different tokenization between systems being compared — perplexity numbers are comparable only under identical tokenization. Flag OOV rate in test set; KN handles OOV poorly unless you reserve a special <UNK> token during training.
```

## Упражнения

1. **Easy.** Обучите триграммную LM на корпусе Shakespeare из 1,000 предложений. Сгенерируйте 20 предложений. Локально они будут правдоподобными, но глобально несвязными. Это каноническая демонстрация.
2. **Medium.** Реализуйте perplexity для вашей KN-модели на отложенном разбиении Shakespeare. Сравните с Laplace. Вы должны увидеть, что KN снижает perplexity на 30-50%.
3. **Hard.** Постройте триграммный корректор орфографии: по слову с ошибкой и его контексту генерируйте исправления и ранжируйте их по контекстной вероятности LM. Оцените на корпусе орфографических ошибок Birkbeck (публичный).

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| N-gram | Последовательность слов | Последовательность из `n` подряд идущих токенов. |
| Smoothing | Избежание нулей | Перераспределение вероятностной массы так, чтобы невиданные события получали ненулевую вероятность. |
| Perplexity | Метрика качества LM | `exp(-average log-prob)` на отложенных данных. Ниже — лучше. |
| Backoff | Откат к более короткому контексту | Если счетчик триграммы равен нулю, используйте биграмму. Katz backoff формализует это. |
| Kneser-Ney | Лучшее сглаживание для n-грамм | Absolute discounting + continuation probability для модели меньшего порядка. |
| Continuation probability | Специфично для KN | `P(w)`, взвешенная по числу контекстов, в которых появляется `w`, а не по сырому счетчику. |

## Дополнительное чтение

- [Jurafsky and Martin — Speech and Language Processing, Chapter 3 (2026 draft)](https://web.stanford.edu/~jurafsky/slp3/3.pdf) — каноническое изложение n-граммных LM и сглаживания.
- [Chen and Goodman (1998). An Empirical Study of Smoothing Techniques for Language Modeling](https://dash.harvard.edu/handle/1/25104739) — статья, закрепившая Kneser-Ney как лучший сглаживатель n-грамм.
- [Kneser and Ney (1995). Improved Backing-off for M-gram Language Modeling](https://ieeexplore.ieee.org/document/479394) — оригинальная статья о KN.
- [KenLM](https://kheafield.com/code/kenlm/) — быстрая production n-граммная LM, все еще используемая в 2026 году для приложений, чувствительных к задержке.
