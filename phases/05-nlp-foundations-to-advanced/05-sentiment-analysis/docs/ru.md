# Анализ тональности

> Каноническая NLP-задача. Большая часть того, что нужно знать о classical text classification, проявляется именно здесь.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Фаза 5 · 02 (BoW + TF-IDF), Фаза 2 · 14 (Naive Bayes)
**Время:** ~75 минут

## Цели обучения

- Реализовывать с нуля multinomial Naive Bayes и логистическую регрессию для классификации текста.
- Обрабатывать область действия отрицания — классический режим отказа.
- Оценивать метриками, важными для несбалансированной тональности.

## Проблема

"The food was not great." Positive или negative?

Sentiment звучит просто. Рецензент сказал, что ему что-то понравилось или не понравилось. Поставьте sentence label. Причина, по которой это стало канонической NLP-задачей, в том, что каждый на вид простой случай скрывает сложный. Negation переворачивает смысл. Sarcasm инвертирует его. "Not bad at all" позитивно, несмотря на два слова с негативным кодом. Emojis несут больше сигнала, чем окружающий текст. Domain vocabulary важен (`tight` в music review против `tight` в fashion review).

Sentiment — рабочая лаборатория для classical NLP. Если вы понимаете, почему у каждого naive baseline есть конкретный failure mode, вы понимаете, зачем была изобретена каждая более богатая модель. В этом уроке мы построим Naive Bayes baseline с нуля, добавим logistic regression и назовем ловушки, которые превращают production sentiment в задачу уровня compliance.

## Концепция

![Конвейер анализа тональности: токены → признаки → классификатор → метка](./assets/sentiment.svg)

Classical sentiment — это рецепт из двух шагов.

1. **Represent.** Превратить текст в feature vector. BoW, TF-IDF или n-grams.
2. **Classify.** Обучить linear model (Naive Bayes, logistic regression, SVM) на labeled examples.

Naive Bayes — самая простая модель, которая работает. Предположите, что каждая feature независима при условии label. Оцените `P(word | positive)` и `P(word | negative)` по counts. На inference перемножьте probabilities. "Naive" independence assumption смехотворно неверно, но результаты удивительно сильные. Причина: с sparse text features и умеренным объемом данных классификатору важнее, к какой стороне склоняется каждое слово, чем насколько сильно.

Logistic regression исправляет independence assumption. Она учит weight на feature, включая negative weights. `not good` как bigram feature получает negative weight. Naive Bayes не может сделать это для bigrams, которых он никогда не видел размеченными.

## Построение

### Шаг 1: настоящий mini-dataset

```python
POSITIVE = [
    "absolutely loved this movie",
    "beautiful cinematography and a great story",
    "one of the best films of the year",
    "brilliant acting from the lead",
    "heartwarming and funny",
]

NEGATIVE = [
    "boring and far too long",
    "not worth your time",
    "the plot made no sense",
    "terrible acting, awful script",
    "i want my two hours back",
]
```

Маленький специально. В реальной работе используют десятки тысяч examples (IMDb, SST-2, Yelp polarity). Математика идентична.

### Шаг 2: multinomial Naive Bayes с нуля

```python
import math
from collections import Counter


def train_nb(docs_by_class, vocab, alpha=1.0):
    class_priors = {}
    class_word_probs = {}
    total_docs = sum(len(d) for d in docs_by_class.values())

    for cls, docs in docs_by_class.items():
        class_priors[cls] = len(docs) / total_docs
        counts = Counter()
        for doc in docs:
            for token in doc:
                counts[token] += 1
        total = sum(counts.values()) + alpha * len(vocab)
        class_word_probs[cls] = {
            w: (counts[w] + alpha) / total for w in vocab
        }
    return class_priors, class_word_probs


def predict_nb(doc, class_priors, class_word_probs):
    scores = {}
    for cls in class_priors:
        s = math.log(class_priors[cls])
        for token in doc:
            if token in class_word_probs[cls]:
                s += math.log(class_word_probs[cls][token])
        scores[cls] = s
    return max(scores, key=scores.get)
```

Additive smoothing (alpha=1.0) — это Laplace smoothing. Без него слово, не встречавшееся в классе, имеет probability zero, и log взрывается. `alpha=0.01` часто используется на практике. `alpha=1.0` — teaching default.

### Шаг 3: logistic regression с нуля

```python
import numpy as np


def sigmoid(x):
    return 1.0 / (1.0 + np.exp(-np.clip(x, -20, 20)))


def train_lr(X, y, epochs=500, lr=0.05, l2=0.01):
    n_features = X.shape[1]
    w = np.zeros(n_features)
    b = 0.0
    for _ in range(epochs):
        logits = X @ w + b
        preds = sigmoid(logits)
        err = preds - y
        grad_w = X.T @ err / len(y) + l2 * w
        grad_b = err.mean()
        w -= lr * grad_w
        b -= lr * grad_b
    return w, b


def predict_lr(X, w, b):
    return (sigmoid(X @ w + b) >= 0.5).astype(int)
```

L2 regularization здесь важна. Text features разреженные; без L2 модель memorizes training examples. Начните с `0.01` и настраивайте.

### Шаг 4: обработка negation (failure mode)

Рассмотрим "not good" и "not bad". BoW classifier видит `{not, good}` и `{not, bad}` и учится по тому, что чаще встречалось при обучении. Bigram classifier видит `not_good` и `not_bad` и учит их как отдельные features. Обычно этого достаточно.

Более грубое исправление, которое работает, когда нет bigrams: **negation scoping**. Добавляйте prefix к tokens после negation word до следующего punctuation.

```python
NEGATION_WORDS = {"not", "no", "never", "nor", "none", "nothing", "neither"}
NEGATION_TERMINATORS = {".", "!", "?", ",", ";"}


def apply_negation(tokens):
    out = []
    negate = False
    for token in tokens:
        if token in NEGATION_TERMINATORS:
            negate = False
            out.append(token)
            continue
        if token in NEGATION_WORDS:
            negate = True
            out.append(token)
            continue
        out.append(f"NOT_{token}" if negate else token)
    return out
```

```python
>>> apply_negation(["not", "good", "at", "all", ".", "but", "funny"])
['not', 'NOT_good', 'NOT_at', 'NOT_all', '.', 'but', 'funny']
```

Теперь `good` и `NOT_good` — разные features. Классификатор может назначить им противоположные weights. Три строки preprocessing, измеримый прирост accuracy на sentiment benchmarks.

### Шаг 5: evaluation metrics, которые важны

Accuracy alone вводит в заблуждение, если classes are imbalanced. Реальные sentiment corpora обычно на 70-80% positive или на 70-80% negative; constant-majority classifier получает 80% accuracy и бесполезен. Сообщайте все следующее:

- **Per-class precision and recall.** Одна пара на class. Macro-average их, чтобы получить одно число, уважающее class balance.
- **Macro-F1 (primary metric for imbalanced data).** Среднее per-class F1 scores с равным весом. Используйте это вместо accuracy, когда classes are imbalanced.
- **Weighted-F1 (alternative).** То же, что macro, но взвешенное по class frequency. Сообщайте вместе с macro-F1, когда сам imbalance имеет business meaning.
- **Confusion matrix.** Raw counts. Всегда смотрите перед доверием любому scalar metric; она показывает, какие пары classes модель путает.
- **Per-class error samples.** Возьмите 5 wrong predictions на class. Прочитайте их. Ничто не заменяет чтение настоящих errors.

Для сильно imbalanced data (> 95-5 ratio) сообщайте **AUROC** и **AUPRC** вместо accuracy. AUPRC более чувствителен к minority class, который обычно и важен (spam, fraud, rare sentiment).

**Common bug to avoid.** Reporting micro-F1 instead of macro-F1 on imbalanced data дает число, которое выглядит высоким, потому что dominated by majority class. Macro-F1 заставляет увидеть minority-class performance.

```python
def evaluate(y_true, y_pred):
    tp = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 1)
    fp = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 1)
    fn = sum(1 for t, p in zip(y_true, y_pred) if t == 1 and p == 0)
    tn = sum(1 for t, p in zip(y_true, y_pred) if t == 0 and p == 0)
    precision = tp / (tp + fp) if tp + fp else 0
    recall = tp / (tp + fn) if tp + fn else 0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0
    return {"tp": tp, "fp": fp, "tn": tn, "fn": fn, "precision": precision, "recall": recall, "f1": f1}
```

## Использование

scikit-learn делает это правильно в шесть строк.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.pipeline import Pipeline

pipe = Pipeline([
    ("tfidf", TfidfVectorizer(ngram_range=(1, 2), min_df=2, sublinear_tf=True, stop_words=None)),
    ("clf", LogisticRegression(C=1.0, max_iter=1000)),
])
pipe.fit(X_train, y_train)
print(pipe.score(X_test, y_test))
```

Три вещи, на которые нужно обратить внимание. `stop_words=None` сохраняет negations. `ngram_range=(1, 2)` добавляет bigrams, так что `not_good` становится feature. `sublinear_tf=True` dampens repeated words. Эти три flags — разница между 75%-accurate baseline и 85%-accurate baseline на SST-2.

### Когда переходить к transformer

- Sarcasm detection. Classical models здесь ломаются. Точка.
- Long reviews, где sentiment shifts mid-document.
- Aspect-based sentiment. "Camera was great but battery was terrible." Нужно привязать sentiment к aspects. Только transformers или structured output models.
- Non-English, low-resource languages. Multilingual BERT бесплатно дает zero-shot baseline.

Если вам нужно что-то из перечисленного, переходите к фазе 7 (transformers deep dive). Иначе Naive Bayes или logistic regression на TF-IDF плюс bigrams плюс negation handling — ваш production baseline 2026 года.

### Reproducibility trap (снова)

Retraining sentiment models — рутина. Re-evaluating them — нет. Accuracy numbers, опубликованные в статьях, используют specific splits, specific preprocessing, specific tokenizers. Если сравнивать новую модель с baseline без identical pipeline, вы получите misleading deltas. Всегда регенерируйте baseline на своем pipeline, а не берите число из статьи.

## Доставка

Сохраните как `outputs/prompt-sentiment-baseline.md`:

```markdown
---
name: sentiment-baseline
description: Design a sentiment analysis baseline for a new dataset.
phase: 5
lesson: 05
---

Given a dataset description (domain, language, size, label granularity, latency budget), you output:

1. Feature extraction recipe. Specify tokenizer, n-gram range, stopword policy (usually keep), negation handling (scoped prefix or bigrams).
2. Classifier. Naive Bayes for baseline, logistic regression for production, transformer only if the domain needs sarcasm / aspects / cross-lingual.
3. Evaluation plan. Report precision, recall, F1, confusion matrix, and per-class error samples (not just scalars).
4. One failure mode to monitor post-deployment. Domain drift and sarcasm are the top two.

Refuse to recommend dropping stopwords for sentiment tasks. Refuse to report accuracy as the sole metric when classes are imbalanced (e.g., 90% positive). Flag subword-rich languages as needing FastText or transformer embeddings over word-level TF-IDF.
```

## Упражнения

1. **Легко.** Добавьте `apply_negation` как preprocessing step в scikit-learn pipeline и измерьте F1 delta на небольшом sentiment dataset.
2. **Средне.** Реализуйте class-weighted logistic regression (передайте `class_weight="balanced"` в scikit-learn или выведите gradient самостоятельно). Измерьте effect на synthetic 90-10 class imbalance.
3. **Сложно.** Постройте sarcasm detector, обучив второй classifier на residuals sentiment model. Задокументируйте experimental setup. Предупредите читателя, когда accuracy ниже chance (chance-level on 2-class sarcasm is ~50%, и большинство первых попыток оказываются там).

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| Polarity | Positive or negative | Binary label; иногда расширяется до neutral или fine-grained (5-star). |
| Aspect-based sentiment | Per-aspect polarity | Приписывает sentiment конкретным entities или attributes, упомянутым в тексте. |
| Область действия отрицания | Переворот соседних токенов | Добавлять префикс `NOT_` к токенам после "not" до пунктуации. |
| Laplace smoothing | Adding 1 to counts | Предотвращает zero-probability features в Naive Bayes. |
| L2 regularization | Shrinking weights | Добавляет `lambda * sum(w^2)` к loss. Необходима для sparse text features. |

## Дополнительное чтение

- [Pang and Lee (2008). Opinion Mining and Sentiment Analysis](https://www.cs.cornell.edu/home/llee/opinion-mining-sentiment-analysis-survey.html) — foundational survey. Длинный, но первые четыре sections покрывают всю classical часть.
- [Wang and Manning (2012). Baselines and Bigrams: Simple, Good Sentiment and Topic Classification](https://aclanthology.org/P12-2018/) — статья, показавшая, что bigrams + Naive Bayes трудно обойти на short text.
- [scikit-learn text feature extraction docs](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — reference для `CountVectorizer`, `TfidfVectorizer` и каждого параметра, который вы будете настраивать.
