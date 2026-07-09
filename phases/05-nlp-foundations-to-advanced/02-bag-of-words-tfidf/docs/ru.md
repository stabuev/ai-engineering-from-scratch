# Bag of Words, TF-IDF и представление текста

> Сначала считайте, потом думайте. В 2026 году TF-IDF все еще обходит embeddings на четко определенных задачах.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Фаза 5 · 01 (Text Processing), Фаза 2 · 02 (Linear Regression from Scratch)
**Время:** ~75 минут

## Цели обучения

- Строить словарь, матрицу bag-of-words и веса TF-IDF, затем L2-нормировать и оценивать косинусной близостью.
- Объяснять, что измеряют TF, DF и IDF.
- Говорить, почему TF-IDF все еще бьет эмбеддинги на четко поставленных задачах.

## Проблема

Модели нужны числа. У вас есть строки.

Каждый NLP pipeline должен ответить на один и тот же вопрос. Как превратить поток токенов переменной длины в вектор фиксированного размера, который сможет принять классификатор. Первым ответом области был самый простой рабочий вариант. Посчитать слова. Сделать вектор.

Этот вектор вынес на себе больше production NLP, чем любая embedding-модель. Спам-фильтры, тематические классификаторы, обнаружение аномалий в логах, ранжирование поиска (до BM25), первая волна sentiment analysis, первое десятилетие академических NLP benchmark'ов. В 2026 году практики все еще сначала берут его для узких задач классификации. Он быстрый, интерпретируемый и часто неотличим от embedding-модели на 400M параметров в задачах, где важно наличие слов.

В этом уроке мы построим bag of words, затем TF-IDF, с нуля. Затем покажем, как scikit-learn делает то же самое в три строки. Затем назовем failure mode, из-за которого приходится переходить к embeddings.

## Концепция

![Поток представления BoW и TF-IDF](./assets/bow-tfidf.svg)

**Bag of Words (BoW)** отбрасывает порядок. Для каждого документа посчитайте, сколько раз встречается каждое слово словаря. Длина вектора равна размеру словаря. Позиция `i` — это счетчик слова `i`.

**TF-IDF** перевзвешивает BoW. Слово, которое встречается в каждом документе, малоинформативно, поэтому его вес уменьшается. Слово, редкое во всем корпусе, но частое в одном документе, является сигналом, поэтому его вес увеличивается.

```
TF-IDF(w, d) = TF(w, d) * IDF(w)
             = count(w in d) / |d| * log(N / df(w))
```

Где `TF` — term frequency в документе, `df` — document frequency (сколько документов содержит слово), `N` — общее число документов. `log` удерживает вес в ограниченных пределах для повсеместных слов.

Ключевое свойство: оба метода создают разреженные векторы с интерпретируемыми осями. Можно посмотреть на веса обученного классификатора и прочитать, какие слова толкают документ к каждому классу. С 768-мерным BERT embedding так сделать нельзя.

## Построение

### Шаг 1: построить словарь

```python
def build_vocab(docs):
    vocab = {}
    for doc in docs:
        for token in doc:
            if token not in vocab:
                vocab[token] = len(vocab)
    return vocab
```

Вход: список токенизированных документов (подойдет любой word-level tokenizer; `code/main.py` в этом уроке использует упрощенный вариант с приведением к lowercase). Выход: dict `{word: index}`. Стабильный порядок вставки означает, что слово с индексом 0 — первое слово, встреченное в первом документе. Конвенции различаются; scikit-learn сортирует по алфавиту.

### Шаг 2: bag of words

```python
def bag_of_words(docs, vocab):
    matrix = [[0] * len(vocab) for _ in docs]
    for i, doc in enumerate(docs):
        for token in doc:
            if token in vocab:
                matrix[i][vocab[token]] += 1
    return matrix
```

```python
>>> docs = [["cat", "sat", "on", "mat"], ["cat", "cat", "ran"]]
>>> vocab = build_vocab(docs)
>>> bag_of_words(docs, vocab)
[[1, 1, 1, 1, 0], [2, 0, 0, 0, 1]]
```

Строки — документы. Столбцы — индексы словаря. Элемент `[i][j]` означает "сколько раз слово `j` встречается в документе `i`." В Doc 1 `cat` встречается дважды, потому что так и есть. В Doc 0 `ran` встречается ноль раз, потому что его там нет.

### Шаг 3: term frequency и document frequency

```python
import math


def term_frequency(doc_bow, doc_length):
    return [c / doc_length if doc_length else 0 for c in doc_bow]


def document_frequency(bow_matrix):
    df = [0] * len(bow_matrix[0])
    for row in bow_matrix:
        for j, count in enumerate(row):
            if count > 0:
                df[j] += 1
    return df


def inverse_document_frequency(df, n_docs):
    return [math.log((n_docs + 1) / (d + 1)) + 1 for d in df]
```

Здесь стоит назвать два приема сглаживания. `(n+1)/(d+1)` предотвращает `log(x/0)`. Завершающий `+1` гарантирует, что слово в каждом документе все еще имеет IDF 1 (а не 0), как в настройке scikit-learn по умолчанию. Другие реализации используют сырой `log(N/df)`. Оба варианта работают; сглаженная версия дружелюбнее.

### Шаг 4: TF-IDF

```python
def tfidf(bow_matrix):
    n_docs = len(bow_matrix)
    df = document_frequency(bow_matrix)
    idf = inverse_document_frequency(df, n_docs)
    out = []
    for row in bow_matrix:
        length = sum(row)
        tf = term_frequency(row, length)
        out.append([tf_j * idf_j for tf_j, idf_j in zip(tf, idf)])
    return out
```

```python
>>> docs = [
...     ["the", "cat", "sat"],
...     ["the", "dog", "sat"],
...     ["the", "cat", "ran"],
... ]
>>> vocab = build_vocab(docs)
>>> bow = bag_of_words(docs, vocab)
>>> tfidf(bow)
```

Три документа, пять слов словаря (`the`, `cat`, `sat`, `dog`, `ran`). `the` встречается во всех трех, поэтому его IDF низкий. `dog` встречается в одном, поэтому его IDF высокий. Векторы разреженные (большинство элементов малы), а дискриминативные слова выделяются.

### Шаг 5: L2-нормализовать строки

```python
def l2_normalize(matrix):
    out = []
    for row in matrix:
        norm = math.sqrt(sum(x * x for x in row))
        out.append([x / norm if norm else 0 for x in row])
    return out
```

Без нормализации более длинный документ получает больший вектор и доминирует в оценках similarity. L2 normalization помещает каждый документ на единичную гиперсферу. Cosine similarity между строками теперь является просто dot product.

## Использование

В scikit-learn есть production-версия.

```python
from sklearn.feature_extraction.text import CountVectorizer, TfidfVectorizer

docs = ["the cat sat on the mat", "the dog sat on the mat", "the cat ran"]

bow_vectorizer = CountVectorizer()
bow = bow_vectorizer.fit_transform(docs)
print(bow_vectorizer.get_feature_names_out())
print(bow.toarray())

tfidf_vectorizer = TfidfVectorizer()
tfidf = tfidf_vectorizer.fit_transform(docs)
print(tfidf.toarray().round(3))
```

`CountVectorizer` делает tokenization, vocabulary и BoW за один вызов. `TfidfVectorizer` добавляет IDF weighting и L2 normalization. Оба возвращают sparse matrices. Для 100k документов плотная версия не поместится в память; оставайтесь в sparse-представлении, пока классификатор не потребует dense.

Параметры, которые меняют все:

| Аргумент | Эффект |
|-----|--------|
| `ngram_range=(1, 2)` | Включить bigrams. Обычно улучшает классификацию. |
| `min_df=2` | Убрать слова, встречающиеся менее чем в 2 документах. Сокращает словарь на шумных данных. |
| `max_df=0.95` | Убрать слова, встречающиеся более чем в 95% документов. Приближает удаление stopwords без жестко заданного списка. |
| `stop_words="english"` | Встроенный список stopwords в scikit-learn. Зависит от задачи — sentiment analysis *не должен* удалять отрицания. |
| `sublinear_tf=True` | Использовать `1 + log(tf)` вместо сырого `tf`. Помогает, когда термин много раз повторяется в одном документе. |

### Когда TF-IDF все еще выигрывает (на 2026 год)

- Spam detection, topic labeling, log anomaly flagging. Важно наличие слова; семантические нюансы не важны.
- Режимы с малым количеством данных (сотни размеченных примеров). TF-IDF плюс logistic regression не требует затрат на pretraining.
- Везде, где важна latency. TF-IDF плюс linear model отвечает за микросекунды. Прогон документа через transformer для embedding занимает 10-100ms.
- Системы, которые должны объяснять свои предсказания. Посмотрите coefficients классификатора. Главные positive words и есть причина.

### Где TF-IDF ломается

Failure из-за семантической слепоты. Рассмотрим два документа:

- "The movie was not good at all."
- "The movie was excellent."

Один — негативный отзыв. Другой — позитивный. Их TF-IDF overlap ровно `{the, movie, was}`. Bag-of-words classifier должен запомнить, что слово `not` рядом с `good` переворачивает метку. Он может выучить это на достаточном количестве данных, но никогда не так элегантно, как модель, понимающая синтаксис.

Другая ошибка: out-of-vocabulary words при inference. BoW model, обученная на IMDb reviews, не знает, что делать с `Zoomer-approved`, если этот token никогда не встречался при обучении. Subword embeddings (урок 04) справляются с этим. TF-IDF — нет.

### Гибрид: эмбеддинги, взвешенные TF-IDF

Прагматичный default 2026 года для medium-data classification: использовать TF-IDF weights как attention over word embeddings.

```python
def tfidf_weighted_embedding(doc, tfidf_scores, embedding_table, dim):
    vec = [0.0] * dim
    total_weight = 0.0
    for token in doc:
        if token not in embedding_table or token not in tfidf_scores:
            continue
        weight = tfidf_scores[token]
        emb = embedding_table[token]
        for i in range(dim):
            vec[i] += weight * emb[i]
        total_weight += weight
    if total_weight == 0:
        return vec
    return [v / total_weight for v in vec]
```

Вы получаете semantic capacity от embeddings и акцент на rare-word от TF-IDF. Классификатор обучается на pooled vector. Для sentiment, topic и intent classification ниже примерно 50k размеченных примеров это превосходит каждый подход по отдельности.

## Доставка

Сохраните как `outputs/prompt-vectorization-picker.md`:

```markdown
---
name: vectorization-picker
description: Given a text-classification task, recommend BoW, TF-IDF, embeddings, or a hybrid.
phase: 5
lesson: 02
---

You recommend a text-vectorization strategy. Given a task description, output:

1. Representation (BoW, TF-IDF, transformer embeddings, or a hybrid). Explain why in one sentence.
2. Specific vectorizer configuration. Name the library. Quote the arguments (`ngram_range`, `min_df`, `max_df`, `sublinear_tf`, `stop_words`).
3. One failure mode to test before shipping.

Refuse to recommend embeddings when the user has under 500 labeled examples unless they show evidence of semantic failure in a TF-IDF baseline. Refuse to remove stopwords for sentiment analysis (negations carry signal). Flag class imbalance as needing more than a vectorizer change.

Example input: "Classifying 30k customer support tickets into 12 categories. Most tickets are 2-3 sentences. English only. Need explainability for audit logs."

Example output:

- Representation: TF-IDF. 30k examples is not small; explainability requirement rules out dense embeddings.
- Config: `TfidfVectorizer(ngram_range=(1, 2), min_df=3, max_df=0.95, sublinear_tf=True, stop_words=None)`. Keep stopwords because category keywords sometimes are stopwords ("not working" vs "working").
- Failure to test: verify `min_df=3` does not drop rare category keywords. Run `get_feature_names_out` filtered by class and eyeball.
```

## Упражнения

1. **Легко.** Реализуйте `cosine_similarity(doc_vec_a, doc_vec_b)` на L2-normalized TF-IDF output. Проверьте, что идентичные документы дают score 1.0, а документы с непересекающимся словарем дают 0.0.
2. **Средне.** Добавьте поддержку `n-gram` в `bag_of_words`. Параметр `n` создает counts over `n`-grams. Проверьте, что `n=2` на `["the", "cat", "sat"]` создает bigram counts для `["the cat", "cat sat"]`.
3. **Сложно.** Постройте TF-IDF-weighted-embedding hybrid выше, используя GloVe 100d vectors (download once, cache). Сравните classification accuracy с plain TF-IDF и plain mean-pooled embeddings на датасете 20 Newsgroups. Сообщите, что где побеждает.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| BoW | Word frequency vector | Счетчики слов словаря в одном документе. Отбрасывает порядок. |
| TF | Term frequency | Счетчик слова в документе, опционально нормализованный на длину документа. |
| DF | Document frequency | Количество документов, содержащих слово хотя бы один раз. |
| IDF | Inverse document frequency | Сглаженный `log(N / df)`. Уменьшает вес слов, которые встречаются везде. |
| Sparse vector | Mostly zeros | Словарь обычно содержит 10k-100k слов; большинство отсутствует в конкретном документе. |
| Cosine similarity | Vector angle | Dot product L2-normalized vectors. 1 — идентичны, 0 — ортогональны. |

## Дополнительное чтение

- [scikit-learn — feature extraction from text](https://scikit-learn.org/stable/modules/feature_extraction.html#text-feature-extraction) — каноническая API reference, плюс notes on every knob.
- [Salton, G., & Buckley, C. (1988). Term-weighting approaches in automatic text retrieval](https://www.sciencedirect.com/science/article/pii/0306457388900210) — статья, которая сделала TF-IDF default на десятилетие.
- ["Why TF-IDF Still Beats Embeddings" — Ashfaque Thonikkadavan (Medium)](https://medium.com/@cmtwskb/why-tf-idf-still-beats-embeddings-ad85c123e1b2) — взгляд 2026 года на то, когда старый метод выигрывает и почему.
