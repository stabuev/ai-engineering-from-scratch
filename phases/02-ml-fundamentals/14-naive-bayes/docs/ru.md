# Наивный Байес

> «Наивное» предположение неверно, но все равно работает. В этом его красота.

**Тип:** Практика
**Язык:** Python
**Требования:** Фаза 2, Уроки 01-07 (классификация, теорема Байеса)
**Время:** ~75 минут

## Цели обучения

- Реализовать Multinomial Naive Bayes с нуля с Laplace smoothing для классификации текста
- Объяснить, почему наивное предположение независимости математически неверно, но на практике дает правильное ранжирование классов
- Сравнить варианты Multinomial, Bernoulli и Gaussian Naive Bayes и выбрать подходящий для заданного типа признаков
- Оценить Naive Bayes относительно logistic regression на высокоразмерных разреженных данных и объяснить bias-variance tradeoff

## Проблема

Вам нужно классифицировать текст. Письма — на spam или not-spam. Отзывы клиентов — на positive или negative. Support tickets — по категориям. У вас тысячи признаков (по одному на слово) и ограниченные training data.

Большинство классификаторов здесь задыхается. Logistic regression нужны достаточные samples, чтобы надежно оценить тысячи weights. Decision trees делят по одному слову за раз и сильно переобучаются. KNN в 10 000 измерениях бессмысленен, потому что каждая точка примерно одинаково далека от любой другой.

Naive Bayes справляется. Он делает математически неверное предположение (что каждый признак независим от каждого другого при условии класса), но все равно превосходит «более умные» модели на text classification, особенно с малыми training sets. Он обучается за один проход по данным. Масштабируется до миллионов признаков. Выдает probability estimates (хотя часто плохо calibrated из-за assumption independence).

Понимание того, почему неверное предположение дает хорошие predictions, учит фундаментальной вещи о machine learning: лучшая модель — не самая «правильная», а та, у которой лучший bias-variance tradeoff для ваших данных.

## Концепция

### Теорема Байеса (быстрый повтор)

Теорема Байеса переворачивает условные вероятности:

```
P(class | features) = P(features | class) * P(class) / P(features)
```

Нам нужна `P(class | features)` — вероятность, что документ принадлежит классу при данных словах. Ее можно вычислить из:
- `P(features | class)` — likelihood увидеть эти слова в документах этого класса
- `P(class)` — prior probability класса (насколько spam распространен вообще?)
- `P(features)` — evidence, одинаковый для всех классов, поэтому при сравнении его можно игнорировать

Побеждает класс с максимальной `P(class | features)`.

### Наивное предположение независимости

Точное вычисление `P(features | class)` требует оценки joint probability всех признаков вместе. При vocabulary из 10 000 слов нужно оценить распределение по 2^10,000 возможных combinations. Невозможно.

Наивное предположение: каждый признак условно независим при заданном классе.

```
P(w1, w2, ..., wn | class) = P(w1 | class) * P(w2 | class) * ... * P(wn | class)
```

Вместо одного невозможного joint distribution вы оцениваете n простых per-feature distributions. Каждому нужен только count.

Это предположение очевидно неверно. Слова "machine" и "learning" не независимы ни в каком документе. Но классификатору не нужны корректные probability estimates. Ему нужно корректное ranking — у какого класса probability максимальна. Independence assumption вносит систематические ошибки, но они похожим образом влияют на все классы, поэтому ranking остается правильным.

### Почему это все равно работает

Три причины:

1. **Ranking over calibration.** Классификации нужно только, чтобы top-ranked class был верным. Даже если P(spam) = 0.99999 при истинной вероятности 0.7, классификатор все равно правильно выбирает spam. Нам не нужны точные probabilities. Нам нужен правильный winner.

2. **High bias, low variance.** Independence assumption — сильный prior. Он жестко ограничивает модель и предотвращает overfitting. При ограниченных training data модель, которая слегка неверна, но стабильна, лучше теоретически правильной, но wildly unstable. Это bias-variance tradeoff в действии.

3. **Feature redundancy cancels out.** Коррелированные признаки дают redundant evidence. Классификатор double-counts эту evidence, но он double-counts ее и для правильного класса. Если "machine" и "learning" всегда появляются вместе, оба дают evidence за "tech" class. NB считает их дважды, но считает дважды за правильный класс.

Четвертая практическая причина: Naive Bayes крайне быстр. Training — один проход по данным с подсчетом frequencies. Prediction — matrix multiplication. Можно обучаться на миллионе документов за секунды. Эта скорость позволяет быстрее итерировать, пробовать больше feature sets и запускать больше experiments, чем с медленными моделями.

### Математика шаг за шагом

Разберем конкретный пример. Пусть есть два класса: spam и not-spam. Vocabulary содержит три слова: "free", "money", "meeting".

Training data:
- Spam emails упоминают "free" 80 раз, "money" 60 раз, "meeting" 10 раз (150 total words)
- Not-spam emails упоминают "free" 5 раз, "money" 10 раз, "meeting" 100 раз (115 total words)
- 40% emails — spam, 60% — not-spam

С Laplace smoothing (alpha=1):

```
P(free | spam)    = (80 + 1) / (150 + 3) = 81/153 = 0.529
P(money | spam)   = (60 + 1) / (150 + 3) = 61/153 = 0.399
P(meeting | spam) = (10 + 1) / (150 + 3) = 11/153 = 0.072

P(free | not-spam)    = (5 + 1) / (115 + 3) = 6/118 = 0.051
P(money | not-spam)   = (10 + 1) / (115 + 3) = 11/118 = 0.093
P(meeting | not-spam) = (100 + 1) / (115 + 3) = 101/118 = 0.856
```

Новое email содержит: "free" (2 раза), "money" (1 раз), "meeting" (0 раз).

```
log P(spam | email) = log(0.4) + 2*log(0.529) + 1*log(0.399) + 0*log(0.072)
                    = -0.916 + 2*(-0.637) + (-0.919) + 0
                    = -3.109

log P(not-spam | email) = log(0.6) + 2*log(0.051) + 1*log(0.093) + 0*log(0.856)
                        = -0.511 + 2*(-2.976) + (-2.375) + 0
                        = -8.838
```

Spam побеждает с большим отрывом. Слово "free", встречающееся дважды, — сильное evidence за spam. Заметьте, что отсутствие "meeting" дает нулевой вклад в обе log sums (0 * log(P)) — в Multinomial NB отсутствующие слова не влияют. Bernoulli NB явно моделирует отсутствие слов.

### Три варианта

Naive Bayes бывает в трех вариантах. Каждый по-своему моделирует `P(feature | class)`.

#### Multinomial Naive Bayes

Моделирует каждый признак как count. Лучше всего подходит для text data, где признаки — word frequencies или TF-IDF values.

```
P(word_i | class) = (count of word_i in class + alpha) / (total words in class + alpha * vocab_size)
```

`alpha` — Laplace smoothing (объясняется ниже). Это рабочая лошадка text classification.

#### Gaussian Naive Bayes

Моделирует каждый признак нормальным распределением. Лучше всего подходит для continuous features.

```
P(x_i | class) = (1 / sqrt(2 * pi * var)) * exp(-(x_i - mean)^2 / (2 * var))
```

Для каждого класса оцениваются mean и variance по каждому признаку. Это хорошо работает, когда признаки внутри каждого класса действительно похожи на bell curve.

#### Bernoulli Naive Bayes

Моделирует каждый признак как binary (present or absent). Лучше всего подходит для коротких текстов или binary feature vectors.

```
P(word_i | class) = (docs in class containing word_i + alpha) / (total docs in class + 2 * alpha)
```

В отличие от Multinomial, Bernoulli явно штрафует отсутствие слова. Если "free" обычно встречается в spam, но в этом email отсутствует, Bernoulli считает это evidence против spam.

### Когда использовать какой вариант

| Вариант | Тип признаков | Лучше всего для | Пример |
|---------|---------------|-----------------|--------|
| Multinomial | Counts или frequencies | Text classification, bag-of-words | Email spam, topic classification |
| Gaussian | Continuous values | Табличные данные с roughly normal features | Iris classification, sensor data |
| Bernoulli | Binary (0/1) | Короткий текст, binary feature vectors | SMS spam, presence/absence features |

### Laplace Smoothing

Что происходит, когда слово встречается в test data, но никогда не встречалось в training data для конкретного класса?

Без smoothing: `P(word | class) = 0/N = 0`. Один ноль, умноженный на весь product, делает `P(class | features) = 0` независимо от всех остальных evidence. Одно unseen word уничтожает prediction, каким бы сильным ни было остальное evidence.

Laplace smoothing добавляет небольшой count `alpha` (обычно 1) к каждому feature count:

```
P(word_i | class) = (count(word_i, class) + alpha) / (total_words_in_class + alpha * vocab_size)
```

При alpha=1 каждое слово получает хотя бы крошечную probability. Слово "discombobulate" в test email больше не убивает spam probability. У smoothing есть Bayesian interpretation: он эквивалентен uniform Dirichlet prior на word distributions.

Более высокий alpha означает более сильное smoothing (более uniform distributions). Более низкий alpha означает, что модель сильнее доверяет data. Alpha — гиперпараметр, который нужно подбирать.

Эффект alpha:

| Alpha | Эффект | Когда использовать |
|-------|--------|--------------------|
| 0.001 | Почти нет smoothing, доверяем data | Очень большой training set, unseen features не ожидаются |
| 0.1 | Легкое smoothing | Большой training set |
| 1.0 | Стандартное Laplace smoothing | Стартовый default |
| 10.0 | Сильное smoothing, выравнивает distributions | Очень маленький training set, ожидается много unseen features |

### Вычисления в log-space

Перемножение сотен probabilities (каждая меньше 1) вызывает floating-point underflow. Product становится нулем в floating point, хотя истинное значение — очень маленькое положительное число.

Решение: работать в log space. Вместо перемножения probabilities складывать их логарифмы:

```
log P(class | x1, x2, ..., xn) = log P(class) + sum_i log P(xi | class)
```

Это превращает prediction в dot product:

```
log_scores = X @ log_feature_probs.T + log_class_priors
prediction = argmax(log_scores)
```

Matrix multiplication. Поэтому prediction в Naive Bayes такой быстрый — это та же операция, что и в single-layer linear model.

### Naive Bayes vs Logistic Regression

Оба являются linear classifiers для текста. Отличие в том, что они моделируют.

| Аспект | Naive Bayes | Logistic Regression |
|--------|-------------|---------------------|
| Тип | Generative (models P(X\|Y)) | Discriminative (models P(Y\|X)) |
| Training | Count frequencies | Optimize loss function |
| Small data | Лучше (strong prior помогает) | Хуже (недостаточно для оценки weights) |
| Large data | Хуже (wrong assumption hurts) | Лучше (flexible boundary) |
| Features | Assumes independence | Handles correlations |
| Speed | Single pass, очень быстро | Iterative optimization |
| Calibration | Poor probabilities | Better probabilities |

Правило: начинайте с Naive Bayes. Если данных достаточно и NB вышел на плато, переходите к logistic regression.

### Classification Pipeline

```mermaid
flowchart LR
    A[Raw Text] --> B[Tokenize]
    B --> C[Build Vocabulary]
    C --> D[Count Word Frequencies]
    D --> E[Apply Smoothing]
    E --> F[Compute Log Probabilities]
    F --> G[Predict: argmax P class given words]

    style A fill:#f9f,stroke:#333
    style G fill:#9f9,stroke:#333
```

На практике мы работаем в log space, чтобы избежать floating-point underflow. Вместо перемножения множества маленьких probabilities складываем их логарифмы:

```
log P(class | features) = log P(class) + sum_i log P(feature_i | class)
```

## Соберите это

Код в `code/naive_bayes.py` реализует MultinomialNB и GaussianNB с нуля.

### MultinomialNB

Реализация с нуля:

1. **fit(X, y)**: для каждого класса посчитать frequency каждого feature. Добавить Laplace smoothing. Вычислить log probabilities. Сохранить class priors (log class frequencies).

2. **predict_log_proba(X)**: для каждого sample вычислить log P(class) + sum log P(feature_i | class) для всех classes. Это matrix multiplication: X @ log_probs.T + log_priors.

3. **predict(X)**: вернуть class с максимальной log probability.

```python
class MultinomialNB:
    def __init__(self, alpha=1.0):
        self.alpha = alpha

    def fit(self, X, y):
        classes = np.unique(y)
        n_classes = len(classes)
        n_features = X.shape[1]

        self.classes_ = classes
        self.class_log_prior_ = np.zeros(n_classes)
        self.feature_log_prob_ = np.zeros((n_classes, n_features))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.class_log_prior_[i] = np.log(X_c.shape[0] / X.shape[0])
            counts = X_c.sum(axis=0) + self.alpha
            self.feature_log_prob_[i] = np.log(counts / counts.sum())

        return self
```

Ключевая идея: после fitting prediction — это просто matrix multiplication плюс bias. Поэтому Naive Bayes такой быстрый.

### GaussianNB

Для continuous features оцениваем mean и variance по class и feature:

```python
class GaussianNB:
    def __init__(self):
        pass

    def fit(self, X, y):
        classes = np.unique(y)
        self.classes_ = classes
        self.means_ = np.zeros((len(classes), X.shape[1]))
        self.vars_ = np.zeros((len(classes), X.shape[1]))
        self.priors_ = np.zeros(len(classes))

        for i, c in enumerate(classes):
            X_c = X[y == c]
            self.means_[i] = X_c.mean(axis=0)
            self.vars_[i] = X_c.var(axis=0) + 1e-9
            self.priors_[i] = X_c.shape[0] / X.shape[0]

        return self
```

Prediction использует Gaussian PDF по каждому feature, перемножая по features (складывая в log space).

### Demo: Text Classification

Код генерирует synthetic bag-of-words data, имитирующие два класса (tech articles vs sports articles). У каждого класса свое word frequency distribution. MultinomialNB классифицирует их по word counts.

Synthetic data работает так: мы создаем 200 «слов» (feature columns). Words 0-39 имеют high frequency в tech articles и low в sports. Words 80-119 имеют high frequency в sports и low в tech. Words 40-79 имеют medium frequency в обоих. Это создает реалистичный сценарий, где некоторые words — сильные class indicators, а остальные noise.

### Demo: Continuous Features

Код генерирует Iris-like data (3 classes, 4 features, Gaussian clusters). GaussianNB классифицирует по per-class mean и variance. У каждого class свой center (mean vector) и spread (variance), имитируя real-world data, где measurements систематически отличаются между categories.

Код также показывает:
- **Smoothing comparison:** training MultinomialNB с разными alpha values, чтобы увидеть влияние smoothing strength на accuracy.
- **Training size experiment:** как NB accuracy растет при увеличении training data от 20 до 1600 samples. NB достигает приличной accuracy даже с очень малым числом samples — это его главное преимущество.
- **Confusion matrix:** per-class precision, recall и F1 score, чтобы показать, где NB ошибается.

### Скорость prediction

Naive Bayes prediction — matrix multiplication. Для n samples с d features и k classes:
- MultinomialNB: одно matrix multiply (n x d) @ (d x k) = O(n * d * k)
- GaussianNB: n * k Gaussian PDF evaluations, каждая по d features = O(n * d * k)

Оба линейны по каждой размерности. Сравните это с KNN (требует distance computation до всех training points) или SVM с RBF kernel (требует kernel evaluation против всех support vectors). NB на порядки быстрее во время prediction.

## Используйте это

Со sklearn оба варианта — one-liners:

```python
from sklearn.naive_bayes import GaussianNB, MultinomialNB

gnb = GaussianNB()
gnb.fit(X_train, y_train)
print(f"GaussianNB accuracy: {gnb.score(X_test, y_test):.3f}")

mnb = MultinomialNB(alpha=1.0)
mnb.fit(X_train_counts, y_train)
print(f"MultinomialNB accuracy: {mnb.score(X_test_counts, y_test):.3f}")
```

Для text classification со sklearn:

```python
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("vectorizer", CountVectorizer()),
    ("classifier", MultinomialNB(alpha=1.0)),
])

text_clf.fit(train_texts, train_labels)
accuracy = text_clf.score(test_texts, test_labels)
```

Код в `naive_bayes.py` сравнивает implementations from scratch со sklearn на тех же данных, чтобы проверить correctness.

### TF-IDF с Naive Bayes

Сырые word counts дают каждому слову одинаковый вес на occurrence. Но частые слова вроде "the" и "is" часто встречаются в каждом class и не несут информации. TF-IDF (Term Frequency - Inverse Document Frequency) снижает вес common words и повышает вес rare, discriminative words.

```python
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.naive_bayes import MultinomialNB
from sklearn.pipeline import Pipeline

text_clf = Pipeline([
    ("tfidf", TfidfVectorizer()),
    ("classifier", MultinomialNB(alpha=0.1)),
])
```

TF-IDF values неотрицательны, поэтому работают с MultinomialNB. Комбинация TF-IDF + MultinomialNB — один из сильнейших baselines для text classification. Она часто превосходит более сложные модели на datasets с менее чем 10 000 training samples.

### BernoulliNB для короткого текста

Для коротких текстов (tweets, SMS, chat messages) BernoulliNB может превосходить MultinomialNB. В коротких текстах low word counts, поэтому frequency information, на которую опирается MultinomialNB, шумная. BernoulliNB смотрит только на presence or absence, что надежнее для короткого текста.

```python
from sklearn.naive_bayes import BernoulliNB
from sklearn.feature_extraction.text import CountVectorizer

text_clf = Pipeline([
    ("vectorizer", CountVectorizer(binary=True)),
    ("classifier", BernoulliNB(alpha=1.0)),
])
```

Флаг `binary=True` в CountVectorizer превращает все counts в 0/1. Без него BernoulliNB тоже работает, но видит counts, для которых он не проектировался.

### Калибровка NB probabilities

NB probabilities плохо calibrated. Когда NB говорит P(spam) = 0.95, истинная вероятность может быть 0.7. Если нужны надежные probability estimates (например, чтобы задать threshold или объединить с другими моделями), используйте sklearn `CalibratedClassifierCV`:

```python
from sklearn.calibration import CalibratedClassifierCV

calibrated_nb = CalibratedClassifierCV(MultinomialNB(), cv=5, method="sigmoid")
calibrated_nb.fit(X_train, y_train)
proba = calibrated_nb.predict_proba(X_test)
```

Это обучает logistic regression поверх raw scores NB с использованием cross-validation. Полученные probabilities намного ближе к истинным class frequencies.

### Частые ловушки

1. **Отрицательные значения признаков.** MultinomialNB требует неотрицательные features. Если есть negative values (например, TF-IDF с некоторыми settings или standardized features), используйте GaussianNB или сдвиньте features в positive.

2. **Zero variance features.** GaussianNB делит на variance. Если feature имеет zero variance для class (все значения одинаковы), probability computation ломается. Код добавляет маленький smoothing term (1e-9) ко всем variances.

3. **Class imbalance.** Если 99% emails not-spam, prior P(not-spam) = 0.99 настолько силен, что подавляет likelihood evidence. Можно вручную задать class priors или использовать class_prior parameter в sklearn.

4. **Feature scaling.** MultinomialNB не требует scaling (работает на counts). GaussianNB тоже не требует scaling (оценивает per-feature statistics). Это преимущество перед logistic regression и SVM, чувствительными к feature scales.

## Доведите до результата

Этот урок создает:
- `outputs/skill-naive-bayes-chooser.md` — decision skill для выбора правильного NB variant
- `code/naive_bayes.py` — MultinomialNB и GaussianNB с нуля со сравнением sklearn

### Когда Naive Bayes ломается

NB ломается, когда independence assumption вызывает неверное ranking (а не только неверные probabilities). Это происходит, когда:

1. **Сильные feature interactions.** Если class зависит от комбинации двух features, но не от каждого по отдельности (XOR-like patterns), NB полностью пропустит это. Каждый feature сам по себе не дает evidence, и NB не может нелинейно их комбинировать.

2. **Сильно коррелированные features с противоположной evidence.** Если feature A говорит "spam", а feature B говорит "not-spam", но A и B perfectly correlated (в реальности всегда согласуются), NB увидит конфликтующую evidence там, где ее нет.

3. **Очень большие training sets.** При достаточном количестве данных discriminative models вроде logistic regression учат истинную decision boundary и превосходят NB. Independence assumption, помогавшая на малых данных, теперь ограничивает модель.

На практике эти failure modes редки для text classification. Text features многочисленны, individually weak, а ошибки independence assumption обычно компенсируются. Для tabular data с небольшим числом сильно коррелированных features сначала рассмотрите logistic regression или tree-based models.

## Упражнения

1. **Smoothing experiment.** Обучите MultinomialNB на text data с alpha values 0.01, 0.1, 1.0, 10.0 и 100.0. Постройте accuracy vs alpha. Где performance достигает пика? Почему слишком высокий alpha вредит?

2. **Feature independence test.** Возьмите реальный text dataset. Выберите два явно коррелированных слова ("machine" и "learning"). Вычислите P(word1 | class) * P(word2 | class) и сравните с P(word1 AND word2 | class). Насколько ошибается independence assumption? Влияет ли это на classification accuracy?

3. **Bernoulli implementation.** Расширьте код классом BernoulliNB. Преобразуйте bag-of-words в binary (present/absent) и сравните accuracy с MultinomialNB на text data. Когда Bernoulli выигрывает?

4. **NB vs Logistic Regression.** Обучите обе модели на text data. Начните со 100 training samples и увеличивайте до 10 000. Постройте accuracy vs training set size для обеих. В какой точке Logistic Regression обгоняет Naive Bayes?

5. **Spam filter.** Постройте полный spam classifier: tokenize raw email text, build vocabulary, create bag-of-words features, train MultinomialNB, evaluate with precision and recall (not just accuracy — почему?).

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|--------|-------------|------------------------------|
| Naive Bayes | «Простой probabilistic classifier» | Классификатор, применяющий теорему Байеса с предположением, что features conditionally independent given class |
| Conditional independence | «Features не влияют друг на друга» | P(A, B \| C) = P(A \| C) * P(B \| C) — знание B не сообщает ничего нового об A, если C известно |
| Laplace smoothing | «Add-one smoothing» | Добавление малого count к каждому feature, чтобы zero probabilities не доминировали в prediction |
| Prior | «Что вы думали до данных» | P(class) — probability каждого class до наблюдения features |
| Likelihood | «Насколько data подходит» | P(features \| class) — probability наблюдать эти features, если class известен |
| Posterior | «Что вы думаете после данных» | P(class \| features) — обновленная probability class после наблюдения features |
| Generative model | «Моделирует, как генерируются данные» | Модель, изучающая P(X \| Y) и P(Y), затем использующая теорему Байеса для P(Y \| X) |
| Discriminative model | «Моделирует decision boundary» | Модель, напрямую изучающая P(Y \| X), не моделируя генерацию X |
| Log probability | «Избежать underflow» | Работа с log P вместо P, чтобы произведение многих малых чисел не стало нулем в floating point |

## Дополнительное чтение

- [scikit-learn Naive Bayes docs](https://scikit-learn.org/stable/modules/naive_bayes.html) — все три варианта с математическими деталями
- [McCallum and Nigam, A Comparison of Event Models for Naive Bayes Text Classification (1998)](https://www.cs.cmu.edu/~knigam/papers/multinomial-aaaiws98.pdf) — классическое сравнение Multinomial и Bernoulli для текста
- [Rennie et al., Tackling the Poor Assumptions of Naive Bayes Text Classifiers (2003)](https://people.csail.mit.edu/jrennie/papers/icml03-nb.pdf) — улучшения NB для текста
- [Ng and Jordan, On Discriminative vs. Generative Classifiers (2001)](https://ai.stanford.edu/~ang/papers/nips01-discriminativegenerative.pdf) — доказывает, что NB сходится быстрее LR на меньшем числе данных
