# Конструирование и отбор признаков

> Хороший признак стоит тысячи точек данных.

**Тип:** Практика
**Языки:** Python
**Требования:** Фаза 1 (статистика для ML, линейная алгебра), Фаза 2 Уроки 1-7
**Время:** ~90 минут

## Цели обучения

- Реализовать числовые преобразования (стандартизация, min-max scaling, log transform, binning) и объяснить, когда каждое уместно
- Построить one-hot, label и target encoding для категориальных признаков и определить риск утечки данных в target encoding
- Собрать TF-IDF-векторизатор с нуля и объяснить, почему он лучше сырых счетчиков слов для классификации текста
- Применить filter-based feature selection (variance threshold, correlation, mutual information), чтобы снизить размерность

## Проблема

У вас есть набор данных. Вы выбираете алгоритм. Обучаете его. Результаты посредственные. Пробуете более сложный алгоритм. Все еще посредственно. Тратите неделю на подбор гиперпараметров. Небольшое улучшение.

Затем кто-то преобразует сырые данные в более хорошие признаки, и простая логистическая регрессия обгоняет ваш настроенный ансамбль градиентного бустинга.

Так происходит постоянно. В классическом ML представление данных важнее выбора алгоритма. Модель цены дома с признаками «площадь» и «число спален» обгонит модель с признаком «адрес как сырая строка», каким бы сложным ни был learner. Алгоритм может работать только с тем, что вы ему дали.

Feature engineering — процесс преобразования сырых данных в представления, где моделям легче находить паттерны. Feature selection — процесс удаления признаков, которые добавляют шум без сигнала. Вместе это самая высокоэффективная деятельность в классическом ML.

## Концепция

### Pipeline признаков

```mermaid
flowchart LR
    A[Сырые данные] --> B[Обработать пропуски]
    B --> C[Числовые преобразования]
    B --> D[Кодирование категорий]
    B --> E[Текстовые признаки]
    C --> F[Взаимодействия признаков]
    D --> F
    E --> F
    F --> G[Отбор признаков]
    G --> H[Данные, готовые для модели]
```

### Числовые признаки

Сырые числа редко готовы для модели. Частые преобразования:

**Масштабирование:** приводит признаки к одному диапазону, чтобы distance-based алгоритмы (K-Means, KNN, SVM) относились ко всем признакам одинаково. Min-max scaling отображает в [0, 1]. Стандартизация (z-score) отображает к mean=0, std=1.

**Логарифмическое преобразование:** сжимает распределения с правой асимметрией (доход, население, счетчики слов). Превращает мультипликативные зависимости в аддитивные.

**Binning:** превращает непрерывные значения в категории. Полезно, когда зависимость между признаком и target нелинейная, но ступенчатая (например, возрастные группы).

**Полиномиальные признаки:** создает члены x^2, x^3, x1*x2. Позволяет линейным моделям улавливать нелинейные зависимости ценой роста числа признаков.

### Категориальные признаки

Моделям нужны числа. Категории нужно кодировать.

**One-hot encoding:** создает бинарный столбец для каждой категории. `color = red/blue/green` превращается в три столбца: `is_red`, `is_blue`, `is_green`. Хорошо работает для признаков с низкой кардинальностью, но взрывается при большом числе категорий.

**Label encoding:** отображает каждую категорию в целое число: red=0, blue=1, green=2. Вводит ложный порядок (модель может подумать, что green > blue > red). Уместно только для моделей на деревьях, которые делят по отдельным значениям.

**Target encoding:** заменяет каждую категорию средним значением целевой переменной для этой категории. Мощно, но опасно: высокий риск data leakage. Нужно вычислять только на обучающих данных и применять к тестовым.

### Текстовые признаки

**Count vectorizer:** считает, сколько раз каждое слово встречается в документе. «the cat sat on the mat» превращается в {the: 2, cat: 1, sat: 1, on: 1, mat: 1}.

**TF-IDF:** Term Frequency-Inverse Document Frequency. Взвешивает слова по тому, насколько они уникальны среди документов. Частые слова вроде "the" получают малый вес. Редкие отличительные слова получают высокий вес.

```
TF(word, doc) = count(word in doc) / total words in doc
IDF(word) = log(total docs / docs containing word)
TF-IDF = TF * IDF
```

### Пропущенные значения

В реальных данных есть дыры. Стратегии:

- **Удалить строки:** только когда пропусков мало и они случайны
- **Mean/median imputation:** просто, сохраняет форму распределения (медиана устойчивее к выбросам)
- **Mode imputation:** для категориальных признаков
- **Indicator column:** добавить бинарный столбец `was_this_missing` перед заполнением. Сам факт пропуска может быть информативен
- **Forward/backward fill:** для временных рядов

### Взаимодействия признаков

Иногда зависимость находится в комбинации. «Рост» и «вес» по отдельности менее предсказательны, чем «BMI = weight / height^2». Feature interactions умножают пространство признаков, поэтому используйте доменное знание, чтобы выбирать правильные.

### Отбор признаков

Больше признаков — не всегда лучше. Нерелевантные признаки добавляют шум, увеличивают время обучения и могут вызывать переобучение.

**Filter methods (до модели):**
- Correlation: удалить признаки, сильно коррелирующие друг с другом (избыточные)
- Mutual information: измеряет, насколько знание признака снижает неопределенность о target
- Variance threshold: удалить признаки, которые почти не меняются

**Wrapper methods (model-based):**
- L1 regularization (Lasso): приводит веса нерелевантных признаков ровно к нулю
- Recursive feature elimination: обучить, удалить наименее важный признак, повторить

**Почему selection важен:** модель с 10 хорошими признаками обычно обгонит модель с 10 хорошими и 90 шумовыми. Шумовые признаки дают модели возможность переобучиться на паттерны обучающих данных, которые не обобщаются.

## Соберите это

### Шаг 1: числовые преобразования с нуля

```python
import math


def min_max_scale(values):
    min_val = min(values)
    max_val = max(values)
    if max_val == min_val:
        return [0.0] * len(values)
    return [(v - min_val) / (max_val - min_val) for v in values]


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    variance = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(variance) if variance > 0 else 1.0
    return [(v - mean) / std for v in values]


def log_transform(values):
    return [math.log(v + 1) for v in values]


def bin_values(values, n_bins=5):
    min_val = min(values)
    max_val = max(values)
    bin_width = (max_val - min_val) / n_bins
    if bin_width == 0:
        return [0] * len(values)
    result = []
    for v in values:
        bin_idx = int((v - min_val) / bin_width)
        bin_idx = min(bin_idx, n_bins - 1)
        result.append(bin_idx)
    return result


def polynomial_features(row, degree=2):
    n = len(row)
    result = list(row)
    if degree >= 2:
        for i in range(n):
            result.append(row[i] ** 2)
        for i in range(n):
            for j in range(i + 1, n):
                result.append(row[i] * row[j])
    return result
```

### Шаг 2: категориальное кодирование с нуля

```python
def one_hot_encode(values):
    categories = sorted(set(values))
    cat_to_idx = {cat: i for i, cat in enumerate(categories)}
    n_cats = len(categories)

    encoded = []
    for v in values:
        row = [0] * n_cats
        row[cat_to_idx[v]] = 1
        encoded.append(row)

    return encoded, categories


def label_encode(values):
    categories = sorted(set(values))
    cat_to_int = {cat: i for i, cat in enumerate(categories)}
    return [cat_to_int[v] for v in values], cat_to_int


def target_encode(feature_values, target_values, smoothing=10):
    global_mean = sum(target_values) / len(target_values)

    category_stats = {}
    for feat, target in zip(feature_values, target_values):
        if feat not in category_stats:
            category_stats[feat] = {"sum": 0.0, "count": 0}
        category_stats[feat]["sum"] += target
        category_stats[feat]["count"] += 1

    encoding = {}
    for cat, stats in category_stats.items():
        cat_mean = stats["sum"] / stats["count"]
        weight = stats["count"] / (stats["count"] + smoothing)
        encoding[cat] = weight * cat_mean + (1 - weight) * global_mean

    return [encoding[v] for v in feature_values], encoding
```

### Шаг 3: текстовые признаки с нуля

```python
def count_vectorize(documents):
    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    vectors = []
    for doc in documents:
        vec = [0] * len(vocab)
        for word in doc.lower().split():
            vec[vocab[word]] += 1
        vectors.append(vec)

    return vectors, vocab


def tfidf(documents):
    n_docs = len(documents)

    vocab = {}
    idx = 0
    for doc in documents:
        for word in doc.lower().split():
            if word not in vocab:
                vocab[word] = idx
                idx += 1

    doc_freq = {}
    for doc in documents:
        seen = set()
        for word in doc.lower().split():
            if word not in seen:
                doc_freq[word] = doc_freq.get(word, 0) + 1
                seen.add(word)

    vectors = []
    for doc in documents:
        words = doc.lower().split()
        word_count = len(words)
        tf_map = {}
        for word in words:
            tf_map[word] = tf_map.get(word, 0) + 1

        vec = [0.0] * len(vocab)
        for word, count in tf_map.items():
            tf = count / word_count
            idf = math.log(n_docs / doc_freq[word])
            vec[vocab[word]] = tf * idf
        vectors.append(vec)

    return vectors, vocab
```

### Шаг 4: заполнение пропусков с нуля

```python
def impute_mean(values):
    present = [v for v in values if v is not None]
    if not present:
        return [0.0] * len(values), 0.0
    mean = sum(present) / len(present)
    return [v if v is not None else mean for v in values], mean


def impute_median(values):
    present = sorted(v for v in values if v is not None)
    if not present:
        return [0.0] * len(values), 0.0
    n = len(present)
    if n % 2 == 0:
        median = (present[n // 2 - 1] + present[n // 2]) / 2
    else:
        median = present[n // 2]
    return [v if v is not None else median for v in values], median


def impute_mode(values):
    present = [v for v in values if v is not None]
    if not present:
        return values, None
    counts = {}
    for v in present:
        counts[v] = counts.get(v, 0) + 1
    mode = max(counts, key=counts.get)
    return [v if v is not None else mode for v in values], mode


def add_missing_indicator(values):
    return [0 if v is not None else 1 for v in values]
```

### Шаг 5: отбор признаков с нуля

```python
def correlation(x, y):
    n = len(x)
    mean_x = sum(x) / n
    mean_y = sum(y) / n
    cov = sum((xi - mean_x) * (yi - mean_y) for xi, yi in zip(x, y)) / n
    std_x = math.sqrt(sum((xi - mean_x) ** 2 for xi in x) / n)
    std_y = math.sqrt(sum((yi - mean_y) ** 2 for yi in y) / n)
    if std_x == 0 or std_y == 0:
        return 0.0
    return cov / (std_x * std_y)


def mutual_information(feature, target, n_bins=10):
    feat_min = min(feature)
    feat_max = max(feature)
    bin_width = (feat_max - feat_min) / n_bins if feat_max != feat_min else 1.0
    feat_binned = [
        min(int((f - feat_min) / bin_width), n_bins - 1) for f in feature
    ]

    n = len(feature)
    target_classes = sorted(set(target))

    feat_bins = sorted(set(feat_binned))
    p_feat = {}
    for b in feat_bins:
        p_feat[b] = feat_binned.count(b) / n

    p_target = {}
    for t in target_classes:
        p_target[t] = target.count(t) / n

    mi = 0.0
    for b in feat_bins:
        for t in target_classes:
            joint_count = sum(
                1 for fb, tv in zip(feat_binned, target) if fb == b and tv == t
            )
            p_joint = joint_count / n
            if p_joint > 0:
                mi += p_joint * math.log(p_joint / (p_feat[b] * p_target[t]))

    return mi


def variance_threshold(features, threshold=0.01):
    n_features = len(features[0])
    n_samples = len(features)
    selected = []

    for j in range(n_features):
        col = [features[i][j] for i in range(n_samples)]
        mean = sum(col) / n_samples
        var = sum((v - mean) ** 2 for v in col) / n_samples
        if var >= threshold:
            selected.append(j)

    return selected


def remove_correlated(features, threshold=0.9):
    n_features = len(features[0])
    n_samples = len(features)

    to_remove = set()
    for i in range(n_features):
        if i in to_remove:
            continue
        col_i = [features[r][i] for r in range(n_samples)]
        for j in range(i + 1, n_features):
            if j in to_remove:
                continue
            col_j = [features[r][j] for r in range(n_samples)]
            corr = abs(correlation(col_i, col_j))
            if corr >= threshold:
                to_remove.add(j)

    return [i for i in range(n_features) if i not in to_remove]
```

### Шаг 6: полный pipeline и demo

```python
import random


def make_housing_data(n=200, seed=42):
    random.seed(seed)
    data = []
    for _ in range(n):
        sqft = random.uniform(500, 5000)
        bedrooms = random.choice([1, 2, 3, 4, 5])
        age = random.uniform(0, 50)
        neighborhood = random.choice(["downtown", "suburbs", "rural"])
        has_pool = random.choice([True, False])

        sqft_with_missing = sqft if random.random() > 0.05 else None
        age_with_missing = age if random.random() > 0.08 else None

        price = (
            50 * sqft
            + 20000 * bedrooms
            - 1000 * age
            + (50000 if neighborhood == "downtown" else 10000 if neighborhood == "suburbs" else 0)
            + (15000 if has_pool else 0)
            + random.gauss(0, 20000)
        )

        data.append({
            "sqft": sqft_with_missing,
            "bedrooms": bedrooms,
            "age": age_with_missing,
            "neighborhood": neighborhood,
            "has_pool": has_pool,
            "price": price,
        })
    return data


if __name__ == "__main__":
    data = make_housing_data(200)

    print("=== Raw Data Sample ===")
    for row in data[:3]:
        print(f"  {row}")

    sqft_raw = [d["sqft"] for d in data]
    age_raw = [d["age"] for d in data]
    prices = [d["price"] for d in data]

    print("\n=== Missing Value Handling ===")
    sqft_missing = sum(1 for v in sqft_raw if v is None)
    age_missing = sum(1 for v in age_raw if v is None)
    print(f"  sqft missing: {sqft_missing}/{len(sqft_raw)}")
    print(f"  age missing: {age_missing}/{len(age_raw)}")

    sqft_indicator = add_missing_indicator(sqft_raw)
    age_indicator = add_missing_indicator(age_raw)
    sqft_imputed, sqft_fill = impute_median(sqft_raw)
    age_imputed, age_fill = impute_mean(age_raw)
    print(f"  sqft filled with median: {sqft_fill:.0f}")
    print(f"  age filled with mean: {age_fill:.1f}")

    print("\n=== Numerical Transforms ===")
    sqft_scaled = standardize(sqft_imputed)
    age_scaled = min_max_scale(age_imputed)
    sqft_log = log_transform(sqft_imputed)
    age_binned = bin_values(age_imputed, n_bins=5)
    print(f"  sqft standardized: mean={sum(sqft_scaled)/len(sqft_scaled):.4f}, std={math.sqrt(sum(v**2 for v in sqft_scaled)/len(sqft_scaled)):.4f}")
    print(f"  age min-max: [{min(age_scaled):.2f}, {max(age_scaled):.2f}]")
    print(f"  age bins: {sorted(set(age_binned))}")

    print("\n=== Categorical Encoding ===")
    neighborhoods = [d["neighborhood"] for d in data]

    ohe, ohe_cats = one_hot_encode(neighborhoods)
    print(f"  One-hot categories: {ohe_cats}")
    print(f"  Sample encoding: {neighborhoods[0]} -> {ohe[0]}")

    le, le_map = label_encode(neighborhoods)
    print(f"  Label encoding map: {le_map}")

    te, te_map = target_encode(neighborhoods, prices, smoothing=10)
    print(f"  Target encoding: {({k: round(v) for k, v in te_map.items()})}")

    print("\n=== Text Features ===")
    descriptions = [
        "large modern house with pool",
        "small cozy cottage near downtown",
        "spacious family home with large yard",
        "modern apartment downtown with view",
        "rustic cabin in rural area",
    ]
    cv, cv_vocab = count_vectorize(descriptions)
    print(f"  Vocabulary size: {len(cv_vocab)}")
    print(f"  Doc 0 non-zero features: {sum(1 for v in cv[0] if v > 0)}")

    tf, tf_vocab = tfidf(descriptions)
    print(f"  TF-IDF vocabulary size: {len(tf_vocab)}")
    top_words = sorted(tf_vocab.keys(), key=lambda w: tf[0][tf_vocab[w]], reverse=True)[:3]
    print(f"  Doc 0 top TF-IDF words: {top_words}")

    print("\n=== Polynomial Features ===")
    sample_row = [sqft_scaled[0], age_scaled[0]]
    poly = polynomial_features(sample_row, degree=2)
    print(f"  Input: {[round(v, 4) for v in sample_row]}")
    print(f"  Polynomial: {[round(v, 4) for v in poly]}")
    print(f"  Features: [x1, x2, x1^2, x2^2, x1*x2]")

    print("\n=== Feature Selection ===")
    feature_matrix = [
        [sqft_scaled[i], age_scaled[i], float(sqft_indicator[i]), float(age_indicator[i])]
        + ohe[i]
        for i in range(len(data))
    ]

    print(f"  Total features: {len(feature_matrix[0])}")

    surviving_var = variance_threshold(feature_matrix, threshold=0.01)
    print(f"  After variance threshold (0.01): {len(surviving_var)} features kept")

    surviving_corr = remove_correlated(feature_matrix, threshold=0.9)
    print(f"  After correlation filter (0.9): {len(surviving_corr)} features kept")

    binary_prices = [1 if p > sum(prices) / len(prices) else 0 for p in prices]
    print("\n  Mutual information with target:")
    feature_names = ["sqft", "age", "sqft_missing", "age_missing"] + [f"neigh_{c}" for c in ohe_cats]
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        mi = mutual_information(col, binary_prices, n_bins=10)
        print(f"    {feature_names[j]}: MI={mi:.4f}")

    print("\n  Correlation with price:")
    for j in range(len(feature_matrix[0])):
        col = [feature_matrix[i][j] for i in range(len(feature_matrix))]
        corr = correlation(col, prices)
        print(f"    {feature_names[j]}: r={corr:.4f}")
```

## Используйте это

Со scikit-learn эти преобразования собираются в composable pipelines:

```python
from sklearn.preprocessing import StandardScaler, OneHotEncoder, PolynomialFeatures
from sklearn.impute import SimpleImputer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.feature_selection import mutual_info_classif, VarianceThreshold
from sklearn.compose import ColumnTransformer
from sklearn.pipeline import Pipeline

numeric_pipe = Pipeline([
    ("imputer", SimpleImputer(strategy="median")),
    ("scaler", StandardScaler()),
])

categorical_pipe = Pipeline([
    ("encoder", OneHotEncoder(sparse_output=False)),
])

preprocessor = ColumnTransformer([
    ("num", numeric_pipe, ["sqft", "age"]),
    ("cat", categorical_pipe, ["neighborhood"]),
])
```

Реализации с нуля показывают, что именно происходит внутри каждого transform. Библиотечные версии добавляют обработку пограничных случаев, поддержку sparse matrices и композицию pipeline, но математика та же.

## Доведите до результата

Этот урок создает:
- `outputs/prompt-feature-engineer.md` — промпт для систематического конструирования признаков из сырых данных

## Упражнения

1. Добавьте robust scaling (с использованием медианы и interquartile range вместо среднего и стандартного отклонения) к числовым преобразованиям. Сравните его со standard scaling на данных с экстремальными выбросами.
2. Реализуйте leave-one-out target encoding: для каждой строки вычисляйте средний target без target-значения этой же строки. Покажите, как это снижает переобучение по сравнению с naive target encoding.
3. Постройте автоматический pipeline отбора признаков, объединяющий variance threshold, correlation filtering и mutual information ranking. Примените его к housing dataset и сравните качество модели (используйте простую линейную регрессию) на всех признаках и выбранных признаках.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|--------|-------------|------------------------------|
| Feature engineering | «Делать новые столбцы» | Преобразование сырых данных в представления, раскрывающие паттерны для модели |
| Стандартизация | «Сделать нормальным» | Вычитание среднего и деление на стандартное отклонение, чтобы признак имел mean=0 и std=1 |
| One-hot encoding | «Сделать dummy variables» | Создание одного бинарного столбца на категорию, где ровно один столбец равен 1 для каждой строки |
| Target encoding | «Кодировать через ответ» | Замена каждой категории средним target-значением для этой категории со сглаживанием для предотвращения переобучения |
| TF-IDF | «Модные счетчики слов» | Term Frequency умноженная на Inverse Document Frequency: слова взвешиваются по их отличительности в корпусе |
| Imputation | «Заполнение пустот» | Замена пропущенных значений оцененными значениями (средним, медианой, модой или предсказанием модели) |
| Feature selection | «Выкидывание плохих столбцов» | Удаление признаков, добавляющих шум или избыточность, с сохранением только тех, где есть сигнал о target |
| Mutual information | «Насколько одно говорит о другом» | Мера уменьшения неопределенности о переменной Y при наблюдении переменной X |
| Data leakage | «Случайное жульничество» | Использование при обучении информации, недоступной во время предсказания, что дает ложно оптимистичные результаты |

## Дополнительное чтение

- [Feature Engineering and Selection (Max Kuhn & Kjell Johnson)](http://www.feat.engineering/) — бесплатная онлайн-книга, покрывающая всю область feature engineering
- [scikit-learn Preprocessing Guide](https://scikit-learn.org/stable/modules/preprocessing.html) — практический справочник по стандартным преобразованиям
- [Target Encoding Done Right (Micci-Barreca, 2001)](https://dl.acm.org/doi/10.1145/507533.507538) — оригинальная статья о target encoding со сглаживанием
