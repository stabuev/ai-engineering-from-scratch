# Оценка моделей

> Модель настолько хороша, насколько хорош способ ее измерения.

**Тип:** Практика
**Языки:** Python
**Требования:** Фаза 1 (вероятность и распределения, статистика для ML), Фаза 2 Уроки 1-8
**Время:** ~90 минут

## Цели обучения

- Реализовать K-fold и stratified K-fold cross-validation с нуля и объяснить, почему стратификация важна для несбалансированных данных
- Вычислять precision, recall, F1, AUC-ROC и регрессионные метрики (MSE, RMSE, MAE, R-squared) с нуля
- Интерпретировать learning curves, чтобы диагностировать высокое смещение или высокую дисперсию модели
- Определять частые ошибки оценки: data leakage, неправильный выбор метрики и загрязнение тестовой выборки

## Проблема

Вы обучили модель. Она получает 95% accuracy на ваших данных. Это хорошо?

Может быть. А может, нет. Если 95% данных принадлежат одному классу, модель, которая всегда предсказывает этот класс, получает 95% accuracy и при этом совершенно бесполезна. Если вы оценивали на тех же данных, на которых обучались, число 95% бессмысленно: модель просто запомнила ответы. Если в наборе данных есть временная компонента, а вы случайно перемешали данные перед разбиением, модель может использовать будущие данные, чтобы предсказывать прошлое.

Оценка моделей — место, где ломается большинство ML-проектов. Неправильная метрика делает плохую модель хорошей на вид. Неправильное разбиение позволяет модели «жульничать». Неправильное сравнение заставляет выбрать худшую модель. Правильная оценка не опциональна. Это разница между моделью, которая работает в production, и моделью, которая падает при встрече с реальными данными.

## Концепция

### Train, Validation, Test

```mermaid
flowchart LR
    A[Full Dataset] --> B[Train Set 60-70%]
    A --> C[Validation Set 15-20%]
    A --> D[Test Set 15-20%]
    B --> E[Fit Model]
    E --> C
    C --> F[Tune Hyperparameters]
    F --> E
    F --> G[Final Model]
    G --> D
    D --> H[Report Performance]
```

Три разбиения, три назначения:

- **Training set**: модель учится на этих данных. Она видит эти примеры во время обучения.
- **Validation set**: используется для настройки гиперпараметров и выбора между моделями. Модель не обучается на этих данных, но ваши решения зависят от них.
- **Test set**: используется ровно один раз, в самом конце, чтобы сообщить финальное качество. Если вы посмотрели на test performance, а затем вернулись менять модель, это уже не test set. Он стал второй validation set.

Test set — ваша hold-out гарантия, что сообщенное качество отражает поведение модели на действительно невидимых данных.

### K-Fold Cross-Validation

На малых наборах данных одно train/validation-разбиение тратит данные впустую и дает шумные оценки. K-fold cross-validation использует все данные и для обучения, и для валидации:

```mermaid
flowchart TB
    subgraph Fold1["Fold 1"]
        direction LR
        V1["Val"] --- T1a["Train"] --- T1b["Train"] --- T1c["Train"] --- T1d["Train"]
    end
    subgraph Fold2["Fold 2"]
        direction LR
        T2a["Train"] --- V2["Val"] --- T2b["Train"] --- T2c["Train"] --- T2d["Train"]
    end
    subgraph Fold3["Fold 3"]
        direction LR
        T3a["Train"] --- T3b["Train"] --- V3["Val"] --- T3c["Train"] --- T3d["Train"]
    end
    subgraph Fold4["Fold 4"]
        direction LR
        T4a["Train"] --- T4b["Train"] --- T4c["Train"] --- V4["Val"] --- T4d["Train"]
    end
    subgraph Fold5["Fold 5"]
        direction LR
        T5a["Train"] --- T5b["Train"] --- T5c["Train"] --- T5d["Train"] --- V5["Val"]
    end
    Fold1 --> R["Average scores"]
    Fold2 --> R
    Fold3 --> R
    Fold4 --> R
    Fold5 --> R
```

1. Разделить данные на K равных folds
2. Для каждого fold обучаться на K-1 folds и валидироваться на оставшемся fold
3. Усреднить K validation scores

K=5 или K=10 — стандартные варианты. Каждая точка данных используется для валидации ровно один раз. Средний score — более стабильная оценка, чем любое одиночное разбиение.

**Stratified K-fold**: сохраняет распределение классов в каждом fold. Если набор данных состоит из 70% класса A и 30% класса B, каждый fold будет иметь примерно такое же соотношение. Это важно для несбалансированных данных, где случайное разбиение может поместить все minority samples в один fold.

### Метрики классификации

**Матрица ошибок (confusion matrix)** — основа. Для бинарной классификации:

|  | Предсказан Positive | Предсказан Negative |
|--|---|---|
| Фактически Positive | True Positive (TP) | False Negative (FN) |
| Фактически Negative | False Positive (FP) | True Negative (TN) |

Из этой матрицы следуют все остальные метрики:

- **Accuracy** = (TP + TN) / (TP + TN + FP + FN). Доля правильных предсказаний. Вводит в заблуждение при дисбалансе классов.
- **Precision** = TP / (TP + FP). Среди всех предсказанных positive сколько действительно positive? Используйте, когда false positives дороги (например, спам-фильтр пометил настоящее письмо как спам).
- **Recall** (sensitivity) = TP / (TP + FN). Среди всех фактических positive сколько мы нашли? Используйте, когда false negatives дороги (например, скрининг рака пропустил опухоль).
- **F1 score** = 2 * precision * recall / (precision + recall). Гармоническое среднее precision и recall. Балансирует их, когда ни одна метрика явно не доминирует.
- **AUC-ROC**: Area Under the Receiver Operating Characteristic curve. Строит true positive rate против false positive rate при разных порогах классификации. AUC = 0.5 означает случайное угадывание, AUC = 1.0 — идеальное разделение. Метрика не зависит от порога: она измеряет, насколько хорошо модель ранжирует positive выше negative независимо от выбранной отсечки.

### Метрики регрессии

- **MSE** (Mean Squared Error) = mean((y_true - y_pred)^2). Квадратично штрафует большие ошибки. Чувствительна к выбросам.
- **RMSE** (Root Mean Squared Error) = sqrt(MSE). В тех же единицах, что целевая переменная. Проще интерпретировать, чем MSE.
- **MAE** (Mean Absolute Error) = mean(|y_true - y_pred|). Все ошибки учитывает линейно. Устойчивее к выбросам, чем MSE.
- **R-squared** = 1 - SS_res / SS_tot, где SS_res = sum((y_true - y_pred)^2), а SS_tot = sum((y_true - y_mean)^2). Доля дисперсии, объясненная моделью. R^2 = 1.0 — идеально. R^2 = 0.0 означает, что модель не лучше постоянного предсказания среднего. R^2 может быть отрицательным, если модель хуже среднего.

### Learning Curves

Постройте training и validation scores как функцию размера обучающей выборки:

- **High bias (underfitting)**: обе кривые сходятся к низкому score. Добавление данных не поможет. Нужна более сложная модель.
- **High variance (overfitting)**: training score высокий, а validation score намного ниже. Разрыв между ними большой. Добавление данных должно помочь.

### Validation Curves

Постройте training и validation scores как функцию гиперпараметра:

- При низкой сложности: оба score низкие (underfitting)
- При правильной сложности: оба score высокие и близки
- При высокой сложности: training score остается высоким, но validation score падает (overfitting)

Оптимальное значение гиперпараметра там, где validation score максимален.

### Частые ошибки оценки

**Data leakage**: информация из test set просачивается в обучение. Примеры: scaler fitted на полном наборе до разбиения, включение будущих данных в прогнозирование временных рядов, использование признака, полученного из target. Всегда сначала разделяйте, затем предобрабатывайте.

**Дисбаланс классов**: 99% транзакций легитимны, 1% — fraud. Модель, всегда предсказывающая «легитимно», получает 99% accuracy. Используйте precision, recall, F1 или AUC-ROC.

**Неправильная метрика**: оптимизация accuracy, когда нужно оптимизировать recall (медицинская диагностика), или оптимизация RMSE при тяжелых выбросах (лучше MAE).

**Нет stratified splits**: при дисбалансе случайное разбиение может поместить слишком мало minority samples в validation fold и дать нестабильные оценки.

**Слишком частое тестирование**: каждый раз, когда вы смотрите на test performance и корректируете модель, вы переобучаетесь на test set. Test set используется один раз.

## Соберите это

### Шаг 1: train/validation/test split

```python
import random
import math


def train_val_test_split(X, y, train_ratio=0.6, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    train_end = int(n * train_ratio)
    val_end = int(n * (train_ratio + val_ratio))

    train_idx = indices[:train_end]
    val_idx = indices[train_end:val_end]
    test_idx = indices[val_end:]

    X_train = [X[i] for i in train_idx]
    y_train = [y[i] for i in train_idx]
    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]
    X_test = [X[i] for i in test_idx]
    y_test = [y[i] for i in test_idx]

    return X_train, y_train, X_val, y_val, X_test, y_test
```

### Шаг 2: K-fold и stratified K-fold cross-validation

```python
def kfold_split(n, k=5, seed=42):
    random.seed(seed)
    indices = list(range(n))
    random.shuffle(indices)

    fold_size = n // k
    folds = []

    for i in range(k):
        start = i * fold_size
        end = start + fold_size if i < k - 1 else n
        val_idx = indices[start:end]
        train_idx = indices[:start] + indices[end:]
        folds.append((train_idx, val_idx))

    return folds


def stratified_kfold_split(y, k=5, seed=42):
    random.seed(seed)

    class_indices = {}
    for i, label in enumerate(y):
        class_indices.setdefault(label, []).append(i)

    for label in class_indices:
        random.shuffle(class_indices[label])

    folds = [{"train": [], "val": []} for _ in range(k)]

    for label, indices in class_indices.items():
        fold_size = len(indices) // k
        for i in range(k):
            start = i * fold_size
            end = start + fold_size if i < k - 1 else len(indices)
            val_part = indices[start:end]
            train_part = indices[:start] + indices[end:]
            folds[i]["val"].extend(val_part)
            folds[i]["train"].extend(train_part)

    return [(f["train"], f["val"]) for f in folds]


def cross_validate(X, y, model_fn, k=5, metric_fn=None, stratified=False):
    n = len(X)

    if stratified:
        folds = stratified_kfold_split(y, k)
    else:
        folds = kfold_split(n, k)

    scores = []
    for train_idx, val_idx in folds:
        X_train = [X[i] for i in train_idx]
        y_train = [y[i] for i in train_idx]
        X_val = [X[i] for i in val_idx]
        y_val = [y[i] for i in val_idx]

        model = model_fn()
        model.fit(X_train, y_train)
        predictions = [model.predict(x) for x in X_val]

        if metric_fn:
            score = metric_fn(y_val, predictions)
        else:
            score = sum(1 for yt, yp in zip(y_val, predictions) if yt == yp) / len(y_val)
        scores.append(score)

    return scores
```

### Шаг 3: матрица ошибок и метрики классификации

```python
def confusion_matrix(y_true, y_pred):
    tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
    tn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 0)
    fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)
    fn = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 0)
    return tp, tn, fp, fn


def accuracy(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    total = tp + tn + fp + fn
    return (tp + tn) / total if total > 0 else 0.0


def precision(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fp) if (tp + fp) > 0 else 0.0


def recall(y_true, y_pred):
    tp, tn, fp, fn = confusion_matrix(y_true, y_pred)
    return tp / (tp + fn) if (tp + fn) > 0 else 0.0


def f1_score(y_true, y_pred):
    p = precision(y_true, y_pred)
    r = recall(y_true, y_pred)
    return 2 * p * r / (p + r) if (p + r) > 0 else 0.0


def roc_curve(y_true, y_scores):
    thresholds = sorted(set(y_scores), reverse=True)
    tpr_list = []
    fpr_list = []

    total_positives = sum(y_true)
    total_negatives = len(y_true) - total_positives

    for threshold in thresholds:
        y_pred = [1 if s >= threshold else 0 for s in y_scores]
        tp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 1 and yp == 1)
        fp = sum(1 for yt, yp in zip(y_true, y_pred) if yt == 0 and yp == 1)

        tpr = tp / total_positives if total_positives > 0 else 0.0
        fpr = fp / total_negatives if total_negatives > 0 else 0.0

        tpr_list.append(tpr)
        fpr_list.append(fpr)

    return fpr_list, tpr_list, thresholds


def auc_roc(y_true, y_scores):
    fpr_list, tpr_list, _ = roc_curve(y_true, y_scores)

    pairs = sorted(zip(fpr_list, tpr_list))
    fpr_sorted = [p[0] for p in pairs]
    tpr_sorted = [p[1] for p in pairs]

    area = 0.0
    for i in range(1, len(fpr_sorted)):
        width = fpr_sorted[i] - fpr_sorted[i - 1]
        height = (tpr_sorted[i] + tpr_sorted[i - 1]) / 2
        area += width * height

    return area
```

### Шаг 4: регрессионные метрики

```python
def mse(y_true, y_pred):
    n = len(y_true)
    return sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred)) / n


def rmse(y_true, y_pred):
    return math.sqrt(mse(y_true, y_pred))


def mae(y_true, y_pred):
    n = len(y_true)
    return sum(abs(yt - yp) for yt, yp in zip(y_true, y_pred)) / n


def r_squared(y_true, y_pred):
    mean_y = sum(y_true) / len(y_true)
    ss_res = sum((yt - yp) ** 2 for yt, yp in zip(y_true, y_pred))
    ss_tot = sum((yt - mean_y) ** 2 for yt in y_true)
    if ss_tot == 0:
        return 0.0
    return 1.0 - ss_res / ss_tot
```

### Шаг 5: learning curves

```python
def learning_curve(X, y, model_fn, metric_fn, train_sizes=None, val_ratio=0.2, seed=42):
    random.seed(seed)
    n = len(X)
    indices = list(range(n))
    random.shuffle(indices)

    val_size = int(n * val_ratio)
    val_idx = indices[:val_size]
    pool_idx = indices[val_size:]

    X_val = [X[i] for i in val_idx]
    y_val = [y[i] for i in val_idx]

    if train_sizes is None:
        train_sizes = [int(len(pool_idx) * r) for r in [0.1, 0.2, 0.4, 0.6, 0.8, 1.0]]

    train_scores = []
    val_scores = []

    for size in train_sizes:
        subset = pool_idx[:size]
        X_train = [X[i] for i in subset]
        y_train = [y[i] for i in subset]

        model = model_fn()
        model.fit(X_train, y_train)

        train_pred = [model.predict(x) for x in X_train]
        val_pred = [model.predict(x) for x in X_val]

        train_scores.append(metric_fn(y_train, train_pred))
        val_scores.append(metric_fn(y_val, val_pred))

    return train_sizes, train_scores, val_scores
```

### Шаг 6: простой классификатор для тестов и полный demo

```python
class SimpleLogistic:
    def __init__(self, lr=0.1, epochs=100):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def sigmoid(self, z):
        z = max(-500, min(500, z))
        return 1.0 / (1.0 + math.exp(-z))

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                z = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                pred = self.sigmoid(z)
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j]
                self.bias += self.lr * error

    def predict_proba(self, x):
        z = sum(w * xi for w, xi in zip(self.weights, x)) + self.bias
        return self.sigmoid(z)

    def predict(self, x):
        return 1 if self.predict_proba(x) >= 0.5 else 0


class SimpleLinearRegression:
    def __init__(self, lr=0.001, epochs=200):
        self.lr = lr
        self.epochs = epochs
        self.weights = None
        self.bias = 0.0

    def fit(self, X, y):
        n_features = len(X[0])
        self.weights = [0.0] * n_features
        self.bias = 0.0
        n = len(X)

        for _ in range(self.epochs):
            for xi, yi in zip(X, y):
                pred = sum(w * x for w, x in zip(self.weights, xi)) + self.bias
                error = yi - pred
                for j in range(n_features):
                    self.weights[j] += self.lr * error * xi[j] / n
                self.bias += self.lr * error / n

    def predict(self, x):
        return sum(w * xi for w, xi in zip(self.weights, x)) + self.bias


def standardize(values):
    n = len(values)
    mean = sum(values) / n
    var = sum((v - mean) ** 2 for v in values) / n
    std = math.sqrt(var) if var > 0 else 1.0
    return [(v - mean) / std for v in values], mean, std


def make_classification_data(n=300, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.gauss(0, 1)
        x2 = random.gauss(0, 1)
        label = 1 if (x1 + x2 + random.gauss(0, 0.5)) > 0 else 0
        X.append([x1, x2])
        y.append(label)
    return X, y


def make_regression_data(n=200, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        x1 = random.uniform(0, 10)
        x2 = random.uniform(0, 5)
        target = 3 * x1 + 2 * x2 + random.gauss(0, 2)
        X.append([x1, x2])
        y.append(target)
    return X, y


def make_imbalanced_data(n=300, minority_ratio=0.05, seed=42):
    random.seed(seed)
    X = []
    y = []
    for _ in range(n):
        if random.random() < minority_ratio:
            x1 = random.gauss(3, 0.5)
            x2 = random.gauss(3, 0.5)
            label = 1
        else:
            x1 = random.gauss(0, 1)
            x2 = random.gauss(0, 1)
            label = 0
        X.append([x1, x2])
        y.append(label)
    return X, y


if __name__ == "__main__":
    X_clf, y_clf = make_classification_data(300)

    print("=== Train/Validation/Test Split ===")
    X_train, y_train, X_val, y_val, X_test, y_test = train_val_test_split(X_clf, y_clf)
    print(f"  Train: {len(X_train)}, Val: {len(X_val)}, Test: {len(X_test)}")
    print(f"  Train class distribution: {sum(y_train)}/{len(y_train)} positive")
    print(f"  Val class distribution: {sum(y_val)}/{len(y_val)} positive")

    model = SimpleLogistic(lr=0.1, epochs=200)
    model.fit(X_train, y_train)

    print("\n=== Classification Metrics ===")
    y_pred = [model.predict(x) for x in X_test]
    tp, tn, fp, fn = confusion_matrix(y_test, y_pred)
    print(f"  Confusion matrix: TP={tp}, TN={tn}, FP={fp}, FN={fn}")
    print(f"  Accuracy:  {accuracy(y_test, y_pred):.4f}")
    print(f"  Precision: {precision(y_test, y_pred):.4f}")
    print(f"  Recall:    {recall(y_test, y_pred):.4f}")
    print(f"  F1 Score:  {f1_score(y_test, y_pred):.4f}")

    y_scores = [model.predict_proba(x) for x in X_test]
    auc = auc_roc(y_test, y_scores)
    print(f"  AUC-ROC:   {auc:.4f}")

    print("\n=== K-Fold Cross-Validation (K=5) ===")
    cv_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
    )
    mean_cv = sum(cv_scores) / len(cv_scores)
    std_cv = math.sqrt(sum((s - mean_cv) ** 2 for s in cv_scores) / len(cv_scores))
    print(f"  Fold scores: {[round(s, 4) for s in cv_scores]}")
    print(f"  Mean: {mean_cv:.4f} (+/- {std_cv:.4f})")

    print("\n=== Stratified K-Fold Cross-Validation (K=5) ===")
    strat_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        k=5,
        metric_fn=accuracy,
        stratified=True,
    )
    strat_mean = sum(strat_scores) / len(strat_scores)
    strat_std = math.sqrt(sum((s - strat_mean) ** 2 for s in strat_scores) / len(strat_scores))
    print(f"  Fold scores: {[round(s, 4) for s in strat_scores]}")
    print(f"  Mean: {strat_mean:.4f} (+/- {strat_std:.4f})")

    print("\n=== Imbalanced Data: Why Accuracy Lies ===")
    X_imb, y_imb = make_imbalanced_data(300, minority_ratio=0.05)
    positives = sum(y_imb)
    print(f"  Class distribution: {positives} positive, {len(y_imb) - positives} negative ({positives/len(y_imb)*100:.1f}% positive)")

    always_negative = [0] * len(y_imb)
    print(f"  Always-negative baseline:")
    print(f"    Accuracy:  {accuracy(y_imb, always_negative):.4f}")
    print(f"    Precision: {precision(y_imb, always_negative):.4f}")
    print(f"    Recall:    {recall(y_imb, always_negative):.4f}")
    print(f"    F1 Score:  {f1_score(y_imb, always_negative):.4f}")

    X_tr_i, y_tr_i, X_v_i, y_v_i, X_te_i, y_te_i = train_val_test_split(X_imb, y_imb)
    model_imb = SimpleLogistic(lr=0.5, epochs=500)
    model_imb.fit(X_tr_i, y_tr_i)
    y_pred_imb = [model_imb.predict(x) for x in X_te_i]
    print(f"\n  Trained model on imbalanced data:")
    print(f"    Accuracy:  {accuracy(y_te_i, y_pred_imb):.4f}")
    print(f"    Precision: {precision(y_te_i, y_pred_imb):.4f}")
    print(f"    Recall:    {recall(y_te_i, y_pred_imb):.4f}")
    print(f"    F1 Score:  {f1_score(y_te_i, y_pred_imb):.4f}")

    print("\n=== Regression Metrics ===")
    X_reg, y_reg = make_regression_data(200)

    col0 = [x[0] for x in X_reg]
    col1 = [x[1] for x in X_reg]
    col0_s, m0, s0 = standardize(col0)
    col1_s, m1, s1 = standardize(col1)
    X_reg_scaled = [[col0_s[i], col1_s[i]] for i in range(len(X_reg))]

    X_tr_r, y_tr_r, X_v_r, y_v_r, X_te_r, y_te_r = train_val_test_split(X_reg_scaled, y_reg)
    reg_model = SimpleLinearRegression(lr=0.01, epochs=500)
    reg_model.fit(X_tr_r, y_tr_r)
    y_pred_r = [reg_model.predict(x) for x in X_te_r]

    print(f"  MSE:       {mse(y_te_r, y_pred_r):.4f}")
    print(f"  RMSE:      {rmse(y_te_r, y_pred_r):.4f}")
    print(f"  MAE:       {mae(y_te_r, y_pred_r):.4f}")
    print(f"  R-squared: {r_squared(y_te_r, y_pred_r):.4f}")

    mean_baseline = [sum(y_tr_r) / len(y_tr_r)] * len(y_te_r)
    print(f"\n  Mean baseline:")
    print(f"    MSE:       {mse(y_te_r, mean_baseline):.4f}")
    print(f"    R-squared: {r_squared(y_te_r, mean_baseline):.4f}")

    print("\n=== Learning Curve ===")
    sizes, train_sc, val_sc = learning_curve(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=200),
        metric_fn=accuracy,
    )
    print(f"  {'Size':>6} {'Train':>8} {'Val':>8}")
    for s, tr, va in zip(sizes, train_sc, val_sc):
        print(f"  {s:>6} {tr:>8.4f} {va:>8.4f}")

    print("\n=== Statistical Model Comparison ===")
    model_a_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=100),
        k=5, metric_fn=accuracy,
    )
    model_b_scores = cross_validate(
        X_clf, y_clf,
        model_fn=lambda: SimpleLogistic(lr=0.1, epochs=500),
        k=5, metric_fn=accuracy,
    )
    diffs = [a - b for a, b in zip(model_a_scores, model_b_scores)]
    mean_diff = sum(diffs) / len(diffs)
    std_diff = math.sqrt(sum((d - mean_diff) ** 2 for d in diffs) / len(diffs))
    t_stat = mean_diff / (std_diff / math.sqrt(len(diffs))) if std_diff > 0 else 0.0
    print(f"  Model A (100 epochs) mean: {sum(model_a_scores)/len(model_a_scores):.4f}")
    print(f"  Model B (500 epochs) mean: {sum(model_b_scores)/len(model_b_scores):.4f}")
    print(f"  Mean difference: {mean_diff:.4f}")
    print(f"  Paired t-statistic: {t_stat:.4f}")
    print(f"  (|t| > 2.78 for significance at p<0.05 with df=4)")
```

### Ожидаемый вывод

Запустите `code/evaluation.py` — последние строки должны быть такими:

```
     240   0.8792   0.8500

=== Statistical Model Comparison ===
  Model A (100 epochs) mean: 0.8733
  Model B (500 epochs) mean: 0.8733
  Mean difference: 0.0000
  Paired t-statistic: 0.0000
  (|t| > 2.78 for significance at p<0.05 with df=4)
```

## Используйте это

В scikit-learn оценка встроена в workflow:

```python
from sklearn.model_selection import cross_val_score, StratifiedKFold, learning_curve
from sklearn.metrics import (
    accuracy_score, precision_score, recall_score, f1_score,
    roc_auc_score, confusion_matrix, mean_squared_error, r2_score,
)
from sklearn.linear_model import LogisticRegression

model = LogisticRegression()
scores = cross_val_score(model, X, y, cv=StratifiedKFold(5), scoring="f1")
```

Реализации с нуля показывают, что именно делает cross-validation (никакой магии, только циклы и отслеживание индексов), как считается каждая метрика (просто подсчет TP/FP/TN/FN) и почему важна стратификация (сохранение соотношения классов в каждом fold). Библиотечные версии добавляют параллелизм, больше scoring options и интеграцию с pipelines.

## Доведите до результата

Этот урок создает:
- `outputs/skill-evaluation.md` — skill по стратегии оценки моделей классификации и регрессии

## Упражнения

1. Реализуйте precision-recall curves: постройте precision vs recall при разных порогах. Вычислите average precision (площадь под PR curve). Сравните PR curve с ROC curve на несбалансированном наборе данных и объясните, когда каждая информативнее.
2. Постройте nested cross-validation loop: внешний цикл оценивает качество модели, внутренний подбирает гиперпараметры. Используйте его, чтобы честно сравнить две модели без утечки validation data в оценку.
3. Реализуйте permutation test для сравнения моделей: перемешайте labels, переобучите и измерьте качество. Повторите 100 раз, чтобы построить null distribution. Вычислите p-value для наблюдаемого качества модели относительно этого распределения.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|--------|-------------|------------------------------|
| Переобучение | «Запоминание обучающих данных» | Модель улавливает шум в обучающих данных: хорошо работает на training, но плохо на невидимых данных |
| Cross-validation | «Тестирование на разных поднаборах» | Систематическое вращение части данных, используемой для валидации, с усреднением результатов по всем вращениям |
| Precision | «Сколько предсказанных positive корректны» | TP / (TP + FP): доля positive-предсказаний, которые действительно positive |
| Recall | «Сколько фактических positive мы нашли» | TP / (TP + FN): доля фактических positive, которые были правильно найдены |
| AUC-ROC | «Насколько хорошо модель разделяет классы» | Площадь под кривой true positive rate vs false positive rate по всем порогам, от 0.5 (случайно) до 1.0 (идеально) |
| R-squared | «Сколько дисперсии объяснено» | 1 - (сумма квадратов остатков / полная сумма квадратов): доля дисперсии target, захваченная моделью |
| Data leakage | «Модель сжульничала» | Использование при обучении информации, недоступной во время предсказания, что ведет к оптимистичной оценке |
| Learning curve | «Как качество меняется с большим объемом данных» | График training и validation scores от размера обучающего набора, показывающий underfitting или overfitting |
| Stratified split | «Сохранить соотношение классов» | Разбиение данных так, чтобы каждый subset имел ту же долю каждого класса, что и полный набор |

## Дополнительное чтение

- [scikit-learn Model Selection Guide](https://scikit-learn.org/stable/model_selection.html) — полный справочник по cross-validation, metrics и hyperparameter tuning
- [Beyond Accuracy: Precision and Recall (Google ML Crash Course)](https://developers.google.com/machine-learning/crash-course/classification/precision-and-recall) — ясное объяснение с интерактивными примерами
- [A Survey of Cross-Validation Procedures (Arlot & Celisse, 2010)](https://projecteuclid.org/journals/statistics-surveys/volume-4/issue-none/A-survey-of-cross-validation-procedures-for-model-selection/10.1214/09-SS054.full) — строгий разбор того, когда и почему разные CV-стратегии работают
