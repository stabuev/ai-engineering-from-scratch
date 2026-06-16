# Нормы и расстояния

> Ваша функция расстояния определяет, что значит "похожий". Выберите неправильно — и все последующие компоненты сломаются.

**Тип:** Практика
**Язык:** Python
**Предварительные требования:** Фаза 1, уроки 01 (интуиция линейной алгебры), 02 (векторы, матрицы и операции)
**Время:** ~90 минут

## Цели обучения

- Реализовать функции расстояния L1, L2, cosine, Mahalanobis, Jaccard и edit distance с нуля
- Выбирать подходящую метрику расстояния для заданной ML-задачи и объяснять, почему альтернативы не подходят
- Связать нормы L1 и L2 с регуляризацией LASSO и Ridge и их геометрическими областями ограничений
- Показать, как один и тот же датасет дает разных ближайших соседей при разных метриках

## Проблема

У вас есть два вектора. Возможно, это word embeddings. Возможно, это профили пользователей. Возможно, это массивы пикселей. Вам нужно понять: насколько они близки?

Ответ полностью зависит от выбранной функции расстояния. Две точки данных могут быть ближайшими соседями по одной метрике и далеко друг от друга по другой. Ваш KNN-классификатор, рекомендательная система, vector database, алгоритм кластеризации, loss function — все они зависят от этого выбора. Ошибитесь, и модель будет оптимизировать не то.

Не существует универсально лучшего расстояния. L2 работает для пространственных данных. Cosine similarity доминирует в NLP. Jaccard работает с множествами. Edit distance работает со строками. Mahalanobis учитывает корреляции. Wasserstein перемещает вероятностную массу. Каждая из них кодирует разное предположение о том, что значит "похожий".

Этот урок строит все основные функции расстояния с нуля, показывает, когда каждая из них является правильным инструментом, и демонстрирует, как одни и те же данные дают полностью разных ближайших соседей в зависимости от выбранной метрики.

## Концепция

### Нормы: измерение величины вектора

Норма измеряет "размер" вектора. Любую функцию расстояния между двумя векторами можно записать как норму их разности: d(a, b) = ||a - b||. Поэтому понимание норм — это понимание расстояний.

### L1-норма (Manhattan distance)

L1-норма суммирует абсолютные значения всех компонент.

```
||x||_1 = |x_1| + |x_2| + ... + |x_n|
```

Она называется Manhattan distance, потому что измеряет, сколько вы пройдете по городской сетке, где можно двигаться только вдоль осей. Без диагоналей.

```
Point A = (1, 1)
Point B = (4, 5)

L1 distance = |4-1| + |5-1| = 3 + 4 = 7

On a grid, you walk 3 blocks east and 4 blocks north.
```

Когда использовать L1:
- Разреженные данные высокой размерности (текстовые признаки, one-hot encodings)
- Когда нужна устойчивость к выбросам (одно огромное различие не доминирует)
- Задачи отбора признаков (L1 regularization способствует разреженности)

Связь с L1 regularization (Lasso): добавление ||w||_1 к loss function штрафует сумму абсолютных значений весов. Это выталкивает малые веса ровно в ноль, выполняя автоматический отбор признаков. L1 penalty создает ромбовидные области ограничений в пространстве весов, а углы ромбов лежат на осях, где некоторые веса равны нулю.

Связь с loss functions: Mean Absolute Error (MAE) — это среднее L1 distance между predictions и targets. Она штрафует все ошибки линейно, поэтому более устойчива к выбросам по сравнению с MSE.

### L2-норма (Euclidean distance)

L2-норма — это расстояние по прямой. Квадратный корень из суммы квадратов компонент.

```
||x||_2 = sqrt(x_1^2 + x_2^2 + ... + x_n^2)
```

Это расстояние, которое вы учили на геометрии. Пифагор в n измерениях.

```
Point A = (1, 1)
Point B = (4, 5)

L2 distance = sqrt((4-1)^2 + (5-1)^2) = sqrt(9 + 16) = sqrt(25) = 5.0

The straight line, cutting diagonally through the grid.
```

Когда использовать L2:
- Непрерывные данные низкой и средней размерности
- Когда масштабы признаков сопоставимы
- Физические расстояния (пространственные данные, показания сенсоров)
- Сходство изображений на уровне пикселей

Связь с L2 regularization (Ridge): добавление ||w||_2^2 к loss function штрафует большие веса. В отличие от L1, оно не выталкивает веса в ноль. Оно пропорционально сжимает все веса к нулю. L2 penalty создает круглые области ограничений, поэтому на осях нет углов. Веса становятся малыми, но редко ровно нулевыми.

Связь с loss functions: Mean Squared Error (MSE) — это среднее квадратов L2 distances. Возведение в квадрат штрафует большие ошибки сильнее, чем малые.

```
MAE (L1 loss):  |y - y_hat|         Linear penalty. Robust to outliers.
MSE (L2 loss):  (y - y_hat)^2       Quadratic penalty. Sensitive to outliers.
```

### Lp-нормы: общее семейство

L1 и L2 — частные случаи Lp-нормы:

```
||x||_p = (|x_1|^p + |x_2|^p + ... + |x_n|^p)^(1/p)
```

Разные значения p дают разные формы "единичных шаров" (множества всех точек на расстоянии 1 от начала координат):

```
p=1:    Diamond shape      (corners on axes)
p=2:    Circle/sphere      (the usual round ball)
p=3:    Superellipse       (rounded square)
p=inf:  Square/hypercube   (flat sides along axes)
```

### L-infinity-норма (Chebyshev distance)

Когда p стремится к бесконечности, Lp-норма сходится к максимальной абсолютной компоненте.

```
||x||_inf = max(|x_1|, |x_2|, ..., |x_n|)
```

Расстояние между двумя точками определяется единственным измерением, где они различаются сильнее всего. Все остальные измерения игнорируются.

```
Point A = (1, 1)
Point B = (4, 5)

L-inf distance = max(|4-1|, |5-1|) = max(3, 4) = 4
```

Когда использовать L-infinity:
- Когда важно максимальное отклонение по любому одному измерению
- Игровые доски (король в шахматах двигается в L-infinity: один шаг в любом направлении стоит 1)
- Производственные допуски (каждое измерение должно быть в пределах спецификации)

### Cosine similarity и cosine distance

Cosine similarity измеряет угол между двумя векторами, игнорируя их величины.

```
cos_sim(a, b) = (a . b) / (||a||_2 * ||b||_2)
```

Она лежит в диапазоне от -1 (противоположные направления) до +1 (одно направление). Перпендикулярные векторы имеют cosine similarity 0.

Cosine distance превращает ее в расстояние: cosine_distance = 1 - cosine_similarity. Диапазон от 0 (одинаковое направление) до 2 (противоположное направление).

```
a = (1, 0)    b = (1, 1)

cos_sim = (1*1 + 0*1) / (1 * sqrt(2)) = 1/sqrt(2) = 0.707
cos_dist = 1 - 0.707 = 0.293
```

Почему cosine доминирует в NLP и embeddings: в тексте длина документа не должна влиять на similarity. Документ о кошках, который вдвое длиннее другого документа о кошках, все равно должен быть "похожим". Cosine similarity игнорирует величину (длину) и смотрит только на направление. Два документа с одинаковым распределением слов, но разной длиной, указывают в одном направлении и получают cosine similarity 1.0.

Когда использовать cosine similarity:
- Сходство текстов (TF-IDF vectors, word embeddings, sentence embeddings)
- Любая область, где величина — шум, а направление — сигнал
- Рекомендательные системы (векторы предпочтений пользователей)
- Поиск по embeddings (vector databases почти всегда используют cosine или dot product)

### Dot product similarity против cosine similarity

Dot product двух векторов:

```
a . b = a_1*b_1 + a_2*b_2 + ... + a_n*b_n
      = ||a|| * ||b|| * cos(angle)
```

Cosine similarity — это dot product, нормированный на обе величины. Когда оба вектора уже нормированы до единичной длины (magnitude = 1), dot product и cosine similarity идентичны.

```
If ||a|| = 1 and ||b|| = 1:
    a . b = cos(angle between a and b)
```

Когда они различаются: dot product включает информацию о величине. Вектор с большей magnitude получает более высокий score dot product. Это важно в некоторых поисковых системах, где вы хотите, чтобы "популярные" элементы ранжировались выше. Magnitude работает как неявный сигнал качества или важности.

```
a = (3, 0)    b = (1, 0)    c = (0, 1)

dot(a, b) = 3     dot(a, c) = 0
cos(a, b) = 1.0   cos(a, c) = 0.0

Both agree on direction, but dot product also reflects magnitude.
```

На практике:
- Используйте cosine similarity, когда нужно сходство только по направлению
- Используйте dot product, когда величины несут осмысленную информацию
- Многие vector databases (Pinecone, Weaviate, Qdrant) позволяют выбирать между ними
- Если ваши embeddings L2-normalized, выбор не имеет значения

### Mahalanobis distance

Euclidean distance обращается со всеми измерениями одинаково. Но если признаки коррелированы или имеют разные масштабы, L2 дает вводящие в заблуждение результаты.

Mahalanobis distance учитывает ковариационную структуру данных.

```
d_M(x, y) = sqrt((x - y)^T * S^(-1) * (x - y))
```

где S — ковариационная матрица данных.

Интуитивно: Mahalanobis distance сначала декоррелирует и нормализует данные (whitening), затем вычисляет L2 distance в этом преобразованном пространстве. Если S — единичная матрица (некоррелированные признаки с единичной дисперсией), Mahalanobis distance сводится к Euclidean distance.

```
Example: height and weight are correlated.
Someone 6'2" and 180 lbs is not unusual.
Someone 5'0" and 180 lbs is unusual.

Euclidean distance might say they are equally far from the mean.
Mahalanobis distance correctly identifies the second as an outlier
because it accounts for the height-weight correlation.
```

Когда использовать Mahalanobis distance:
- Обнаружение выбросов (точки с большим Mahalanobis distance от среднего — выбросы)
- Классификация, когда признаки имеют разные масштабы и корреляции
- Когда у вас достаточно данных, чтобы надежно оценить ковариационную матрицу
- Контроль качества в производстве (многомерный мониторинг процесса)

### Jaccard similarity (для множеств)

Jaccard similarity измеряет пересечение двух множеств.

```
J(A, B) = |A intersect B| / |A union B|
```

Она лежит в диапазоне от 0 (нет пересечения) до 1 (множества совпадают). Jaccard distance = 1 - Jaccard similarity.

```
A = {cat, dog, fish}
B = {cat, bird, fish, snake}

Intersection = {cat, fish}         size = 2
Union = {cat, dog, fish, bird, snake}  size = 5

Jaccard similarity = 2/5 = 0.4
Jaccard distance = 0.6
```

Когда использовать Jaccard:
- Сравнение множеств тегов, категорий или признаков
- Сходство документов на основе наличия слов (не частоты)
- Обнаружение почти полных дублей (MinHash approximation of Jaccard)
- Сравнение бинарных векторов признаков (данные присутствия/отсутствия)
- Оценка моделей сегментации (Intersection over Union = Jaccard)

### Edit distance (Levenshtein distance)

Edit distance считает минимальное число односимвольных операций, нужных, чтобы превратить одну строку в другую. Операции: вставка, удаление или замена.

```
"kitten" -> "sitting"

kitten -> sitten  (substitute k -> s)
sitten -> sittin  (substitute e -> i)
sittin -> sitting (insert g)

Edit distance = 3
```

Вычисляется с помощью динамического программирования. Заполните матрицу, где ячейка (i, j) — edit distance между первыми i символами строки A и первыми j символами строки B.

```
        ""  s  i  t  t  i  n  g
    ""   0  1  2  3  4  5  6  7
    k    1  1  2  3  4  5  6  7
    i    2  2  1  2  3  4  5  6
    t    3  3  2  1  2  3  4  5
    t    4  4  3  2  1  2  3  4
    e    5  5  4  3  2  2  3  4
    n    6  6  5  4  3  3  2  3
```

Когда использовать edit distance:
- Проверка и исправление орфографии
- Выравнивание последовательностей ДНК (со взвешенными операциями)
- Нечеткое сопоставление строк
- Дедупликация грязных текстовых данных

### KL Divergence (не distance, но используется как distance)

KL divergence измеряет, насколько одно распределение вероятностей отличается от другого. Это покрыто в уроке 09, но относится к этому обсуждению, потому что люди используют ее как "расстояние", несмотря на то что она им не является.

```
D_KL(P || Q) = sum(p(x) * log(p(x) / q(x)))
```

Критическое свойство: KL divergence НЕ симметрична.

```
D_KL(P || Q) != D_KL(Q || P)
```

Это значит, что она не выполняет базовое требование метрики расстояния. Она также не удовлетворяет неравенству треугольника. Это divergence, не distance.

Forward KL (D_KL(P || Q)) — "ищет среднее": Q пытается покрыть все моды P.
Reverse KL (D_KL(Q || P)) — "ищет моду": Q фокусируется на одной моде P.

Где вы увидите KL divergence:
- VAEs (KL-слагаемое в ELBO подталкивает latent distribution к prior)
- Knowledge distillation (student пытается приблизиться к teacher's distribution)
- RLHF (KL penalty удерживает fine-tuned model рядом с base model)
- Policy gradient methods (ограничение обновлений политики)

### Wasserstein distance (Earth Mover's Distance)

Wasserstein distance измеряет минимальную "работу", нужную, чтобы превратить одно распределение вероятностей в другое. Думайте так: если одно распределение — куча земли, а другое — яма, сколько земли нужно переместить и насколько далеко?

```
W(P, Q) = inf over all transport plans gamma of E[d(x, y)]
```

Для одномерных распределений это упрощается до интеграла абсолютной разности cumulative distribution functions:

```
W_1(P, Q) = integral |CDF_P(x) - CDF_Q(x)| dx
```

Почему Wasserstein важен:
- Это настоящая метрика (симметрична и удовлетворяет неравенству треугольника)
- Он дает градиенты даже когда распределения не пересекаются (KL divergence уходит в бесконечность)
- Это свойство сделало его центральным для Wasserstein GANs (WGANs), которые решили нестабильность обучения оригинальных GANs

```
Distributions with no overlap:

P: [1, 0, 0, 0, 0]    Q: [0, 0, 0, 0, 1]

KL divergence: infinity (log of zero)
Wasserstein: 4 (move all mass 4 bins)

Wasserstein gives a meaningful gradient. KL does not.
```

Когда использовать Wasserstein:
- Обучение GAN (WGAN, WGAN-GP)
- Сравнение распределений, которые могут не пересекаться
- Задачи optimal transport
- Поиск изображений (сравнение цветовых гистограмм)

### Почему разным задачам нужны разные расстояния

| Задача | Лучшее расстояние | Почему |
|------|--------------|-----|
| Сходство текстов | Cosine | Величина — шум, направление — смысл |
| Сравнение пикселей изображений | L2 | Пространственные отношения важны, признаки имеют сопоставимый масштаб |
| Разреженные признаки высокой размерности | L1 | Устойчива, не усиливает редкие большие различия |
| Пересечение множеств (теги, категории) | Jaccard | Данные естественно представлены множествами, а не векторами |
| Сопоставление строк | Edit distance | Операции соответствуют человеческой интуиции редактирования |
| Обнаружение выбросов | Mahalanobis | Учитывает корреляции и масштабы признаков |
| Сравнение распределений | KL divergence | Измеряет потерю информации при использовании Q вместо P |
| Обучение GAN | Wasserstein | Дает градиенты, даже когда распределения не пересекаются |
| Embeddings (vector DB) | Cosine или dot product | Embeddings обучаются кодировать смысл в направлении |
| Рекомендации | Dot product | Величина может кодировать популярность или уверенность |
| Последовательности ДНК | Взвешенное edit distance | Цена замены зависит от пары нуклеотидов |
| Контроль качества производства | L-infinity | Важно худшее отклонение по любому измерению |

### Связь с loss functions

Loss functions — это функции расстояния, примененные к predictions и targets.

```
Loss function       Distance it uses       Behavior
MSE                 L2 squared             Penalizes large errors heavily
MAE                 L1                     Penalizes all errors equally
Huber loss          L1 for large errors,   Best of both: robust to outliers,
                    L2 for small errors    smooth gradient near zero
Cross-entropy       KL divergence          Measures distribution mismatch
Hinge loss          max(0, margin - d)     Only penalizes below margin
Triplet loss        L2 (typically)         Pulls positives close, pushes
                                           negatives away
Contrastive loss    L2                     Similar pairs close, dissimilar
                                           pairs beyond margin
```

### Связь с regularization

Regularization добавляет штраф за норму весов к loss function.

```
L1 regularization (Lasso):   loss + lambda * ||w||_1
  -> Sparse weights. Some weights become exactly zero.
  -> Automatic feature selection.
  -> Solution has corners (non-differentiable at zero).

L2 regularization (Ridge):   loss + lambda * ||w||_2^2
  -> Small weights. All weights shrink toward zero.
  -> No feature selection (nothing goes to exactly zero).
  -> Smooth solution everywhere.

Elastic Net:                  loss + lambda_1 * ||w||_1 + lambda_2 * ||w||_2^2
  -> Combines sparsity of L1 with stability of L2.
  -> Groups of correlated features are kept or dropped together.
```

Почему L1 дает разреженность, а L2 нет: представьте область ограничений в двумерном пространстве весов. L1 — ромб, L2 — круг. Линии уровня loss function (эллипсы) с большей вероятностью коснутся ромба в углу, где один вес равен нулю. Они касаются круга в гладкой точке, где оба веса ненулевые.

### Nearest neighbor search

Каждая функция расстояния задает задачу nearest neighbor search: по query point найти ближайшие точки в датасете.

Точный nearest neighbor search имеет сложность O(n * d) на запрос в датасете из n точек с d измерениями. Для больших датасетов это слишком медленно.

Алгоритмы approximate nearest neighbor (ANN) обменивают небольшую долю точности на огромный выигрыш в скорости:

```
Algorithm         Approach                      Used by
KD-trees          Axis-aligned space partition   scikit-learn (low-dim)
Ball trees        Nested hyperspheres            scikit-learn (medium-dim)
LSH               Random hash projections        Near-duplicate detection
HNSW              Hierarchical navigable         FAISS, Qdrant, Weaviate
                  small-world graph
IVF               Inverted file index with       FAISS (billion-scale)
                  cluster-based search
Product quant.    Compress vectors, search       FAISS (memory-constrained)
                  in compressed space
```

HNSW (Hierarchical Navigable Small World) — доминирующий алгоритм в современных vector databases. Он строит многоуровневый граф, где каждый узел соединяется со своими approximate nearest neighbors. Поиск начинается на верхнем уровне (разреженный, длинные переходы) и спускается к нижнему уровню (плотный, короткие переходы).

## Реализуйте

### Шаг 1: Все нормы и функции расстояния

См. `code/distances.py` для полной реализации. Каждая функция построена с нуля, используя только базовую математику Python.

### Шаг 2: Одни данные, разные distances, разные neighbors

Демо в `distances.py` создает датасет, выбирает query point и показывает, как nearest neighbor меняется в зависимости от метрики расстояния. Точка, которая "ближайшая" по L1, может не быть ближайшей по L2 или cosine.

### Шаг 3: Embedding similarity search

Код включает mock embedding similarity search, который находит наиболее похожие "документы" на query с помощью cosine similarity vs L2 distance, показывая, что ранжирования могут различаться.

### Ожидаемый вывод

Запустите `code/distances.py` — последние строки должны быть такими:

```
  After L2 regularization (50 steps):
    Weights: [-0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0, -0.0, -0.0, 0.0]
    Zeros:   0/10
    L2 norm: 0.0000

  L1 drives 'small' weights to exactly zero (sparsity).
  L2 shrinks all weights but none reach exactly zero.
```

## Используйте

Самое распространенное практическое применение: поиск похожих элементов в vector database.

```python
import numpy as np

def cosine_similarity_matrix(X):
    norms = np.linalg.norm(X, axis=1, keepdims=True)
    norms = np.where(norms == 0, 1, norms)
    X_normalized = X / norms
    return X_normalized @ X_normalized.T

embeddings = np.random.randn(1000, 768)

sim_matrix = cosine_similarity_matrix(embeddings)

query_idx = 0
similarities = sim_matrix[query_idx]
top_k = np.argsort(similarities)[::-1][1:6]
print(f"Top 5 most similar to item 0: {top_k}")
print(f"Similarities: {similarities[top_k]}")
```

Когда вы вызываете `model.encode(text)`, а затем ищете в vector database, именно это происходит под капотом. Embedding model отображает текст в векторы. Vector database вычисляет cosine similarity (или dot product) между вашим query vector и каждым stored vector, используя ANN algorithms, чтобы не проверять их все.

## Упражнения

1. Вычислите L1, L2 и L-infinity distances между (1, 2, 3) и (4, 0, 6). Проверьте, что L-inf <= L2 <= L1 всегда выполняется для любой пары точек. Докажите, почему этот порядок гарантирован.

2. Создайте два вектора, где cosine similarity высокая (> 0.9), но L2 distance большая (> 10). Объясните геометрически, что происходит. Затем создайте два вектора, где cosine similarity низкая (< 0.3), но L2 distance малая (< 0.5).

3. Реализуйте функцию, которая принимает датасет и query point и возвращает nearest neighbor по L1, L2, cosine и Mahalanobis distance. Найдите датасет, где все четыре метрики расходятся в том, какая точка ближайшая.

4. Вычислите Wasserstein distance между [0.5, 0.5, 0, 0] и [0, 0, 0.5, 0.5] вручную методом CDF. Затем вычислите его между [0.25, 0.25, 0.25, 0.25] и [0, 0, 0.5, 0.5]. Какое больше и почему?

5. Реализуйте MinHash для approximate Jaccard similarity. Сгенерируйте 100 случайных множеств, вычислите exact Jaccard для всех пар и сравните с MinHash approximation при 50, 100 и 200 hash functions. Постройте график ошибки аппроксимации.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Norm | "Размер вектора" | Функция, отображающая вектор в неотрицательный скаляр и удовлетворяющая неравенству треугольника, абсолютной однородности и равенству нулю только для нулевого вектора |
| L1 norm | "Manhattan distance" | Сумма абсолютных значений компонент. Дает разреженность в оптимизации. Устойчива к выбросам |
| L2 norm | "Euclidean distance" | Квадратный корень из суммы квадратов компонент. Расстояние по прямой в евклидовом пространстве |
| Lp norm | "Обобщенная норма" | p-й корень из суммы p-х степеней абсолютных компонент. L1 и L2 — частные случаи |
| L-infinity norm | "Max norm" или "Chebyshev distance" | Максимальное абсолютное значение компоненты. Предел Lp при p, стремящемся к бесконечности |
| Cosine similarity | "Угол между векторами" | Dot product, нормированный на обе величины. Диапазон от -1 до +1. Игнорирует длину вектора |
| Cosine distance | "1 минус cosine similarity" | Превращает cosine similarity в расстояние. Диапазон от 0 до 2 |
| Dot product | "Ненормированный cosine" | Сумма покомпонентных произведений. Равен cosine similarity, умноженной на обе величины |
| Mahalanobis distance | "Расстояние с учетом корреляций" | L2 distance в пространстве, которое было whitened (декоррелировано и нормализовано) с помощью ковариационной матрицы данных |
| Jaccard similarity | "Пересечение множеств" | Размер intersection, деленный на размер union. Для множеств, не векторов |
| Edit distance | "Levenshtein distance" | Минимальное число вставок, удалений и замен, чтобы превратить одну строку в другую |
| KL divergence | "Расстояние между распределениями" | Не настоящее расстояние (не симметрична). Измеряет дополнительные биты от использования Q для кодирования P |
| Wasserstein distance | "Earth mover's distance" | Минимальная работа для переноса массы из одного распределения в другое. Настоящая метрика |
| Approximate nearest neighbor | "ANN search" | Алгоритмы (HNSW, LSH, IVF), которые находят приблизительно ближайшие точки намного быстрее точного поиска |
| HNSW | "Алгоритм vector DB" | Hierarchical Navigable Small World graph. Многоуровневый граф для быстрого approximate nearest neighbor search |
| L1 regularization | "Lasso" | Добавление L1 norm весов к loss. Доводит веса до нуля (разреженность) |
| L2 regularization | "Ridge" или "weight decay" | Добавление squared L2 norm весов к loss. Сжимает веса к нулю без разреженности |
| Elastic Net | "L1 + L2" | Комбинирует L1 и L2 regularization. Лучше работает с коррелированными группами признаков, чем каждая по отдельности |

## Дополнительные материалы

- [FAISS: A Library for Efficient Similarity Search](https://github.com/facebookresearch/faiss) - библиотека Meta для billion-scale ANN search
- [Wasserstein GAN (Arjovsky et al., 2017)](https://arxiv.org/abs/1701.07875) - статья, которая ввела Earth Mover's distance в GANs
- [Locality-Sensitive Hashing (Indyk & Motwani, 1998)](https://dl.acm.org/doi/10.1145/276698.276876) - фундаментальный ANN-алгоритм
- [Efficient Estimation of Word Representations (Mikolov et al., 2013)](https://arxiv.org/abs/1301.3781) - Word2Vec, где cosine similarity стала стандартом для embeddings
- [sklearn.neighbors documentation](https://scikit-learn.org/stable/modules/neighbors.html) - практическое руководство по метрикам расстояния и алгоритмам соседей в scikit-learn
