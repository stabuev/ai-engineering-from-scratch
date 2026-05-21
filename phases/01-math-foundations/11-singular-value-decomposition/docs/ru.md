# Разложение по сингулярным значениям

> SVD — швейцарский нож линейной алгебры. Оно есть у каждой матрицы. Оно нужно каждому data scientist.

**Тип:** Практика
**Языки:** Python, Julia
**Предварительные требования:** Phase 1, Lessons 01 (Linear Algebra Intuition), 02 (Vectors & Matrices Operations), 03 (Matrix Transformations)
**Время:** ~120 минут

## Цели обучения

- Реализовать SVD через power iteration и объяснить геометрический смысл U, Sigma и V^T
- Применить truncated SVD для сжатия изображений и измерить соотношение степени сжатия и ошибки реконструкции
- Вычислить псевдообратную матрицу Мура-Пенроуза через SVD для решения переопределенных задач least squares
- Связать SVD с PCA, рекомендательными системами (latent factors) и Latent Semantic Analysis в NLP

## Проблема

У вас есть матрица 1000x2000. Возможно, это оценки пользователей для фильмов. Возможно, это таблица частот документ-термин. Возможно, это значения пикселей изображения. Вам нужно сжать ее, удалить шум, найти скрытую структуру или решить с ней задачу least squares. Eigendecomposition работает только для квадратных матриц. И даже тогда оно требует, чтобы у матрицы был полный набор линейно независимых собственных векторов.

SVD работает с любой матрицей. Любой формы. Любого ранга. Без условий. Оно раскладывает матрицу на три множителя, которые раскрывают геометрию того, что матрица делает с пространством. Это самая общая и самая полезная факторизация во всей линейной алгебре.

## Концепция

### Что SVD делает геометрически

Любая матрица, независимо от формы, выполняет три операции подряд: поворот, масштабирование, поворот. SVD делает это разложение явным.

```
A = U * Sigma * V^T

      m x n     m x m    m x n    n x n
     (any)    (rotate)  (scale)  (rotate)
```

Для любой матрицы A SVD раскладывает ее на:
- V^T поворачивает векторы во входном пространстве (n-мерном)
- Sigma масштабирует вдоль каждой оси (растягивает или сжимает)
- U поворачивает результат в выходное пространство (m-мерное)

```mermaid
graph LR
    A["Input space (n-dim)\nData cloud\n(arbitrary orientation)"] -->|"V^T\n(rotate)"| B["Scaled space\nAligned with axes\nthen scaled by Sigma"]
    B -->|"U\n(rotate)"| C["Output space (m-dim)\nRotated to output\norientation"]
```

Думайте об этом так. Вы даете SVD матрицу. Оно говорит: "Эта матрица берет сферу входов, сначала поворачивает ее на V^T, затем растягивает в эллипсоид с помощью Sigma, затем поворачивает эллипсоид с помощью U." Сингулярные значения — это длины осей эллипсоида.

### Полное разложение

Для матрицы A формы m x n:

```
A = U * Sigma * V^T

where:
  U     is m x m, orthogonal (U^T U = I)
  Sigma is m x n, diagonal (singular values on the diagonal)
  V     is n x n, orthogonal (V^T V = I)

The singular values sigma_1 >= sigma_2 >= ... >= sigma_r > 0
where r = rank(A)
```

Столбцы U называются левыми сингулярными векторами. Столбцы V называются правыми сингулярными векторами. Диагональные элементы Sigma называются сингулярными значениями. Они всегда неотрицательны и обычно отсортированы по убыванию.

### Левые сингулярные векторы, сингулярные значения, правые сингулярные векторы

У каждого компонента SVD есть отдельный геометрический смысл.

**Правые сингулярные векторы (столбцы V):** Они образуют ортонормированный базис входного пространства (R^n). Это направления во входном пространстве, которые матрица отображает в ортогональные направления в выходном пространстве. Думайте о них как о естественной системе координат для области определения.

**Сингулярные значения (диагональ Sigma):** Это коэффициенты масштабирования. i-е сингулярное значение говорит, насколько матрица растягивает векторы вдоль i-го правого сингулярного вектора. Нулевое сингулярное значение означает, что матрица полностью сминает это направление.

**Левые сингулярные векторы (столбцы U):** Они образуют ортонормированный базис выходного пространства (R^m). i-й левый сингулярный вектор — это направление в выходном пространстве, куда попадает i-й правый сингулярный вектор (после масштабирования).

Связь между ними:

```
A * v_i = sigma_i * u_i

The matrix A takes the i-th right singular vector v_i,
scales it by sigma_i, and maps it to the i-th left singular vector u_i.
```

Это дает покоординатную картину того, что делает любая матрица.

### Форма через внешние произведения

SVD можно записать как сумму матриц ранга 1:

```
A = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T + ... + sigma_r * u_r * v_r^T

Each term sigma_i * u_i * v_i^T is a rank-1 matrix (an outer product).
The full matrix is the sum of r such matrices, where r is the rank.
```

Эта форма лежит в основе низкоранговой аппроксимации. Каждый член добавляет один слой структуры. Первый член улавливает самый важный паттерн. Второй — следующий по важности. И так далее. Усечение этой суммы дает наилучшую возможную аппроксимацию для любого заданного ранга.

```
Rank-1 approx:    A_1 = sigma_1 * u_1 * v_1^T
                  (captures the dominant pattern)

Rank-2 approx:    A_2 = sigma_1 * u_1 * v_1^T + sigma_2 * u_2 * v_2^T
                  (captures the two most important patterns)

Rank-k approx:    A_k = sum of top k terms
                  (optimal by the Eckart-Young theorem)
```

### Связь с eigendecomposition

SVD и eigendecomposition глубоко связаны. Сингулярные значения и векторы A напрямую получаются из собственных значений и собственных векторов A^T A и A A^T.

```
A^T A = V * Sigma^T * U^T * U * Sigma * V^T
      = V * Sigma^T * Sigma * V^T
      = V * D * V^T

where D = Sigma^T * Sigma is a diagonal matrix with sigma_i^2 on the diagonal.

So:
- The right singular vectors (V) are eigenvectors of A^T A
- The singular values squared (sigma_i^2) are eigenvalues of A^T A

Similarly:
A A^T = U * Sigma * V^T * V * Sigma^T * U^T
      = U * Sigma * Sigma^T * U^T

So:
- The left singular vectors (U) are eigenvectors of A A^T
- The eigenvalues of A A^T are also sigma_i^2
```

Эта связь говорит о трех вещах:
1. Сингулярные значения всегда вещественные и неотрицательные (это квадратные корни из собственных значений положительно полуопределенной матрицы).
2. SVD можно вычислять через eigendecomposition матрицы A^T A, но это возводит condition number в квадрат и теряет численную точность. Специализированные алгоритмы SVD избегают этого.
3. Когда A квадратная и симметричная положительно полуопределенная, SVD и eigendecomposition совпадают.

### Truncated SVD: низкоранговая аппроксимация

Теорема Эккарта-Янга-Мирского утверждает, что лучшая аппроксимация A ранга k (и в норме Фробениуса, и в спектральной норме) получается, если оставить только верхние k сингулярных значений и соответствующие им векторы:

```
A_k = U_k * Sigma_k * V_k^T

where:
  U_k     is m x k  (first k columns of U)
  Sigma_k is k x k  (top-left k x k block of Sigma)
  V_k     is n x k  (first k columns of V)

Approximation error = sigma_{k+1}  (in spectral norm)
                    = sqrt(sigma_{k+1}^2 + ... + sigma_r^2)  (in Frobenius norm)
```

Это не просто "хорошая" аппроксимация. Это доказуемо лучшая возможная аппроксимация ранга k. Никакая другая матрица ранга k не ближе к A.

| Компонент | Относительная величина | Оставлен в аппроксимации ранга 3? |
|-----------|-------------------|------------------------|
| sigma_1 | Наибольшая | Да |
| sigma_2 | Большая | Да |
| sigma_3 | Средне-большая | Да |
| sigma_4 | Средняя | Нет (ошибка) |
| sigma_5 | Средне-малая | Нет (ошибка) |
| sigma_6 | Малая | Нет (ошибка) |
| sigma_7 | Очень малая | Нет (ошибка) |
| sigma_8 | Крошечная | Нет (ошибка) |

Оставляем top 3: A_3 улавливает три крупнейших сингулярных значения. Ошибка = оставшиеся значения (от sigma_4 до sigma_8).

Если сингулярные значения быстро убывают, малое k улавливает большую часть матрицы. Если они убывают медленно, у матрицы нет низкоранговой структуры.

### Сжатие изображений с помощью SVD

Изображение в оттенках серого — это матрица интенсивностей пикселей. Изображение 800x600 содержит 480,000 значений. SVD позволяет аппроксимировать его гораздо меньшим числом значений.

```
Original image: 800 x 600 = 480,000 values

SVD with rank k:
  U_k:      800 x k values
  Sigma_k:  k values
  V_k:      600 x k values
  Total:    k * (800 + 600 + 1) = k * 1401 values

  k=10:   14,010 values   (2.9% of original)
  k=50:   70,050 values  (14.6% of original)
  k=100: 140,100 values  (29.2% of original)

  The compression ratio improves as k gets smaller,
  but visual quality degrades.
```

Ключевая идея: у естественных изображений сингулярные значения быстро убывают. Первые несколько сингулярных значений улавливают крупную структуру (формы, градиенты). Более поздние улавливают мелкие детали и шум. Усечение на ранге 50 часто дает изображение, которое выглядит почти идентичным оригиналу, используя на 85% меньше памяти.

### SVD для рекомендательных систем

Netflix Prize сделал эту идею знаменитой. У вас есть матрица пользователь-фильм с оценками, где большинство элементов отсутствует.

```
             Movie1  Movie2  Movie3  Movie4  Movie5
  User1      [  5      ?       3       ?       1  ]
  User2      [  ?      4       ?       2       ?  ]
  User3      [  3      ?       5       ?       ?  ]
  User4      [  ?      ?       ?       4       3  ]

  ? = unknown rating
```

Идея: эта матрица оценок имеет низкий ранг. Вкусы пользователей не полностью независимы. Есть несколько latent factors (боевик против драмы, старое против нового, интеллектуальное против visceral), которые объясняют большинство предпочтений.

SVD на (заполненной) матрице оценок раскладывает ее на:
- U: профили пользователей в пространстве latent factors
- Sigma: важность каждого latent factor
- V^T: профили фильмов в пространстве latent factors

Предсказанная оценка пользователя для фильма — это dot product его пользовательского профиля с профилем фильма (взвешенный сингулярными значениями). Низкоранговая аппроксимация заполняет отсутствующие элементы.

На практике используют варианты вроде incremental SVD Саймона Фанка или ALS (alternating least squares), которые напрямую работают с пропущенными данными. Но основная идея та же: разложение по latent factors через SVD.

### SVD в NLP: Latent Semantic Analysis

Latent Semantic Analysis (LSA), также называемый Latent Semantic Indexing (LSI), применяет SVD к матрице термин-документ.

```
             Doc1   Doc2   Doc3   Doc4
  "cat"      [  3      0      1      0  ]
  "dog"      [  2      0      0      1  ]
  "fish"     [  0      4      1      0  ]
  "pet"      [  1      1      1      1  ]
  "ocean"    [  0      3      0      0  ]

After SVD with rank k=2:

  Each document becomes a point in 2D "concept space."
  Each term becomes a point in the same 2D space.
  Documents about similar topics cluster together.
  Terms with similar meanings cluster together.

  "cat" and "dog" end up near each other (land pets).
  "fish" and "ocean" end up near each other (water concepts).
  Doc1 and Doc3 cluster if they share similar topics.
```

LSA был одним из первых успешных методов для извлечения семантической близости из сырого текста. Он работает потому, что синонимичные термины обычно появляются в похожих документах, поэтому SVD группирует их в одни и те же скрытые измерения. Современные word embeddings (Word2Vec, GloVe) можно рассматривать как потомков этой идеи.

### SVD для удаления шума

В зашумленных данных сигнал сосредоточен в верхних сингулярных значениях, а шум размазан по всем сингулярным значениям. Усечение удаляет шумовой уровень.

**Сингулярные значения чистого сигнала:**

| Компонент | Величина | Тип |
|-----------|-----------|------|
| sigma_1 | Очень большая | Сигнал |
| sigma_2 | Большая | Сигнал |
| sigma_3 | Средняя | Сигнал |
| sigma_4 | Близка к нулю | Пренебрежимо мала |
| sigma_5 | Близка к нулю | Пренебрежимо мала |

**Сингулярные значения зашумленного сигнала (шум добавляется ко всем):**

| Компонент | Величина | Тип |
|-----------|-----------|------|
| sigma_1 | Очень большая | Сигнал |
| sigma_2 | Большая | Сигнал |
| sigma_3 | Средняя | Сигнал |
| sigma_4 | Малая | Шум |
| sigma_5 | Малая | Шум |
| sigma_6 | Малая | Шум |
| sigma_7 | Малая | Шум |

```mermaid
graph TD
    A["All singular values"] --> B{"Clear gap?"}
    B -->|"Above gap"| C["Signal: keep these (top k)"]
    B -->|"Below gap"| D["Noise: discard these"]
    C --> E["Reconstruct with A_k to get denoised version"]
```

Это используется в обработке сигналов, научных измерениях и очистке данных. Каждый раз, когда у вас есть матрица, испорченная аддитивным шумом, truncated SVD дает принципиальный способ отделить сигнал от шума.

### Псевдообратная матрица через SVD

Псевдообратная матрица Мура-Пенроуза A+ обобщает обращение матриц на неквадратные и сингулярные матрицы. SVD делает ее вычисление тривиальным.

```
If A = U * Sigma * V^T, then:

A+ = V * Sigma+ * U^T

where Sigma+ is formed by:
  1. Transpose Sigma (swap rows and columns)
  2. Replace each non-zero diagonal entry sigma_i with 1/sigma_i
  3. Leave zeros as zeros

For A (m x n):      A+ is (n x m)
For Sigma (m x n):  Sigma+ is (n x m)
```

Псевдообратная матрица решает задачи least squares. Если Ax = b не имеет точного решения (переопределенная система), то x = A+ b — решение least squares (минимизирует ||Ax - b||).

```
Overdetermined system (more equations than unknowns):

  [1  1]         [3]
  [2  1] x   =   [5]       No exact solution exists.
  [3  1]         [6]

  x_ls = A+ b = V * Sigma+ * U^T * b

  This gives the x that minimizes the sum of squared residuals.
  Same result as the normal equations (A^T A)^(-1) A^T b,
  but numerically more stable.
```

### Преимущества численной устойчивости

Вычисление eigendecomposition для A^T A возводит сингулярные значения в квадрат (собственные значения A^T A равны sigma_i^2). Это возводит condition number в квадрат и усиливает численные ошибки.

```
Example:
  A has singular values [1000, 1, 0.001]
  Condition number of A: 1000 / 0.001 = 10^6

  A^T A has eigenvalues [10^6, 1, 10^{-6}]
  Condition number of A^T A: 10^6 / 10^{-6} = 10^{12}

  Computing SVD directly: works with condition number 10^6
  Computing via A^T A:     works with condition number 10^{12}
                           (6 extra digits of precision lost)
```

Современные алгоритмы SVD (бидиагонализация Голуба-Кахана) работают напрямую с A, никогда не формируя A^T A. Поэтому всегда стоит предпочитать `np.linalg.svd(A)` вместо `np.linalg.eig(A.T @ A)`.

### Связь с PCA

PCA — это SVD на центрированных данных. Это не аналогия. Это буквально то же вычисление.

```
Given data matrix X (n_samples x n_features), centered (mean subtracted):

Covariance matrix: C = (1/(n-1)) * X^T X

PCA finds eigenvectors of C. But:

  X = U * Sigma * V^T    (SVD of X)

  X^T X = V * Sigma^2 * V^T

  C = (1/(n-1)) * V * Sigma^2 * V^T

So the principal components are exactly the right singular vectors V.
The explained variance for each component is sigma_i^2 / (n-1).

In sklearn, PCA is implemented using SVD, not eigendecomposition.
It is faster and more numerically stable.
```

Это означает, что все, что вы изучили о снижении размерности в Lesson 10, под капотом является SVD. PCA — самое распространенное применение SVD в машинном обучении.

## Реализуйте

### Шаг 1: SVD с нуля через power iteration

Идея: чтобы найти крупнейшее сингулярное значение и его векторы, используйте power iteration на A^T A (или A A^T). Затем дефлируйте матрицу и повторяйте для следующего сингулярного значения.

```python
import numpy as np

def power_iteration(M, num_iters=100):
    n = M.shape[1]
    v = np.random.randn(n)
    v = v / np.linalg.norm(v)

    for _ in range(num_iters):
        Mv = M @ v
        v = Mv / np.linalg.norm(Mv)

    eigenvalue = v @ M @ v
    return eigenvalue, v

def svd_from_scratch(A, k=None):
    m, n = A.shape
    if k is None:
        k = min(m, n)

    sigmas = []
    us = []
    vs = []

    A_residual = A.copy().astype(float)

    for _ in range(k):
        AtA = A_residual.T @ A_residual
        eigenvalue, v = power_iteration(AtA, num_iters=200)

        if eigenvalue < 1e-10:
            break

        sigma = np.sqrt(eigenvalue)
        u = A_residual @ v / sigma

        sigmas.append(sigma)
        us.append(u)
        vs.append(v)

        A_residual = A_residual - sigma * np.outer(u, v)

    U = np.column_stack(us) if us else np.empty((m, 0))
    S = np.array(sigmas)
    V = np.column_stack(vs) if vs else np.empty((n, 0))

    return U, S, V
```

### Шаг 2: Тестирование и сравнение с NumPy

```python
np.random.seed(42)
A = np.random.randn(5, 4)

U_ours, S_ours, V_ours = svd_from_scratch(A)
U_np, S_np, Vt_np = np.linalg.svd(A, full_matrices=False)

print("Our singular values:", np.round(S_ours, 4))
print("NumPy singular values:", np.round(S_np, 4))

A_reconstructed = U_ours @ np.diag(S_ours) @ V_ours.T
print(f"Reconstruction error: {np.linalg.norm(A - A_reconstructed):.8f}")
```

### Шаг 3: Demo сжатия изображений

```python
def compress_image_svd(image_matrix, k):
    U, S, Vt = np.linalg.svd(image_matrix, full_matrices=False)
    compressed = U[:, :k] @ np.diag(S[:k]) @ Vt[:k, :]
    return compressed

image = np.random.seed(42)
rows, cols = 200, 300
image = np.random.randn(rows, cols)

for k in [1, 5, 10, 20, 50]:
    compressed = compress_image_svd(image, k)
    error = np.linalg.norm(image - compressed) / np.linalg.norm(image)
    original_size = rows * cols
    compressed_size = k * (rows + cols + 1)
    ratio = compressed_size / original_size
    print(f"k={k:>3d}  error={error:.4f}  storage={ratio:.1%}")
```

### Шаг 4: Удаление шума

```python
np.random.seed(42)
clean = np.outer(np.sin(np.linspace(0, 4*np.pi, 100)),
                 np.cos(np.linspace(0, 2*np.pi, 80)))
noise = 0.3 * np.random.randn(100, 80)
noisy = clean + noise

U, S, Vt = np.linalg.svd(noisy, full_matrices=False)
denoised = U[:, :5] @ np.diag(S[:5]) @ Vt[:5, :]

print(f"Noisy error:    {np.linalg.norm(noisy - clean):.4f}")
print(f"Denoised error: {np.linalg.norm(denoised - clean):.4f}")
print(f"Improvement:    {(1 - np.linalg.norm(denoised - clean) / np.linalg.norm(noisy - clean)):.1%}")
```

### Шаг 5: Псевдообратная матрица

```python
A = np.array([[1, 1], [2, 1], [3, 1]], dtype=float)
b = np.array([3, 5, 6], dtype=float)

U, S, Vt = np.linalg.svd(A, full_matrices=False)
S_inv = np.diag(1.0 / S)
A_pinv = Vt.T @ S_inv @ U.T

x_svd = A_pinv @ b
x_lstsq = np.linalg.lstsq(A, b, rcond=None)[0]
x_pinv = np.linalg.pinv(A) @ b

print(f"SVD pseudoinverse solution:  {x_svd}")
print(f"np.linalg.lstsq solution:   {x_lstsq}")
print(f"np.linalg.pinv solution:    {x_pinv}")
```

## Используйте

Полные рабочие демо находятся в `code/svd.py`. Запустите его, чтобы увидеть применение SVD к сжатию изображений, рекомендательным системам, latent semantic analysis и удалению шума.

```bash
python svd.py
```

Версия на Julia в `code/svd.jl` демонстрирует те же концепции с использованием нативной функции Julia `svd()` и пакета `LinearAlgebra`.

```bash
julia svd.jl
```

## Итоговые артефакты

Этот урок создает:
- `outputs/skill-svd.md` — skill для понимания, когда и как применять SVD в реальных проектах

## Упражнения

1. Реализуйте полное SVD с нуля без power iteration. Вместо этого вычислите eigendecomposition A^T A, чтобы получить V и сингулярные значения, затем вычислите U = A V Sigma^{-1}. Сравните численную точность с вашей версией на power iteration и с NumPy.

2. Загрузите реальное изображение в оттенках серого (или преобразуйте изображение в grayscale). Сожмите его на рангах 1, 5, 10, 25, 50, 100. Для каждого ранга вычислите степень сжатия и относительную ошибку. Найдите ранг, на котором изображение становится визуально приемлемым.

3. Постройте маленькую рекомендательную систему. Создайте матрицу оценок пользователь-фильм размером 10x8 с некоторыми известными элементами. Заполните пропуски средними по строкам. Вычислите SVD и восстановите аппроксимацию ранга 3. Используйте восстановленную матрицу, чтобы предсказать отсутствующие оценки. Проверьте, что предсказания разумны.

4. Создайте матрицу документ-термин размером 100x50 с 3 синтетическими темами. У каждой темы 5 связанных терминов. Добавьте шум. Примените SVD и проверьте, что верхние 3 сингулярных значения намного больше остальных. Спроецируйте документы в 3D latent space и проверьте, что документы одной темы кластеризуются вместе.

5. Сгенерируйте чистую низкоранговую матрицу (ранг 3, размер 50x40) и добавьте гауссов шум разных уровней (sigma = 0.1, 0.5, 1.0, 2.0). Для каждого уровня шума найдите оптимальный ранг усечения, перебирая k от 1 до 40 и измеряя ошибку реконструкции относительно чистой матрицы. Постройте график того, как оптимальное k меняется с уровнем шума.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| SVD | "Разложить любую матрицу" | Разложить A на U Sigma V^T, где U и V ортогональны, а Sigma диагональна с неотрицательными элементами. Работает для любой матрицы любой формы. |
| Singular value | "Насколько важен этот компонент" | i-й диагональный элемент Sigma. Измеряет, насколько матрица растягивает вдоль i-го главного направления. Всегда неотрицателен, отсортирован по убыванию. |
| Left singular vector | "Выходное направление" | Столбец U. Направление в выходном пространстве, в которое отображается i-й правый сингулярный вектор (после масштабирования на sigma_i). |
| Right singular vector | "Входное направление" | Столбец V. Направление во входном пространстве, которое матрица отображает в i-й левый сингулярный вектор (после масштабирования на sigma_i). |
| Truncated SVD | "Низкоранговая аппроксимация" | Оставить только верхние k сингулярных значений и их векторы. Дает доказуемо лучшую аппроксимацию исходной матрицы ранга k (теорема Эккарта-Янга). |
| Rank | "Истинная размерность" | Число ненулевых сингулярных значений. Показывает, сколько независимых направлений матрица реально использует. |
| Pseudoinverse | "Обобщенная обратная" | V Sigma+ U^T. Инвертирует ненулевые сингулярные значения, оставляет нули нулями. Решает задачи least squares для неквадратных или сингулярных матриц. |
| Condition number | "Насколько чувствительно к ошибкам" | sigma_max / sigma_min. Большой condition number означает, что малые изменения входа вызывают большие изменения выхода. SVD показывает это напрямую. |
| Latent factor | "Скрытая переменная" | Измерение в низкоранговом пространстве, найденное SVD. В рекомендациях latent factor может соответствовать жанровому предпочтению. В NLP — теме. |
| Frobenius norm | "Общий размер матрицы" | Квадратный корень из суммы квадратов элементов. Равна квадратному корню из суммы квадратов сингулярных значений. Используется для измерения ошибки аппроксимации. |
| Eckart-Young theorem | "SVD дает лучшее сжатие" | Для любого целевого ранга k truncated SVD минимизирует ошибку аппроксимации среди всех возможных матриц ранга k. |
| Power iteration | "Найти крупнейший собственный вектор" | Многократно умножать случайный вектор на матрицу и нормировать. Сходится к собственному вектору с крупнейшим собственным значением. Строительный блок многих алгоритмов SVD. |

## Дополнительные материалы

- [Gilbert Strang: Linear Algebra and Its Applications, Chapter 7](https://math.mit.edu/~gs/linearalgebra/) - подробное изложение SVD с приложениями
- [3Blue1Brown: But what is the SVD?](https://www.youtube.com/watch?v=vSczTbgc8Rc) - геометрическая интуиция для SVD
- [We Recommend a Singular Value Decomposition](https://www.ams.org/publicoutreach/feature-column/fcarc-svd) - доступный обзор от American Mathematical Society
- [Netflix Prize and Matrix Factorization](https://sifter.org/~simon/journal/20061211.html) - оригинальный блог-пост Саймона Фанка об SVD для рекомендаций
- [Latent Semantic Analysis](https://en.wikipedia.org/wiki/Latent_semantic_analysis) - исходное NLP-приложение SVD
- [Numerical Linear Algebra by Trefethen and Bau](https://people.maths.ox.ac.uk/trefethen/text.html) - золотой стандарт для понимания алгоритмов SVD и их численных свойств
