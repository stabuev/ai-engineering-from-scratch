# Интуиция линейной алгебры

> Любая AI-модель — это просто матричная математика в модной шляпе.

**Тип:** Изучение
**Языки:** Python, Julia
**Предварительные требования:** Phase 0
**Время:** ~60 минут

## Цели обучения

- Реализовать операции с векторами и матрицами (сложение, скалярное произведение, умножение матриц) с нуля на Python
- Объяснить геометрически, что делают скалярное произведение, проекция и процесс Грама-Шмидта
- Определять линейную независимость, ранг и базис набора векторов с помощью приведения строк
- Связать понятия линейной алгебры с их применением в AI: embeddings, attention scores и LoRA

## Проблема

Откройте любую статью по ML. Уже на первой странице вы увидите векторы, матрицы, скалярные произведения и преобразования. Без интуиции линейной алгебры это просто символы. С ней вы видите, что нейронная сеть действительно делает: перемещает точки в пространстве.

Вам не нужно быть математиком. Вам нужно увидеть, что эти операции означают геометрически, а затем реализовать их самостоятельно.

## Концепция

### Векторы — это точки (и направления)

Вектор — это просто список чисел. Но эти числа что-то означают: это координаты в пространстве.

**2D-вектор [3, 2]:**

| x | y | Точка |
|---|---|-------|
| 3 | 2 | Вектор указывает из начала координат (0,0) в точку (3, 2) на плоскости |

Вектор имеет длину sqrt(3^2 + 2^2) = sqrt(13) и направлен вверх и вправо.

В AI векторы представляют все:
- Слово → вектор из 768 чисел (его "смысл" в embedding space)
- Изображение → вектор из миллионов значений пикселей
- Пользователь → вектор предпочтений

### Матрицы — это преобразования

Матрица преобразует один вектор в другой. Она может вращать, масштабировать, растягивать или проецировать.

```mermaid
graph LR
    subgraph Before
        A["Point A"]
        B["Point B"]
    end
    subgraph Matrix["Matrix Multiplication"]
        M["M (transformation)"]
    end
    subgraph After
        A2["Point A'"]
        B2["Point B'"]
    end
    A --> M
    B --> M
    M --> A2
    M --> B2
```

В AI матрицы И ЕСТЬ модель:
- Веса нейронной сети → матрицы, которые преобразуют вход в выход
- Attention scores → матрицы, которые решают, на чем сфокусироваться
- Embeddings → матрицы, которые отображают слова в векторы

### Скалярное произведение измеряет сходство

Скалярное произведение двух векторов показывает, насколько они похожи.

```
a · b = a₁×b₁ + a₂×b₂ + ... + aₙ×bₙ

Same direction:      a · b > 0  (similar)
Perpendicular:       a · b = 0  (unrelated)
Opposite direction:  a · b < 0  (dissimilar)
```

Именно так буквально работают поисковые системы, рекомендательные системы и RAG: находят векторы с высокими скалярными произведениями.

### Линейная независимость

Векторы линейно независимы, если ни один вектор в наборе нельзя записать как комбинацию остальных. Если v1, v2, v3 независимы, они натягивают 3D-пространство. Если один является комбинацией остальных, они натягивают только плоскость.

Почему это важно для AI: ваша матрица признаков должна иметь линейно независимые столбцы. Если два признака идеально коррелируют (линейно зависимы), модель не может различить их эффекты. Это вызывает мультиколлинеарность в регрессии: матрица весов становится нестабильной, и небольшие изменения входа приводят к резким скачкам выхода.

**Конкретный пример:**

```
v1 = [1, 0, 0]
v2 = [0, 1, 0]
v3 = [2, 1, 0]   # v3 = 2*v1 + v2
```

v1 и v2 независимы: ни один не является скалярным кратным или комбинацией другого. Но v3 = 2*v1 + v2, поэтому {v1, v2, v3} — зависимый набор. Все три вектора лежат в xy-плоскости. Как бы вы их ни комбинировали, вы не сможете получить [0, 0, 1]. У вас три вектора, но только две степени свободы.

В датасете: если feature_3 = 2*feature_1 + feature_2, добавление feature_3 не дает модели никакой новой информации. Хуже того, это делает нормальные уравнения вырожденными: уникального решения для весов не существует.

### Базис и ранг

Базис — это минимальный набор линейно независимых векторов, который натягивает все пространство. Количество базисных векторов равно размерности пространства.

Стандартный базис для 3D-пространства — {[1,0,0], [0,1,0], [0,0,1]}. Но любые три независимых вектора в 3D образуют допустимый базис. Выбор базиса — это выбор системы координат.

Ранг матрицы = количество линейно независимых столбцов = количество линейно независимых строк. Если rank < min(rows, cols), матрица имеет неполный ранг. Это означает:
- Система имеет бесконечно много решений (или ни одного)
- Информация теряется при преобразовании
- Матрицу нельзя обратить

| Ситуация | Ранг | Что это значит для ML |
|-----------|------|---------------------|
| Полный ранг (rank = min(m, n)) | Максимально возможный | Существует уникальное решение методом наименьших квадратов. Модель хорошо обусловлена. |
| Неполный ранг (rank < min(m, n)) | Ниже максимума | Признаки избыточны. Существует бесконечно много решений для весов. Нужна регуляризация. |
| Ранг 1 | 1 | Каждый столбец — масштабированная копия одного вектора. Все данные лежат на прямой. |
| Почти неполный ранг (малые сингулярные значения) | Численно низкий | Матрица плохо обусловлена. Крошечный шум во входе вызывает большие изменения выхода. Используйте усечение SVD или ridge regression. |

### Проекция

Проекция вектора **a** на вектор **b** дает компоненту **a** в направлении **b**:

```
proj_b(a) = (a dot b / b dot b) * b
```

Остаток (a - proj_b(a)) перпендикулярен b. Это ортогональное разложение лежит в основе подгонки методом наименьших квадратов.

Проекция встречается в ML повсюду:
- Linear regression минимизирует расстояние от наблюдений до пространства столбцов: решение И ЕСТЬ проекция
- PCA проецирует данные на направления максимальной дисперсии
- Attention в transformers вычисляет проекции queries на keys

```mermaid
graph LR
    subgraph Projection["Projection of a onto b"]
        direction TB
        O["Origin"] --> |"b (direction)"| B["b"]
        O --> |"a (original)"| A["a"]
        O --> |"proj_b(a)"| P["projection"]
        A -.-> |"residual (perpendicular)"| P
    end
```

**Пример:** a = [3, 4], b = [1, 0]

proj_b(a) = (3*1 + 4*0) / (1*1 + 0*0) * [1, 0] = 3 * [1, 0] = [3, 0]

Проекция отбрасывает y-компоненту. Это снижение размерности в простейшей форме: отбросить направления, которые вас не интересуют.

### Процесс Грама-Шмидта

Преобразование любого набора независимых векторов в ортонормированный базис. Ортонормированный означает, что каждый вектор имеет длину 1, а каждая пара векторов перпендикулярна.

Алгоритм:
1. Возьмите первый вектор и нормализуйте его
2. Возьмите второй вектор, вычтите его проекцию на первый, нормализуйте
3. Возьмите третий вектор, вычтите его проекции на все предыдущие векторы, нормализуйте
4. Повторите для оставшихся векторов

```
Input:  v1, v2, v3, ... (linearly independent)

u1 = v1 / |v1|

w2 = v2 - (v2 dot u1) * u1
u2 = w2 / |w2|

w3 = v3 - (v3 dot u1) * u1 - (v3 dot u2) * u2
u3 = w3 / |w3|

Output: u1, u2, u3, ... (orthonormal basis)
```

Так внутри работает QR-разложение. Q — это ортонормированный базис, R хранит коэффициенты проекций. QR-разложение используется для:
- Решения систем линейных уравнений (более устойчиво, чем метод Гаусса)
- Вычисления собственных значений (QR algorithm)
- Регрессии методом наименьших квадратов (стандартный численный метод)

## Соберите это

### Шаг 1: Векторы с нуля (Python)

```python
class Vector:
    def __init__(self, components):
        self.components = list(components)
        self.dim = len(self.components)

    def __add__(self, other):
        return Vector([a + b for a, b in zip(self.components, other.components)])

    def __sub__(self, other):
        return Vector([a - b for a, b in zip(self.components, other.components)])

    def dot(self, other):
        return sum(a * b for a, b in zip(self.components, other.components))

    def magnitude(self):
        return sum(x**2 for x in self.components) ** 0.5

    def normalize(self):
        mag = self.magnitude()
        return Vector([x / mag for x in self.components])

    def cosine_similarity(self, other):
        return self.dot(other) / (self.magnitude() * other.magnitude())

    def __repr__(self):
        return f"Vector({self.components})"


a = Vector([1, 2, 3])
b = Vector([4, 5, 6])

print(f"a + b = {a + b}")
print(f"a · b = {a.dot(b)}")
print(f"|a| = {a.magnitude():.4f}")
print(f"cosine similarity = {a.cosine_similarity(b):.4f}")
```

### Шаг 2: Матрицы с нуля (Python)

```python
class Matrix:
    def __init__(self, rows):
        self.rows = [list(row) for row in rows]
        self.shape = (len(self.rows), len(self.rows[0]))

    def __matmul__(self, other):
        if isinstance(other, Vector):
            return Vector([
                sum(self.rows[i][j] * other.components[j] for j in range(self.shape[1]))
                for i in range(self.shape[0])
            ])
        rows = []
        for i in range(self.shape[0]):
            row = []
            for j in range(other.shape[1]):
                row.append(sum(
                    self.rows[i][k] * other.rows[k][j]
                    for k in range(self.shape[1])
                ))
            rows.append(row)
        return Matrix(rows)

    def transpose(self):
        return Matrix([
            [self.rows[j][i] for j in range(self.shape[0])]
            for i in range(self.shape[1])
        ])

    def __repr__(self):
        return f"Matrix({self.rows})"


rotation_90 = Matrix([[0, -1], [1, 0]])
point = Vector([3, 1])

rotated = rotation_90 @ point
print(f"Original: {point}")
print(f"Rotated 90°: {rotated}")
```

### Шаг 3: Почему это важно для AI

```python
import random

random.seed(42)
weights = Matrix([[random.gauss(0, 0.1) for _ in range(3)] for _ in range(2)])
input_vector = Vector([1.0, 0.5, -0.3])

output = weights @ input_vector
print(f"Input (3D): {input_vector}")
print(f"Output (2D): {output}")
print("This is what a neural network layer does -- matrix multiplication.")
```

### Шаг 4: Версия на Julia

```julia
a = [1.0, 2.0, 3.0]
b = [4.0, 5.0, 6.0]

println("a + b = ", a + b)
println("a · b = ", a ⋅ b)       # Julia supports unicode operators
println("|a| = ", √(a ⋅ a))
println("cosine = ", (a ⋅ b) / (√(a ⋅ a) * √(b ⋅ b)))

# Matrix-vector multiplication
W = [0.1 -0.2 0.3; 0.4 0.5 -0.1]
x = [1.0, 0.5, -0.3]
println("Wx = ", W * x)
println("This is a neural network layer.")
```

### Шаг 5: Линейная независимость и проекция с нуля (Python)

```python
def is_linearly_independent(vectors):
    n = len(vectors)
    dim = len(vectors[0].components)
    mat = Matrix([v.components[:] for v in vectors])
    rows = [row[:] for row in mat.rows]
    rank = 0
    for col in range(dim):
        pivot = None
        for row in range(rank, len(rows)):
            if abs(rows[row][col]) > 1e-10:
                pivot = row
                break
        if pivot is None:
            continue
        rows[rank], rows[pivot] = rows[pivot], rows[rank]
        scale = rows[rank][col]
        rows[rank] = [x / scale for x in rows[rank]]
        for row in range(len(rows)):
            if row != rank and abs(rows[row][col]) > 1e-10:
                factor = rows[row][col]
                rows[row] = [rows[row][j] - factor * rows[rank][j] for j in range(dim)]
        rank += 1
    return rank == n


def project(a, b):
    scalar = a.dot(b) / b.dot(b)
    return Vector([scalar * x for x in b.components])


def gram_schmidt(vectors):
    orthonormal = []
    for v in vectors:
        w = v
        for u in orthonormal:
            proj = project(w, u)
            w = w - proj
        if w.magnitude() < 1e-10:
            continue
        orthonormal.append(w.normalize())
    return orthonormal


v1 = Vector([1, 0, 0])
v2 = Vector([1, 1, 0])
v3 = Vector([1, 1, 1])
basis = gram_schmidt([v1, v2, v3])
for i, u in enumerate(basis):
    print(f"u{i+1} = {u}")
    print(f"  |u{i+1}| = {u.magnitude():.6f}")

print(f"u1 · u2 = {basis[0].dot(basis[1]):.6f}")
print(f"u1 · u3 = {basis[0].dot(basis[2]):.6f}")
print(f"u2 · u3 = {basis[1].dot(basis[2]):.6f}")
```

### Ожидаемый вывод

Запустите `code/vectors.py` — последние строки должны быть такими:

```
Identity 2x2 rank: 2
[[1,2],[2,4]] rank: 1
[[1,0,0],[0,1,0]] rank: 2

=== Neural Network Layer (Matrix x Vector) ===
Input (3D):  Vector([1.0, 0.5, -0.3])
Output (2D): Vector([-0.019714737127338927, 0.10873956075097067])
^ This is literally what a neural network layer does.
```

## Используйте это

Теперь то же самое с NumPy — именно так вы будете использовать это на практике:

```python
import numpy as np

a = np.array([1, 2, 3], dtype=float)
b = np.array([4, 5, 6], dtype=float)

print(f"a + b = {a + b}")
print(f"a · b = {np.dot(a, b)}")
print(f"|a| = {np.linalg.norm(a):.4f}")
print(f"cosine = {np.dot(a, b) / (np.linalg.norm(a) * np.linalg.norm(b)):.4f}")

W = np.random.randn(2, 3) * 0.1
x = np.array([1.0, 0.5, -0.3])
print(f"Wx = {W @ x}")
```

### Ранг, проекция и QR с NumPy

```python
import numpy as np

A = np.array([[1, 2], [2, 4]])
print(f"Rank: {np.linalg.matrix_rank(A)}")

a = np.array([3, 4])
b = np.array([1, 0])
proj = (np.dot(a, b) / np.dot(b, b)) * b
print(f"Projection of {a} onto {b}: {proj}")

Q, R = np.linalg.qr(np.random.randn(3, 3))
print(f"Q is orthogonal: {np.allclose(Q @ Q.T, np.eye(3))}")
print(f"R is upper triangular: {np.allclose(R, np.triu(R))}")
```

### PyTorch: тензоры — это векторы с autodiff

```python
import torch

x = torch.randn(3, requires_grad=True)
y = torch.tensor([1.0, 0.0, 0.0])

similarity = torch.dot(x, y)
similarity.backward()

print(f"x = {x.data}")
print(f"y = {y.data}")
print(f"dot product = {similarity.item():.4f}")
print(f"d(dot)/dx = {x.grad}")
```

Градиент скалярного произведения по x — это просто y. PyTorch вычислил это автоматически. Каждая операция в нейронной сети строится из таких операций: умножения матриц, скалярных произведений, проекций, — а autodiff отслеживает градиенты через все них.

Вы только что с нуля построили то, что NumPy делает в одну строку. Теперь вы знаете, что происходит под капотом.

## Доведите до результата

Этот урок создает:
- `outputs/prompt-linear-algebra-tutor.md` — prompt для AI assistants, чтобы обучать линейной алгебре через геометрическую интуицию

## Связи

Все в этом уроке связано с конкретными частями современного AI:

| Концепция | Где встречается |
|---------|------------------|
| Скалярное произведение | Attention scores в transformers, cosine similarity в RAG |
| Умножение матриц | Каждый слой нейронной сети, каждое линейное преобразование |
| Линейная независимость | Feature selection, избегание мультиколлинеарности |
| Ранг | Определение разрешимости системы, LoRA (low-rank adaptation) |
| Проекция | Linear regression (проекция на пространство столбцов), PCA |
| Gram-Schmidt / QR | Численные решатели, вычисление собственных значений |
| Ортонормированный базис | Устойчивые численные вычисления, whitening transforms |

LoRA заслуживает отдельного упоминания. Она дообучает большие языковые модели, раскладывая обновления весов на матрицы низкого ранга. Вместо обновления матрицы весов 4096x4096 (16M параметров) LoRA обновляет две матрицы размера 4096x16 и 16x4096 (131K параметров). Ограничение rank-16 означает, что LoRA предполагает: обновление весов живет в 16-мерном подпространстве полного 4096-мерного пространства. Это линейная алгебра, выполняющая реальную работу.

## Упражнения

1. Реализуйте `Vector.angle_between(other)`, который возвращает угол в градусах между двумя векторами
2. Создайте 2D-матрицу масштабирования, которая удваивает x-координату и утраивает y-координату, затем примените ее к вектору [1, 1]
3. Для 5 случайных word-like vectors (размерность 50) найдите две самые похожие с помощью cosine similarity
4. Проверьте, что результат Gram-Schmidt действительно ортонормирован: убедитесь, что каждая пара имеет скалярное произведение 0, а каждый вектор имеет длину 1
5. Создайте матрицу 3x3 с рангом 2. Проверьте с помощью метода `rank()`. Затем объясните, какой геометрический объект натягивают столбцы.
6. Спроецируйте вектор [1, 2, 3] на [1, 1, 1]. Что результат представляет геометрически?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Вектор | "Стрелка" | Список чисел, представляющий точку или направление в n-мерном пространстве |
| Матрица | "Таблица чисел" | Преобразование, которое отображает векторы из одного пространства в другое |
| Скалярное произведение | "Умножить и сложить" | Мера того, насколько два вектора сонаправлены: основа similarity search |
| Embedding | "Какая-то AI-магия" | Вектор, который представляет смысл чего-либо (слова, изображения, пользователя) |
| Линейная независимость | "Они не перекрываются" | Ни один вектор в наборе нельзя записать как комбинацию остальных |
| Ранг | "Сколько измерений" | Количество линейно независимых столбцов (или строк) в матрице |
| Проекция | "Тень" | Компонента одного вектора в направлении другого |
| Базис | "Оси координат" | Минимальный набор независимых векторов, натягивающий пространство |
| Ортонормированный | "Перпендикулярные единичные векторы" | Векторы, которые взаимно перпендикулярны и каждый имеет длину 1 |
