# Правило цепочки и автоматическое дифференцирование

> Правило цепочки - это двигатель каждой нейронной сети, которая учится.

**Тип:** Практика
**Язык:** Python
**Предварительные требования:** Phase 1, Lesson 04 (Derivatives & Gradients)
**Время:** ~90 минут

## Цели обучения

- Построить минимальный autograd engine (класс Value), который записывает операции и вычисляет градиенты через reverse-mode autodiff
- Реализовать forward и backward passes через вычислительный граф с помощью topological sort
- Построить и обучить multi-layer perceptron на XOR, используя только autograd engine, написанный с нуля
- Проверить корректность autodiff с помощью gradient checking относительно численных finite differences

## Проблема

Вы умеете вычислять производные простых функций. Но нейронная сеть - не простая функция. Это сотни функций, составленных вместе: matrix multiply, add bias, apply activation, matrix multiply again, softmax, cross-entropy loss. Выход - это функция от функции от функции.

Чтобы обучить сеть, вам нужен градиент loss по каждому отдельному весу. Делать это вручную невозможно для миллионов параметров. Делать это численно (finite differences) слишком медленно.

Правило цепочки дает математику. Automatic differentiation дает алгоритм. Вместе они позволяют вычислять точные градиенты через произвольные композиции функций за время, пропорциональное одному forward pass.

Так работают PyTorch, TensorFlow и JAX. Вы построите миниатюрную версию с нуля.

## Концепция

### Правило цепочки

Если `y = f(g(x))`, производная `y` по `x` равна:

```
dy/dx = dy/dg * dg/dx = f'(g(x)) * g'(x)
```

Умножайте производные вдоль цепочки. Каждое звено добавляет свою локальную производную.

Пример: `y = sin(x^2)`

```
g(x) = x^2       g'(x) = 2x
f(g) = sin(g)     f'(g) = cos(g)

dy/dx = cos(x^2) * 2x
```

Для более глубоких композиций цепочка удлиняется:

```
y = f(g(h(x)))

dy/dx = f'(g(h(x))) * g'(h(x)) * h'(x)
```

Каждый слой нейронной сети - одно звено этой цепочки.

### Вычислительные графы

Вычислительный граф делает правило цепочки наглядным. Каждая операция становится узлом. Данные текут вперед через граф. Градиенты текут назад.

**Forward pass (вычисление значений):**

```mermaid
graph TD
    x1["x1 = 2"] --> mul["* (multiply)"]
    x2["x2 = 3"] --> mul
    mul -->|"a = 6"| add["+ (add)"]
    b["b = 1"] --> add
    add -->|"c = 7"| relu["relu"]
    relu -->|"y = 7"| y["output y"]
```

**Backward pass (вычисление градиентов):**

```mermaid
graph TD
    dy["dy/dy = 1"] -->|"relu'(c)=1 since c>0"| dc["dy/dc = 1"]
    dc -->|"dc/da = 1"| da["dy/da = 1"]
    dc -->|"dc/db = 1"| db["dy/db = 1"]
    da -->|"da/dx1 = x2 = 3"| dx1["dy/dx1 = 3"]
    da -->|"da/dx2 = x1 = 2"| dx2["dy/dx2 = 2"]
```

Backward pass применяет правило цепочки в каждом узле, распространяя градиенты от выхода к входам.

### Forward Mode и Reverse Mode

Есть два способа применить правило цепочки через граф.

**Forward mode** начинает с входов и проталкивает производные вперед. Он вычисляет `dx/dx = 1` и распространяет производную через каждую операцию. Хорош, когда входов мало, а выходов много.

```
Forward mode: seed dx/dx = 1, propagate forward

  x = 2       (dx/dx = 1)
  a = x^2     (da/dx = 2x = 4)
  y = sin(a)  (dy/dx = cos(a) * da/dx = cos(4) * 4 = -2.615)
```

**Reverse mode** начинает с выхода и тянет градиенты назад. Он вычисляет `dy/dy = 1` и распространяет их через каждую операцию в обратном порядке. Хорош, когда входов много, а выходов мало.

```
Reverse mode: seed dy/dy = 1, propagate backward

  y = sin(a)  (dy/dy = 1)
  a = x^2     (dy/da = cos(a) = cos(4) = -0.654)
  x = 2       (dy/dx = dy/da * da/dx = -0.654 * 4 = -2.615)
```

У нейронных сетей миллионы входов (весов) и один выход (loss). Reverse mode вычисляет все градиенты за один backward pass. Поэтому backpropagation использует reverse mode.

| Режим | Начальное значение | Направление | Лучший случай |
|------|------|-----------|-----------|
| Forward | `dx_i/dx_i = 1` | От входа к выходу | Мало входов, много выходов |
| Reverse | `dy/dy = 1` | От выхода к входу | Много входов, мало выходов (нейронные сети) |

### Дуальные числа для Forward Mode

Forward mode можно элегантно реализовать с помощью дуальных чисел. Дуальное число имеет вид `a + b*epsilon`, где `epsilon^2 = 0`.

```
Dual number: (value, derivative)

(2, 1) means: value is 2, derivative w.r.t. x is 1

Arithmetic rules:
  (a, a') + (b, b') = (a+b, a'+b')
  (a, a') * (b, b') = (a*b, a'*b + a*b')
  sin(a, a')         = (sin(a), cos(a)*a')
```

Инициализируйте входную переменную производной 1. Производная автоматически распространяется через каждую операцию.

### Построение Autograd Engine

Autograd engine нужны три вещи:

1. **Обертка значений.** Обернуть каждое число в объект, который хранит значение и градиент.
2. **Запись графа.** Каждая операция записывает свои входы и локальную функцию градиента.
3. **Backward pass.** Выполнить topological sort графа, затем пройти его в обратном порядке, применяя правило цепочки в каждом узле.

Именно это делает `autograd` в PyTorch. Класс `torch.Tensor` оборачивает значения, записывает операции при `requires_grad=True` и вычисляет градиенты при вызове `.backward()`.

### Как PyTorch Autograd работает под капотом

Когда вы пишете код PyTorch:

```python
x = torch.tensor(2.0, requires_grad=True)
y = x ** 2 + 3 * x + 1
y.backward()
print(x.grad)  # 7.0 = 2*x + 3 = 2*2 + 3
```

PyTorch внутри:

1. Создает узел `Tensor` для `x` с `requires_grad=True`
2. Каждая операция (`**`, `*`, `+`) создает новый узел и записывает backward function
3. `y.backward()` запускает reverse-mode autodiff по записанному графу
4. `grad_fn` каждого узла вычисляет локальные градиенты и передает их родительским узлам
5. Градиенты накапливаются в атрибутах `.grad` через сложение (а не замену)

Граф динамический (define-by-run). Новый граф строится на каждом forward pass. Поэтому PyTorch поддерживает control flow (if/else, loops) внутри моделей.

## Соберите это

### Шаг 1: Класс Value

```python
class Value:
    def __init__(self, data, children=(), op=''):
        self.data = data
        self.grad = 0.0
        self._backward = lambda: None
        self._prev = set(children)
        self._op = op

    def __repr__(self):
        return f"Value(data={self.data:.4f}, grad={self.grad:.4f})"
```

Каждый `Value` хранит числовые данные, свой градиент (сначала ноль), backward function и указатели на дочерние узлы, которые его породили.

### Шаг 2: Арифметические операции с отслеживанием градиентов

```python
    def __add__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data + other.data, (self, other), '+')
        def _backward():
            self.grad += out.grad
            other.grad += out.grad
        out._backward = _backward
        return out

    def __mul__(self, other):
        other = other if isinstance(other, Value) else Value(other)
        out = Value(self.data * other.data, (self, other), '*')
        def _backward():
            self.grad += other.data * out.grad
            other.grad += self.data * out.grad
        out._backward = _backward
        return out

    def relu(self):
        out = Value(max(0, self.data), (self,), 'relu')
        def _backward():
            self.grad += (1.0 if out.data > 0 else 0.0) * out.grad
        out._backward = _backward
        return out
```

Каждая операция создает closure, который знает, как вычислить локальные градиенты и умножить их на upstream gradient (`out.grad`). `+=` обрабатывает случай, когда значение используется в нескольких операциях.

### Шаг 3: Backward pass

```python
    def backward(self):
        topo = []
        visited = set()
        def build_topo(v):
            if v not in visited:
                visited.add(v)
                for child in v._prev:
                    build_topo(child)
                topo.append(v)
        build_topo(self)

        self.grad = 1.0
        for v in reversed(topo):
            v._backward()
```

Topological sort гарантирует, что градиент каждого узла полностью вычислен до того, как он распространится к его дочерним узлам. Начальный градиент равен 1.0 (dy/dy = 1).

### Шаг 4: Больше операций для полноценного engine

Базовый класс Value поддерживает сложение, умножение и relu. Настоящему autograd engine нужно больше. Вот операции, которые нужны для построения нейронных сетей:

```python
    def __neg__(self):
        return self * -1

    def __sub__(self, other):
        return self + (-other)

    def __radd__(self, other):
        return self + other

    def __rmul__(self, other):
        return self * other

    def __rsub__(self, other):
        return other + (-self)

    def __pow__(self, n):
        out = Value(self.data ** n, (self,), f'**{n}')
        def _backward():
            self.grad += n * (self.data ** (n - 1)) * out.grad
        out._backward = _backward
        return out

    def __truediv__(self, other):
        return self * (other ** -1) if isinstance(other, Value) else self * (Value(other) ** -1)

    def exp(self):
        import math
        e = math.exp(self.data)
        out = Value(e, (self,), 'exp')
        def _backward():
            self.grad += e * out.grad
        out._backward = _backward
        return out

    def log(self):
        import math
        out = Value(math.log(self.data), (self,), 'log')
        def _backward():
            self.grad += (1.0 / self.data) * out.grad
        out._backward = _backward
        return out

    def tanh(self):
        import math
        t = math.tanh(self.data)
        out = Value(t, (self,), 'tanh')
        def _backward():
            self.grad += (1 - t ** 2) * out.grad
        out._backward = _backward
        return out
```

**Почему важна каждая операция:**

| Операция | Правило backward | Где используется |
|-----------|--------------|---------|
| `__sub__` | Переиспользует add + neg | Вычисление loss (pred - target) |
| `__pow__` | n * x^(n-1) | Polynomial activations (полиномиальные активации), MSE (error^2) |
| `__truediv__` | Переиспользует mul + pow(-1) | Normalization, масштабирование learning rate |
| `exp` | exp(x) * upstream | Softmax, log-likelihood |
| `log` | (1/x) * upstream | Cross-entropy loss, log probabilities |
| `tanh` | (1 - tanh^2) * upstream | Классическая activation function |

Умная часть: `__sub__` и `__truediv__` определены через уже существующие операции. Они получают правильные градиенты бесплатно, потому что правило цепочки композируется через базовые операции add/mul/pow.

### Шаг 5: Мини-MLP с нуля

С полноценным классом Value можно построить нейронную сеть. Без PyTorch. Без NumPy. Только Values и правило цепочки.

```python
import random

class Neuron:
    def __init__(self, n_inputs):
        self.w = [Value(random.uniform(-1, 1)) for _ in range(n_inputs)]
        self.b = Value(0.0)

    def __call__(self, x):
        act = sum((wi * xi for wi, xi in zip(self.w, x)), self.b)
        return act.tanh()

    def parameters(self):
        return self.w + [self.b]

class Layer:
    def __init__(self, n_inputs, n_outputs):
        self.neurons = [Neuron(n_inputs) for _ in range(n_outputs)]

    def __call__(self, x):
        return [n(x) for n in self.neurons]

    def parameters(self):
        return [p for n in self.neurons for p in n.parameters()]

class MLP:
    def __init__(self, sizes):
        self.layers = [Layer(sizes[i], sizes[i+1]) for i in range(len(sizes)-1)]

    def __call__(self, x):
        for layer in self.layers:
            x = layer(x)
        return x[0] if len(x) == 1 else x

    def parameters(self):
        return [p for layer in self.layers for p in layer.parameters()]
```

`Neuron` вычисляет `tanh(w1*x1 + w2*x2 + ... + b)`. `Layer` - это список нейронов. `MLP` складывает слои. Каждый вес - это `Value`, поэтому вызов `loss.backward()` распространяет градиенты к каждому параметру.

**Обучение на XOR:**

```python
random.seed(42)
model = MLP([2, 4, 1])  # 2 inputs, 4 hidden neurons, 1 output

xs = [[0, 0], [0, 1], [1, 0], [1, 1]]
ys = [-1, 1, 1, -1]  # XOR pattern (using -1/1 for tanh)

for step in range(100):
    preds = [model(x) for x in xs]
    loss = sum((p - y) ** 2 for p, y in zip(preds, ys))

    for p in model.parameters():
        p.grad = 0.0
    loss.backward()

    lr = 0.05
    for p in model.parameters():
        p.data -= lr * p.grad

    if step % 20 == 0:
        print(f"step {step:3d}  loss = {loss.data:.4f}")

print("\nPredictions after training:")
for x, y in zip(xs, ys):
    print(f"  input={x}  target={y:2d}  pred={model(x).data:6.3f}")
```

Это micrograd. Полный цикл обучения нейронной сети на чистом Python с automatic differentiation. Каждый коммерческий deep learning framework делает то же самое в большом масштабе.

### Шаг 6: Gradient checking

Как понять, что ваш autodiff правильный? Сравнить его с численными производными. Это gradient checking.

```python
def gradient_check(build_expr, x_val, h=1e-7):
    x = Value(x_val)
    y = build_expr(x)
    y.backward()
    autodiff_grad = x.grad

    y_plus = build_expr(Value(x_val + h)).data
    y_minus = build_expr(Value(x_val - h)).data
    numerical_grad = (y_plus - y_minus) / (2 * h)

    diff = abs(autodiff_grad - numerical_grad)
    return autodiff_grad, numerical_grad, diff
```

Проверьте на сложном выражении:

```python
def expr(x):
    return (x ** 3 + x * 2 + 1).tanh()

ad, num, diff = gradient_check(expr, 0.5)
print(f"Autodiff:  {ad:.8f}")
print(f"Numerical: {num:.8f}")
print(f"Difference: {diff:.2e}")
# Difference should be < 1e-5
```

Gradient checking необходим при реализации новых операций. Если в вашем backward pass есть ошибка, численная проверка ее поймает. Каждая серьезная реализация deep learning запускает gradient checks во время разработки.

**Когда использовать gradient checking:**

| Ситуация | Делать gradient check? |
|-----------|-------------------|
| Добавляете новую операцию в autograd | Да, всегда |
| Отлаживаете training loop, который не сходится | Да, сначала проверьте градиенты |
| Production training | Нет, слишком медленно (2x forward passes на параметр) |
| Unit tests для autograd code | Да, автоматизируйте |

### Шаг 7: Проверка относительно ручного вычисления

```python
x1 = Value(2.0)
x2 = Value(3.0)
a = x1 * x2          # a = 6.0
b = a + Value(1.0)    # b = 7.0
y = b.relu()          # y = 7.0

y.backward()

print(f"y = {y.data}")          # 7.0
print(f"dy/dx1 = {x1.grad}")   # 3.0 (= x2)
print(f"dy/dx2 = {x2.grad}")   # 2.0 (= x1)
```

Ручная проверка: `y = relu(x1*x2 + 1)`. Так как `x1*x2 + 1 = 7 > 0`, relu является identity.
`dy/dx1 = x2 = 3`. `dy/dx2 = x1 = 2`. Engine совпадает.

### Ожидаемый вывод

Запустите `code/autodiff.py` — последние строки должны быть такими:

```
  DONE

=== Verify against PyTorch ===
  Our engine: dy/dx1=3.0, dy/dx2=2.0
  PyTorch:    dy/dx1=3.0, dy/dx2=2.0
  MATCH

All demos passed.
```

## Используйте это

### Проверка относительно PyTorch

```python
import torch

x1 = torch.tensor(2.0, requires_grad=True)
x2 = torch.tensor(3.0, requires_grad=True)
a = x1 * x2
b = a + 1.0
y = torch.relu(b)
y.backward()

print(f"PyTorch dy/dx1 = {x1.grad.item()}")  # 3.0
print(f"PyTorch dy/dx2 = {x2.grad.item()}")  # 2.0
```

Те же градиенты. Ваш engine вычисляет тот же результат, что и PyTorch, потому что математика та же: reverse-mode autodiff через правило цепочки.

### Более сложное выражение

```python
a = Value(2.0)
b = Value(-3.0)
c = Value(10.0)
f = (a * b + c).relu()  # relu(2*(-3) + 10) = relu(4) = 4

f.backward()
print(f"df/da = {a.grad}")  # -3.0 (= b)
print(f"df/db = {b.grad}")  #  2.0 (= a)
print(f"df/dc = {c.grad}")  #  1.0
```

## Доведите до результата

Этот урок создает:
- `outputs/skill-autodiff.md` -- skill для построения и отладки autograd systems
- `code/autodiff.py` -- минимальный autograd engine, который можно расширять

Класс Value, построенный здесь, является основой цикла обучения нейронной сети в Phase 3.

## Упражнения

1. Добавьте `__pow__` в класс Value, чтобы можно было вычислять `x ** n`. Проверьте, что `d/dx(x^3)` при `x=2` равна `12.0`.

2. Добавьте `tanh` как activation function. Проверьте, что `tanh'(0) = 1` и `tanh'(2) = 0.0707` (примерно).

3. Постройте вычислительный граф для одного нейрона: `y = relu(w1*x1 + w2*x2 + b)`. Вычислите все пять градиентов и проверьте относительно PyTorch.

4. Реализуйте forward-mode autodiff с помощью дуальных чисел. Создайте класс `Dual` и проверьте, что он дает те же производные, что и ваш reverse-mode engine.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Chain rule | "Умножить производные" | Производная составных функций равна произведению локальной производной каждой функции, вычисленной в правильной точке |
| Computational graph | "Диаграмма сети" | Направленный ациклический граф, где узлы - операции, а ребра переносят значения (вперед) или градиенты (назад) |
| Forward mode | "Проталкивать производные вперед" | Autodiff, который распространяет производные от входов к выходам. Один проход на входную переменную. |
| Reverse mode | "Backpropagation" | Autodiff, который распространяет градиенты от выходов к входам. Один проход на выходную переменную. |
| Autograd | "Автоматические градиенты" | Система, которая записывает операции над значениями, строит граф и вычисляет точные градиенты через правило цепочки |
| Dual numbers | "Значение плюс производная" | Числа вида a + b*epsilon (epsilon^2 = 0), которые несут информацию о производной через арифметические операции |
| Topological sort | "Порядок зависимостей" | Упорядочивание узлов графа так, чтобы каждый узел шел после всех своих зависимостей. Нужно для правильного распространения градиентов. |
| Gradient accumulation | "Добавлять, а не заменять" | Когда значение идет в несколько операций, его градиент - это сумма всех входящих вкладов градиента |
| Dynamic graph | "Define by run" | Вычислительный граф, перестраиваемый на каждом forward pass, что позволяет использовать Python control flow внутри моделей (стиль PyTorch) |
| Gradient checking | "Численная проверка" | Сравнение autodiff gradients с численными finite-difference gradients для проверки корректности. Необходимо для отладки. |
| MLP | "Multi-layer perceptron" | Нейронная сеть с одним или несколькими скрытыми слоями нейронов. Каждый нейрон вычисляет взвешенную сумму плюс bias, затем применяет activation function. |
| Neuron | "Взвешенная сумма + activation" | Базовая единица: output = activation(w1*x1 + w2*x2 + ... + b). Веса и bias - обучаемые параметры. |

## Дополнительное чтение

- [3Blue1Brown: Backpropagation calculus](https://www.youtube.com/watch?v=tIeHLnjs5U8) -- визуальное объяснение правила цепочки в нейронных сетях
- [PyTorch Autograd mechanics](https://pytorch.org/docs/stable/notes/autograd.html) -- как работает настоящая система
- [Baydin et al., Automatic Differentiation in Machine Learning: a Survey](https://arxiv.org/abs/1502.05767) -- подробный обзор
