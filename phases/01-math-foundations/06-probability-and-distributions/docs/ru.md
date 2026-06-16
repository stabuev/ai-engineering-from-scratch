# Вероятность и распределения

> Вероятность - это язык, на котором AI выражает неопределенность.

**Тип:** Изучение
**Язык:** Python
**Предварительные требования:** Phase 1, Lessons 01-04
**Время:** ~75 минут

## Цели обучения

- Реализовать PMFs и PDFs с нуля для Bernoulli, categorical, Poisson, uniform и normal distributions
- Вычислять expected value, variance и использовать Central Limit Theorem, чтобы объяснить, почему Gaussians встречаются повсюду
- Построить функции softmax и log-softmax с приемом численной устойчивости (вычитание max logit)
- Вычислять cross-entropy loss из logits и связывать его с negative log-likelihood

## Проблема

Классификатор выдает `[0.03, 0.91, 0.06]`. Language model выбирает следующее слово из 50,000 кандидатов. Diffusion model генерирует изображения, сэмплируя из выученных распределений. Все это - вероятность в действии.

Каждое предсказание модели - это вероятностное распределение. Каждая loss function измеряет, насколько предсказанное распределение далеко от истинного. Каждый шаг обучения корректирует параметры, чтобы одно распределение стало больше похоже на другое. Без вероятности вы не сможете прочитать ни одной ML-статьи, отладить ни одной модели или понять, почему training loss стал NaN.

## Концепция

### События, пространство элементарных исходов и вероятность

Пространство элементарных исходов S - это множество всех возможных исходов. Событие - это подмножество пространства исходов. Вероятность отображает события в числа между 0 и 1.

```
Coin flip:
  S = {H, T}
  P(H) = 0.5,  P(T) = 0.5

Single die roll:
  S = {1, 2, 3, 4, 5, 6}
  P(even) = P({2, 4, 6}) = 3/6 = 0.5
```

Три аксиомы определяют всю теорию вероятностей:
1. P(A) >= 0 для любого события A
2. P(S) = 1 (что-то всегда происходит)
3. P(A or B) = P(A) + P(B), когда A и B не могут произойти одновременно

Все остальное (теорема Байеса, математические ожидания, распределения) следует из этих трех правил.

### Условная вероятность и независимость

P(A|B) - это вероятность A при условии, что B произошло.

```
P(A|B) = P(A and B) / P(B)

Example: deck of cards
  P(King | Face card) = P(King and Face card) / P(Face card)
                      = (4/52) / (12/52)
                      = 4/12 = 1/3
```

Два события независимы, если знание одного ничего не говорит о другом:

```
Independent:   P(A|B) = P(A)
Equivalent to: P(A and B) = P(A) * P(B)
```

Подбрасывания монеты независимы. Вытягивание карт без возвращения - нет.

### Probability Mass Functions (функции вероятности) и Probability Density Functions (функции плотности)

Дискретные случайные величины имеют probability mass function (PMF). У каждого исхода есть конкретная вероятность, которую можно прочитать напрямую.

```
PMF: P(X = k)

Fair die:
  P(X = 1) = 1/6
  P(X = 2) = 1/6
  ...
  P(X = 6) = 1/6

  Sum of all probabilities = 1
```

Непрерывные случайные величины имеют probability density function (PDF). Плотность в одной точке не является вероятностью. Вероятность получается интегрированием плотности по интервалу.

```
PDF: f(x)

P(a <= X <= b) = integral of f(x) from a to b

f(x) can be greater than 1 (density, not probability)
integral from -inf to +inf of f(x) dx = 1
```

Это различие важно в ML. Выходы классификации - это PMFs (дискретные варианты). Latent spaces в VAE используют PDFs (непрерывные).

### Распространенные распределения

**Bernoulli:** одно испытание, два исхода. Моделирует binary classification.

```
P(X = 1) = p
P(X = 0) = 1 - p
Mean = p,  Variance = p(1-p)
```

**Categorical:** одно испытание, k исходов. Моделирует multi-class classification (softmax output).

```
P(X = i) = p_i,  where sum of p_i = 1
Example: P(cat) = 0.7,  P(dog) = 0.2,  P(bird) = 0.1
```

**Uniform:** все исходы одинаково вероятны. Используется для random initialization.

```
Discrete: P(X = k) = 1/n for k in {1, ..., n}
Continuous: f(x) = 1/(b-a) for x in [a, b]
```

**Normal (Gaussian):** колоколообразная кривая. Параметризуется средним (mu) и дисперсией (sigma^2).

```
f(x) = (1 / sqrt(2*pi*sigma^2)) * exp(-(x - mu)^2 / (2*sigma^2))

Standard normal: mu = 0, sigma = 1
  68% of data within 1 sigma
  95% within 2 sigma
  99.7% within 3 sigma
```

**Poisson:** количества редких событий на фиксированном интервале. Моделирует интенсивности событий.

```
P(X = k) = (lambda^k * e^(-lambda)) / k!
Mean = lambda,  Variance = lambda
```

### Математическое ожидание и дисперсия

Математическое ожидание - это взвешенный средний исход.

```
Discrete:   E[X] = sum of x_i * P(X = x_i)
Continuous: E[X] = integral of x * f(x) dx
```

Дисперсия измеряет разброс вокруг среднего.

```
Var(X) = E[(X - E[X])^2] = E[X^2] - (E[X])^2
Standard deviation = sqrt(Var(X))
```

В ML математическое ожидание появляется как loss function (средняя потеря по распределению данных). Дисперсия говорит об устойчивости модели. Высокая variance градиентов означает шумное обучение.

### Совместные и маргинальные распределения

Совместное распределение P(X, Y) описывает две случайные величины вместе.

Пример совместной PMF (X = weather, Y = umbrella):

| | Y=0 (no umbrella) | Y=1 (umbrella) | Marginal P(X) |
|---|---|---|---|
| X=0 (sun) | 0.40 | 0.10 | P(X=0) = 0.50 |
| X=1 (rain) | 0.05 | 0.45 | P(X=1) = 0.50 |
| **Marginal P(Y)** | P(Y=0) = 0.45 | P(Y=1) = 0.55 | 1.00 |

Маргинальное распределение суммирует по другой переменной:

```
P(X = x) = sum over all y of P(X = x, Y = y)
```

Суммы строк и столбцов в таблице выше - это маргиналы.

### Почему нормальное распределение появляется повсюду

Central Limit Theorem: сумма (или среднее) многих независимых случайных величин сходится к нормальному распределению независимо от исходного распределения.

```
Roll 1 die:  uniform distribution (flat)
Average of 2 dice:  triangular (peaked)
Average of 30 dice: nearly perfect bell curve

This works for ANY starting distribution.
```

Вот почему:
- Ошибки измерений примерно нормальны (много маленьких независимых источников)
- Инициализации весов в нейронных сетях используют normal distributions
- Шум градиентов в SGD примерно нормален (сумма многих sample gradients)
- Нормальное распределение - это распределение с максимальной энтропией при заданных среднем и дисперсии

### Логарифмы вероятностей

Сырые вероятности вызывают численные проблемы. Перемножение множества малых вероятностей быстро приводит к underflow до нуля.

```
P(sentence) = P(word1) * P(word2) * ... * P(word_n)
            = 0.01 * 0.003 * 0.02 * ...
            -> 0.0 (underflow after ~30 terms)
```

Логарифмы вероятностей решают это. Умножения превращаются в сложения.

```
log P(sentence) = log P(word1) + log P(word2) + ... + log P(word_n)
                = -4.6 + -5.8 + -3.9 + ...
                -> finite number (no underflow)
```

Правила:
- log(a * b) = log(a) + log(b)
- log probabilities всегда <= 0 (так как 0 < P <= 1)
- Более отрицательное значение = менее вероятно
- Cross-entropy loss - это отрицательный логарифм вероятности правильного класса

### Softmax как вероятностное распределение

Нейронные сети выдают сырые scores (logits). Softmax преобразует их в корректное вероятностное распределение.

```
softmax(z_i) = exp(z_i) / sum(exp(z_j) for all j)

Properties:
  - All outputs are in (0, 1)
  - All outputs sum to 1
  - Preserves relative ordering of inputs
  - exp() amplifies differences between logits
```

Прием softmax: вычесть max logit перед возведением в экспоненту, чтобы предотвратить overflow.

```
z = [100, 101, 102]
exp(102) = overflow

z_shifted = z - max(z) = [-2, -1, 0]
exp(0) = 1  (safe)

Same result, no overflow.
```

Log-softmax объединяет softmax и log для численной устойчивости. PyTorch использует это внутри для cross-entropy loss.

### Sampling

Sampling означает получение случайных значений из распределения. В ML:
- Dropout случайно выбирает, какие нейроны обнулить
- Data augmentation сэмплирует случайные преобразования
- Language models сэмплируют следующий token из предсказанного распределения
- Diffusion models сэмплируют шум и постепенно denoise

Sampling из произвольных распределений требует техник вроде inverse transform sampling, rejection sampling или reparameterization trick (используется в VAEs).

## Соберите это

### Шаг 1: Основы вероятности

```python
import math
import random

def factorial(n):
    result = 1
    for i in range(2, n + 1):
        result *= i
    return result

def combinations(n, k):
    return factorial(n) // (factorial(k) * factorial(n - k))

def conditional_probability(p_a_and_b, p_b):
    return p_a_and_b / p_b

p_king_given_face = conditional_probability(4/52, 12/52)
print(f"P(King | Face card) = {p_king_given_face:.4f}")
```

### Шаг 2: PMF и PDF с нуля

```python
def bernoulli_pmf(k, p):
    return p if k == 1 else (1 - p)

def categorical_pmf(k, probs):
    return probs[k]

def poisson_pmf(k, lam):
    return (lam ** k) * math.exp(-lam) / factorial(k)

def uniform_pdf(x, a, b):
    if a <= x <= b:
        return 1.0 / (b - a)
    return 0.0

def normal_pdf(x, mu, sigma):
    coeff = 1.0 / (sigma * math.sqrt(2 * math.pi))
    exponent = -0.5 * ((x - mu) / sigma) ** 2
    return coeff * math.exp(exponent)
```

### Шаг 3: Математическое ожидание и дисперсия

```python
def expected_value(values, probabilities):
    return sum(v * p for v, p in zip(values, probabilities))

def variance(values, probabilities):
    mu = expected_value(values, probabilities)
    return sum(p * (v - mu) ** 2 for v, p in zip(values, probabilities))

die_values = [1, 2, 3, 4, 5, 6]
die_probs = [1/6] * 6
mu = expected_value(die_values, die_probs)
var = variance(die_values, die_probs)
print(f"Die: E[X] = {mu:.4f}, Var(X) = {var:.4f}, SD = {var**0.5:.4f}")
```

### Шаг 4: Sampling из распределений

```python
def sample_bernoulli(p, n=1):
    return [1 if random.random() < p else 0 for _ in range(n)]

def sample_categorical(probs, n=1):
    cumulative = []
    total = 0
    for p in probs:
        total += p
        cumulative.append(total)
    samples = []
    for _ in range(n):
        r = random.random()
        for i, c in enumerate(cumulative):
            if r <= c:
                samples.append(i)
                break
    return samples

def sample_normal_box_muller(mu, sigma, n=1):
    samples = []
    for _ in range(n):
        u1 = random.random()
        u2 = random.random()
        z = math.sqrt(-2 * math.log(u1)) * math.cos(2 * math.pi * u2)
        samples.append(mu + sigma * z)
    return samples
```

### Шаг 5: Softmax и log probabilities

```python
def softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    exps = [math.exp(z) for z in shifted]
    total = sum(exps)
    return [e / total for e in exps]

def log_softmax(logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = max_logit + math.log(sum(math.exp(z) for z in shifted))
    return [z - log_sum_exp for z in logits]

def cross_entropy_loss(logits, target_index):
    log_probs = log_softmax(logits)
    return -log_probs[target_index]
```

### Шаг 6: Демонстрация Central Limit Theorem

```python
def demonstrate_clt(dist_fn, n_samples, n_averages):
    averages = []
    for _ in range(n_averages):
        samples = [dist_fn() for _ in range(n_samples)]
        averages.append(sum(samples) / len(samples))
    return averages
```

### Шаг 7: Визуализация

```python
import matplotlib.pyplot as plt

xs = [mu + sigma * (i - 500) / 100 for i in range(1001)]
ys = [normal_pdf(x, mu, sigma) for x, mu, sigma in ...]
plt.plot(xs, ys)
```

Полные реализации со всеми визуализациями находятся в `code/probability.py`.

### Ожидаемый вывод

Запустите `code/probability.py` — последние строки должны быть такими:

```
  As n grows, std shrinks and distribution approaches normal.

--- Visualization ---
  Saved: probability_distributions.png

============================================================
All probability computations complete.
============================================================
```

## Используйте это

С NumPy и SciPy все выше сводится к one-liners:

```python
import numpy as np
from scipy import stats

normal = stats.norm(loc=0, scale=1)
samples = normal.rvs(size=10000)
print(f"Mean: {np.mean(samples):.4f}, Std: {np.std(samples):.4f}")
print(f"P(X < 1.96) = {normal.cdf(1.96):.4f}")

logits = np.array([2.0, 1.0, 0.1])
from scipy.special import softmax, log_softmax
probs = softmax(logits)
log_probs = log_softmax(logits)
print(f"Softmax: {probs}")
print(f"Log-softmax: {log_probs}")
```

Вы построили это с нуля. Теперь вы знаете, что делают библиотечные вызовы.

## Упражнения

1. Реализуйте inverse transform sampling для exponential distribution. Проверьте, сэмплируя 10,000 значений и сравнивая histogram с истинной PDF.

2. Постройте таблицу joint distribution для двух loaded dice. Вычислите marginal distributions и проверьте, независимы ли dice.

3. Вычислите cross-entropy loss для 5-классового classifier, который выдает logits `[2.0, 0.5, -1.0, 3.0, 0.1]`, когда правильный класс имеет индекс 3. Затем проверьте ответ с `nn.CrossEntropyLoss` в PyTorch.

4. Напишите функцию, которая принимает список log probabilities и возвращает наиболее вероятную sequence, суммарную log probability и эквивалентную raw probability. Проверьте ее на предложении из 50 слов, где каждое слово имеет вероятность 0.01.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Sample space | "Все возможности" | Множество S всех возможных исходов эксперимента |
| PMF | "Функция вероятности" | Функция, которая дает точную вероятность каждого дискретного исхода; сумма равна 1 |
| PDF | "Кривая вероятности" | Функция плотности для непрерывных переменных. Интегрируйте ее по интервалу, чтобы получить вероятность |
| Conditional probability | "Вероятность при условии чего-то" | P(A\|B) = P(A and B) / P(B). Основа Bayesian thinking и теоремы Байеса |
| Independence | "Они не влияют друг на друга" | P(A and B) = P(A) * P(B). Знание одного события ничего не говорит о другом |
| Expected value | "Среднее" | Сумма всех исходов, взвешенная вероятностями. Loss function является expected value |
| Variance | "Насколько разбросано" | Ожидаемое квадратичное отклонение от среднего. High variance = шумные, нестабильные оценки |
| Normal distribution | "Колоколообразная кривая" | f(x) = (1/sqrt(2*pi*sigma^2)) * exp(-(x-mu)^2/(2*sigma^2)). Появляется повсюду благодаря CLT |
| Central Limit Theorem | "Средние становятся нормальными" | Среднее многих независимых samples сходится к normal distribution независимо от исходного распределения |
| Joint distribution | "Две переменные вместе" | P(X, Y) описывает вероятность каждой комбинации исходов X и Y |
| Marginal distribution | "Просуммировать по другой переменной" | P(X) = sum_y P(X, Y). Восстанавливает распределение одной переменной из совместного |
| Log probability | "Логарифм вероятности" | log P(x). Превращает произведения в суммы, предотвращая numerical underflow в длинных последовательностях |
| Softmax | "Превратить scores в вероятности" | softmax(z_i) = exp(z_i) / sum(exp(z_j)). Отображает вещественные logits в корректное вероятностное распределение |
| Cross-entropy | "Loss function" | -sum(p_true * log(p_predicted)). Измеряет, насколько различаются два распределения. Чем ниже, тем лучше |
| Logits | "Сырые выходы модели" | Ненормированные scores до softmax. Названы по logistic function |
| Sampling | "Получение случайных значений" | Генерация значений согласно вероятностному распределению. Так модели генерируют output |

## Дополнительное чтение

- [3Blue1Brown: But what is the Central Limit Theorem?](https://www.youtube.com/watch?v=zeJD6dqJ5lo) - визуальное доказательство того, почему средние становятся нормальными
- [Stanford CS229 Probability Review](https://cs229.stanford.edu/section/cs229-prob.pdf) - краткий справочник, покрывающий все здесь и больше
- [The Log-Sum-Exp Trick](https://gregorygundersen.com/blog/2020/02/09/log-sum-exp/) - почему численная устойчивость важна и как ее достичь
