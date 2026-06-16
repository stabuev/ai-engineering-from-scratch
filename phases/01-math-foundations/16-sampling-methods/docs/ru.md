# Методы сэмплирования

> Сэмплирование — это способ, которым ИИ исследует пространство возможностей.

**Тип:** Практика
**Язык:** Python
**Предварительные требования:** Фаза 1, уроки 06-07 (вероятность, теорема Байеса)
**Время:** ~120 минут

## Цели обучения

- Реализовать inverse CDF, rejection sampling и importance sampling с нуля, используя только равномерные случайные числа
- Построить temperature, top-k и top-p (nucleus) sampling для генерации токенов языковой моделью
- Объяснить reparameterization trick и почему он позволяет выполнять backpropagation через сэмплирование в VAEs
- Запустить Metropolis-Hastings MCMC для сэмплирования из ненормированного целевого распределения

## Проблема

Языковая модель завершает обработку вашего запроса и выдает вектор из 50,000 logits. По одному для каждого токена в словаре. Теперь ей нужно выбрать один. Как?

Если она всегда выбирает токен с максимальной вероятностью, каждый ответ одинаковый. Детерминированный. Скучный. Если она выбирает равномерно случайно, на выходе бессмыслица. Ответ находится где-то между этими крайностями, и этим «где-то» управляет сэмплирование.

Сэмплирование не ограничено генерацией текста. Обучение с подкреплением оценивает градиенты политики, сэмплируя траектории. VAEs учат латентные представления, сэмплируя из выученных распределений и распространяя градиенты через случайность. Диффузионные модели генерируют изображения, сэмплируя шум и итеративно удаляя его. Методы Монте-Карло оценивают интегралы, для которых нет аналитического решения. Алгоритмы MCMC исследуют высокоразмерные апостериорные распределения, которые невозможно перебрать.

Каждая генеративная система ИИ — это система сэмплирования. Стратегия сэмплирования определяет качество, разнообразие и управляемость результата. В этом уроке строятся все основные методы сэмплирования с нуля: от равномерных случайных чисел до техник, на которых работают современные LLMs и генеративные модели.

## Концепция

### Почему сэмплирование важно

Сэмплирование появляется в четырех фундаментальных ролях в ИИ и машинном обучении:

**Генерация.** Языковые модели, диффузионные модели и GANs создают результат через сэмплирование. Алгоритм сэмплирования напрямую управляет креативностью, связностью и разнообразием. Temperature, top-k и nucleus sampling — это ручки, которые инженеры крутят каждый день.

**Обучение.** Стохастический градиентный спуск сэмплирует мини-батчи. Dropout сэмплирует нейроны для отключения. Аугментация данных сэмплирует случайные преобразования. Importance sampling перевзвешивает примеры, чтобы уменьшить дисперсию градиента в обучении с подкреплением (PPO, TRPO).

**Оценивание.** У многих величин в ML нет аналитического решения: ожидаемая функция потерь по распределению данных, нормировочная константа energy-based модели, модельная вероятность (evidence) в байесовском выводе. Оценка Монте-Карло приближает все это усреднением по сэмплам.

**Исследование.** Алгоритмы MCMC исследуют апостериорные распределения в байесовском выводе. Эволюционные стратегии сэмплируют возмущения параметров. Thompson sampling балансирует исследование и использование в задачах многоруких бандитов.

Ключевая сложность: напрямую можно сэмплировать только из простых распределений (равномерного, нормального). Для всего остального нужен метод, который превращает простые сэмплы в сэмплы из целевого распределения.

### Равномерное случайное сэмплирование

Каждый метод сэмплирования начинается здесь. Генератор равномерных случайных чисел выдает значения в [0, 1), где каждый подинтервал одинаковой длины имеет одинаковую вероятность.

```
U ~ Uniform(0, 1)

P(a <= U <= b) = b - a    for 0 <= a <= b <= 1

Properties:
  E[U] = 0.5
  Var(U) = 1/12
```

Чтобы сэмплировать равномерно из дискретного множества из n элементов, сгенерируйте U и верните floor(n * U). Чтобы сэмплировать из непрерывного диапазона [a, b], вычислите a + (b - a) * U.

Ключевая идея: одно равномерное случайное число содержит ровно столько случайности, сколько нужно, чтобы получить один сэмпл из любого распределения. Весь трюк — найти правильное преобразование.

### Метод inverse CDF (inverse transform sampling)

Кумулятивная функция распределения (CDF) отображает значения в вероятности:

```
F(x) = P(X <= x)

Properties:
  F is non-decreasing
  F(-inf) = 0
  F(+inf) = 1
  F maps the real line to [0, 1]
```

Inverse CDF отображает вероятности обратно в значения. Если U ~ Uniform(0, 1), то X = F_inverse(U) следует целевому распределению.

```
Algorithm:
  1. Generate u ~ Uniform(0, 1)
  2. Return F_inverse(u)

Why it works:
  P(X <= x) = P(F_inverse(U) <= x) = P(U <= F(x)) = F(x)
```

**Пример экспоненциального распределения:**

```
PDF: f(x) = lambda * exp(-lambda * x),   x >= 0
CDF: F(x) = 1 - exp(-lambda * x)

Solve F(x) = u for x:
  u = 1 - exp(-lambda * x)
  exp(-lambda * x) = 1 - u
  x = -ln(1 - u) / lambda

Since (1 - U) and U have the same distribution:
  x = -ln(u) / lambda
```

Это идеально работает, когда F_inverse можно записать аналитически. Для нормального распределения нет аналитической inverse CDF, поэтому используются другие методы (Box-Muller или численная аппроксимация).

**Дискретная версия:** для дискретных распределений построите CDF как кумулятивную сумму, сгенерируйте U и найдите первый индекс, где кумулятивная сумма превышает U. Именно так работает `sample_categorical` в уроке 06.

### Rejection Sampling

Когда CDF нельзя обратить, но целевую PDF можно вычислять с точностью до константы, работает rejection sampling.

```
Target distribution: p(x)  (can evaluate, possibly unnormalized)
Proposal distribution: q(x)  (can sample from)
Bound: M such that p(x) <= M * q(x) for all x

Algorithm:
  1. Sample x ~ q(x)
  2. Sample u ~ Uniform(0, 1)
  3. If u < p(x) / (M * q(x)), accept x
  4. Otherwise, reject and go to step 1

Acceptance rate = 1/M
```

Чем плотнее граница M, тем выше доля принятых сэмплов. В малых размерностях (1-3) rejection sampling работает хорошо. В высоких размерностях доля принятия падает экспоненциально, потому что большая часть объема предлагающего распределения отвергается. Это проклятие размерности для rejection sampling.

**Пример: сэмплирование из усеченного нормального распределения.** Используйте равномерное предлагающее распределение на усеченном диапазоне. Огибающая M — это максимум нормальной PDF на этом диапазоне.

**Пример: сэмплирование из полукруга.** Предлагайте точки равномерно в ограничивающем прямоугольнике. Принимайте, если точка попала внутрь полукруга. Так метод Монте-Карло вычисляет pi: доля принятия равна отношению площадей pi/4.

### Importance Sampling

Иногда вам не нужны сэмплы из целевого распределения p(x). Нужно оценить матожидание относительно p(x), а сэмплы есть из другого распределения q(x).

```
Goal: estimate E_p[f(x)] = integral of f(x) * p(x) dx

Rewrite:
  E_p[f(x)] = integral of f(x) * (p(x)/q(x)) * q(x) dx
            = E_q[f(x) * w(x)]

where w(x) = p(x) / q(x)  are the importance weights.

Estimator:
  E_p[f(x)] ~ (1/N) * sum(f(x_i) * w(x_i))    where x_i ~ q(x)
```

Это критично в обучении с подкреплением. В PPO (Proximal Policy Optimization) вы собираете траектории под старой стратегией pi_old, но хотите оптимизировать новую стратегию pi_new. Importance weight равен pi_new(a|s) / pi_old(a|s). PPO обрезает эти веса, чтобы новая стратегия не отклонялась слишком далеко от старой.

Дисперсия оценки importance sampling зависит от того, насколько q похоже на p. Если q сильно отличается от p, несколько сэмплов получают огромные веса и доминируют в оценке. Self-normalized importance sampling делит на сумму весов, чтобы уменьшить эту проблему:

```
E_p[f(x)] ~ sum(w_i * f(x_i)) / sum(w_i)
```

### Monte Carlo Estimation

Оценка Монте-Карло приближает интегралы усреднением случайных сэмплов. Закон больших чисел гарантирует сходимость.

```
Goal: estimate I = integral of g(x) dx over domain D

Method:
  1. Sample x_1, ..., x_N uniformly from D
  2. I ~ (Volume of D / N) * sum(g(x_i))

Error: O(1 / sqrt(N))   regardless of dimension
```

Скорость убывания ошибки не зависит от размерности. Поэтому методы Монте-Карло доминируют в высоких размерностях, где интегрирование по сетке невозможно.

**Оценка pi:**

```
Sample (x, y) uniformly from [-1, 1] x [-1, 1]
Count how many fall inside the unit circle: x^2 + y^2 <= 1
pi ~ 4 * (count inside) / (total count)
```

**Оценка матожиданий:**

```
E[f(X)] ~ (1/N) * sum(f(x_i))    where x_i ~ p(x)

The sample mean converges to the true expectation.
Variance of the estimator = Var(f(X)) / N
```

### Markov Chain Monte Carlo (MCMC): Metropolis-Hastings

MCMC строит цепь Маркова, стационарное распределение которой является целевым распределением p(x). После достаточного числа шагов сэмплы из цепи являются (приблизительно) сэмплами из p(x).

```
Target: p(x)  (known up to a normalizing constant)
Proposal: q(x'|x)  (how to propose the next state given the current state)

Metropolis-Hastings algorithm:
  1. Start at some x_0
  2. For t = 1, 2, ..., T:
     a. Propose x' ~ q(x'|x_t)
     b. Compute acceptance ratio:
        alpha = [p(x') * q(x_t|x')] / [p(x_t) * q(x'|x_t)]
     c. Accept with probability min(1, alpha):
        - If u < alpha (u ~ Uniform(0,1)): x_{t+1} = x'
        - Otherwise: x_{t+1} = x_t
  3. Discard first B samples (burn-in)
  4. Return remaining samples
```

Для симметричных предлагающих распределений (q(x'|x) = q(x|x')) отношение упрощается до p(x')/p(x). Это исходный алгоритм Metropolis.

**Почему это работает.** Правило принятия обеспечивает detailed balance: вероятность быть в x и перейти в x' равна вероятности быть в x' и перейти в x. Detailed balance означает, что p(x) является стационарным распределением цепи.

**Практические соображения:**
- Burn-in: отбросьте ранние сэмплы до того, как цепь достигнет равновесия
- Thinning: сохраняйте каждый k-й сэмпл, чтобы уменьшить автокорреляцию
- Масштаб предлагающего распределения: слишком маленький — цепь движется медленно (высокая доля принятия, медленное исследование); слишком большой — большинство кандидатов отвергается (низкая доля принятия, застревание на месте)
- Оптимальная доля принятия для гауссовского proposal в высоких размерностях примерно 0.234

### Gibbs Sampling

Gibbs sampling — специальный случай MCMC для многомерных распределений. Вместо предложения по всем размерностям сразу он обновляет одну переменную за раз из ее условного распределения.

```
Target: p(x_1, x_2, ..., x_d)

Algorithm:
  For each iteration t:
    Sample x_1^{t+1} ~ p(x_1 | x_2^t, x_3^t, ..., x_d^t)
    Sample x_2^{t+1} ~ p(x_2 | x_1^{t+1}, x_3^t, ..., x_d^t)
    ...
    Sample x_d^{t+1} ~ p(x_d | x_1^{t+1}, x_2^{t+1}, ..., x_{d-1}^{t+1})
```

Gibbs sampling требует, чтобы можно было сэмплировать из каждого условного распределения p(x_i | x_{-i}). Для многих моделей это просто:
- Байесовские сети: условные распределения следуют из структуры графа
- Gaussian mixtures: условные распределения являются гауссовскими
- Модели Изинга: условное распределение каждого спина зависит только от соседей

Доля принятия всегда 1 (каждый кандидат принимается), потому что сэмплирование из точного условного распределения автоматически удовлетворяет detailed balance.

**Ограничение.** Когда переменные сильно коррелированы, Gibbs sampling смешивается медленно: обновление одной переменной за раз не может делать большие диагональные перемещения по распределению.

### Temperature Sampling (используется в LLMs)

Языковые модели выдают logits z_1, ..., z_V для каждого токена в словаре. Softmax превращает их в вероятности. Temperature масштабирует logits перед softmax:

```
p_i = exp(z_i / T) / sum(exp(z_j / T))

T = 1.0: standard softmax (original distribution)
T -> 0:  argmax (deterministic, always picks highest logit)
T -> inf: uniform (all tokens equally likely)
T < 1.0: sharpens the distribution (more confident, less diverse)
T > 1.0: flattens the distribution (less confident, more diverse)
```

**Почему это работает.** Деление logits на T < 1 усиливает различия между logits. Если z_1 = 2 и z_2 = 1, деление на T = 0.5 дает z_1/T = 4 и z_2/T = 2, увеличивая разрыв. После softmax токен с максимальным logit получает гораздо большую долю вероятности.

**На практике:**
- T = 0.0: greedy decoding, лучше для фактических вопросов и ответов
- T = 0.3-0.7: умеренная креативность, хорошо для генерации кода
- T = 0.7-1.0: баланс, хорошо для обычного диалога
- T = 1.0-1.5: творческое письмо, мозговой штурм
- T > 1.5: все более случайно, редко полезно

Temperature не меняет, какие токены возможны. Она меняет вероятностную массу, выделенную каждому токену.

### Top-k Sampling

Top-k sampling ограничивает набор кандидатов k токенами с наибольшими вероятностями, затем перенормирует и сэмплирует из этого ограниченного множества.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Keep only the top k tokens
  4. Renormalize: p_i' = p_i / sum(p_j for j in top-k)
  5. Sample from the renormalized distribution

k = 1:  greedy decoding
k = V:  no filtering (standard sampling)
k = 40: typical setting, removes long tail of unlikely tokens
```

Top-k не дает модели выбирать крайне маловероятные токены (опечатки, бессмыслицу), которые находятся в длинном хвосте распределения словаря. Проблема: k фиксировано независимо от контекста. Когда модель уверена (один токен имеет вероятность 95%), k = 40 все равно допускает 39 альтернатив. Когда модель не уверена (вероятность размазана по 1000 токенам), k = 40 отрезает правдоподобные варианты.

### Top-p (Nucleus) Sampling

Top-p sampling динамически подстраивает размер набора кандидатов. Вместо фиксированного числа токенов он оставляет минимальный набор токенов, кумулятивная вероятность которых превышает p.

```
Algorithm:
  1. Compute softmax probabilities for all V tokens
  2. Sort tokens by probability (descending)
  3. Find smallest k such that sum of top-k probabilities >= p
  4. Keep only those k tokens
  5. Renormalize and sample

p = 0.9:  keeps tokens covering 90% of probability mass
p = 1.0:  no filtering
p = 0.1:  very restrictive, nearly greedy
```

Когда модель уверена, nucleus sampling оставляет мало токенов (возможно, 2-3). Когда модель не уверена, оставляет много (возможно, 200). Благодаря этому адаптивному поведению nucleus sampling обычно дает текст лучше, чем top-k.

**Распространенные сочетания:**
- Temperature 0.7 + top-p 0.9: хорошая универсальная настройка
- Temperature 0.0 (greedy): лучше для детерминированных задач
- Temperature 1.0 + top-k 50: настройка из оригинальной статьи Fan et al. (2018)

Top-k и top-p можно комбинировать. Сначала примените top-k, затем top-p к оставшемуся множеству.

### Reparameterization Trick (используется в VAEs)

Variational autoencoders (VAEs) учатся, кодируя входы в распределение в латентном пространстве, сэмплируя из этого распределения и декодируя сэмпл обратно. Проблема: через операцию сэмплирования нельзя выполнить backpropagation.

```
Standard sampling (not differentiable):
  z ~ N(mu, sigma^2)

  The randomness blocks gradient flow.
  d/d_mu [sample from N(mu, sigma^2)] = ???
```

Reparameterization trick отделяет случайность от параметров:

```
Reparameterized sampling:
  epsilon ~ N(0, 1)          (fixed random noise, no parameters)
  z = mu + sigma * epsilon   (deterministic function of parameters)

  Now z is a deterministic, differentiable function of mu and sigma.
  d(z)/d(mu) = 1
  d(z)/d(sigma) = epsilon

  Gradients flow through mu and sigma.
```

Это работает, потому что N(mu, sigma^2) имеет то же распределение, что и mu + sigma * N(0, 1). Ключевая идея: перенести случайность в источник без параметров (epsilon), затем выразить сэмпл как дифференцируемое преобразование параметров.

**В цикле обучения VAE:**
1. Encoder выдает mu и log(sigma^2) для каждого входа
2. Сэмплировать epsilon ~ N(0, 1)
3. Вычислить z = mu + sigma * epsilon
4. Decode z для реконструкции входа
5. Выполнить backpropagation через шаги 4, 3, 2, 1 (возможно, потому что шаг 3 дифференцируем)

Без reparameterization trick VAEs нельзя обучать стандартным backpropagation. Эта одна идея сделала VAEs практичными.

### Gumbel-Softmax (дифференцируемое категориальное сэмплирование)

Reparameterization trick работает для непрерывных распределений (гауссовских). Для дискретных категориальных распределений нужен другой подход. Gumbel-Softmax дает дифференцируемую аппроксимацию categorical sampling.

**Gumbel-Max trick (недифференцируемый):**

```
To sample from a categorical distribution with log-probabilities log(p_1), ..., log(p_k):
  1. Sample g_i ~ Gumbel(0, 1) for each category
     (g = -log(-log(u)), where u ~ Uniform(0, 1))
  2. Return argmax(log(p_i) + g_i)

This produces exact categorical samples.
```

**Gumbel-Softmax (дифференцируемая аппроксимация):**

```
Replace the hard argmax with a soft softmax:
  y_i = exp((log(p_i) + g_i) / tau) / sum(exp((log(p_j) + g_j) / tau))

tau (temperature) controls the approximation:
  tau -> 0:  approaches a one-hot vector (hard categorical)
  tau -> inf: approaches uniform (1/k, 1/k, ..., 1/k)
  tau = 1.0: soft approximation
```

Gumbel-Softmax создает непрерывную релаксацию дискретного сэмпла. Результат — это вектор вероятностей (soft one-hot), а не hard one-hot. Градиенты проходят через softmax. Во время прямого прохода в обучении можно использовать straight-through estimator: hard argmax для прямого прохода, но soft Gumbel-Softmax gradients для обратного прохода.

**Применения:**
- Дискретные латентные переменные в VAEs
- Neural architecture search (выбор дискретных операций)
- Механизмы hard attention
- Обучение с подкреплением с дискретными действиями

### Stratified Sampling

Обычный Monte Carlo sampling может случайно оставлять пробелы в пространстве сэмплов. Stratified sampling принудительно обеспечивает равномерное покрытие, разделяя пространство на strata и сэмплируя из каждой.

```
Standard Monte Carlo:
  Sample N points uniformly from [0, 1]
  Some regions may have clusters, others gaps

Stratified sampling:
  Divide [0, 1] into N equal strata: [0, 1/N), [1/N, 2/N), ..., [(N-1)/N, 1)
  Sample one point uniformly within each stratum
  x_i = (i + u_i) / N   where u_i ~ Uniform(0, 1),  i = 0, ..., N-1
```

Stratified sampling всегда имеет меньшую или равную дисперсию по сравнению с обычным Monte Carlo:

```
Var(stratified) <= Var(standard Monte Carlo)

The improvement is largest when f(x) varies smoothly.
For piecewise-constant functions, stratified sampling is exact.
```

**Применения:**
- Численное интегрирование (quasi-Monte Carlo)
- Разбиения обучающих данных (обеспечение баланса классов в каждом fold)
- Importance sampling со stratification (комбинация обеих техник)
- NeRF (Neural Radiance Fields) использует stratified sampling вдоль лучей камеры

### Связь с Diffusion Models

Diffusion models генерируют изображения через процесс сэмплирования. Прямой процесс добавляет Gaussian noise к изображению за T шагов, пока оно не станет чистым шумом. Обратный процесс учится убирать шум, восстанавливая исходное изображение шаг за шагом.

```
Forward process (known):
  x_t = sqrt(alpha_t) * x_{t-1} + sqrt(1 - alpha_t) * epsilon
  where epsilon ~ N(0, I)

  After T steps: x_T ~ N(0, I)  (pure noise)

Reverse process (learned):
  x_{t-1} = (1/sqrt(alpha_t)) * (x_t - (1 - alpha_t)/sqrt(1 - alpha_bar_t) * epsilon_theta(x_t, t)) + sigma_t * z
  where z ~ N(0, I)

  Each denoising step is a sampling step.
```

Связь с методами из этого урока:
- Каждый шаг denoising использует reparameterization trick (сэмплировать шум, применить детерминированное преобразование)
- Noise schedule {alpha_t} управляет формой temperature annealing
- Обучение использует оценку Монте-Карло для аппроксимации ELBO (evidence lower bound)
- Ancestral sampling в diffusion models — это цепь Маркова (каждый шаг зависит только от текущего состояния)

Весь процесс генерации изображения — итеративное сэмплирование: начать с шума и на каждом шаге сэмплировать чуть менее шумную версию, обусловленную выученной denoising model.

## Сборка

### Шаг 1: равномерное распределение и inverse CDF sampling

```python
import math
import random

def sample_uniform(a, b):
    return a + (b - a) * random.random()

def sample_exponential_inverse_cdf(lam):
    u = random.random()
    return -math.log(u) / lam
```

Сгенерируйте 10,000 сэмплов из экспоненциального распределения и проверьте, что среднее равно 1/lambda.

### Шаг 2: Rejection sampling

```python
def rejection_sample(target_pdf, proposal_sample, proposal_pdf, M):
    while True:
        x = proposal_sample()
        u = random.random()
        if u < target_pdf(x) / (M * proposal_pdf(x)):
            return x
```

Используйте rejection sampling, чтобы получить сэмплы из усеченного нормального распределения. Проверьте форму через гистограмму сэмплов.

### Шаг 3: Importance sampling

```python
def importance_sampling_estimate(f, target_pdf, proposal_pdf, proposal_sample, n):
    total = 0
    for _ in range(n):
        x = proposal_sample()
        w = target_pdf(x) / proposal_pdf(x)
        total += f(x) * w
    return total / n
```

Оцените E[X^2] под нормальным распределением, используя равномерное proposal. Сравните с известным ответом (mu^2 + sigma^2).

### Шаг 4: оценка pi методом Монте-Карло

```python
def monte_carlo_pi(n):
    inside = 0
    for _ in range(n):
        x = random.uniform(-1, 1)
        y = random.uniform(-1, 1)
        if x*x + y*y <= 1:
            inside += 1
    return 4 * inside / n
```

### Шаг 5: Metropolis-Hastings MCMC

```python
def metropolis_hastings(target_log_pdf, proposal_sample, proposal_log_pdf, x0, n_samples, burn_in):
    samples = []
    x = x0
    for i in range(n_samples + burn_in):
        x_new = proposal_sample(x)
        log_alpha = (target_log_pdf(x_new) + proposal_log_pdf(x, x_new)
                     - target_log_pdf(x) - proposal_log_pdf(x_new, x))
        if math.log(random.random()) < log_alpha:
            x = x_new
        if i >= burn_in:
            samples.append(x)
    return samples
```

Сэмплируйте из бимодального распределения (смеси двух гауссиан). Визуализируйте траекторию цепи.

### Шаг 6: Gibbs sampling

```python
def gibbs_sampling_2d(conditional_x_given_y, conditional_y_given_x, x0, y0, n_samples, burn_in):
    x, y = x0, y0
    samples = []
    for i in range(n_samples + burn_in):
        x = conditional_x_given_y(y)
        y = conditional_y_given_x(x)
        if i >= burn_in:
            samples.append((x, y))
    return samples
```

### Шаг 7: Temperature sampling

```python
def softmax(logits):
    max_l = max(logits)
    exps = [math.exp(z - max_l) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def temperature_sample(logits, temperature):
    scaled = [z / temperature for z in logits]
    probs = softmax(scaled)
    return sample_from_probs(probs)
```

Покажите, как temperature меняет выходное распределение для набора token logits.

### Шаг 8: Top-k и top-p sampling

```python
def top_k_sample(logits, k):
    indexed = sorted(enumerate(logits), key=lambda x: -x[1])
    top = indexed[:k]
    top_logits = [l for _, l in top]
    probs = softmax(top_logits)
    idx = sample_from_probs(probs)
    return top[idx][0]

def top_p_sample(logits, p):
    probs = softmax(logits)
    indexed = sorted(enumerate(probs), key=lambda x: -x[1])
    cumsum = 0
    selected = []
    for token_idx, prob in indexed:
        cumsum += prob
        selected.append((token_idx, prob))
        if cumsum >= p:
            break
    sel_probs = [pr for _, pr in selected]
    total = sum(sel_probs)
    sel_probs = [pr / total for pr in sel_probs]
    idx = sample_from_probs(sel_probs)
    return selected[idx][0]
```

### Шаг 9: Reparameterization trick

```python
def reparam_sample(mu, sigma):
    epsilon = random.gauss(0, 1)
    return mu + sigma * epsilon

def reparam_gradient(mu, sigma, epsilon):
    dz_dmu = 1.0
    dz_dsigma = epsilon
    return dz_dmu, dz_dsigma
```

Продемонстрируйте, что градиенты проходят через репараметризованный сэмпл, но не через прямое сэмплирование.

### Шаг 10: Gumbel-Softmax

```python
def gumbel_sample():
    u = random.random()
    return -math.log(-math.log(u))

def gumbel_softmax(logits, temperature):
    gumbels = [math.log(p) + gumbel_sample() for p in logits]
    return softmax([g / temperature for g in gumbels])
```

Покажите, как уменьшение temperature заставляет результат приближаться к one-hot vector.

Полные реализации со всеми визуализациями находятся в `code/sampling.py`.

### Ожидаемый вывод

Запустите `code/sampling.py` — последние строки должны быть такими:

```
--- 14. Visualizations ---
  Saved: sampling_methods.png

=================================================================
All sampling methods complete.
=================================================================
```

## Использование

С NumPy и SciPy промышленные версии выглядят так:

```python
import numpy as np

rng = np.random.default_rng(42)

exponential_samples = rng.exponential(scale=2.0, size=10000)
print(f"Exponential mean: {exponential_samples.mean():.4f} (expected 2.0)")

from scipy import stats
normal = stats.norm(loc=0, scale=1)
print(f"CDF at 1.96: {normal.cdf(1.96):.4f}")
print(f"Inverse CDF at 0.975: {normal.ppf(0.975):.4f}")

logits = np.array([2.0, 1.0, 0.5, 0.1, -1.0])
temperature = 0.7
scaled = logits / temperature
probs = np.exp(scaled - scaled.max()) / np.exp(scaled - scaled.max()).sum()
token = rng.choice(len(logits), p=probs)
print(f"Sampled token index: {token}")
```

Для MCMC в масштабе используйте специализированные библиотеки:
- PyMC: полноценное байесовское моделирование с NUTS (adaptive HMC)
- emcee: ансамблевый MCMC sampler
- NumPyro/JAX: MCMC с ускорением на GPU

Вы построили это с нуля. Теперь вы знаете, что делают вызовы библиотек.

## Упражнения

1. Реализуйте inverse CDF sampling для распределения Cauchy. CDF: F(x) = 0.5 + arctan(x)/pi. Сгенерируйте 10,000 сэмплов и постройте гистограмму рядом с истинной PDF. Обратите внимание на тяжелые хвосты (экстремальные значения далеко от центра).

2. Используйте rejection sampling, чтобы сгенерировать сэмплы из распределения Beta(2, 5) с proposal Uniform(0, 1). Постройте принятые сэмплы рядом с истинной Beta PDF. Какова теоретическая доля принятия?

3. Оцените интеграл sin(x) от 0 до pi с помощью Монте-Карло с 1,000, 10,000 и 100,000 сэмплов. Сравните ошибку на каждом уровне. Проверьте, что ошибка масштабируется как O(1/sqrt(N)).

4. Реализуйте Metropolis-Hastings для сэмплирования из двумерного распределения p(x, y), пропорционального exp(-(x^2 * y^2 + x^2 + y^2 - 8*x - 8*y) / 2). Постройте сэмплы и траекторию цепи. Поэкспериментируйте с разными стандартными отклонениями proposal.

5. Соберите полную демонстрацию генерации текста: дан словарь из 10 слов с logits, генерируйте последовательности из 20 токенов с использованием (a) greedy, (b) temperature=0.7, (c) top-k=3, (d) top-p=0.9. Сравните разнообразие результатов по 5 запускам.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Sampling | «Вытягивание случайных значений» | Генерация значений согласно вероятностному распределению. Механизм за всей генеративной ИИ-системой |
| Uniform distribution | «Все одинаково вероятно» | Каждое значение в [a, b] имеет одинаковую плотность вероятности 1/(b-a). Стартовая точка всех методов сэмплирования |
| Inverse CDF | «Преобразование вероятности» | F_inverse(U) превращает равномерный сэмпл в сэмпл из любого распределения с известной CDF. Точно и эффективно |
| Rejection sampling | «Предложить и принять/отклонить» | Генерировать из простого proposal, принимать с вероятностью, пропорциональной отношению target/proposal. Точно, но тратит сэмплы впустую |
| Importance sampling | «Перевзвесить сэмплы» | Оценивать матожидания под p(x), используя сэмплы из q(x) и вес p(x)/q(x) для каждого сэмпла. Ключ к PPO в RL |
| Monte Carlo | «Усреднить случайные сэмплы» | Приближать интегралы средними по сэмплам. Ошибка O(1/sqrt(N)) независимо от размерности |
| MCMC | «Сходящееся случайное блуждание» | Построить цепь Маркова, стационарное распределение которой является целевым. Metropolis-Hastings — базовый алгоритм |
| Metropolis-Hastings | «Принимать вверх, иногда вниз» | Предлагать ходы и принимать на основе отношения плотностей. Detailed balance обеспечивает сходимость к целевому распределению |
| Gibbs sampling | «По одной переменной за раз» | Обновлять каждую переменную из ее условного распределения при фиксированных остальных. Доля принятия 100% |
| Temperature | «Ручка уверенности» | Делит logits на T перед softmax. T<1 делает распределение резче (больше уверенности), T>1 сглаживает (больше разнообразия) |
| Top-k sampling | «Оставить k лучших» | Обнулить все, кроме k токенов с наибольшей вероятностью, перенормировать, сэмплировать. Фиксированный размер набора кандидатов |
| Nucleus sampling (top-p) | «Оставить вероятные» | Оставить минимальный набор токенов, кумулятивная вероятность которых превышает p. Адаптивный размер набора кандидатов |
| Reparameterization trick | «Вынести случайность наружу» | Записать z = mu + sigma * epsilon, где epsilon ~ N(0,1). Делает сэмплирование дифференцируемым. Важно для обучения VAE |
| Gumbel-Softmax | «Мягкое категориальное сэмплирование» | Дифференцируемая аппроксимация категориального сэмплирования через Gumbel noise и softmax with temperature |
| Stratified sampling | «Принудительное покрытие» | Разделить пространство сэмплов на strata и сэмплировать из каждой. Всегда меньшая дисперсия, чем у наивного Монте-Карло |
| Burn-in | «Период прогрева» | Начальные MCMC-сэмплы, отброшенные до достижения стационарного распределения цепью |
| Detailed balance | «Условие обратимости» | p(x) * T(x->y) = p(y) * T(y->x). Достаточное условие, чтобы p было стационарным распределением цепи Маркова |
| Diffusion sampling | «Итеративное denoising» | Генерировать данные, начиная с шума и применяя выученные шаги denoising. Каждый шаг — условная операция сэмплирования |

## Дополнительное чтение

- [Holbrook (2023): The Metropolis-Hastings Algorithm](https://arxiv.org/abs/2304.07010) - подробное руководство по основам MCMC
- [Jang, Gu, Poole (2017): Categorical Reparameterization with Gumbel-Softmax](https://arxiv.org/abs/1611.01144) - оригинальная статья о Gumbel-Softmax
- [Holtzman et al. (2020): The Curious Case of Neural Text Degeneration](https://arxiv.org/abs/1904.09751) - статья о nucleus (top-p) sampling
- [Kingma & Welling (2014): Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) - статья о VAE, вводящая reparameterization trick
- [Ho, Jain, Abbeel (2020): Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) - DDPM связывает сэмплирование с генерацией изображений
