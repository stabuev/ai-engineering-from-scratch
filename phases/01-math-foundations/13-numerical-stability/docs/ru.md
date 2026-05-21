# Численная устойчивость

> Floating point — дырявая абстракция. Она укусит вас во время обучения, и вы не увидите, как это случится.

**Тип:** Практика
**Язык:** Python
**Предварительные требования:** Фаза 1, уроки 01-04
**Время:** ~120 минут

## Цели обучения

- Реализовать численно устойчивые softmax и log-sum-exp с помощью трюка вычитания максимума
- Находить overflow, underflow и catastrophic cancellation в вычислениях с floating point
- Сравнивать аналитические градиенты с численными градиентами через центральные конечные разности
- Объяснить, почему bfloat16 предпочтительнее float16 для обучения и как loss scaling предотвращает gradient underflow

## Проблема

Ваша модель обучается три часа, затем loss становится NaN. Вы добавляете оператор печати. Logits нормальные на шаге 9,000. На шаге 9,001 они становятся `inf`. К шагу 9,002 каждый gradient — `nan`, и обучение мертво.

Или: модель дообучилась до конца, но accuracy на 2% ниже, чем заявлено в статье. Вы проверяете все. Архитектура совпадает. Гиперпараметры совпадают. Данные совпадают. Проблема в том, что статья использовала float32, а вы использовали float16 без правильного масштабирования. Тридцать два бита накопленной ошибки округления тихо съели вашу accuracy.

Или: вы реализуете cross-entropy loss с нуля. Он работает на малых logits. Когда logits превышают 100, он возвращает `inf`. Softmax переполнился, потому что `exp(100)` больше, чем может представить float32. Каждый ML-фреймворк обрабатывает это двухстрочным трюком. Вы не знали, что этот трюк существует.

Численная устойчивость — не теоретическая забота. Это разница между успешным запуском обучения и запуском, который тихо проваливается. Каждая серьезная ML-ошибка, которую вы будете отлаживать, в конце концов сведется к floating point.

## Концепция

### IEEE 754: как компьютеры хранят вещественные числа

Компьютеры хранят вещественные числа как значения floating point согласно стандарту IEEE 754. Float состоит из трех частей: знаковый бит, экспонента и мантисса (significand).

```
Float32 layout (32 bits total):
[1 sign] [8 exponent] [23 mantissa]

Value = (-1)^sign * 2^(exponent - 127) * 1.mantissa
```

Мантисса определяет точность (сколько значащих цифр). Экспонента определяет диапазон (насколько большим или малым может быть число).

```
Format     Bits   Exponent  Mantissa  Decimal digits  Range (approx)
float64    64     11        52        ~15-16          +/- 1.8e308
float32    32     8         23        ~7-8            +/- 3.4e38
float16    16     5         10        ~3-4            +/- 65,504
bfloat16   16     8         7         ~2-3            +/- 3.4e38
```

float32 дает примерно 7 десятичных цифр точности. Это значит, что он может различить 1.0000001 и 1.0000002, но не 1.00000001 и 1.00000002. После 7 цифр все превращается в шум округления.

float16 дает примерно 3 цифры. Самое большое число, которое он может представить, — 65,504. Для ML это тревожно мало: logits, gradients и активации регулярно превышают это значение.

bfloat16 — ответ Google на проблему диапазона float16. У него та же 8-битная экспонента, что и у float32 (тот же диапазон, до 3.4e38), но только 7 бит мантиссы (меньше точности, чем у float16). Для обучения neural networks диапазон важнее точности, поэтому bfloat16 обычно выигрывает.

### Почему 0.1 + 0.2 != 0.3

Число 0.1 невозможно точно представить в двоичном floating point. В системе счисления с основанием 2 это периодическая дробь:

```
0.1 in binary = 0.0001100110011001100110011... (repeating forever)
```

Float32 обрезает ее до 23 бит мантиссы. Сохраненное значение примерно равно 0.100000001490116. Аналогично, 0.2 хранится примерно как 0.200000002980232. Их сумма равна 0.300000004470348, а не 0.3.

```
In Python:
>>> 0.1 + 0.2
0.30000000000000004

>>> 0.1 + 0.2 == 0.3
False
```

Это важно для ML, потому что:

1. Сравнения loss вроде `if loss < threshold` могут давать неправильные ответы
2. Накопление множества малых значений (gradient updates на тысячах шагов) уходит от истинной суммы
3. Checksums и тесты воспроизводимости падают, если сравнивать числа float через `==`

Исправление: никогда не сравнивайте числа float через `==`. Используйте `abs(a - b) < epsilon` или `math.isclose()`.

### Catastrophic cancellation

Когда вы вычитаете два почти равных числа floating point, значащие цифры сокращаются, и остается шум округления, вынесенный в ведущие цифры.

```
a = 1.0000001    (stored as 1.00000011920929 in float32)
b = 1.0000000    (stored as 1.00000000000000 in float32)

True difference:  0.0000001
Computed:         0.00000011920929

Relative error: 19.2%
```

Это 19% относительной ошибки от одного вычитания. В ML это происходит, когда вы:

- Вычисляете дисперсию данных с большим средним: `E[x^2] - E[x]^2`, когда E[x] велико
- Вычитаете почти равные log-probabilities
- Вычисляете градиенты через finite differences со слишком малым epsilon

Исправление: переписывайте формулы так, чтобы избегать вычитания больших, почти равных чисел. Для дисперсии используйте Welford algorithm или сначала центрируйте данные. Для log-probabilities работайте в log-space на всем протяжении.

### Overflow и underflow

Overflow происходит, когда результат слишком велик для представления. Underflow происходит, когда он слишком мал (ближе к нулю, чем наименьшее представимое положительное число).

```
Float32 boundaries:
  Maximum:  3.4028235e+38
  Minimum positive (normal): 1.175e-38
  Minimum positive (denorm): 1.401e-45
  Overflow:  anything > 3.4e38 becomes inf
  Underflow: anything < 1.4e-45 becomes 0.0
```

Функция `exp()` — основной источник overflow в ML:

```
exp(88.7)  = 3.40e+38   (barely fits in float32)
exp(89.0)  = inf         (overflow)
exp(-87.3) = 1.18e-38   (barely above underflow)
exp(-104)  = 0.0         (underflow to zero)
```

Функция `log()` упирается в другую сторону:

```
log(0.0)   = -inf
log(-1.0)  = nan
log(1e-45) = -103.3      (fine)
log(1e-46) = -inf        (input underflowed to 0, then log(0) = -inf)
```

В ML `exp()` появляется в softmax, sigmoid и вероятностных вычислениях. `log()` появляется в cross-entropy, log-likelihoods и KL divergence. Комбинация `log(exp(x))` — минное поле без правильных приемов.

### Log-sum-exp trick

Вычислять `log(sum(exp(x_i)))` напрямую численно опасно. Если хотя бы один `x_i` велик, `exp(x_i)` переполняется. Если все `x_i` очень отрицательные, все `exp(x_i)` дают underflow до нуля, и `log(0)` становится `-inf`.

Трюк: вычесть максимальное значение перед вычислением экспонент.

```
log(sum(exp(x_i))) = max(x) + log(sum(exp(x_i - max(x))))
```

Почему это работает: после вычитания `max(x)` крупнейшая экспонента равна `exp(0) = 1`. Overflow невозможен. По крайней мере один член суммы равен 1, значит сумма не меньше 1, а `log(1) = 0`. Underflow до `-inf` невозможен.

Доказательство:

```
log(sum(exp(x_i)))
= log(sum(exp(x_i - c + c)))                    (add and subtract c)
= log(sum(exp(x_i - c) * exp(c)))               (exp(a+b) = exp(a)*exp(b))
= log(exp(c) * sum(exp(x_i - c)))               (factor out exp(c))
= c + log(sum(exp(x_i - c)))                    (log(a*b) = log(a) + log(b))
```

Положите `c = max(x)`, и overflow исчезает.

Этот трюк появляется в ML повсюду:
- Нормализация softmax
- Вычисление cross-entropy loss
- Суммирование log-probability в sequence models
- Смеси гауссиан
- Вариационный вывод

### Почему softmax нужен max-subtraction trick

Softmax превращает logits в вероятности:

```
softmax(x_i) = exp(x_i) / sum(exp(x_j))
```

Без трюка logits [100, 101, 102] вызывают overflow:

```
exp(100) = 2.69e43
exp(101) = 7.31e43
exp(102) = 1.99e44
sum      = 2.99e44

These overflow float32 (max ~3.4e38)? No, 2.69e43 < 3.4e38? Actually:
exp(88.7) is already at the float32 limit.
exp(100) = inf in float32.
```

С трюком вычитаем max(x) = 102:

```
exp(100 - 102) = exp(-2) = 0.135
exp(101 - 102) = exp(-1) = 0.368
exp(102 - 102) = exp(0)  = 1.000
sum = 1.503

softmax = [0.090, 0.245, 0.665]
```

Вероятности идентичны. Вычисление безопасно. Это не оптимизация. Это требование корректности.

### NaN и Inf: обнаружение и предотвращение

`nan` (Not a Number) и `inf` (infinity) вирусно распространяются через вычисления. Один `nan` в gradient update делает вес равным `nan`, что делает каждый следующий выход равным `nan`. Обучение умирает за один шаг.

Как появляется `inf`:
- `exp()` от большого положительного числа
- Деление на ноль: `1.0 / 0.0`
- `float32` overflow при накоплениях

Как появляется `nan`:
- `0.0 / 0.0`
- `inf - inf`
- `inf * 0`
- `sqrt()` от отрицательного числа
- `log()` от отрицательного числа
- Любая арифметика с уже существующим `nan`

Обнаружение:

```python
import math

math.isnan(x)       # True if x is nan
math.isinf(x)       # True if x is +inf or -inf
math.isfinite(x)    # True if x is neither nan nor inf
```

Стратегии предотвращения:

1. Ограничивайте входы для `exp()`: `exp(clamp(x, -80, 80))`
2. Добавляйте epsilon к знаменателям: `x / (y + 1e-8)`
3. Добавляйте epsilon внутрь `log()`: `log(x + 1e-8)`
4. Используйте устойчивые реализации (log-sum-exp, stable softmax)
5. Gradient clipping, чтобы предотвратить взрыв весов
6. Во время отладки проверяйте `nan`/`inf` после каждого forward pass

### Numerical gradient checking

Аналитические градиенты (из backpropagation) могут содержать ошибки. Numerical gradient checking проверяет их, вычисляя градиенты через finite differences.

Формула центральной разности:

```
df/dx ~= (f(x + h) - f(x - h)) / (2h)
```

Она имеет точность O(h^2), намного лучше forward difference `(f(x+h) - f(x)) / h`, который имеет только O(h).

Выбор h: слишком большой — аппроксимация неверна. Слишком маленький — catastrophic cancellation разрушает ответ. Обычно используют `h = 1e-5` до `1e-7`.

Проверка: вычислите относительную разницу между аналитическими и численными градиентами.

```
relative_error = |grad_analytical - grad_numerical| / max(|grad_analytical|, |grad_numerical|, 1e-8)
```

Практические правила:
- relative_error < 1e-7: идеально, градиент правильный
- relative_error < 1e-5: приемлемо, вероятно правильно
- relative_error > 1e-3: что-то не так
- relative_error > 1: градиент полностью неверный

Всегда проверяйте градиенты при реализации нового layer или loss function. PyTorch предоставляет для этого `torch.autograd.gradcheck()`.

### Mixed precision training

Современные GPU имеют специализированное оборудование (Tensor Cores), которое вычисляет matrix multiplications в float16 в 2-8 раз быстрее float32. Mixed precision training использует это:

```
1. Maintain float32 master copy of weights
2. Forward pass in float16 (fast)
3. Compute loss in float32 (prevents overflow)
4. Backward pass in float16 (fast)
5. Scale gradients to float32
6. Update float32 master weights
```

Проблема обучения в чистом float16: gradients часто очень малы (1e-8 или меньше). Float16 превращает в ноль все ниже ~6e-8. Модель перестает учиться, потому что все gradient updates равны нулю.

Исправление — loss scaling:

```
1. Multiply loss by a large scale factor (e.g., 1024)
2. Backward pass computes gradients of (loss * 1024)
3. All gradients are 1024x larger (pushed above float16 underflow)
4. Divide gradients by 1024 before updating weights
5. Net effect: same update, but no underflow
```

Dynamic loss scaling автоматически настраивает коэффициент масштабирования. Начните с большого значения (65536). Если gradients переполняются до `inf`, уменьшите его вдвое. Если N шагов проходят без overflow, удвойте его.

### bfloat16 vs float16: почему bfloat16 выигрывает для обучения

```
float16:   [1 sign] [5 exponent]  [10 mantissa]
bfloat16:  [1 sign] [8 exponent]  [7 mantissa]
```

float16 имеет больше точности (10 бит мантиссы против 7), но ограниченный диапазон (max ~65,504). bfloat16 имеет меньше точности, но тот же диапазон, что float32 (max ~3.4e38).

Для обучения neural networks:

- Активации и logits регулярно превышают 65,504 во время всплесков обучения. float16 переполняется; bfloat16 справляется.
- Loss scaling обязателен с float16, но обычно не нужен с bfloat16, потому что его диапазон покрывает спектр величин градиента.
- bfloat16 — простое усечение float32: отбросить нижние 16 бит мантиссы. Конвертация тривиальна и без потерь в экспоненте.

float16 предпочтителен для inference, где значения ограничены и точность важнее. bfloat16 предпочтителен для training, где диапазон важнее. Поэтому TPUs и современные GPU NVIDIA (A100, H100) имеют native bfloat16 support.

### Gradient clipping

Exploding gradients возникают, когда градиенты экспоненциально растут через многие layers (часто в RNNs, глубоких networks и transformers). Один большой gradient может испортить все веса за один шаг.

Два типа clipping:

**Clip by value:** ограничить каждый элемент градиента независимо.

```
grad = clamp(grad, -max_val, max_val)
```

Просто, но может изменить направление вектора градиента.

**Clip by norm:** масштабировать весь вектор градиента так, чтобы его норма не превышала threshold.

```
if ||grad|| > max_norm:
    grad = grad * (max_norm / ||grad||)
```

Сохраняет направление градиента. Именно это делает `torch.nn.utils.clip_grad_norm_()`. Это стандартный выбор.

Типичные значения: `max_norm=1.0` для transformers, `max_norm=0.5` для RL, `max_norm=5.0` для более простых networks.

Gradient clipping — не hack. Это механизм безопасности. Без него один batch с выбросом может дать gradient, достаточно большой, чтобы разрушить недели обучения.

### Normalization layers как численные стабилизаторы

Batch normalization, layer normalization и RMS normalization обычно представляют как regularizers, которые помогают обучению сходиться. Они также являются численными стабилизаторами.

Без normalization активации могут экспоненциально расти или уменьшаться через layers:

```
Layer 1: values in [0, 1]
Layer 5: values in [0, 100]
Layer 10: values in [0, 10,000]
Layer 50: values in [0, inf]
```

Normalization центрирует и масштабирует активации на каждом layer:

```
LayerNorm(x) = (x - mean(x)) / (std(x) + epsilon) * gamma + beta
```

`epsilon` (обычно 1e-5) предотвращает деление на ноль, когда все активации одинаковы. Обучаемые параметры `gamma` и `beta` позволяют сети восстановить любой scale, который ей нужен.

Это удерживает значения в численно безопасном диапазоне по всей network, предотвращая и overflow в forward pass, и gradient explosion в backward pass.

### Распространенные численные ошибки в ML

**Ошибка: Loss становится NaN после нескольких epochs.**
Причина: logits выросли слишком сильно, softmax переполнился. Или learning rate слишком высок, и веса разошлись.
Исправление: используйте stable softmax (max subtraction), уменьшите learning rate, добавьте gradient clipping.

**Ошибка: Loss застрял на log(num_classes).**
Причина: выходы модели близки к равномерным вероятностям. Часто означает, что gradients vanishing или модель вообще не учится.
Исправление: проверьте корректность labels, проверьте loss function, проверьте dead ReLUs.

**Ошибка: Validation accuracy ниже ожидаемой на 1-3%.**
Причина: mixed precision без правильного loss scaling. Gradient underflow тихо зануляет малые обновления.
Исправление: включите dynamic loss scaling или перейдите на bfloat16.

**Ошибка: Gradient norms равны 0.0 для некоторых layers.**
Причина: dead ReLU neurons (все входы отрицательные) или float16 underflow.
Исправление: используйте LeakyReLU или GELU, используйте gradient scaling, проверьте weight initialization.

**Ошибка: Модель работает на одном GPU, но дает другие результаты на другом.**
Причина: недетерминированный порядок накопления floating point. Параллельные редукции на GPU суммируют в разном порядке на разном hardware, а сложение floating point не ассоциативно.
Исправление: принимайте малые различия (1e-6) или установите `torch.use_deterministic_algorithms(True)` и примите штраф по скорости.

**Ошибка: `exp()` возвращает `inf` в loss computation.**
Причина: сырые logits переданы в `exp()` без max-subtraction trick.
Исправление: используйте `torch.nn.functional.log_softmax()`, который реализует log-sum-exp внутри.

**Ошибка: Training расходится после перехода с float32 на float16.**
Причина: float16 не может представить величины градиентов ниже 6e-8 или активации выше 65,504.
Исправление: используйте mixed precision с loss scaling (AMP) или bfloat16.

## Реализуйте

### Шаг 1: Демонстрация пределов точности floating point

```python
print("=== Floating Point Precision ===")
print(f"0.1 + 0.2 = {0.1 + 0.2}")
print(f"0.1 + 0.2 == 0.3? {0.1 + 0.2 == 0.3}")
print(f"Difference: {(0.1 + 0.2) - 0.3:.2e}")
```

### Шаг 2: Реализация naive и stable softmax

```python
import math

def softmax_naive(logits):
    exps = [math.exp(z) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

def softmax_stable(logits):
    max_logit = max(logits)
    exps = [math.exp(z - max_logit) for z in logits]
    total = sum(exps)
    return [e / total for e in exps]

safe_logits = [2.0, 1.0, 0.1]
print(f"Naive:  {softmax_naive(safe_logits)}")
print(f"Stable: {softmax_stable(safe_logits)}")

dangerous_logits = [100.0, 101.0, 102.0]
print(f"Stable: {softmax_stable(dangerous_logits)}")
# softmax_naive(dangerous_logits) would return [nan, nan, nan]
```

### Шаг 3: Реализация stable log-sum-exp

```python
def logsumexp_naive(values):
    return math.log(sum(math.exp(v) for v in values))

def logsumexp_stable(values):
    c = max(values)
    return c + math.log(sum(math.exp(v - c) for v in values))

safe = [1.0, 2.0, 3.0]
print(f"Naive:  {logsumexp_naive(safe):.6f}")
print(f"Stable: {logsumexp_stable(safe):.6f}")

large = [500.0, 501.0, 502.0]
print(f"Stable: {logsumexp_stable(large):.6f}")
# logsumexp_naive(large) returns inf
```

### Шаг 4: Реализация stable cross-entropy

```python
def cross_entropy_naive(true_class, logits):
    probs = softmax_naive(logits)
    return -math.log(probs[true_class])

def cross_entropy_stable(true_class, logits):
    max_logit = max(logits)
    shifted = [z - max_logit for z in logits]
    log_sum_exp = math.log(sum(math.exp(s) for s in shifted))
    log_prob = shifted[true_class] - log_sum_exp
    return -log_prob

logits = [2.0, 5.0, 1.0]
true_class = 1
print(f"Naive:  {cross_entropy_naive(true_class, logits):.6f}")
print(f"Stable: {cross_entropy_stable(true_class, logits):.6f}")
```

### Шаг 5: Gradient checking

```python
def numerical_gradient(f, x, h=1e-5):
    grad = []
    for i in range(len(x)):
        x_plus = x[:]
        x_minus = x[:]
        x_plus[i] += h
        x_minus[i] -= h
        grad.append((f(x_plus) - f(x_minus)) / (2 * h))
    return grad

def check_gradient(analytical, numerical, tolerance=1e-5):
    for i, (a, n) in enumerate(zip(analytical, numerical)):
        denom = max(abs(a), abs(n), 1e-8)
        rel_error = abs(a - n) / denom
        status = "OK" if rel_error < tolerance else "FAIL"
        print(f"  param {i}: analytical={a:.8f} numerical={n:.8f} "
              f"rel_error={rel_error:.2e} [{status}]")

def f(params):
    x, y = params
    return x**2 + 3*x*y + y**3

def f_grad(params):
    x, y = params
    return [2*x + 3*y, 3*x + 3*y**2]

point = [2.0, 1.0]
analytical = f_grad(point)
numerical = numerical_gradient(f, point)
check_gradient(analytical, numerical)
```

## Используйте

### Симуляция mixed precision

```python
import struct

def float32_to_float16_round(x):
    packed = struct.pack('f', x)
    f32 = struct.unpack('f', packed)[0]
    packed16 = struct.pack('e', f32)
    return struct.unpack('e', packed16)[0]

def simulate_bfloat16(x):
    packed = struct.pack('f', x)
    as_int = int.from_bytes(packed, 'little')
    truncated = as_int & 0xFFFF0000
    repacked = truncated.to_bytes(4, 'little')
    return struct.unpack('f', repacked)[0]
```

### Gradient clipping

```python
def clip_by_norm(gradients, max_norm):
    total_norm = math.sqrt(sum(g**2 for g in gradients))
    if total_norm > max_norm:
        scale = max_norm / total_norm
        return [g * scale for g in gradients]
    return gradients

grads = [10.0, 20.0, 30.0]
clipped = clip_by_norm(grads, max_norm=5.0)
print(f"Original norm: {math.sqrt(sum(g**2 for g in grads)):.2f}")
print(f"Clipped norm:  {math.sqrt(sum(g**2 for g in clipped)):.2f}")
print(f"Direction preserved: {[c/clipped[0] for c in clipped]} == {[g/grads[0] for g in grads]}")
```

### Обнаружение NaN/Inf

```python
def check_tensor(name, values):
    has_nan = any(math.isnan(v) for v in values)
    has_inf = any(math.isinf(v) for v in values)
    if has_nan or has_inf:
        print(f"WARNING {name}: nan={has_nan} inf={has_inf}")
        return False
    return True

check_tensor("good", [1.0, 2.0, 3.0])
check_tensor("bad",  [1.0, float('nan'), 3.0])
check_tensor("ugly", [1.0, float('inf'), 3.0])
```

См. `code/numerical.py` для полных реализаций с демонстрацией всех граничных случаев.

## Итоговые артефакты

Этот урок создает:
- `code/numerical.py` со stable softmax, log-sum-exp, cross-entropy, gradient checking и симуляцией mixed precision
- `outputs/prompt-numerical-debugger.md` для диагностики NaN/Inf и численных проблем в обучении

Эти устойчивые реализации снова появятся в фазе 3 при построении training loop и в фазе 4 при реализации attention mechanisms.

## Упражнения

1. **Catastrophic cancellation.** Вычислите дисперсию для [1000000.0, 1000001.0, 1000002.0] с помощью наивной формулы `E[x^2] - E[x]^2` в float32. Затем вычислите ее с помощью Welford's online algorithm. Сравните ошибки с истинной дисперсией (0.6667).

2. **Охота за точностью.** Найдите наименьшее положительное значение float32 `x`, такое что `1.0 + x == 1.0` в Python. Это machine epsilon. Проверьте, что оно совпадает с `numpy.finfo(numpy.float32).eps`.

3. **Граничные случаи log-sum-exp.** Проверьте вашу функцию `logsumexp_stable` на: (a) все значения равны, (b) одно значение намного больше остальных, (c) все значения очень отрицательные (-1000). Убедитесь, что она дает правильные результаты там, где наивная версия ломается.

4. **Gradient checking слоя neural network.** Реализуйте один linear layer `y = Wx + b` и его аналитический backward pass. Используйте `numerical_gradient`, чтобы проверить корректность для матрицы весов 3x2.

5. **Эксперимент с loss scaling.** Смоделируйте обучение с float16: создайте случайные градиенты в диапазоне [1e-9, 1e-3], конвертируйте в float16 и измерьте, какая доля стала нулем. Затем примените loss scaling (умножьте на 1024), конвертируйте в float16, масштабируйте обратно и снова измерьте долю нулей.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| IEEE 754 | "Стандарт float" | Международный стандарт, определяющий двоичные форматы floating point, правила округления и специальные значения (inf, nan). Его реализуют все современные CPU и GPU. |
| Machine epsilon | "Предел точности" | Наименьшее значение e, такое что 1.0 + e != 1.0 в данном формате float. Для float32 это примерно 1.19e-7. |
| Catastrophic cancellation | "Потеря точности при вычитании" | Когда вычитаются почти равные floating point numbers, значащие цифры сокращаются, и шум округления доминирует в результате. |
| Overflow | "Число слишком большое" | Результат превышает максимальное представимое значение и становится inf. exp(89) переполняет float32. |
| Underflow | "Число слишком маленькое" | Результат ближе к нулю, чем наименьшее представимое положительное число, и становится 0.0. exp(-104) дает underflow в float32. |
| Log-sum-exp trick | "Сначала вычесть max" | Вычисление log(sum(exp(x))) через вынесение exp(max(x)), чтобы предотвратить overflow и underflow. Используется в softmax, cross-entropy и вычислениях log-probability. |
| Stable softmax | "Softmax, который не взрывается" | Вычитание max(logits) перед вычислением экспонент. Численно идентичный результат, overflow невозможен. |
| Gradient checking | "Проверить backprop" | Сравнение analytical gradients from backpropagation с numerical gradients from finite differences, чтобы ловить ошибки реализации. |
| Mixed precision | "Float16 forward, float32 backward" | Использование чисел lower-precision для операций, критичных по скорости, и чисел higher-precision для численно чувствительных операций. Типичное ускорение 2-3x. |
| Loss scaling | "Предотвратить gradient underflow" | Умножение loss на большую константу перед backprop, чтобы gradients оставались в представимом диапазоне float16, затем деление на ту же константу перед обновлением весов. |
| bfloat16 | "Brain floating point" | 16-битный формат Google с 8 exponent bits (тот же диапазон, что float32) и 7 mantissa bits (меньше точности, чем float16). Предпочтителен для training. |
| Gradient clipping | "Ограничить норму градиента" | Масштабирование вектора градиента так, чтобы его норма не превышала threshold. Предотвращает порчу весов из-за exploding gradients. |
| NaN | "Not a Number" | Специальное значение float от неопределенных операций (0/0, inf-inf, sqrt(-1)). Распространяется через всю последующую арифметику. |
| Inf | "Infinity" | Специальное значение float от overflow или division by zero. Может комбинироваться и давать NaN (inf - inf, inf * 0). |
| Numerical gradient | "Brute force derivative" | Аппроксимация derivative через вычисление f(x+h) и f(x-h) и деление на 2h. Медленно, но надежно для проверки. |

## Дополнительные материалы

- [What Every Computer Scientist Should Know About Floating-Point Arithmetic (Goldberg 1991)](https://docs.oracle.com/cd/E19957-01/806-3568/ncg_goldberg.html) -- исчерпывающий справочник, плотный, но полный
- [Mixed Precision Training (Micikevicius et al., 2018)](https://arxiv.org/abs/1710.03740) -- статья NVIDIA, которая ввела loss scaling для float16 training
- [AMP: Automatic Mixed Precision (PyTorch docs)](https://pytorch.org/docs/stable/amp.html) -- практическое руководство по mixed precision в PyTorch
- [bfloat16 format (Google Cloud TPU docs)](https://cloud.google.com/tpu/docs/bfloat16) -- почему Google выбрал этот формат для TPUs
- [Kahan Summation (Wikipedia)](https://en.wikipedia.org/wiki/Kahan_summation_algorithm) -- алгоритм для уменьшения ошибки округления при суммировании floating point
