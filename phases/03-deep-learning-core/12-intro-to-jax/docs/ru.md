# Введение в JAX

> PyTorch изменяет тензоры. TensorFlow строит графы. JAX компилирует чистые функции. Именно последнее меняет то, как вы думаете о глубоком обучении.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 03, уроки 01-10, базовый NumPy
**Время:** ~90 минут

## Цели обучения

- Писать код нейронных сетей в стиле чистых функций, используя функциональный API JAX (jax.numpy, jax.grad, jax.jit, jax.vmap)
- Объяснять ключевое архитектурное различие между eager-мутацией в PyTorch и моделью функциональной компиляции в JAX
- Применять JIT-компиляцию и векторизацию vmap, чтобы ускорять циклы обучения по сравнению с наивным Python
- Обучить простую сеть в JAX и сопоставить явное управление состоянием с объектно-ориентированным подходом PyTorch

## Проблема

Вы уже умеете строить нейронные сети в PyTorch. Вы определяете `nn.Module`, вызываете `.backward()`, делаете шаг оптимизатора. Это работает. Миллионы людей этим пользуются.

Но в PyTorch есть ограничение, встроенное в саму его основу: он трассирует операции eager-режимом, по одной, в Python. Каждый `tensor + tensor` — это отдельный запуск ядра. Каждый шаг обучения заново интерпретирует один и тот же Python-код. Это нормально работает до тех пор, пока вам не нужно обучать модель с 540 миллиардами параметров на 2 048 TPU. Тогда накладные расходы вас уничтожают.

Google DeepMind обучает Gemini на JAX. Anthropic обучала Claude на JAX. Это не маленькие задачи — это крупнейшие запуски обучения нейронных сетей на Земле. Они выбрали JAX, потому что он рассматривает ваш цикл обучения как компилируемую программу, а не как последовательность Python-вызовов.

JAX — это NumPy с тремя сверхспособностями: автоматическое дифференцирование, JIT-компиляция в XLA и автоматическая векторизация. Вы пишете функцию, которая обрабатывает один пример. JAX дает вам функцию, которая обрабатывает батч, считает градиенты, компилируется в машинный код и работает на нескольких устройствах. И все это без изменения исходной функции.

## Концепция

### Философия JAX

JAX — функциональный фреймворк. Нет классов, нет изменяемого состояния, нет метода `.backward()`. Вместо этого:

| PyTorch | JAX |
|---------|-----|
| Класс `nn.Module` с состоянием | Чистая функция: `f(params, x) -> y` |
| `loss.backward()` | `jax.grad(loss_fn)(params, x, y)` |
| Eager-выполнение | JIT-компиляция через XLA |
| Ручной цикл `for x in batch:` | Автовекторизация `jax.vmap(f)` |
| `DataParallel` / `FSDP` | Автопараллелизм `jax.pmap(f)` |
| Изменяемый `model.parameters()` | Неизменяемый pytree массивов |

Это не стилистическое предпочтение. Это ограничение компилятора. JIT-компиляция требует чистых функций: одни и те же входы всегда дают одни и те же выходы, без побочных эффектов. Именно это ограничение делает возможными ускорения в 100 раз.

### jax.numpy: знакомая поверхность

JAX заново реализует API NumPy для ускорителей:

```python
import jax.numpy as jnp

a = jnp.array([1.0, 2.0, 3.0])
b = jnp.array([4.0, 5.0, 6.0])
c = jnp.dot(a, b)
```

Те же имена функций. Те же правила broadcasting. Та же семантика срезов. Но массивы живут на GPU/TPU, и каждая операция трассируется компилятором.

Одно критическое отличие: массивы JAX неизменяемы. Нельзя написать `a[0] = 5`. Вместо этого: `a = a.at[0].set(5)`. Первую неделю это кажется неудобным, а потом становится понятно: неизменяемость — это то, что делает трансформации вроде `grad`, `jit` и `vmap` компонуемыми.

### jax.grad: функциональное автодифференцирование

PyTorch прикрепляет градиенты к тензорам (`.grad`). JAX прикрепляет градиенты к функциям.

```python
import jax

def f(x):
    return x ** 2

df = jax.grad(f)
df(3.0)
```

`jax.grad` принимает функцию и возвращает новую функцию, которая вычисляет градиент. Нет вызова `.backward()`. Нет вычислительного графа, сохраненного на тензорах. Градиент — это просто еще одна функция, которую можно вызвать, скомпоновать или JIT-скомпилировать.

Это компонуется произвольно:

```python
d2f = jax.grad(jax.grad(f))
d2f(3.0)
```

Вторые производные. Третьи производные. Якобианы. Гессианы. Все через композицию `grad`. PyTorch тоже может это делать (`torch.autograd.functional.hessian`), но это надстройка. В JAX это фундамент.

Ограничение: `grad` работает только с чистыми функциями. Никаких print-выражений внутри (они выполняются во время трассировки, а не исполнения). Никакой мутации внешнего состояния. Никакой генерации случайных чисел без явного управления ключами.

### jit: компиляция в XLA

```python
@jax.jit
def train_step(params, x, y):
    loss = loss_fn(params, x, y)
    return loss

fast_step = jax.jit(train_step)
```

При первом вызове JAX трассирует функцию: записывает, какие операции происходят, не выполняя их. Затем он передает эту трассу в XLA (Accelerated Linear Algebra), компилятор Google для TPU и GPU. XLA объединяет операции, устраняет лишние копирования памяти и генерирует оптимизированный машинный код.

Последующие вызовы полностью обходят Python. Скомпилированный код выполняется на ускорителе со скоростью C++.

Когда JIT помогает:
- Шаги обучения (одно и то же вычисление повторяется тысячи раз)
- Инференс (та же модель, другие входы)
- Любая функция, вызываемая больше одного раза с входами похожей формы

Когда JIT мешает:
- Функции с Python control flow, зависящим от значений (`if x > 0`, где x — трассируемый массив)
- Одноразовые вычисления (накладные расходы компиляции превышают время выполнения)
- Отладка (трассировка скрывает фактическое исполнение)

Ограничение на control flow реально. `jax.lax.cond` заменяет `if/else`. `jax.lax.scan` заменяет циклы `for`. Это не опционально — это цена компиляции.

### vmap: автоматическая векторизация

Вы пишете функцию, которая обрабатывает один пример:

```python
def predict(params, x):
    return jnp.dot(params['w'], x) + params['b']
```

`vmap` поднимает ее до обработки батча:

```python
batch_predict = jax.vmap(predict, in_axes=(None, 0))
```

`in_axes=(None, 0)` означает: не батчить по `params` (они общие), батчить по оси 0 у `x`. Нет ручного цикла `for`. Нет reshape. Нет протаскивания размерности батча через код. JAX сам находит размерность батча и векторизует все вычисление.

Это не синтаксический сахар. `vmap` генерирует объединенный векторизованный код, который работает в 10-100 раз быстрее Python-цикла. И он компонуется с `jit` и `grad`:

```python
per_example_grads = jax.vmap(jax.grad(loss_fn), in_axes=(None, 0, 0))
```

Градиенты по отдельным примерам. Одна строка. В PyTorch это почти невозможно без хаков.

### pmap: параллелизм данных между устройствами

```python
parallel_step = jax.pmap(train_step, axis_name='devices')
```

`pmap` реплицирует функцию на все доступные устройства (GPU/TPU) и разбивает батч. Внутри функции `jax.lax.pmean` и `jax.lax.psum` синхронизируют градиенты между устройствами.

Google обучает Gemini на тысячах чипов TPU v5e, используя `pmap` (и его преемник `shard_map`). Программная модель: напишите версию для одного устройства, оберните в `pmap`, готово.

### Pytrees: универсальная структура данных

JAX работает с "pytrees" — вложенными комбинациями списков, кортежей, словарей и массивов. Параметры вашей модели — это pytree:

```python
params = {
    'layer1': {'w': jnp.zeros((784, 256)), 'b': jnp.zeros(256)},
    'layer2': {'w': jnp.zeros((256, 128)), 'b': jnp.zeros(128)},
    'layer3': {'w': jnp.zeros((128, 10)),  'b': jnp.zeros(10)},
}
```

Каждая трансформация JAX — `grad`, `jit`, `vmap` — умеет обходить pytrees. `jax.tree.map(f, tree)` применяет `f` к каждому листу. Так оптимизаторы обновляют все параметры сразу:

```python
params = jax.tree.map(lambda p, g: p - lr * g, params, grads)
```

Нет метода `.parameters()`. Нет регистрации параметров. Структура дерева и есть модель.

### Функциональный против объектно-ориентированного

PyTorch хранит состояние внутри объектов:

```python
class Model(nn.Module):
    def __init__(self):
        self.linear = nn.Linear(784, 10)

    def forward(self, x):
        return self.linear(x)
```

JAX использует чистые функции с явным состоянием:

```python
def predict(params, x):
    return jnp.dot(x, params['w']) + params['b']
```

Параметры передаются внутрь. Ничего не хранится. Ничего не мутируется. Это делает каждую функцию тестируемой, компонуемой и компилируемой. Это также означает, что вы управляете параметрами сами — или используете библиотеку вроде Flax или Equinox.

### Экосистема JAX

JAX дает вам примитивы. Библиотеки дают эргономику:

| Библиотека | Роль | Стиль |
|---------|------|-------|
| **Flax** (Google) | Слои нейронных сетей | `nn.Module` с явным состоянием |
| **Equinox** (Patrick Kidger) | Слои нейронных сетей | На основе pytree, Pythonic |
| **Optax** (DeepMind) | Оптимизаторы + расписания LR | Компонуемые градиентные преобразования |
| **Orbax** (Google) | Чекпоинтинг | Сохранение/восстановление pytrees |
| **CLU** (Google) | Метрики + логирование | Утилиты для цикла обучения |

Optax — стандартная библиотека оптимизаторов. Она отделяет преобразование градиентов (Adam, SGD, clipping) от обновления параметров, делая композицию тривиальной:

```python
optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adam(learning_rate=1e-3),
)
```

### Когда использовать JAX, а когда PyTorch

| Фактор | JAX | PyTorch |
|--------|-----|---------|
| Поддержка TPU | Первоклассная (Google построила и то и другое) | Поддерживается сообществом (torch_xla) |
| Поддержка GPU | Хорошая (CUDA через XLA) | Лучшая в классе (нативный CUDA) |
| Отладка | Сложная (трассировка + компиляция) | Простая (eager, построчно) |
| Экосистема | Ориентирована на исследования (Flax, Equinox) | Огромная (HuggingFace, torchvision и т. д.) |
| Найм | Нишевая (Google/DeepMind/Anthropic) | Массовая (везде) |
| Крупномасштабное обучение | Лучше (XLA, pmap, mesh) | Хорошо (FSDP, DeepSpeed) |
| Скорость прототипирования | Медленнее (функциональные накладные расходы) | Быстрее (мутируй и запускай) |
| Production-инференс | TensorFlow Serving, Vertex AI | TorchServe, Triton, ONNX |
| Кто использует | DeepMind (Gemini), Anthropic (Claude) | Meta (Llama), OpenAI (GPT), Stability AI |

Честный ответ: используйте PyTorch, если у вас нет конкретной причины использовать JAX. Такие причины: доступ к TPU, необходимость градиентов по отдельным примерам, обучение на многих устройствах в огромном масштабе или работа в Google/DeepMind/Anthropic.

### Случайные числа в JAX

В JAX нет глобального случайного состояния. Каждая случайная операция требует явного PRNG-ключа:

```python
key = jax.random.PRNGKey(42)
key1, key2 = jax.random.split(key)
w = jax.random.normal(key1, shape=(784, 256))
```

Сначала это раздражает. Но это гарантирует воспроизводимость между устройствами и компиляциями — свойство, которое PyTorch `torch.manual_seed` не может гарантировать в multi-GPU настройках.

## Соберите это

### Шаг 1: настройка и данные

Мы обучим 3-слойный MLP на MNIST, используя JAX и Optax. 784 входа, два скрытых слоя по 256 и 128 нейронов, 10 выходных классов.

```python
import jax
import jax.numpy as jnp
from jax import random
import optax

def get_mnist_data():
    from sklearn.datasets import fetch_openml
    mnist = fetch_openml('mnist_784', version=1, as_frame=False, parser='auto')
    X = mnist.data.astype('float32') / 255.0
    y = mnist.target.astype('int')
    X_train, X_test = X[:60000], X[60000:]
    y_train, y_test = y[:60000], y[60000:]
    return X_train, y_train, X_test, y_test
```

### Шаг 2: инициализация параметров

Никакого класса. Просто функция, которая возвращает pytree:

```python
def init_params(key):
    k1, k2, k3 = random.split(key, 3)
    scale1 = jnp.sqrt(2.0 / 784)
    scale2 = jnp.sqrt(2.0 / 256)
    scale3 = jnp.sqrt(2.0 / 128)
    params = {
        'layer1': {
            'w': scale1 * random.normal(k1, (784, 256)),
            'b': jnp.zeros(256),
        },
        'layer2': {
            'w': scale2 * random.normal(k2, (256, 128)),
            'b': jnp.zeros(128),
        },
        'layer3': {
            'w': scale3 * random.normal(k3, (128, 10)),
            'b': jnp.zeros(10),
        },
    }
    return params
```

He-инициализация, выполненная вручную. Три PRNG-ключа, отделенные от одного seed. Каждый вес — неизменяемый массив во вложенном dict.

### Шаг 3: прямой проход

```python
def forward(params, x):
    x = jnp.dot(x, params['layer1']['w']) + params['layer1']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer2']['w']) + params['layer2']['b']
    x = jax.nn.relu(x)
    x = jnp.dot(x, params['layer3']['w']) + params['layer3']['b']
    return x

def loss_fn(params, x, y):
    logits = forward(params, x)
    one_hot = jax.nn.one_hot(y, 10)
    return -jnp.mean(jnp.sum(jax.nn.log_softmax(logits) * one_hot, axis=-1))
```

Чистые функции. Параметры на вход, предсказание на выход. Нет `self`, нет сохраненного состояния. `loss_fn` вычисляет cross-entropy с нуля: softmax, log, отрицательное среднее.

### Шаг 4: JIT-скомпилированный шаг обучения

```python
@jax.jit
def train_step(params, opt_state, x, y):
    loss, grads = jax.value_and_grad(loss_fn)(params, x, y)
    updates, opt_state = optimizer.update(grads, opt_state, params)
    params = optax.apply_updates(params, updates)
    return params, opt_state, loss

@jax.jit
def accuracy(params, x, y):
    logits = forward(params, x)
    preds = jnp.argmax(logits, axis=-1)
    return jnp.mean(preds == y)
```

`jax.value_and_grad` возвращает и значение loss, и градиенты за один проход. Декоратор `@jax.jit` компилирует обе функции в XLA. После первого вызова каждый шаг обучения выполняется без обращения к Python.

### Шаг 5: цикл обучения

```python
optimizer = optax.adam(learning_rate=1e-3)

X_train, y_train, X_test, y_test = get_mnist_data()
X_train, X_test = jnp.array(X_train), jnp.array(X_test)
y_train, y_test = jnp.array(y_train), jnp.array(y_test)

key = random.PRNGKey(0)
params = init_params(key)
opt_state = optimizer.init(params)

batch_size = 128
n_epochs = 10

for epoch in range(n_epochs):
    key, subkey = random.split(key)
    perm = random.permutation(subkey, len(X_train))
    X_shuffled = X_train[perm]
    y_shuffled = y_train[perm]

    epoch_loss = 0.0
    n_batches = len(X_train) // batch_size
    for i in range(n_batches):
        start = i * batch_size
        xb = X_shuffled[start:start + batch_size]
        yb = y_shuffled[start:start + batch_size]
        params, opt_state, loss = train_step(params, opt_state, xb, yb)
        epoch_loss += loss

    train_acc = accuracy(params, X_train[:5000], y_train[:5000])
    test_acc = accuracy(params, X_test, y_test)
    print(f"Epoch {epoch + 1:2d} | Loss: {epoch_loss / n_batches:.4f} | "
          f"Train Acc: {train_acc:.4f} | Test Acc: {test_acc:.4f}")
```

10 эпох. ~97% test accuracy. Первая эпоха медленная (JIT-компиляция). Эпохи 2-10 быстрые.

Обратите внимание, чего не хватает: нет `.zero_grad()`, нет `.backward()`, нет `.step()`. Все обновление — один вызов составленной функции. Градиенты вычисляются, преобразуются Adam и применяются к параметрам — все внутри `train_step`.

## Используйте это

### Flax: стандарт Google

Flax — самая распространенная библиотека нейронных сетей для JAX. Она возвращает `nn.Module`, но с явным управлением состоянием:

```python
import flax.linen as nn

class MLP(nn.Module):
    @nn.compact
    def __call__(self, x):
        x = nn.Dense(256)(x)
        x = nn.relu(x)
        x = nn.Dense(128)(x)
        x = nn.relu(x)
        x = nn.Dense(10)(x)
        return x

model = MLP()
params = model.init(jax.random.PRNGKey(0), jnp.ones((1, 784)))
logits = model.apply(params, x_batch)
```

Та же структура, что в PyTorch, но `params` отделены от модели. `model.init()` создает params. `model.apply(params, x)` запускает прямой проход. У объекта модели нет состояния.

### Equinox: Pythonic-альтернатива

Equinox (от Patrick Kidger) представляет модели как pytrees:

```python
import equinox as eqx

model = eqx.nn.MLP(
    in_size=784, out_size=10, width_size=256, depth=2,
    activation=jax.nn.relu, key=jax.random.PRNGKey(0)
)
logits = model(x)
```

Сама модель — pytree. `.apply()` не нужен. Параметры — это просто листья модели. Это ближе к тому, как мыслит JAX.

### Optax: компонуемые оптимизаторы

Optax отделяет преобразование градиентов от обновления:

```python
schedule = optax.warmup_cosine_decay_schedule(
    init_value=0.0, peak_value=1e-3,
    warmup_steps=1000, decay_steps=50000
)

optimizer = optax.chain(
    optax.clip_by_global_norm(1.0),
    optax.adamw(learning_rate=schedule, weight_decay=0.01),
)
```

Gradient clipping, warmup learning rate, weight decay — все составлено как цепочка преобразований. Каждое преобразование видит градиенты, изменяет их и передает следующему. Нет монолитного класса оптимизатора.

## Доведите до поставки

**Установка:**

```bash
pip install jax jaxlib optax flax
```

Для поддержки GPU:

```bash
pip install jax[cuda12]
```

Для TPU (Google Cloud):

```bash
pip install jax[tpu] -f https://storage.googleapis.com/jax-releases/libtpu_releases.html
```

**Подводные камни производительности:**

- Первый JIT-вызов медленный (компиляция). Прогрейте перед benchmarking.
- Избегайте Python-циклов по массивам JAX внутри JIT. Используйте `jax.lax.scan` или `jax.lax.fori_loop`.
- `jax.debug.print()` работает внутри JIT. Обычный `print()` — нет.
- Профилируйте с `jax.profiler` или TensorBoard. Компиляция XLA может скрывать bottlenecks.
- JAX по умолчанию заранее выделяет 75% GPU-памяти. Установите `XLA_PYTHON_CLIENT_PREALLOCATE=false`, чтобы отключить это.

**Чекпоинтинг:**

```python
import orbax.checkpoint as ocp
checkpointer = ocp.PyTreeCheckpointer()
checkpointer.save('/tmp/model', params)
restored = checkpointer.restore('/tmp/model')
```

**Этот урок создает:**
- `outputs/prompt-jax-optimizer.md` — prompt для выбора правильной конфигурации оптимизатора JAX
- `outputs/skill-jax-patterns.md` — skill, описывающий функциональные паттерны в JAX

## Упражнения

1. Добавьте dropout в MLP. В JAX dropout требует PRNG-ключ: протащите ключ через прямой проход и разделяйте его для каждого dropout-слоя. Сравните test accuracy с ним и без него.

2. Используйте `jax.vmap`, чтобы вычислить градиенты по отдельным примерам для батча из 32 изображений MNIST. Вычислите норму градиента для каждого примера. У каких примеров градиенты самые большие и почему?

3. Замените ручную функцию прямого прохода на обобщенную `mlp_forward(params, x)`, которая работает для любого числа слоев. Используйте `jax.tree.leaves`, чтобы автоматически определить глубину.

4. Измерьте шаг обучения с `@jax.jit` и без него. Засеките время 100 шагов каждого варианта. Насколько велико ускорение на вашем железе? Каковы накладные расходы компиляции при первом вызове?

5. Реализуйте gradient clipping, составив `optax.chain(optax.clip_by_global_norm(1.0), optax.adam(1e-3))`. Обучите с clipping и без него. Постройте график нормы градиента по ходу обучения, чтобы увидеть эффект.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| XLA | "То, что делает JAX быстрым" | Accelerated Linear Algebra — компилятор, который объединяет операции и генерирует оптимизированные GPU/TPU-ядра из вычислительного графа |
| JIT | "Just-in-time compilation" | JAX трассирует функцию при первом вызове, компилирует в XLA, а затем запускает скомпилированную версию при последующих вызовах |
| Чистая функция | "Нет побочных эффектов" | Функция, чей выход зависит только от входов: без глобального состояния, без мутации, без случайности без явных ключей |
| vmap | "Auto-batching" | Преобразует функцию, обрабатывающую один пример, в функцию, обрабатывающую батч, без переписывания |
| pmap | "Auto-parallelism" | Реплицирует функцию на несколько устройств и разбивает входной батч |
| Pytree | "Вложенный dict массивов" | Любая вложенная структура из списков, кортежей, словарей и массивов, которую JAX может обходить и преобразовывать |
| Трассировка | "Запись вычисления" | JAX выполняет функцию с абстрактными значениями, чтобы построить вычислительный граф, не считая реальные результаты |
| Функциональное автодифференцирование | "grad от функции" | Вычисление производных путем преобразования функций, а не путем прикрепления хранилища градиентов к тензорам |
| Optax | "Библиотека оптимизаторов JAX" | Компонуемая библиотека градиентных преобразований — Adam, SGD, clipping, scheduling, — которые соединяются в цепочки |
| Flax | "JAX-овский nn.Module" | Библиотека нейронных сетей Google для JAX, добавляющая абстракции слоев при сохранении явного состояния |

## Дополнительное чтение

- Документация JAX: https://jax.readthedocs.io/ — официальная документация с отличными tutorials по grad, jit и vmap
- "JAX: composable transformations of Python+NumPy programs" (Bradbury et al., 2018) — исходная статья, объясняющая философию дизайна
- Документация Flax: https://flax.readthedocs.io/ — библиотека нейронных сетей Google для JAX
- Patrick Kidger, "Equinox: neural networks in JAX via callable PyTrees and filtered transformations" (2021) — Pythonic-альтернатива Flax
- DeepMind, "Optax: composable gradient transformation and optimisation" — стандартная библиотека оптимизаторов
- "You Don't Know JAX" (Colin Raffel, 2020) — практическое руководство по подводным камням и паттернам JAX от одного из авторов T5
