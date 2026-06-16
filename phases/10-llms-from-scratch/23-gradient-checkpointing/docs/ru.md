# Gradient Checkpointing и Activation Recomputation

> Backprop хранит каждую промежуточную активацию. При 70B параметров и контексте 128K это 3 TB активаций на rank. Checkpointing обменивает FLOPs на память: не сохранять, а пересчитывать. Вопрос в том, какие сегменты отбрасывать, и ответ не сводится к "все".

**Тип:** Build
**Языки:** Python (с numpy, опционально torch)
**Предварительные требования:** Phase 10 Lesson 04 (Pre-Training Mini-GPT), Phase 10 Lesson 05 (Scaling & Distributed)
**Время:** ~70 минут

## Цели обучения

- Реализовывать gradient checkpointing: хранить входы сегментов и пересчитывать промежуточные активации на обратном проходе.
- Считать компромисс FLOP против памяти и находить оптимальный размер сегмента (правило sqrt-L).
- Различать full, selective и block checkpointing — и когда вместо этого выгружать на CPU.

## Проблема

При обучении transformer для каждого слоя сохраняет входы всех операций, которые дифференцируются в backward: входы attention, проекции Q/K/V, выход softmax, входы FFN, выходы нормализаций и residual stream. Для слоя со скрытой размерностью `d`, длиной последовательности `L` и batch `B` это порядка `12 * B * L * d` чисел с плавающей точкой на слой.

Для `d=8192, L=8192, B=1` это 800 MB на слой в BF16. Модель из 64 слоев требует 51 GB только под активации — еще до умножения на размер microbatch, до добавления промежуточных значений attention-softmax (`L^2` на head) и до учета частичных копий из tensor parallelism.

Счет приходит с двух сторон: BF16-веса вместе с состоянием оптимизатора могут поместиться в 80GB, но активации выводят вас за предел. Gradient checkpointing (он же activation recomputation) — стандартное решение. Большую часть активаций отбрасывают, а во время backward заново выполняют forward, чтобы восстановить их. Цена: дополнительные FLOPs. Выигрыш: память снижается пропорционально отношению числа checkpoint-сегментов к общему числу слоев.

При наивной реализации checkpointing добавляет примерно 33% FLOPs forward pass на шаг. При хорошей реализации — selective checkpointing по "smart selection" из Korthikanti et al. — можно сэкономить 5x памяти с накладными расходами меньше 5% FLOPs. А с FP8 matmuls, FSDP offload и expert-parallel MoE это действительно важно: нельзя позволить себе ни лишнюю память, ни впустую потраченные вычисления.

## Концепция

### Что на самом деле нужно backward

`output = layer(input)`. Backward должен получить `grad_input` и `grad_params`. Для их вычисления ему нужны:

- `input` (чтобы вычислить `grad_params = input.T @ grad_output` для линейных слоев)
- некоторые промежуточные значения производных активаций (производная ReLU/GELU/softmax зависит от значения активации)

Forward pass автоматически сохраняет это в autograd graph. Каждый `tensor.retain_grad()` и каждая операция, которой нужен ее вход, удерживают ссылку.

### Наивный full checkpointing

Разбейте сеть на `N` сегментов. Во время forward сохраняйте только *input* каждого сегмента. Когда backward нужны промежуточные значения, заново выполните forward pass сегмента, материализуйте их и затем дифференцируйте.

Пример: 32-слойный transformer разбит на 32 сегмента по 1 слою.

- Память: 32 входа слоев (мало) вместо 32 * (объем активаций на слой) (очень много).
- Дополнительные вычисления: 1 дополнительный forward на сегмент, то есть примерно на 33% больше forward FLOPs в сумме (поскольку backward стоит 2x forward, полный шаг становится 1 + 1 + 2 = 4 единицы вместо 1 + 2 = 3).

Это исходный рецепт Chen et al. 2016: один checkpoint каждые `sqrt(L)` слоев, чтобы сбалансировать память и вычисления. Для L=64 это 8 checkpoints.

### Selective Checkpointing (Korthikanti 2022)

Не все активации стоят одинаково. Выход attention softmax имеет размер `B*L*L*heads` и растет *квадратично* с длиной последовательности. Скрытая активация FFN имеет размер `B*L*4d` и растет линейно. Для длинных последовательностей softmax доминирует.

Selective checkpointing сохраняет дешевые для хранения активации (линейные проекции, residuals) и пересчитывает только дорогие (attention). Вы платите минимум FLOPs за пересчет, но экономите O(L^2) памяти.

Megatron-Core реализует это как "selective" activation recomputation. Такой подход используется в большинстве frontier training runs начиная с 2024 года.

### Offload

Альтернатива recompute: отправлять активации в CPU RAM между forward и backward. Для этого нужна пропускная способность PCIe; это выгодно, когда свободная пропускная способность превышает стоимость rematerialization. Часто применяют смешанные стратегии: для одних слоев checkpointing, для других offload.

FSDP2 предоставляет offload как первоклассную опцию. Offload особенно полезен, когда GPU упирается в память, но у передачи CPU-GPU есть запас.

### Модель стоимости recompute

FLOPs на шаг при наивном checkpointing каждые `k` слоев из `L`:

```
flops_fwd_normal = L * f_layer
flops_bwd_normal = 2 * L * f_layer
flops_total_normal = 3 * L * f_layer

flops_fwd_ckpt = L * f_layer
flops_recompute = L * f_layer  # one extra forward per layer in the segment
flops_bwd_ckpt = 2 * L * f_layer
flops_total_ckpt = 4 * L * f_layer
overhead = 4 / 3 - 1 = 0.33 = 33%
```

При selective checkpointing пересчитывается только attention kernel, а не весь слой:

```
flops_recompute_selective = L * f_attention ~= L * f_layer * 0.15
overhead_selective = (3 + 0.15) / 3 - 1 = 0.05 = 5%
```

### Модель экономии памяти

Объем активаций на слой: `A`. Для `L` слоев общий объем памяти под активации: `L * A`.

Full checkpoint (segment size 1): хранить только `L * input_volume` (~`L * 1/10 A` для стандартного transformer). Экономия примерно `9 * L * A * 1/10`.

Checkpoint каждые `k` слоев: хранить `L/k * A` плюс значения для `k-1` слоев внутри активного сегмента.

При `k = sqrt(L)` память и стоимость recompute обе масштабируются как `sqrt(L)` — оптимальный компромисс для слоев с одинаковой стоимостью.

### Когда не стоит делать checkpoint

- Самые внутренние слои pipeline stage, которые уже находятся in-flight. Их все равно придется завершить.
- Первый и последний слои, если они доминируют по вычислениям внутри stage (в transformers это редкость).
- Attention kernels, уже использующие FlashAttention: Flash уже быстро пересчитывает softmax, поэтому дополнительный checkpointing на уровне слоя дает мало пользы сверху.

### Паттерны реализации

1. **Function wrapper:** оберните сегмент в `torch.utils.checkpoint.checkpoint(fn, input)`. PyTorch сохраняет только `input`, а все остальное пересчитывает на backward.

2. **Decorator-based:** помечайте слои как checkpointable; trainer на этапе конфигурации решает, какие сегменты будут обернуты.

3. **Manual explicit recompute:** напишите backward pass самостоятельно, вызывая custom `recompute_forward`, который дублирует forward с сохраненным input.

Все три варианта дают одинаковый функциональный результат. Wrappers — стандартная идиома.

### Взаимодействие с TP / PP / FP8

- **Tensor parallel:** checkpoint inputs должны быть собраны или заново рассеяны при recompute; учитывайте стоимость коммуникации.
- **Pipeline parallel:** типичный паттерн — checkpoint forward каждого pipeline-stage, чтобы microbatches в обратном порядке могли переиспользовать память активаций.
- **FP8 recompute:** amax histories, обновленные во время recompute, должны совпадать с исходным forward, иначе FP8 scale будет дрейфовать. Большинство фреймворков делают snapshot scale.

## Соберите это

### Шаг 1: Toy model с сегментами

```python
import numpy as np


def linear_forward(x, w, b):
    return x @ w + b


def relu(x):
    return np.maximum(x, 0)


def layer_forward(x, w1, b1, w2, b2):
    h = relu(linear_forward(x, w1, b1))
    return linear_forward(h, w2, b2)


def model_forward(x, params):
    activations = [x]
    h = x
    for w1, b1, w2, b2 in params:
        h = layer_forward(h, w1, b1, w2, b2)
        activations.append(h)
    return h, activations
```

### Шаг 2: Наивный backward, которому нужны все активации

```python
def model_backward(grad_output, activations, params):
    grads = [None] * len(params)
    g = grad_output
    for i in range(len(params) - 1, -1, -1):
        w1, b1, w2, b2 = params[i]
        x_in = activations[i]
        h_pre = linear_forward(x_in, w1, b1)
        h = relu(h_pre)
        gh = g @ w2.T
        gw2 = h.T @ g
        gb2 = g.sum(axis=0)
        g_pre = gh * (h_pre > 0)
        gx = g_pre @ w1.T
        gw1 = x_in.T @ g_pre
        gb1 = g_pre.sum(axis=0)
        grads[i] = (gw1, gb1, gw2, gb2)
        g = gx
    return g, grads
```

### Шаг 3: Память при checkpoint-every-k

```python
def model_forward_checkpointed(x, params, k=4):
    saved_inputs = [x]
    h = x
    for i, (w1, b1, w2, b2) in enumerate(params):
        h = layer_forward(h, w1, b1, w2, b2)
        if (i + 1) % k == 0:
            saved_inputs.append(h)
    return h, saved_inputs


def model_backward_checkpointed(grad_output, saved_inputs, params, k=4):
    grads = [None] * len(params)
    g = grad_output
    segments = [(j * k, min((j + 1) * k, len(params))) for j in range(len(saved_inputs))]
    for seg_idx in range(len(saved_inputs) - 1, -1, -1):
        start, end = segments[seg_idx]
        if start >= end:
            continue
        x_in = saved_inputs[seg_idx]
        _, seg_acts = model_forward(x_in, params[start:end])
        g, seg_grads = model_backward(g, seg_acts, params[start:end])
        for j, gr in enumerate(seg_grads):
            grads[start + j] = gr
    return g, grads
```

### Шаг 4: Модель стоимости

```python
def checkpoint_cost(n_layers, segment_size, flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }


def selective_checkpoint_cost(n_layers, attention_fraction=0.15,
                              flops_per_layer=1.0):
    fwd = n_layers * flops_per_layer
    recompute = n_layers * attention_fraction * flops_per_layer
    bwd = 2 * n_layers * flops_per_layer
    return {
        "fwd": fwd,
        "recompute": recompute,
        "bwd": bwd,
        "total": fwd + recompute + bwd,
        "overhead_vs_no_ckpt": (fwd + recompute + bwd) / (fwd + bwd) - 1.0,
    }
```

### Шаг 5: Оценка памяти

```python
def activation_memory_mb(n_layers, hidden=8192, seq=8192,
                        batch=1, bytes_per_value=2):
    per_layer = 12 * batch * seq * hidden * bytes_per_value
    return n_layers * per_layer / 1e6


def memory_after_checkpoint(n_layers, segment_size, hidden=8192,
                           seq=8192, batch=1, bytes_per_value=2):
    n_seg = max(1, n_layers // segment_size)
    saved = (n_seg + segment_size) * 1 * batch * seq * hidden * bytes_per_value
    return saved / 1e6
```

### Шаг 6: Оптимальный размер сегмента

```python
def optimal_segment(n_layers):
    return int(round(np.sqrt(n_layers)))
```

### Шаг 7: Решение для selective checkpoint

```python
def should_recompute(layer_type, activation_bytes, recompute_flops_ratio):
    if layer_type == "attention" and activation_bytes > 100 * 1e6:
        return True
    if layer_type == "ffn" and activation_bytes > 500 * 1e6:
        return recompute_flops_ratio < 0.1
    return False
```

## Используйте это

- **torch.utils.checkpoint**: `from torch.utils.checkpoint import checkpoint` — каноническая обертка в PyTorch. Оборачивает функцию; сохраняет только inputs, пересчитывает на backward.
- **Megatron-Core activation recomputation**: поддерживает режимы `selective`, `full` и `block`. Стандарт для frontier training начиная с 2024 года.
- **FSDP2 offload**: `module.to_empty(device="cpu")` с `offload_policy` в FSDP2 шардирует активации на CPU вместо recomputing.
- **DeepSpeed ZeRO-Offload**: CPU offload для состояний оптимизатора и активаций, дополняющий checkpointing.

## Доведите до результата

Этот урок создает `outputs/prompt-activation-recompute-policy.md` — prompt, который принимает конфигурацию модели (layers, hidden, seq, batch) и доступную GPU-память, а затем выдает per-layer recompute policy (none / selective / full / offload).

## Упражнения

1. Проверьте корректность. Запустите `model_forward` + `model_backward` (полные активации) и сравните с `model_forward_checkpointed` + `model_backward_checkpointed` (сегменты). Градиенты параметров должны совпадать до машинной точности.

2. Переберите segment size `k` от 1 до `L`. Постройте графики FLOP overhead и памяти. Найдите изгиб кривой.

3. Реализуйте selective checkpointing: сохраняйте input attention-module, но не его intermediates. Измерьте FLOP overhead относительно full-layer checkpointing для 32-слойной модели при seq=8192.

4. Добавьте offload. Сохраняйте segment inputs в симулированный "CPU buffer" (отдельный список). Измерьте "PCIe bandwidth" как bytes/time и найдите точку безубыточности между offload и recompute.

5. Проведите benchmark настоящего PyTorch transformer с `torch.utils.checkpoint` и без него. Измерьте память (через `torch.cuda.max_memory_allocated`) и step time.

## Ключевые термины

| Term | Как обычно говорят | Что это на самом деле означает |
|------|--------------------|--------------------------------|
| Gradient checkpointing | "Экономим память, заново выполняя forward" | Хранить только inputs сегментов; пересчитывать intermediates во время backward, чтобы получить tensors, необходимые для градиентов |
| Activation recomputation | "То же, что checkpointing" | HPC-термин для той же техники |
| Segment size (k) | "Сколько слоев на checkpoint" | Число слоев, чьи intermediates отбрасываются и rematerialized вместе |
| Selective checkpointing | "Трюк Korthikanti" | Пересчитывать только дорогие для хранения активации (attention softmax); дешевые сохранять |
| Full checkpointing | "Наивная версия" | Пересчитывать intermediates каждого слоя в каждом сегменте |
| Block checkpointing | "Coarse-grained" | Делать checkpoint целых transformer blocks; самая крупная гранулярность |
| FLOP overhead | "Налог на вычисления" | Дополнительные FLOPs на шаг = (recompute FLOPs) / (fwd + bwd FLOPs); 33% наивно, 5% selective |
| Activation offload | "Отправить на CPU" | Перемещать активации в CPU RAM на участке forward->backward; альтернатива recompute |
| sqrt-L rule | "Классический optimum" | Для слоев с одинаковой стоимостью оптимальный интервал checkpoint равен sqrt(L) слоям |
| Attention-softmax volume | "Проблема O(L^2)" | L^2 * heads * batch чисел с плавающей точкой; доминирует в памяти активаций на длинных контекстах |

## Дополнительное чтение

- [Chen et al., 2016 -- "Training Deep Nets with Sublinear Memory Cost"](https://arxiv.org/abs/1604.06174) -- исходная статья, формализовавшая gradient checkpointing
- [Korthikanti et al., 2022 -- "Reducing Activation Recomputation in Large Transformer Models"](https://arxiv.org/abs/2205.05198) -- selective activation recomputation и формальный анализ стоимости
- [Pudipeddi et al., 2020 -- "Training Large Neural Networks with Constant Memory using a New Execution Algorithm"](https://arxiv.org/abs/2002.05645) -- альтернативный подход с постоянной памятью через reverse-mode rematerialization
- [Ren et al., 2021 -- "ZeRO-Offload: Democratizing Billion-Scale Model Training"](https://arxiv.org/abs/2101.06840) -- activation offload в масштабе
- [PyTorch torch.utils.checkpoint docs](https://pytorch.org/docs/stable/checkpoint.html) -- стандартный API
- [Megatron-Core activation recomputation documentation](https://docs.nvidia.com/nemo-framework/user-guide/latest/nemotoolkit/features/memory_optimizations.html) -- режимы selective, full и block
