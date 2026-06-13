# Mixture of Experts (MoE)

> Плотный transformer на 70B активирует каждый параметр для каждого токена. MoE на 671B активирует только 37B на токен и обходит его на каждом benchmark. Разреженность — самая важная идея масштабирования десятилетия.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 7 · 05 (полный transformer), Фаза 7 · 07 (GPT)
**Время:** ~45 минут

## Цели обучения

- Заменять плотный FFN маршрутизируемой смесью экспертов и прогонять токены через роутер.
- Объяснять проблему балансировки нагрузки и схемы общих и мелкозернистых экспертов.
- Рассуждать о компромиссе вычислений и памяти, который создаёт разреженность.

## Проблема

FLOPs плотного transformer при inference равны числу его параметров (умножить на 2 для forward pass). Масштабируйте плотную модель, и каждый токен оплачивает полный счет. К 2024 году frontier уперся в compute wall: чтобы стать заметно умнее, нужны были экспоненциально большие FLOPs на токен.

Mixture of Experts разрывает эту связь. Замените каждый FFN на `E` независимых experts + router, который выбирает `k` experts на токен. Total parameters = `E × FFN_size`. Active parameters per token = `k × FFN_size`. Типичная конфигурация 2026 года: `E=256`, `k=8`. Storage масштабируется с `E`, compute масштабируется с `k`.

Frontier 2026 года почти полностью MoE: DeepSeek-V3 (671B total / 37B active), Mixtral 8×22B, Qwen2.5-MoE, Llama 4, Kimi K2, gpt-oss. В независимом leaderboard Artificial Analysis все top 10 open-source models — MoE.

## Концепция

![MoE layer: router selects k of E experts per token](../assets/moe.svg)

### Замена FFN

Плотный transformer block:

```
h = x + attn(norm(x))
h = h + FFN(norm(h))
```

MoE block:

```
h = x + attn(norm(x))
scores = router(norm(h))              # (N_tokens, E)
top_k = argmax_k(scores)              # pick k of E per token
h = h + sum_{e in top_k}(
        gate(scores[e]) * Expert_e(norm(h))
    )
```

Каждый expert — независимый FFN (обычно SwiGLU). Router — один линейный слой. Каждый токен выбирает свои `k` experts и получает gated mixture их выходов.

### Проблема балансировки нагрузки

Если router отправляет 90% токенов через expert 3, остальные experts голодают. Пробовали три исправления:

1. **Auxiliary load-balancing loss** (Switch Transformer, Mixtral). Добавить штраф, пропорциональный variance использования experts. Работает, но добавляет hyperparameter и второй gradient signal.
2. **Expert capacity + token dropping** (ранний Switch). Каждый expert обрабатывает не больше `C × N/E` токенов; overflow tokens пропускают слой. Вредит качеству.
3. **Auxiliary-loss-free balancing** (DeepSeek-V3). Добавить обучаемый per-expert bias, который сдвигает top-k selection router. Bias обновляется вне training loss. Без штрафа на main objective. Большой unlock 2024 года.

Подход DeepSeek-V3: после каждого training step для каждого expert проверить, выше или ниже его usage целевого значения. Сдвинуть bias на `±γ`. Selection использует `scores + bias`. Expert probabilities, используемые для gating, — это исходные `scores` без изменений. Это отделяет routing от expression.

### Shared experts

DeepSeek-V2/V3 также делят experts на *shared* и *routed*. Каждый токен проходит через все shared experts. Routed experts выбираются через top-k. Shared experts захватывают общие знания; routed experts специализируются. V3 использует 1 shared expert плюс top-8 из 256 routed.

### Fine-grained experts

Классический MoE (GShard, Switch): каждый expert такой же широкий, как полный FFN. `E` мал (8–64), `k` мал (1–2).

Современный fine-grained MoE (DeepSeek-V3, Qwen-MoE): каждый expert уже (1/8 размера FFN). `E` велик (256+), `k` больше (8+). Total parameters те же, но combinations масштабируются намного быстрее. `C(256, 8) = 400 trillion` возможных "experts" на токен. Качество растет, latency остается плоской.

### Профиль стоимости

На токен, на слой:

| Конфигурация | Активные параметры / token | Всего параметров |
|--------|-----------------------|--------------|
| Mixtral 8×22B | ~39B | 141B |
| Llama 3 70B (dense) | 70B | 70B |
| DeepSeek-V3 | 37B | 671B |
| Kimi K2 (MoE) | ~32B | 1T |

DeepSeek-V3 обходит Llama 3 70B (dense) почти на каждом benchmark, выполняя **меньше active FLOPs per token**. More parameters = more knowledge. More active FLOPs = more compute per token. MoE разделяет эти величины.

### Подвох: память

Все experts находятся на GPU независимо от того, какие срабатывают. Модель 671B требует ~1.3 TB VRAM для fp16 weights. Frontier MoE deployment требует expert parallelism — shard experts across GPUs, route tokens across the network. Latency определяется all-to-all communication, а не matmul.

## Соберите это

См. `code/main.py`. Компактный MoE layer на чистой stdlib с:

- `n_experts=8` SwiGLU-ish experts (по одному linear для иллюстрации)
- top-k=2 routing
- softmax-normalized gating weights
- auxiliary-loss-free balancing через per-expert bias

### Шаг 1: router

```python
def route(hidden, W_router, top_k, bias):
    scores = [sum(h * w for h, w in zip(hidden, W_router[e])) for e in range(len(W_router))]
    biased = [s + b for s, b in zip(scores, bias)]
    top_idx = sorted(range(len(biased)), key=lambda i: -biased[i])[:top_k]
    # softmax over ORIGINAL scores of the chosen experts
    chosen = [scores[i] for i in top_idx]
    m = max(chosen)
    exps = [math.exp(c - m) for c in chosen]
    s = sum(exps)
    gates = [e / s for e in exps]
    return top_idx, gates
```

Bias влияет на selection, а не на gate weight. Это трюк DeepSeek-V3 — bias исправляет load imbalance, не направляя predictions модели.

### Шаг 2: пропустить 100 токенов через router

Отслеживайте, какие experts срабатывают и как часто. Без bias usage перекошен. С циклом обновления bias (`-γ` для over-used experts, `+γ` для under-used) usage сходится к uniform distribution за несколько итераций.

### Шаг 3: сравнение числа параметров

Напечатайте "dense equivalent" MoE-конфигурации. DeepSeek-V3-shaped: 256 routed + 1 shared, 8 active, d_model=7168. Total parameter count огромен. Active count — седьмая часть плотной Llama 3 70B.

## Используйте это

Загрузка через HuggingFace:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("mistralai/Mixtral-8x22B-v0.1")
```

Production inference 2026 года: vLLM нативно поддерживает MoE routing. SGLang имеет самый быстрый expert-parallel path. Оба автоматически обрабатывают top-k selection и expert parallelism.

**Когда выбирать MoE:**
- Вам нужно frontier quality при меньшей inference cost per token.
- У вас есть VRAM / expert-parallel infrastructure.
- Ваша workload token-heavy (chat, code), а не context-heavy (long docs).

**Когда НЕ выбирать MoE:**
- Edge deployment — вы платите full storage за любой active FLOP.
- Latency-critical single-user serving — expert routing добавляет overhead.
- Малые модели (<7B) — преимущество MoE по качеству появляется только выше compute threshold (~6B active params).

## Доведите до поставки

См. `outputs/skill-moe-configurator.md`. Skill выбирает E, k и layout shared-expert для нового MoE по parameter budget, training tokens и deployment target.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Посмотрите, как auxiliary-loss-free bias update выравнивает expert usage за 50 iterations.
2. **Средне.** Замените learned router на hash-based router (deterministic, no learning). Сравните качество и баланс. Почему learned router лучше?
3. **Сложно.** Реализуйте GRPO-style "rollout-matched routing" (трюк DeepSeek-V3.2): логируйте, какие experts срабатывают во время inference, и принудительно используйте тот же routing во время gradient computation. Измерьте эффект на toy policy-gradient setup.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|------------|-------------------------------|
| Expert | "Одна FFN среди многих" | Независимая feed-forward network; параметры, выделенные для sparse slice FFN-вычисления. |
| Router | "Gate" | Крошечный linear layer, который оценивает каждый токен относительно каждого expert; top-k selection. |
| Top-k routing | "k активных experts на токен" | FFN-вычисление каждого токена проходит ровно через k experts, взвешенных gate. |
| Auxiliary loss | "Штраф за дисбаланс нагрузки" | Дополнительный loss term, который штрафует перекошенное expert usage. |
| Auxiliary-loss-free | "Трюк DeepSeek-V3" | Баланс через per-expert bias только на selection router; без extra gradient. |
| Shared expert | "Всегда включен" | Дополнительный expert, через который проходит каждый токен; захватывает common knowledge. |
| Expert parallelism | "Шардировать по experts" | Распределить разных experts по разным GPUs; route tokens across the network. |
| Sparsity | "Активных параметров меньше, чем всего параметров" | Ratio `k × expert_size / (E × expert_size)`; 37/671 ≈ 5.5% для DeepSeek-V3. |

## Дополнительное чтение

- [Shazeer et al. (2017). Outrageously Large Neural Networks: The Sparsely-Gated Mixture-of-Experts Layer](https://arxiv.org/abs/1701.06538) — идея.
- [Fedus, Zoph, Shazeer (2022). Switch Transformer: Scaling to Trillion Parameter Models with Simple and Efficient Sparsity](https://arxiv.org/abs/2101.03961) — Switch, классический MoE.
- [Jiang et al. (2024). Mixtral of Experts](https://arxiv.org/abs/2401.04088) — Mixtral 8×7B.
- [DeepSeek-AI (2024). DeepSeek-V3 Technical Report](https://arxiv.org/abs/2412.19437) — MLA + auxiliary-loss-free MoE + MTP.
- [Wang et al. (2024). Auxiliary-Loss-Free Load Balancing Strategy for Mixture-of-Experts](https://arxiv.org/abs/2408.15664) — статья о bias-based balancing.
- [Dai et al. (2024). DeepSeekMoE: Towards Ultimate Expert Specialization in Mixture-of-Experts Language Models](https://arxiv.org/abs/2401.06066) — fine-grained + shared-expert split, который использует router этого урока.
- [Kim et al. (2022). DeepSpeed-MoE: Advancing Mixture-of-Experts Inference and Training](https://arxiv.org/abs/2201.05596) — оригинальная статья о shared-expert.
