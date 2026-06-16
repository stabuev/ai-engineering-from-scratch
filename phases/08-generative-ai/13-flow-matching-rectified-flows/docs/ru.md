# Flow Matching и Rectified Flows

> Diffusion models требуют 20-50 sampling steps, потому что идут кривой траекторией от noise к data. Flow matching (Lipman et al., 2023) и rectified flow (Liu et al., 2022) обучают прямые траектории. Более прямые траектории означают fewer steps и faster inference. Stable Diffusion 3, Flux.1 и AudioCraft 2 в 2024 году перешли на flow matching.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 06 (DDPM), Фаза 1 · Calculus
**Время:** ~45 минут

## Цели обучения

- Объяснять, почему flow matching с прямыми путями требует меньше шагов сэмплирования, чем кривой diffusion.
- Реализовывать обучающую потерю flow matching и многошаговое сэмплирование методом Эйлера.
- Связывать rectified flow и consistency distillation с сокращением числа шагов.

## Проблема

Reverse process DDPM — это 1000-step stochastic walk от `N(0, I)` обратно к data distribution. DDIM сжал его до 20-50 deterministic steps. Вам нужно меньше шагов — ideally one. Блокер в том, что ODE, решающая reverse process, stiff; path curved.

Если обучить модель так, чтобы path from noise to data был *straight line*, один Euler step от `t=1` до `t=0` работал бы. Flow matching строит это напрямую: define straight-line interpolation from `x_1 ∼ N(0, I)` to `x_0 ∼ data`, train vector field `v_θ(x, t)` to match its time derivative, integrate at inference.

Rectified flow (Liu 2022) идет дальше: итеративно выпрямляет paths через reflow procedure, которая дает ODE все ближе к linear. После двух reflow iterations 2-step sampler совпадает по quality с 50-step DDPM.

## Концепция

![Flow matching: straight-line interpolation between noise and data](../assets/flow-matching.svg)

### Straight-line flow

Define:

```
x_t = t · x_1 + (1 - t) · x_0,   t ∈ [0, 1]
```

где `x_0 ~ data` и `x_1 ~ N(0, I)`. Time derivative вдоль этой straight line constant:

```
dx_t / dt = x_1 - x_0
```

Define neural vector field `v_θ(x_t, t)` и обучайте match this derivative:

```
L = E_{x_0, x_1, t} || v_θ(x_t, t) - (x_1 - x_0) ||²
```

Это **conditional flow matching** loss (Lipman 2023). Training simulation-free: вы никогда не unroll ODE. Просто sample `(x_0, x_1, t)` and regress.

### Sampling

На inference integrate learned vector field *backwards* in time:

```
x_{t-Δt} = x_t - Δt · v_θ(x_t, t)
```

Start at `x_1 ~ N(0, I)`, Euler-step down to `t=0`.

### Rectified flow (Liu 2022)

Straight-line flow работает, но learned paths *на самом деле не прямые* — они curve, потому что многие `x_0` могут map to same `x_1`. Reflow step у rectified flow:

1. Train flow model v_1 with random pairings.
2. Sample N pairs `(x_1, x_0)` by integrating v_1 from `x_1` to its landing `x_0`.
3. Train v_2 on those paired examples. Поскольку pairs теперь "ODE-matched", straight-line interpolant между ними genuinely flatter.
4. Repeat.

На практике 2 reflow iterations дают near-linear paths, enabling 2-4 step inference. SDXL-Turbo, SD3-Turbo, LCM — all distilled-from-flow-matching models.

### Почему это выиграло для images в 2024

Три причины:

1. **Simulation-free training** — без ODE unrolling during training, trivial to implement.
2. **Better loss geometry** — straight paths have consistent signal-to-noise, while DDPM ε-loss has bad SNR at edges of schedule.
3. **Faster inference** — 4-8 steps at SDXL-Turbo quality; 1 step with consistency distillation.

## Flow matching vs DDPM — точная связь

Flow matching с Gaussian-conditional path — это diffusion *с конкретным noise schedule*. Выберите schedule `x_t = α(t) x_0 + σ(t) x_1`, и flow matching восстановит Stratonovich-reformulated diffusion с `v = α'·x_0 - σ'·x_1`. Для Gaussian paths они algebraically equivalent.

Что добавил flow matching: *clarity* target (plain velocity), cleaner loss и свободу экспериментировать с non-Gaussian interpolants.

## Практика

`code/main.py` реализует 1-D flow matching на two-mode Gaussian mixture. Vector field `v_θ(x, t)` — tiny MLP, trained with straight-line target. На inference integrate 1, 2, 4 и 20 Euler steps и сравните sample quality.

### Шаг 1: training loss

```python
def train_step(x0, net, rng, lr):
    x1 = rng.gauss(0, 1)
    t = rng.random()
    x_t = t * x1 + (1 - t) * x0
    target = x1 - x0
    pred = net_forward(x_t, t)
    loss = (pred - target) ** 2
    # backprop + update
```

### Шаг 2: multi-step inference

```python
def sample(net, num_steps):
    x = rng.gauss(0, 1)
    for i in range(num_steps):
        t = 1.0 - i / num_steps
        dt = 1.0 / num_steps
        x -= dt * net_forward(x, t)
    return x
```

### Шаг 3: compare step counts

Ожидайте, что 4-step sampler уже match 20-step quality — это важно для latency.

## Подводные камни

- **Time parameterization.** Flow matching использует `t ∈ [0, 1]` с `t=0` at data, `t=1` at noise. DDPM использует `t ∈ [0, T]` с `t=0` at data, `t=T` at noise. Same direction, different scale. Papers constantly get this wrong.
- **Schedule choice.** Straight line rectified flow — "the" flow-matching schedule, но можно использовать cosine или logit-normal t-sampling (SD3 does this) for better scale coverage.
- **Reflow cost.** Generating paired dataset for reflow — full inference pass per sample. Делайте reflow только когда реально нужен 1-2 step inference.
- **Classifier-free guidance still applies.** Просто замените ε на v в linear combination: `v_cfg = (1+w) v_cond - w v_uncond`.

## Применение

| Use case | 2026 stack |
|----------|-----------|
| Text-to-image, best quality | Flow matching: SD3, Flux.1-dev |
| Text-to-image, 1-4 steps | Distilled flow matching: Flux.1-schnell, SD3-Turbo, SDXL-Turbo |
| Real-time inference | Consistency distillation from a flow-matched base (LCM, PCM) |
| Audio generation | Flow matching: Stable Audio 2.5, AudioCraft 2 |
| Video generation | Flow matching mixed with diffusion (Sora, Veo, Stable Video) |
| Science / physics (particle trajectories, molecules) | Flow matching + equivariant vector field |

Когда paper говорит "faster than diffusion" in 2025-2026, почти всегда это flow matching + distillation.

## Запуск в продукт

Сохраните `outputs/skill-fm-tuner.md`. Навык принимает diffusion-style model spec и converts it to flow-matching training config: schedule choice, time sampling distribution (uniform / logit-normal), optimizer, reflow plan, target step count, eval protocol.

## Упражнения

1. **Легко.** Запустите `code/main.py` и сравните 1-step vs 20-step MSE vs true data distribution.
2. **Средне.** Перейдите с uniform `t` sampling на logit-normal (concentrates sampling at mid-t). Улучшается ли model quality?
3. **Сложно.** Реализуйте one reflow iteration: generate paired (x_0, x_1) by integrating first model, train second model on pairs, compare 1-step sample quality.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Flow matching | "Straight-line diffusion" | Train `v_θ(x, t)` to match `x_1 - x_0` along an interpolant. |
| Rectified flow | "Reflow" | Iterative procedure that straightens learned flows. |
| Velocity field | "v_θ" | Output of model — direction to move `x_t`. |
| Straight-line interpolant | "The path" | `x_t = (1-t)·x_0 + t·x_1`; trivial target derivative. |
| Euler sampler | "1st order ODE solver" | Simplest integrator; works well when paths are straight. |
| Logit-normal t | "SD3 sampling" | Concentrate `t` sampling toward mid-values where gradients strongest. |
| Consistency distillation | "1-step sampler" | Train student to map any `x_t` directly to `x_0`. |
| CFG with velocity | "v-CFG" | `v_cfg = (1+w) v_cond - w v_uncond`; same trick, new variable. |

## Production note: Flux.1-schnell — flow matching at its fastest

Production win flow matching — Flux.1-schnell: flow-matched DiT distilled to 1-4 inference steps, while keeping Flux-dev-grade quality. Notebook Niels "Run Flux on an 8GB machine" — reference deployment recipe: T5 + CLIP encode, quantized MMDiT denoise (4 steps for schnell vs 50 for dev), VAE decode. Cost accounting:

| Variant | Steps | Latency at 1024² on L4 | Total FLOPs (relative) |
|---------|-------|------------------------|------------------------|
| Flux.1-dev (raw) | 50 | ~15 s | 1.0× |
| Flux.1-schnell | 4 | ~1.2 s | 0.08× (12× faster) |
| SDXL-base | 30 | ~4 s | 0.25× |
| SDXL-Lightning 2-step | 2 | ~0.3 s | 0.03× |

Production rule: **flow-matched base + distillation = default 2026 для fast text-to-image.** Every major vendor ships this combo: SD3-Turbo (SD3 + flow + distillation), Flux-schnell (Flux-dev + rectified-flow straightening), CogView-4-Flash. Pure diffusion bases остались только для legacy checkpoints.

## Дополнительное чтение

- [Liu, Gong, Liu (2022). Flow Straight and Fast: Learning to Generate and Transfer Data with Rectified Flow](https://arxiv.org/abs/2209.03003) — rectified flow.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, rectified flow at scale.
- [Albergo, Vanden-Eijnden (2023). Stochastic Interpolants](https://arxiv.org/abs/2303.08797) — general framework covering FM + diffusion.
- [Song et al. (2023). Consistency Models](https://arxiv.org/abs/2303.01469) — 1-step distillation of diffusion / flow.
- [Sauer et al. (2023). Adversarial Diffusion Distillation (SDXL-Turbo)](https://arxiv.org/abs/2311.17042) — turbo variant.
- [Black Forest Labs (2024). Flux.1 models](https://blackforestlabs.ai/announcing-black-forest-labs/) — flow matching in production.
