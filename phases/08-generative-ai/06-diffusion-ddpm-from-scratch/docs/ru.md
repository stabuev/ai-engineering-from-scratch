# Диффузионные модели — DDPM с нуля

> Ho, Jain, Abbeel (2020) дали области рецепт, от которого она уже не отказалась. Разрушайте данные шумом за тысячу малых шагов. Обучите одну нейросеть предсказывать шум. На inference обратите процесс. Сегодня каждая mainstream модель изображений, видео, 3D и музыки работает на этом цикле, возможно, с flow matching или consistency tricks сверху.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 3 · 02 (Backprop), Фаза 8 · 02 (VAE)
**Время:** ~75 минут

## Проблема

Вам нужен sampler для `p_data(x)`. GAN играют в minimax game, которая часто расходится. VAE производят размытые samples из Gaussian decoder. На самом деле нужна training objective, которая (a) является одной стабильной loss (без saddle point, без minimax), (b) дает lower bound на `log p(x)` (то есть есть likelihoods), и (c) дает samples SOTA-качества.

Sohl-Dickstein et al. (2015) дали теоретический ответ: определить Markov chain `q(x_t | x_{t-1})`, которая постепенно добавляет Gaussian noise, и обучить reverse chain `p_θ(x_{t-1} | x_t)` denoise-ить. Ho, Jain, Abbeel (2020) показали, что loss можно упростить до одной строки — предсказывать шум — и привели математику в порядок. В 2020 это было любопытством. В 2021 дало state-of-the-art samples. В 2022 стало Stable Diffusion. В 2026 это substrate.

## Концепция

![DDPM: forward noise, reverse denoise](../assets/ddpm.svg)

**Forward process `q`.** Добавлять Gaussian noise за `T` малых шагов. Closed form — причина, по которой математика tractable, — состоит в том, что cumulative step тоже Gaussian:

```
q(x_t | x_0) = N( sqrt(α̅_t) · x_0,  (1 - α̅_t) · I )
```

где `α̅_t = ∏_{s=1..t} (1 - β_s)` для schedule `β_t`. Возьмите `β_t` линейно от 1e-4 до 0.02 на T=1000 steps, и `x_T` будет примерно `N(0, I)`.

**Reverse process `p_θ`.** Обучить neural net `ε_θ(x_t, t)`, которая предсказывает добавленный шум. Имея `x_t`, denoise делается так:

```
x_{t-1} = (1 / sqrt(α_t)) · ( x_t - (β_t / sqrt(1 - α̅_t)) · ε_θ(x_t, t) )  +  σ_t · z
```

где `σ_t` — либо `sqrt(β_t)`, либо learned variance. Выражение некрасивое, но это просто алгебра: решить для `x_{t-1}` по posterior `q(x_{t-1} | x_t, x_0)` и заменить `x_0` его noise-predicted estimate.

**Training loss.**

```
L_simple = E_{x_0, t, ε} [ || ε - ε_θ( sqrt(α̅_t) · x_0 + sqrt(1 - α̅_t) · ε,  t ) ||² ]
```

Sample `x_0` из data, выберите random `t`, sample `ε ~ N(0, I)`, посчитайте noisy `x_t` за один раз через closed form и регрессируйте шум. Одна loss, без minimax, без KL, без reparameterization tricks.

**Sampling.** Начните с `x_T ~ N(0, I)`. Итерируйте reverse step от `t = T` до `1`. Готово.

## Почему это работает

Три интуиции:

1. **Denoising is easy; generating is hard.** При `t=T` данные — чистый шум, сеть решает тривиальную задачу. При `t=0` ей нужно очистить только несколько пикселей. На промежуточных `t` задача сложна, но через одни и те же weights идет много gradients со всех noise levels.

2. **Score matching in disguise.** Vincent (2011) доказал, что prediction шума эквивалентен оценке `∇_x log q(x_t | x_0)`, то есть *score*. Reverse SDE использует этот score, чтобы идти вверх по density gradient — guided random walk к high-probability regions.

3. **ELBO сводится к simple MSE.** Полная variational lower bound имеет KL term на каждый timestep. С DDPM parameterization эти KL terms упрощаются до MSE на noise prediction со специальными коэффициентами; Ho отбросил коэффициенты (назвав это "simple" loss), и quality *улучшилось*.

## Практика

`code/main.py` реализует 1-D DDPM. Data — two-mode mixture. "Net" — крошечный MLP, который принимает `(x_t, t)` и выдает predicted noise. Training — one-line loss. Sampling итерирует reverse chain.

### Шаг 1: the forward schedule (closed form)

```python
betas = [1e-4 + (0.02 - 1e-4) * t / (T - 1) for t in range(T)]
alphas = [1 - b for b in betas]
alpha_bars = []
cum = 1.0
for a in alphas:
    cum *= a
    alpha_bars.append(cum)
```

### Шаг 2: sample `x_t` in one shot

```python
def forward_sample(x0, t, alpha_bars, rng):
    a_bar = alpha_bars[t]
    eps = rng.gauss(0, 1)
    x_t = math.sqrt(a_bar) * x0 + math.sqrt(1 - a_bar) * eps
    return x_t, eps
```

### Шаг 3: one training step

```python
def train_step(x0, model, alpha_bars, rng):
    t = rng.randrange(T)
    x_t, eps = forward_sample(x0, t, alpha_bars, rng)
    eps_hat = model_forward(model, x_t, t)
    loss = (eps - eps_hat) ** 2
    return loss, gradient_step(model, ...)
```

### Шаг 4: reverse sampling

```python
def sample(model, alpha_bars, T, rng):
    x = rng.gauss(0, 1)
    for t in range(T - 1, -1, -1):
        eps_hat = model_forward(model, x, t)
        beta_t = 1 - alphas[t]
        x = (x - beta_t / math.sqrt(1 - alpha_bars[t]) * eps_hat) / math.sqrt(alphas[t])
        if t > 0:
            x += math.sqrt(beta_t) * rng.gauss(0, 1)
    return x
```

Для 1-D задачи с 40 timesteps и 24-unit MLP это учит two-mode mixture примерно за ~200 epochs.

## Временное conditioning

Сеть должна знать, какой timestep она denoise-ит. Два стандартных варианта:

- **Sinusoidal embedding.** Как Transformer positional encoding. `embed(t) = [sin(t/ω_0), cos(t/ω_0), sin(t/ω_1), ...]`. Пропустить через MLP, broadcast в сеть.
- **Film / group-norm conditioning.** Project embedding в per-channel scale/bias (FiLM) на каждом block.

Наш игрушечный код использует sinusoidal → concat. Production U-Nets используют FiLM.

## Подводные камни

- **Schedule matters a lot.** Linear `β` — DDPM default, но cosine schedule (Nichol & Dhariwal, 2021) дает лучший FID за тот же compute. Меняйте schedules, если quality вышла на плато.
- **Timestep embedding is fragile.** Передача raw `t` как float работает для toy 1-D, но ломается на images; всегда используйте proper embedding.
- **V-prediction vs ε-prediction.** Для узких режимов (очень малый или очень большой t) у `ε` плохой signal-to-noise. V-prediction (`v = α·ε - σ·x`) стабильнее; SDXL, SD3 и Flux используют его.
- **Classifier-free guidance.** На inference посчитайте conditional и unconditional `ε`, затем `ε_cfg = (1 + w) · ε_cond - w · ε_uncond` с `w ≈ 3-7`. Рассматривается в Lesson 08.
- **1000 steps is a lot.** Production использует DDIM (20-50 steps), DPM-Solver (10-20 steps) или distillation (1-4 steps). См. Lesson 12.

## Применение

| Роль | Typical stack in 2026 |
|------|-----------------------|
| Image pixel-space diffusion (small, toy) | DDPM + U-Net |
| Image latent diffusion | VAE encoder + U-Net or DiT (Lesson 07) |
| Video latent diffusion | Spatiotemporal DiT (Sora, Veo, WAN) |
| Audio latent diffusion | Encodec + diffusion transformer |
| Science (molecules, proteins, physics) | Equivariant diffusion (EDM, RFdiffusion, AlphaFold3) |

Diffusion — универсальный generative backbone. Flow matching (Lesson 13) — конкурент 2024-2026 годов, который обычно выигрывает по inference speed при том же quality.

## Запуск в продукт

Сохраните `outputs/skill-diffusion-trainer.md`. Навык принимает dataset + compute budget и выдает: schedule (linear/cosine/sigmoid), prediction target (ε/v/x), number of steps, guidance scale, sampler family и eval protocol.

## Упражнения

1. **Легко.** Измените T с 40 на 10 в `code/main.py`. Как ухудшается sample quality (visual histogram of outputs)? При каком T two-mode structure collapse-ится?
2. **Средне.** Перейдите с ε-prediction на v-prediction. Заново выведите reverse step. Сравните итоговое sample quality.
3. **Сложно.** Добавьте classifier-free guidance. Condition на class label `c ∈ {0, 1}`, dropping его 10% времени при training, а на sampling используйте `ε = (1+w)·ε_cond - w·ε_uncond`. Измерьте conditional-mode-hit rate при `w = 0, 1, 3, 7`.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Forward process | "Adding noise" | Fixed Markov chain `q(x_t | x_{t-1})`, которая разрушает данные. |
| Reverse process | "Denoising" | Learned chain `p_θ(x_{t-1} | x_t)`, которая восстанавливает данные. |
| β schedule | "The noise ladder" | Per-step variance; linear, cosine или sigmoid. |
| α̅ | "Alpha bar" | Cumulative product `∏(1 - β)`; дает closed-form `x_t` из `x_0`. |
| Simple loss | "MSE on noise" | `||ε - ε_θ(x_t, t)||²`; все variational derivations схлопываются к этому. |
| ε-prediction | "Predict noise" | Output — добавленный шум; стандартный DDPM. |
| V-prediction | "Predict velocity" | Output — `α·ε - σ·x`; лучшее conditioning across t. |
| DDPM | "The paper" | Ho et al. 2020; linear β, 1000 steps, U-Net. |
| DDIM | "Deterministic sampler" | Non-Markov sampler, 20-50 steps, та же training objective. |
| Classifier-free guidance | "CFG" | Смешивает conditional и unconditional noise predictions, чтобы усилить conditioning. |

## Production note: diffusion inference — это проблема числа шагов

DDPM paper запускает T=1000 reverse steps. В production так никто не поставляет. Каждый реальный inference stack выбирает одну из трех стратегий, и каждая ясно ложится на production framing "откуда берется latency":

1. **Faster sampler, same model.** DDIM (20-50 steps), DPM-Solver++ (10-20), UniPC (8-16). Drop-in replacement reverse loop; trained `ε_θ` weights не меняются. Срезает latency в 20-50×.
2. **Distillation.** Обучить student совпадать с teacher за меньшее число шагов: Progressive Distillation (2 → 1), Consistency Models (arbitrary → 1-4), LCM, SDXL-Turbo, SD3-Turbo. Срезает latency еще в 5-10×, требует retraining.
3. **Caching and compilation.** `torch.compile(unet, mode="reduce-overhead")`, diffusion backends TensorRT-LLM, `xformers`/SDPA attention, bf16 weights. Срезает per-step latency примерно в ~2×. Stack-ится с (1) и (2).

Для production diffusion server разговор о budget такой же, как production literature описывает для LLMs: latency = `num_steps × step_cost + VAE_decode`, throughput = `batch_size × (num_steps × step_cost)^-1`. TTFT мал (один step); TPOT-equivalent — полное response time, потому что image generation для пользователя выглядит как "all-at-once".

## Дополнительное чтение

- [Sohl-Dickstein et al. (2015). Deep Unsupervised Learning using Nonequilibrium Thermodynamics](https://arxiv.org/abs/1503.03585) — diffusion paper, опередившая время.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — DDPM.
- [Song, Meng, Ermon (2021). Denoising Diffusion Implicit Models](https://arxiv.org/abs/2010.02502) — DDIM, fewer steps.
- [Nichol & Dhariwal (2021). Improved DDPM](https://arxiv.org/abs/2102.09672) — cosine schedule, learned variance.
- [Dhariwal & Nichol (2021). Diffusion Models Beat GANs on Image Synthesis](https://arxiv.org/abs/2105.05233) — classifier guidance.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Karras et al. (2022). Elucidating the Design Space of Diffusion-Based Generative Models (EDM)](https://arxiv.org/abs/2206.00364) — unified notation, самый чистый рецепт.
