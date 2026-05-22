# Автоэнкодеры и вариационные автоэнкодеры (VAE)

> Обычный autoencoder сжимает, затем восстанавливает. Он запоминает. Он не генерирует. Добавьте один прием — заставьте код выглядеть гауссовым — и получите sampler. Именно этот прием, reparameterization `z = μ + σ·ε`, объясняет, почему у каждой image-модели на latent-diffusion и flow-matching, которой вы пользуетесь в 2026 году, на входе стоит VAE.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 3 · 02 (Backprop), Фаза 3 · 07 (CNNs), Фаза 8 · 01 (Taxonomy)
**Время:** ~75 минут

## Проблема

Сжать MNIST-цифру из 784 пикселей в код из 16 чисел, затем восстановить. Обычный autoencoder отлично справится с reconstruction MSE, но пространство кодов будет комковатым хаосом. Возьмите случайную точку в пространстве кодов, декодируйте ее — и получите шум. У него нет sampler. Это compression model в генеративном костюме.

На самом деле нужно следующее: (a) пространство кодов — чистое, гладкое распределение, из которого можно sample, например isotropic Gaussian `N(0, I)`, (b) декодирование любого sample дает правдоподобную цифру, и (c) encoder и decoder все еще хорошо сжимают. Три цели, одна архитектура, одна loss.

VAE Kingma 2013 решает это так: encoder обучается выдавать *распределение* `q(z|x) = N(μ(x), σ(x)²)`, это распределение притягивается к prior `N(0, I)` через KL penalty, а затем `z` sample-ится из `q(z|x)` перед декодированием. На inference encoder отбрасывают, берут `z ~ N(0, I)` и декодируют. Именно KL penalty заставляет пространство кодов быть структурированным.

В 2026 году VAE редко поставляются как самостоятельные модели — diffusion обогнала их по качеству сырых изображений, — но они остаются главным encoder для каждой latent-diffusion модели (SD 1/2/XL/3, Flux, AudioCraft). Изучая VAE, вы изучаете невидимый первый слой каждой image pipeline, которой пользуетесь.

## Концепция

![Autoencoder vs VAE: the reparameterization trick](../assets/vae.svg)

**Autoencoder.** `z = encoder(x)`, `x̂ = decoder(z)`, loss = `||x - x̂||²`. Пространство кодов неструктурировано.

**VAE encoder.** Выдает два вектора: `μ(x)` и `log σ²(x)`. Они задают `q(z|x) = N(μ, diag(σ²))`.

**Reparameterization trick.** Sampling из `q(z|x)` недифференцируем. Перепишите sample как `z = μ + σ·ε`, где `ε ~ N(0, I)`. Теперь `z` — детерминированная функция от `(μ, σ)` плюс непараметрический шум; градиенты проходят через `μ` и `σ`.

**Loss.** Evidence Lower BOund (ELBO), два слагаемых:

```
loss = reconstruction + β · KL[q(z|x) || N(0, I)]
     = ||x - x̂||²  + β · Σ_i ( σ_i² + μ_i² - log σ_i² - 1 ) / 2
```

Reconstruction тянет `x̂` к `x`. KL тянет `q(z|x)` к prior. Между ними trade-off. Малое β (<1) = более резкие samples, пространство кодов менее гауссово. Большое β (>1) = более чистое пространство кодов, более размытые samples. β-VAE (Higgins 2017) сделала эту ручку известной и запустила исследования disentanglement.

**Sampling.** На inference: взять `z ~ N(0, I)`, прогнать через decoder. Один forward pass — без iterative sampling как в diffusion.

## Практика

`code/main.py` реализует крошечный VAE без numpy или torch. Вход — 8-мерные синтетические данные из 2-component Gaussian mixture в 8-D. Encoder и decoder — MLP с одним hidden layer. Мы реализуем tanh activation, forward pass, loss и рукописный backward pass. Не production — педагогика.

### Шаг 1: encoder forward

```python
def encode(x, enc):
    h = tanh(add(matmul(enc["W1"], x), enc["b1"]))
    mu = add(matmul(enc["W_mu"], h), enc["b_mu"])
    log_sigma2 = add(matmul(enc["W_sig"], h), enc["b_sig"])
    return mu, log_sigma2
```

`log σ²` вместо `σ`, чтобы выход сети был неограниченным (softplus от σ — ловушка: градиенты умирают при σ ≈ 0).

### Шаг 2: reparameterize and decode

```python
def reparameterize(mu, log_sigma2, rng):
    eps = [rng.gauss(0, 1) for _ in mu]
    sigma = [math.exp(0.5 * lv) for lv in log_sigma2]
    return [m + s * e for m, s, e in zip(mu, sigma, eps)]

def decode(z, dec):
    h = tanh(add(matmul(dec["W1"], z), dec["b1"]))
    return add(matmul(dec["W_out"], h), dec["b_out"])
```

### Шаг 3: the ELBO

```python
def elbo(x, x_hat, mu, log_sigma2, beta=1.0):
    recon = sum((a - b) ** 2 for a, b in zip(x, x_hat))
    kl = 0.5 * sum(math.exp(lv) + m * m - lv - 1 for m, lv in zip(mu, log_sigma2))
    return recon + beta * kl, recon, kl
```

Точный closed-form KL, потому что оба распределения гауссовы. Не интегрируйте численно. Люди все еще поставляют код с monte-carlo KL estimates в 2026 году — это в 3x медленнее без причины.

### Шаг 4: generate

```python
def sample(dec, z_dim, rng):
    z = [rng.gauss(0, 1) for _ in range(z_dim)]
    return decode(z, dec)
```

Это и есть генеративная модель. Пять строк.

## Подводные камни

- **Posterior collapse.** KL term так агрессивно тянет `q(z|x) → N(0, I)`, что `z` не несет информации о `x`. Исправление: β-annealing (начать с β=0, поднять до 1), free bits или пропустить KL на неактивных измерениях.
- **Blurry samples.** Gaussian decoder likelihood подразумевает MSE reconstruction, которая Bayes-optimal для L2 (среднее), а среднее набора правдоподобных цифр — размытая цифра. Исправление: discrete decoder (VQ-VAE, NVAE) или использовать VAE только как encoder и поставить diffusion поверх latents (так делает Stable Diffusion).
- **β слишком велика и слишком рано.** См. posterior collapse. Начинайте с β≈0.01 и повышайте.
- **Latent dim слишком мала.** 16-D работает для MNIST, 256-D для ImageNet 256², 2048-D для ImageNet 1024². VAE в Stable Diffusion сжимает 512×512×3 → 64×64×4 (32x downsample factor по площади пространства и 32x по каналам).

## Применение

VAE stack в 2026 году:

| Ситуация | Выбор |
|-----------|------|
| Image-latent encoder для diffusion | Stable Diffusion VAE (`sd-vae-ft-ema`) или Flux VAE |
| Audio-latent encoder | Encodec (Meta), SoundStream или DAC (Descript) |
| Video latents | Spatiotemporal patches Sora, Latte VAE, WAN VAE |
| Disentangled representation learning | β-VAE, FactorVAE, TCVAE |
| Discrete latents (для transformer modelling) | VQ-VAE, RVQ (ResidualVQ) |
| Continuous latents для generation | Plain VAE, затем condition flow/diffusion model в этом latent space |

Latent-diffusion model — это VAE с diffusion model между encoder и decoder. VAE делает грубое сжатие, diffusion model выполняет основную работу. Та же схема для видео (VAE + video-diffusion DiT) и аудио (Encodec + MusicGen transformer).

## Запуск в продукт

Сохраните `outputs/skill-vae-trainer.md`.

Навык принимает: dataset profile + latent-dim target + downstream use (reconstruction, sampling или latent-diffusion input) и выдает: architecture choice (plain/β/VQ/RVQ), β schedule, latent dim, decoder likelihood (Gaussian vs categorical) и evaluation plan (recon MSE, KL per dim, Fréchet distance между `q(z|x)` и `N(0, I)`).

## Упражнения

1. **Легко.** Измените `β` в `code/main.py` на `0.01`, `0.1`, `1.0`, `5.0`. Запишите итоговые reconstruction MSE и KL. Какая β Pareto-best для ваших synthetic data?
2. **Средне.** Замените Gaussian decoder likelihood на Bernoulli likelihood (cross-entropy loss). Сравните качество samples на binarized версии тех же synthetic data.
3. **Сложно.** Расширьте `code/main.py` до mini VQ-VAE: замените continuous `z` на nearest-neighbour lookup в codebook из K=32 entries. Сравните reconstruction MSE и сообщите, сколько codebook entries используется (codebook collapse реален).

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Autoencoder | Encode-decode network | `x → z → x̂`, учит MSE. Не генеративен. |
| VAE | AE with a sampler | Encoder выдает распределение, KL penalty формирует пространство кодов. |
| ELBO | Evidence lower bound | `log p(x) ≥ recon - KL[q(z|x) \|\| p(z)]`; tight, когда `q = p(z|x)`. |
| Reparameterization | `z = μ + σ·ε` | Переписывает стохастический узел как deterministic + pure noise. Позволяет backprop через sampling. |
| Prior | `p(z)` | Целевое распределение для latent, обычно `N(0, I)`. |
| Posterior collapse | "KL term wins" | Encoder игнорирует `x`, выдает prior; decoder вынужден hallucinate. |
| β-VAE | Tunable KL weight | `loss = recon + β·KL`. Чем выше β, тем больше disentanglement, но сильнее blur. |
| VQ-VAE | Discrete latent | Заменяет continuous `z` ближайшим codebook vector; включает transformer modelling. |

## Production note: VAE — самая горячая часть diffusion server

В pipeline Stable Diffusion / Flux / SD3 VAE вызывается дважды на request — один раз для encode (если это img2img / inpainting) и один раз для decode. На 1024² проход decoder часто дает самый большой пик activation-memory во всей pipeline, потому что он upsample-ит `128×128×16` latents обратно в `1024×1024×3`. Два практических следствия:

- **Slice or tile the decode.** `diffusers` предоставляет `pipe.vae.enable_slicing()` и `pipe.vae.enable_tiling()`. Tiling меняет небольшой seam artifact на память `O(tile²)` вместо `O(H·W)`. Необходим для 1024²+ на consumer GPUs.
- **bf16 decoder, fp32 numerics для final resize.** SD 1.x VAE был выпущен в fp32 и *тихо производит NaNs* при приведении к fp16 на 1024²+. SDXL поставляет `madebyollin/sdxl-vae-fp16-fix` — всегда предпочитайте fp16-fix variant или используйте bf16.

## Дополнительное чтение

- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — статья о VAE.
- [Higgins et al. (2017). β-VAE: Learning Basic Visual Concepts with a Constrained Variational Framework](https://openreview.net/forum?id=Sy2fzU9gl) — disentangled β-VAE.
- [van den Oord et al. (2017). Neural Discrete Representation Learning](https://arxiv.org/abs/1711.00937) — VQ-VAE.
- [Vahdat & Kautz (2021). NVAE: A Deep Hierarchical Variational Autoencoder](https://arxiv.org/abs/2007.03898) — state-of-the-art image VAE.
- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion; VAE как encoder.
- [Défossez et al. (2022). High Fidelity Neural Audio Compression](https://arxiv.org/abs/2210.13438) — Encodec, стандарт audio VAE.
