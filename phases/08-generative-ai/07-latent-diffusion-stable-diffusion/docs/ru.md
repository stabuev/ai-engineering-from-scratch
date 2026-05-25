# Latent Diffusion и Stable Diffusion

> Pixel-space diffusion на изображениях 512×512 — вычислительное преступление. Rombach et al. (2022) заметили, что для генерации изображения не нужны все 786k измерений: нужно достаточно измерений для semantic structure и отдельный decoder для остального. Запустите diffusion внутри latent space VAE. Эта одна идея и есть Stable Diffusion.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 02 (VAE), Фаза 8 · 06 (DDPM), Фаза 7 · 09 (ViT)
**Время:** ~75 минут

## Проблема

Pixel-space diffusion на 512² означает, что U-Net работает с tensors формы `[B, 3, 512, 512]`. Каждый sampling step — примерно ~100 GFLOPS для 500M-param U-Net. Пятьдесят шагов — 5 TFLOPS на image. Обучите это на миллиарде images, и compute bill становится абсурдным.

Большая часть этих FLOPs уходит на проталкивание через net перцептивно неважных деталей — high-frequency texture, которую lossy VAE мог бы сжать. Идея Rombach: один раз обучить VAE (*first stage*), заморозить его и запускать diffusion полностью в 4-channel 64×64 latent space (*second stage*). Тот же U-Net. 1/16 пикселей. Примерно ~64x fewer FLOPs при сопоставимом качестве.

Это рецепт Stable Diffusion. SD 1.x / 2.x использовали 860M U-Net поверх `64×64×4` latents, SDXL — 2.6B U-Net поверх `128×128×4`, SD3 заменил U-Net на Diffusion Transformer (DiT) с flow matching. Flux.1-dev (Black Forest Labs, 2024) поставляет 12B-param DiT-MMDiT. Все работают на той же two-stage основе.

## Концепция

![Latent diffusion: VAE compression + diffusion in latent space](../assets/latent-diffusion.svg)

**Две стадии, обученные отдельно.**

1. **Stage 1 — VAE.** Encoder `E(x) → z`, decoder `D(z) → x`. Target compression: 8× downsample по каждой spatial axis + настройка channels так, чтобы total latent size был ~1/16th of pixel count. Loss = reconstruction (L1 + LPIPS perceptual) + KL (малый вес, чтобы `z` не принуждался быть слишком Gaussian, потому что нам не нужен exact sampling из `z`). Часто обучают с adversarial loss, чтобы decoded images были sharp.

2. **Stage 2 — diffusion on `z`.** Считать `z = E(x_real)` данными. Обучить U-Net (или DiT) denoise-ить `z_t`. На inference: sample `z_0` через diffusion, затем `x = D(z_0)`.

**Text conditioning.** Два дополнительных компонента. Frozen text encoder (CLIP-L для SD 1.x, CLIP-L+OpenCLIP-G для SD 2/XL, T5-XXL для SD3 и Flux). Cross-attention injection: каждый U-Net block берет `[Q = image features, K = V = text tokens]` и смешивает их. Tokens — единственный путь, которым text влияет на image.

**Loss function идентична Lesson 06.** Та же DDPM / flow matching MSE на noise. Меняется только data domain.

## Варианты архитектуры

| Model | Year | Backbone | Latent shape | Text encoder | Params |
|-------|------|----------|--------------|--------------|--------|
| SD 1.5 | 2022 | U-Net | 64×64×4 | CLIP-L (77 tokens) | 860M |
| SD 2.1 | 2022 | U-Net | 64×64×4 | OpenCLIP-H | 865M |
| SDXL | 2023 | U-Net + refiner | 128×128×4 | CLIP-L + OpenCLIP-G | 2.6B + 6.6B |
| SDXL-Turbo | 2023 | Distilled | 128×128×4 | same | 1-4 step sampling |
| SD3 | 2024 | MMDiT (multimodal DiT) | 128×128×16 | T5-XXL + CLIP-L + CLIP-G | 2B / 8B |
| Flux.1-dev | 2024 | MMDiT | 128×128×16 | T5-XXL + CLIP-L | 12B |
| Flux.1-schnell | 2024 | MMDiT distilled | 128×128×16 | T5-XXL + CLIP-L | 12B, 1-4 step |

Тренд: заменить U-Net на DiT (transformer по latent patches), масштабировать text encoder (T5 лучше CLIP по prompt adherence), увеличить latent channels (4 → 16 дает больше места для detail).

## Практика

`code/main.py` складывает toy 1-D "VAE" (identity encoder + decoder для демонстрации; настоящий VAE был бы conv net) поверх DDPM из Lesson 06 и добавляет class conditioning с classifier-free guidance. Он показывает, что та же diffusion loss работает и на raw 1-D values, и на encoded values — это ключевой insight.

### Шаг 1: encoder/decoder

```python
def encode(x):    return x * 0.5          # toy "compression" to smaller scale
def decode(z):    return z * 2.0
```

У настоящего VAE есть trained weights. Для педагогики этого linear map достаточно, чтобы показать: diffusion работает на `z` и не заботится об исходном data space.

### Шаг 2: diffusion in `z`-space

Тот же DDPM, что в Lesson 06. Данные, которые видит net, — `z = E(x)`. После sampling `z_0` декодируйте через `D(z_0)`.

### Шаг 3: classifier-free guidance

Во время training отбрасывайте class label в 10% случаев (заменяйте null token). На inference посчитайте `ε_cond` и `ε_uncond`, затем:

```python
eps_cfg = (1 + w) * eps_cond - w * eps_uncond
```

`w = 0` = no guidance (full diversity), `w = 3` = default, `w = 7+` = saturated / over-sharp.

### Шаг 4: text conditioning (concept, not code)

Замените class label на output frozen text encoder. Подайте text embedding в U-Net через cross-attention:

```python
h = h + CrossAttention(Q=h, K=text_embed, V=text_embed)
```

Это единственное существенное отличие class-conditional diffusion model от Stable Diffusion.

## Подводные камни

- **VAE-scale mismatch.** У SD 1.x VAE есть scaling constant (`scaling_factor ≈ 0.18215`), применяемая после encoding. Если забыть ее, U-Net будет обучаться на latents с радикально неверной variance. Каждый checkpoint поставляет ее.
- **Text encoder silently wrong.** SD3 нужен T5-XXL с >=128 tokens, и fallback на CLIP-only lossy. Всегда проверяйте `use_t5=True`, иначе prompt fidelity падает.
- **Mixing latent spaces.** SDXL, SD3, Flux используют разные VAEs. LoRA, обученная на SDXL latents, не заработает на SD3. Hugging Face diffusers 0.30+ отказывается загружать mismatched checkpoints.
- **CFG too high.** `w > 10` дает saturated, oily images и over-fits prompt ценой diversity. Sweet spot — `w = 3-7`.
- **Negative prompts leaking.** Empty negative prompt становится null token; заполненный negative prompt становится `ε_uncond`. Это не одно и то же; некоторые pipelines silently default to null.

## Применение

Production stacks в 2026 году:

| Target | Recommended backbone |
|--------|----------------------|
| Narrow domain, paired data, training a model from scratch | SDXL fine-tune (LoRA / full) — fastest to ship |
| Open-domain text-to-image, open weights | Flux.1-dev (12B, Apache / non-commercial) или SD3.5-Large |
| Fastest inference, open weights | Flux.1-schnell (1-4 step, Apache) или SDXL-Lightning |
| Best prompt adherence, hosted | GPT-Image / DALL-E 3 (still), Midjourney v7, Imagen 4 |
| Edit workflows | Flux.1-Kontext (Dec 2024) — natively accepts image + text |
| Research, baseline | SD 1.5 — ancient but well-studied |

## Запуск в продукт

Сохраните `outputs/skill-sd-prompter.md`. Навык принимает text prompt + target style и выдает: model + checkpoint, CFG scale, sampler, negative prompt, resolution, optional ControlNet/IP-Adapter combo и per-step QA checklist.

## Упражнения

1. **Легко.** Запустите `code/main.py` с guidance `w ∈ {0, 1, 3, 7, 15}`. Запишите mean sample by class. При каком `w` class means расходятся дальше real data means?
2. **Средне.** Замените toy linear encoder на tanh-MLP encoder/decoder pair с reconstruction loss. Переобучите diffusion на новых latents. Меняется ли sample quality?
3. **Сложно.** Настройте реальный Stable Diffusion inference с diffusers: загрузите `sdxl-base`, запустите 30 Euler steps с CFG=7, замерьте время. Теперь переключитесь на `sdxl-turbo` с 4 steps и CFG=0. Тот же subject, другое quality — опишите, что изменилось и почему.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| First stage | "The VAE" | Trained encoder/decoder pair; сжимает 512² до 64². |
| Second stage | "The U-Net" | Diffusion model over the latent space. |
| CFG | "Guidance scale" | `(1+w)·ε_cond - w·ε_uncond`; настраивает conditioning strength. |
| Null token | "Empty prompt embed" | Unconditional embed для `ε_uncond`. |
| Cross-attention | "How text gets in" | Каждый U-Net block attends to text tokens as K and V. |
| DiT | "Diffusion Transformer" | Замена U-Net на transformer over latent patches; лучше масштабируется. |
| MMDiT | "Multi-modal DiT" | Архитектура SD3: text и image streams с joint attention. |
| VAE scaling factor | "Magic number" | Делит latents примерно на ~5.4, чтобы diffusion работала в unit-variance space. |

## Production note: запуск Flux-12B на consumer GPU 8GB

Reference Flux integration — canonical рецепт "у меня consumer GPU, могу ли я это поставить?" Трюк — те же три ручки из production inference literature, примененные к diffusion DiT:

1. **Staggered loading.** У Flux три сети, которым не нужно одновременно жить в VRAM: T5-XXL text encoder (~10 GB in fp32), CLIP-L (small), 12B MMDiT и VAE. Сначала encode prompt, *delete* encoders, load DiT, denoise, *delete* DiT, load VAE, decode. Consumer 8GB GPUs вмещают только одну стадию за раз.
2. **4-bit quantization via bitsandbytes.** `BitsAndBytesConfig(load_in_4bit=True, bnb_4bit_compute_dtype=torch.bfloat16)` на T5 encoder и DiT. Срезает memory 8×, quality drop незаметен для text-to-image по benchmarks Aritra (linked in the notebook).
3. **CPU offload.** `pipe.enable_model_cpu_offload()` auto-swaps modules между CPU и GPU по мере продвижения каждого forward pass. Добавляет 10-20% latency, но вообще позволяет pipeline запуститься.

Memory accounting: `10 GB T5 / 8 = 1.25 GB` quantized, `12 B params × 0.5 bytes = ~6 GB` quantized DiT плюс activations. В терминах stas00 это extreme-end TP=1 inference — no model parallelism, maximum quantization. Для production вы бы запускали TP=2 или TP=4 на H100; для single dev laptop это рецепт.

## Дополнительное чтение

- [Rombach et al. (2022). High-Resolution Image Synthesis with Latent Diffusion Models](https://arxiv.org/abs/2112.10752) — Stable Diffusion.
- [Podell et al. (2023). SDXL: Improving Latent Diffusion Models for High-Resolution Image Synthesis](https://arxiv.org/abs/2307.01952) — SDXL.
- [Peebles & Xie (2023). Scalable Diffusion Models with Transformers (DiT)](https://arxiv.org/abs/2212.09748) — DiT.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — SD3, MMDiT.
- [Ho & Salimans (2022). Classifier-Free Diffusion Guidance](https://arxiv.org/abs/2207.12598) — CFG.
- [Labs (2024). Flux.1 — Black Forest Labs announcement](https://blackforestlabs.ai/announcing-black-forest-labs/) — семейство Flux.1.
- [Hugging Face Diffusers docs](https://huggingface.co/docs/diffusers/index) — reference implementation для каждого checkpoint выше.
