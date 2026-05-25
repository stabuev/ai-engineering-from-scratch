# StyleGAN

> Большинство генераторов подмешивают `z` во все слои одновременно. StyleGAN разделил это: сначала отобразить `z` в промежуточное `w`, затем *внедрять* `w` на каждом уровне разрешения через AdaIN. Одно это изменение распутало latent space и на семь лет сделало фотореалистичные лица практически решенной задачей.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 03 (GANs), Фаза 4 · 08 (Normalization), Фаза 3 · 07 (CNNs)
**Время:** ~45 минут

## Проблема

DCGAN отображает `z` в изображение через стек transposed convolutions. Проблема: `z` управляет всем сразу — pose, lighting, identity, background — все перепутано. Двиньтесь вдоль одной оси `z`, и изменятся все четыре. Нельзя попросить модель "тот же человек, другая pose", потому что representation так не факторизована.

Karras et al. (2019, NVIDIA) предложили: перестать подавать `z` напрямую в conv layers. Подайте learned constant `4×4×512` tensor как вход сети. Обучите 8-layer MLP, который отображает `z ∈ Z → w ∈ W`. Внедряйте `w` на каждом разрешении через *adaptive instance normalization* (AdaIN): нормализуйте каждую conv feature map, затем scale и shift через affine projections of `w`. Добавьте per-layer noise для stochastic detail (skin pores, hair strands).

Результат: `W` имеет примерно ортогональные оси для "high-level style" (pose, identity) и "fine style" (lighting, color). Можно смешивать стили двух изображений: использовать `w` изображения A на low-resolution уровнях и `w` изображения B на high-resolution. Это открыло editing, cross-domain stylization и всю линию исследований "StyleGAN-inversion".

## Концепция

![StyleGAN: mapping network + AdaIN + per-layer noise](../assets/stylegan.svg)

**Mapping network.** `f: Z → W`, 8-layer MLP. `Z = N(0, I)^512`. `W` не обязано быть гауссовым — оно учит форму, адаптированную к данным.

**Synthesis network.** Начинается с learned constant `4×4×512`. Каждый resolution block: `upsample → conv → AdaIN(w_i) → noise → conv → AdaIN(w_i) → noise`. Разрешения удваиваются: 4, 8, 16, 32, 64, 128, 256, 512, 1024.

**AdaIN.**

```
AdaIN(x, y) = y_scale · (x - mean(x)) / std(x) + y_bias
```

где `y_scale` и `y_bias` приходят из affine projections of `w`. Нормализуйте по feature map, затем restyle. "Style" здесь — statistics первого и второго порядка feature map.

**Per-layer noise.** Single-channel Gaussian noise добавляется к каждой feature map и масштабируется learned per-channel factor. Управляет stochastic detail, не влияя на global structure.

**Truncation trick.** На inference sample-им `z`, считаем `w = mapping(z)`, затем `w' = ŵ + ψ·(w - ŵ)`, где `ŵ` — среднее `w` по многим samples. `ψ < 1` меняет diversity на quality. Почти каждое StyleGAN demo использует `ψ ≈ 0.7`.

## StyleGAN 1 → 2 → 3

| Версия | Год | Инновация |
|---------|------|------------|
| StyleGAN | 2019 | Mapping network + AdaIN + noise + progressive growing. |
| StyleGAN2 | 2020 | Weight demodulation заменяет AdaIN (исправляет droplet artifacts); skip/residual architecture; path-length regularization. |
| StyleGAN3 | 2021 | Alias-free convolution + equivariant kernels; устраняет прилипание texture к pixel grid. |
| StyleGAN-XL | 2022 | Class-conditional, 1024², ImageNet. |
| R3GAN | 2024 | Rebrands с более сильной reg; закрывает gap к diffusion на FFHQ-1024 с 20x fewer params. |

В 2026 году StyleGAN3 остается default для (a) narrow-domain photorealism при high FPS, (b) few-shot domain adaptation (обучить на новом dataset из 100 images, freeze mapping), (c) inversion-based editing (найти `w`, восстанавливающий real photo, затем редактировать этот `w`). Для open-domain text-to-image это не инструмент — используйте diffusion.

## Практика

`code/main.py` реализует игрушечный "style-GAN lite" в 1-D: mapping MLP, synthesis function, которая берет learned constant vector и модулирует его scale/bias из `w`, и per-layer noise. Он показывает, что внедрение `w` через affine-modulation не хуже или лучше, чем concatenating `z` во вход generator.

### Шаг 1: mapping network

```python
def mapping(z, M):
    h = z
    for i in range(num_layers):
        h = leaky_relu(add(matmul(M[f"W{i}"], h), M[f"b{i}"]))
    return h
```

### Шаг 2: adaptive instance normalization

```python
def adain(x, w_scale, w_bias):
    mu = mean(x)
    sd = std(x)
    x_norm = [(xi - mu) / (sd + 1e-8) for xi in x]
    return [w_scale * xi + w_bias for xi in x_norm]
```

Per-feature-map scale и bias приходят из `w` через linear projection.

### Шаг 3: per-layer noise

```python
def add_noise(x, sigma, rng):
    return [xi + sigma * rng.gauss(0, 1) for xi in x]
```

Sigma per-channel обучаема.

## Подводные камни

- **Droplet artifacts.** StyleGAN 1 создавал blob-like droplet в feature maps, потому что AdaIN занулял mean. Weight demodulation в StyleGAN 2 исправляет это, масштабируя convolution weights вместо activations.
- **Texture sticking.** В StyleGAN 1 и 2 textures следовали pixel coordinates, а не object coordinates (заметно при interpolation). Alias-free convolutions в StyleGAN 3 исправляют это через windowed sinc filters.
- **Mode coverage.** Truncation `ψ < 0.7` выглядит чисто, но sample-ит из узкого cone; используйте `ψ = 1.0`, если нужна diversity.
- **Inversion is lossy.** Inverting real photo в `W` обычно делается через optimization или encoder (e4e, ReStyle, HyperStyle). Results drift после многих iterations.

## Применение

| Use case | Approach |
|----------|----------|
| Photoreal human faces (anime, product, narrow) | StyleGAN3 FFHQ / custom fine-tune |
| Face editing from a photo | e4e inversion + StyleSpace / InterFaceGAN directions |
| Face swap / reenactment | StyleGAN + encoder + blending |
| Avatar pipelines | StyleGAN3 w/ ADA for low-data fine-tune |
| Domain adaptation from a few images | Freeze mapping network, fine-tune synthesis |
| Multi-modal or text-conditioned generation | Не надо — используйте diffusion |

Для product-grade demos, где ответ — "photo of a person's face", StyleGAN обгоняет diffusion по inference cost (single forward pass, <10ms на 4090) и sharpness при той же планке качества.

## Запуск в продукт

Сохраните `outputs/skill-stylegan-inversion.md`. Навык принимает real photo и выдает: inversion method (e4e / ReStyle / HyperStyle), expected latent loss, editing budget (насколько далеко в `W` можно двигаться до artifacts) и список known-good editing directions (age, expression, pose).

## Упражнения

1. **Легко.** Запустите `code/main.py` с `adain_on=True` и `adain_on=False`. Сравните spread of outputs для fixed latent и perturbed latent.
2. **Средне.** Реализуйте mixing regularization: для training batch посчитайте `w_a`, `w_b` и примените `w_a` для первой половины synthesis, а `w_b` для второй. Учится ли decoder disentangled styles?
3. **Сложно.** Возьмите pretrained StyleGAN3 FFHQ model (ffhq-1024.pkl). Найдите `w` direction, управляющую "smile", обучив SVM на labelled samples; сообщите, насколько далеко можно push до identity drift.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Mapping network | "The MLP" | `f: Z → W`, 8 layers, отделяет latent geometry от data statistics. |
| W space | "The style space" | Output mapping network; примерно disentangled. |
| AdaIN | "Adaptive instance norm" | Normalize feature map, затем scale + shift через `w`-projection. |
| Truncation trick | "Psi" | `w = mean + ψ·(w - mean)`, ψ<1 меняет diversity на quality. |
| Path-length regularization | "PL reg" | Штрафует большие изменения image на единицу изменения `w`; делает `W` smoother. |
| Weight demodulation | "The StyleGAN2 fix" | Нормализует conv weights вместо activations; убирает droplet artifacts. |
| Alias-free | "StyleGAN3's trick" | Windowed sinc filters; устраняет прилипание texture к pixel grid. |
| Inversion | "Find w for a real image" | Optimize или encode `x → w`, чтобы `G(w) ≈ x`. |

## Production note: почему StyleGAN все еще поставляют в 2026 году

StyleGAN3 на 4090 генерирует лицо FFHQ 1024² менее чем за 10 ms — `num_steps = 1`, без VAE decode, без cross-attention pass. В production terms это нижняя latency-граница для любого image generator. Pipeline 50-step SDXL + VAE-decode при том же разрешении — ~3 seconds. Это **300× gap**, и для narrow-domain products (avatar services, ID document pipelines, stock face generation) он выигрывает по TCO.

Два operational consequences:

- **No scheduler, no batcher.** Static batch при target occupancy оптимален. Continuous batching (необходимый для LLMs и diffusion) не дает пользы, потому что каждый request требует одинаковых FLOPs.
- **Truncation `ψ` is the safety knob.** `ψ < 0.7` sample-ит из узкого cone диапазона mapping network. Это единственный рычаг serving layer над sample variance. Понижайте `ψ` при peak load, повышайте для premium users.

## Дополнительное чтение

- [Karras et al. (2019). A Style-Based Generator Architecture for GANs](https://arxiv.org/abs/1812.04948) — StyleGAN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Tov et al. (2021). Designing an Encoder for StyleGAN Image Manipulation](https://arxiv.org/abs/2102.02766) — e4e inversion.
- [Sauer et al. (2022). StyleGAN-XL: Scaling StyleGAN to Large Diverse Datasets](https://arxiv.org/abs/2202.00273) — StyleGAN-XL.
- [Huang et al. (2024). R3GAN: The GAN is dead; long live the GAN!](https://arxiv.org/abs/2501.05441) — modern minimal GAN recipe.
