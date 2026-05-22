# Conditional GANs и Pix2Pix

> Первый большой прорыв 2014-2017 годов — контроль над тем, что создает GAN. Прикрепите метку, изображение или предложение. Pix2Pix сделал версию для изображений и до сих пор обгоняет любые универсальные text-to-image модели на узких image-to-image задачах.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 03 (GANs), Фаза 4 · 06 (U-Net), Фаза 3 · 07 (CNNs)
**Время:** ~75 минут

## Проблема

Unconditional GAN sample-ит произвольные лица. Полезно для demo, бесполезно в production. Вам нужно: *превратить sketch в photo*, *превратить map в aerial photo*, *превратить дневную сцену в ночную*, *раскрасить grayscale image*. Во всех этих случаях дано входное изображение `x`, и нужно выдать `y` с некоторым semantic correspondence. Для одного `x` есть много правдоподобных `y`. Mean-squared error сплющивает их в кашу. Adversarial loss — нет, потому что "выглядит реально" остается резким.

Conditional GAN (Mirza & Osindero, 2014) добавляет condition `c` как вход и в `G`, и в `D`. Pix2Pix (Isola et al., 2017) специализировал это: condition — целое input image, generator — U-Net, discriminator — *patch-based* classifier (PatchGAN), а loss — adversarial + L1. Этот рецепт даже в 2026 году превосходит from-scratch text-to-image models на узких image-to-image доменах, потому что обучается на *paired data* — у вас есть ровно нужный сигнал.

## Концепция

![Pix2Pix: U-Net generator, PatchGAN discriminator](../assets/pix2pix.svg)

**Conditional G.** `G(x, z) → y`. В Pix2Pix `z` — это dropout внутри G (без input noise — Isola обнаружил, что явный noise игнорируется).

**Conditional D.** `D(x, y) → [0, 1]`. Вход — *пара* (condition, output). Это ключевое отличие: D должен судить, согласован ли `y` с `x`, а не просто выглядит ли `y` реальным.

**U-Net generator.** Encoder-decoder со skip connections через bottleneck. Критично для задач, где input и output делят low-level structure (edges, silhouette). Без skips high-frequency detail исчезает.

**PatchGAN discriminator.** Вместо одного real/fake score D выдает сетку `N×N`, где каждая ячейка оценивает receptive field примерно 70×70 пикселей. Затем усреднение. Это Markov random field assumption: realism локален. Обучается быстрее, параметров меньше, output резче.

**Loss.**

```
loss_G = -log D(x, G(x)) + λ · ||y - G(x)||_1
loss_D = -log D(x, y) - log (1 - D(x, G(x)))
```

L1 term стабилизирует обучение и тянет G к известной цели. L1 дает более резкие edges, чем L2 (медианы, не средние). `λ = 100` был default в Pix2Pix.

## CycleGAN — когда нет пар

Pix2Pix требует paired `(x, y)` data. CycleGAN (Zhu et al., 2017) снимает это требование ценой дополнительной loss: *cycle consistency* loss. Два генератора `G: X → Y` и `F: Y → X`. Обучайте их так, чтобы `F(G(x)) ≈ x` и `G(F(y)) ≈ y`. Это позволяет переводить horses to zebras, summer to winter без paired examples.

В 2026 году unpaired image-to-image в основном делают через diffusion (ControlNet, IP-Adapter), а не CycleGAN, но идея cycle-consistency выживает почти в каждой статье по unpaired domain adaptation.

## Практика

`code/main.py` реализует крошечный conditional GAN на 1-D данных. Condition `c` — class label (0 или 1). Задача: произвести sample из conditional distribution для заданного класса.

### Шаг 1: append condition to both G and D inputs

```python
def G(z, c, params):
    return mlp(concat([z, one_hot(c)]), params)

def D(x, c, params):
    return mlp(concat([x, one_hot(c)]), params)
```

One-hot encoding — самый простой способ. Более крупные модели используют learned embeddings, FiLM modulation или cross-attention.

### Шаг 2: train conditional

```python
for step in range(steps):
    x, c = sample_real_conditional()
    noise = sample_noise()
    update_D(x_real=x, x_fake=G(noise, c), c=c)
    update_G(noise, c)
```

Generator должен совпасть с real distribution *для заданного condition*, а не с marginal.

### Шаг 3: verify per-class output

```python
for c in [0, 1]:
    samples = [G(noise, c) for noise in batch]
    mean_c = mean(samples)
    assert_near(mean_c, real_mean_for_class_c)
```

## Подводные камни

- **Condition ignored.** G учится marginalize, D не штрафует, потому что condition signal слабый. Исправление: condition D агрессивнее (early layer, не только late), используйте projection discriminator (Miyato & Koyama 2018).
- **L1 weight too low.** G уходит в произвольные real-looking outputs, а не faithful. Начинайте с λ≈100 для Pix2Pix-style задач.
- **L1 weight too high.** G производит blurry outputs, потому что L1 все еще L_p norm. Anneal down после стабилизации обучения.
- **Ground-truth leakage in D.** Конкатенируйте `(x, y)` как вход D, а не только `y`. Без этого D не может проверять consistency.
- **Mode collapse per class.** Каждый класс может collapse независимо. Запускайте class-conditional diversity checks.

## Применение

Состояние image-to-image задач в 2026 году:

| Задача | Лучший подход |
|------|---------------|
| Sketch → photo, тот же домен, paired data | Pix2Pix / Pix2PixHD (все еще fast, все еще sharp) |
| Sketch → photo, unpaired | ControlNet с Scribble conditioning model |
| Semantic seg → photo | SPADE / GauGAN2 или SD + ControlNet-Seg |
| Style transfer | Diffusion с IP-Adapter или LoRA; GAN methods — legacy |
| Depth → photo | ControlNet-Depth поверх Stable Diffusion |
| Super-resolution | Real-ESRGAN (GAN), ESRGAN-Plus или SD-Upscale (diffusion) |
| Colorization | ColTran, diffusion-based colorizers или Pix2Pix-color |
| Daytime → nighttime, seasons, weather | CycleGAN или ControlNet-based |

Pix2Pix остается правильным инструментом, когда (a) у вас есть тысячи paired examples, (b) задача узкая и повторяемая, и (c) нужен fast inference. На generic open-domain задачах выигрывает diffusion.

## Запуск в продукт

Сохраните `outputs/skill-img2img-chooser.md`. Навык принимает task description, data availability (paired vs unpaired, N samples) и latency/quality budget, затем выдает: approach (Pix2Pix, CycleGAN, ControlNet variant, SDXL + IP-Adapter), training data requirements, inference cost и eval protocol (LPIPS, FID, task-specific).

## Упражнения

1. **Легко.** Измените `code/main.py`, добавив третий class. Подтвердите, что G все еще отображает noise каждого class в правильную mode.
2. **Средне.** Замените L1 на perceptual-style loss в 1-D setting (например, маленький frozen D как feature extractor). Меняет ли это sharpness conditional distribution?
3. **Сложно.** Набросайте CycleGAN в 1-D setting: два distributions, два generators, cycle loss. Покажите, что он учится map между ними без paired data.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Conditional GAN | "GAN with labels" | G(z, c), D(x, c). Обе сети видят condition. |
| Pix2Pix | "Image-to-image GAN" | Paired cGAN с U-Net G и PatchGAN D + L1 loss. |
| U-Net | "Encoder-decoder with skips" | Симметричная conv network; skips сохраняют high-freq. |
| PatchGAN | "Local-realism classifier" | D выдает per-patch score вместо global score. |
| CycleGAN | "Unpaired image translation" | Два G + cycle-consistency loss; без paired data. |
| SPADE | "GauGAN" | Нормализует intermediate activations с semantic map; segmentation-to-image. |
| FiLM | "Feature-wise linear modulation" | Per-feature affine transform из condition; дешевый conditioning. |

## Production note: Pix2Pix как latency-bound baseline

Когда у вас есть paired data и узкая задача (sketch → render, semantic map → photo, day → night), one-shot inference Pix2Pix обгоняет diffusion по latency на порядок. Production-сравнение обычно такое:

| Path | Steps | Typical latency at 512² on a single L4 |
|------|-------|----------------------------------------|
| Pix2Pix (U-Net forward) | 1 | ~30 ms |
| SD-Inpaint or SD-Img2Img | 20 | ~1.2 s |
| SDXL-Turbo Img2Img | 1-4 | ~0.15-0.35 s |
| ControlNet + SDXL base | 20-30 | ~3-5 s |

Pix2Pix выигрывает по throughput в static batches (каждый request — те же FLOPs). Diffusion выигрывает по quality и generalization. Современный ход — часто поставить Pix2Pix-style distilled model для узкой задачи и diffusion fallback для tail inputs.

## Дополнительное чтение

- [Mirza & Osindero (2014). Conditional Generative Adversarial Nets](https://arxiv.org/abs/1411.1784) — статья о cGAN.
- [Isola et al. (2017). Image-to-Image Translation with Conditional Adversarial Networks](https://arxiv.org/abs/1611.07004) — Pix2Pix.
- [Zhu et al. (2017). Unpaired Image-to-Image Translation using Cycle-Consistent Adversarial Networks](https://arxiv.org/abs/1703.10593) — CycleGAN.
- [Wang et al. (2018). High-Resolution Image Synthesis with Conditional GANs](https://arxiv.org/abs/1711.11585) — Pix2PixHD.
- [Park et al. (2019). Semantic Image Synthesis with Spatially-Adaptive Normalization](https://arxiv.org/abs/1903.07291) — SPADE / GauGAN.
- [Miyato & Koyama (2018). cGANs with Projection Discriminator](https://arxiv.org/abs/1802.05637) — projection D.
