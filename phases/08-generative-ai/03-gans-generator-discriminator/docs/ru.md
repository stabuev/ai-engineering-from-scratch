# GANs — генератор против дискриминатора

> Трюк Goodfellow в 2014 году состоял в том, чтобы полностью пропустить плотность. Две сети. Одна делает подделки. Другая их ловит. Они борются, пока подделки не становятся неотличимы от реальных данных. Это не должно работать. Часто и не работает. Но когда работает, samples в узких доменах все еще остаются самыми четкими в литературе.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 3 · 02 (Backprop), Фаза 3 · 08 (Optimizers), Фаза 8 · 02 (VAE)
**Время:** ~75 минут

## Проблема

VAE производят размытые samples, потому что их MSE decoder loss Bayes-optimal для *среднего* изображения, а среднее многих правдоподобных цифр — размытая цифра. Вам нужна loss, которая вознаграждает *правдоподобие*, а не попиксельную близость к одной цели. Closed-form для правдоподобия нет. Его нужно выучить.

Идея Goodfellow: обучить classifier `D(x)` отличать реальные изображения от поддельных. Обучить generator `G(z)` обманывать `D`. Сигнал loss для `G` — это то, что `D` в данный момент считает признаками реальности. Этот сигнал обновляется по мере улучшения `G`, преследуя движущуюся цель. Если обе сети сходятся, `G` выучила распределение данных, ни разу не выписав `log p(x)`.

Это adversarial training. Математика — minimax game:

```
min_G max_D  E_real[log D(x)] + E_fake[log(1 - D(G(z)))]
```

В 2026 году GAN уже не SOTA generator (diffusion и flow matching забрали корону). Но StyleGAN 2/3 остаются самыми резкими face models из когда-либо поставленных, GAN discriminators используются как *perceptual losses* в diffusion training, а adversarial training питает быстрые 1-step distillations (SDXL-Turbo, SD3-Turbo, LCM), которые позволяют поставлять real-time diffusion.

## Концепция

![GAN training: generator and discriminator in minimax](../assets/gan.svg)

**Generator `G(z)`.** Отображает noise vector `z ~ N(0, I)` в sample `x̂`. Сеть формы decoder (dense или transposed conv).

**Discriminator `D(x)`.** Отображает sample в scalar probability (или score). Real → 1, fake → 0.

**Loss.** Два чередующихся обновления:

- **Train `D`:** `loss_D = -[ log D(x) + log(1 - D(G(z))) ]`. Binary cross-entropy для real=1, fake=0.
- **Train `G`:** `loss_G = -log D(G(z))`. Это *non-saturating* форма, которую использовал Goodfellow (исходная `log(1 - D(G(z)))` saturates и убивает gradients, когда `D` уверен).

**Training loop.** Один шаг `D`, один шаг `G`. Повторять.

**Почему это работает.** Если `G` идеально совпадает с `p_data`, то `D` не может быть лучше случайного угадывания и выдает 0.5 везде; `G` больше не получает gradient. Равновесие.

**Почему это ломается.** Mode collapse (`G` находит одну моду, которую `D` не может классифицировать, и штампует ее бесконечно), vanishing gradient (`D` учится слишком быстро, и `log D` saturates), training instability (learning rates, batch sizes, что угодно).

## Варианты, заставившие GAN работать

| Год | Инновация | Что исправила |
|------|------------|-----|
| 2015 | DCGAN | Conv/deconv, batch norm, LeakyReLU — первая стабильная архитектура. |
| 2017 | WGAN, WGAN-GP | Замена BCE на Wasserstein distance + gradient penalty. Исправляет vanishing gradient. |
| 2017 | Spectral normalization | Lipschitz-bound для discriminator. Все еще используется в discriminators 2026 года. |
| 2018 | Progressive GAN | Сначала low-res, затем добавлять слои. Первые мегапиксельные результаты. |
| 2019 | StyleGAN / StyleGAN2 | Mapping network + adaptive instance norm. State of the art для fixed-domain photorealism. |
| 2021 | StyleGAN3 | Alias-free, translation-equivariant — все еще золотой стандарт лиц в 2026 году. |
| 2022 | StyleGAN-XL | Conditional, class-aware, больший масштаб. |
| 2024 | R3GAN | Rebrands с более сильной regularization; работает на 1024² без tricks. |

## Практика

`code/main.py` обучает крошечный GAN на 1-D данных: mixture of two Gaussians. Generator и discriminator — MLP с одним hidden layer. Мы вручную реализуем forward, backward и minimax loop. Цель — увидеть два ключевых режима отказа (mode collapse + vanishing gradient) прямо во время их появления.

### Шаг 1: non-saturating loss

Vanilla Goodfellow loss `log(1 - D(G(z)))` стремится к 0, когда D с высокой уверенностью классифицирует fake от G как fake. В этот момент gradient для G почти нулевой — G не может улучшаться. Non-saturating форма `-log D(G(z))` имеет противоположную асимптоту: она взрывается, когда D уверен, давая G сильный сигнал.

```python
def g_loss(d_fake):
    # maximize log D(G(z))  <=>  minimize -log D(G(z))
    return -sum(math.log(max(p, 1e-8)) for p in d_fake) / len(d_fake)
```

### Шаг 2: one discriminator step per generator step

```python
for step in range(steps):
    # train D
    real_batch = sample_real(batch_size)
    fake_batch = [G(z) for z in sample_noise(batch_size)]
    update_D(real_batch, fake_batch)

    # train G
    fake_batch = [G(z) for z in sample_noise(batch_size)]  # fresh fakes
    update_G(fake_batch)
```

Свежие fakes для G, иначе gradients устаревают.

### Шаг 3: watch for mode collapse

```python
if step % 200 == 0:
    samples = [G(z) for z in sample_noise(500)]
    mode_a = sum(1 for s in samples if s < 0)
    mode_b = 500 - mode_a
    if min(mode_a, mode_b) < 50:
        print("  [!] mode collapse: one mode is starved")
```

Канонический симптом: одна из двух реальных мод перестает генерироваться. Discriminator перестает это исправлять, потому что она никогда не видится как fake.

## Подводные камни

- **Discriminator too strong.** Уменьшите learning rate D в 2-5x или добавьте instance/layer noise. Если D достигает >95% accuracy, G мертв.
- **Generator memorizes a mode.** Добавьте noise на входы D, используйте minibatch-discriminator layer или перейдите на WGAN-GP.
- **Batch norm leaking statistics.** Real batch + fake batch, проходящие через один BN layer, смешивают статистики. Вместо этого используйте instance norm или spectral norm.
- **Inception-score gaming.** FID и IS шумные при малом числе samples. Используйте ≥10k samples на eval.
- **One-shot sampling is a lie for conditional tasks.** Все равно нужны CFG scales, truncation tricks и re-sampling, чтобы получить пригодные outputs.

## Применение

GAN stack в 2026 году:

| Ситуация | Выбор |
|-----------|------|
| Фотореалистичные человеческие лица, fixed pose | StyleGAN3 (самый резкий, самый маленький) |
| Anime / stylized faces | StyleGAN-XL или Stable Diffusion LoRA |
| Image-to-image translation | Pix2Pix / CycleGAN (Фаза 8 · 04) или ControlNet (Фаза 8 · 08) |
| Быстрый 1-step text-to-image | Adversarial distillation of diffusion (SDXL-Turbo, SD3-Turbo) |
| Perceptual loss внутри diffusion trainer | Малый GAN discriminator на image crops |
| Что угодно multi-modal, open-ended | Не надо — используйте diffusion или flow matching |

GAN резкие, но узкие. Как только домен раскрывается — photos, arbitrary text prompts, video — переходите на diffusion. Adversarial trick продолжает жить как компонент (perceptual losses, distillation), а не standalone generator.

## Запуск в продукт

Сохраните `outputs/skill-gan-debugger.md`. Навык принимает failing GAN run (loss curves, sample grid, dataset size) и выдает ранжированный список вероятных причин, one-line fixes и rerun protocol.

## Упражнения

1. **Легко.** Запустите `code/main.py` со штатными настройками. Затем установите `D_LR = 5 * G_LR` и перезапустите. Насколько быстро loss G схлопывается в константу?
2. **Средне.** Замените Goodfellow BCE loss на WGAN loss: `loss_D = E[D(fake)] - E[D(real)]`, `loss_G = -E[D(fake)]`, и clip weights D в `[-0.01, 0.01]`. Стало ли обучение стабильнее? Сравните wall-clock convergence.
3. **Сложно.** Расширьте 1-D пример до 2-D данных (mixture of 8 Gaussians on a ring). Отслеживайте, сколько из 8 modes generator захватывает на шагах 1k, 5k, 10k. Реализуйте minibatch discrimination и измерьте снова.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Generator | "G" | Сеть noise-to-sample, `G: z → x̂`. |
| Discriminator | "D" | Classifier `D: x → [0, 1]`, real vs fake. |
| Minimax | "The game" | `min_G max_D` совместной objective. |
| Non-saturating loss | "The fix" | Использовать `-log D(G(z))` для G вместо `log(1 - D(G(z)))`. |
| Mode collapse | "G memorized one thing" | Generator производит мало разных outputs несмотря на разнообразные data. |
| WGAN | "Wasserstein" | Замена BCE на Earth-Mover distance + gradient penalty; более гладкий gradient. |
| Spectral norm | "Lipschitz trick" | Ограничивает нормы weights D, чтобы ограничить slope; стабилизирует обучение. |
| StyleGAN | "The one that works" | Mapping network + AdaIN; best-in-class для лиц, все еще в 2026 году. |

## Production note: one-shot inference — долгосрочное преимущество GAN

GAN больше не выигрывают по sample quality для open-domain generation, но все еще выигрывают по inference cost. В словаре production-inference литературы у GAN есть:

- **No prefill, no decode stages.** Один forward pass `G(z)`. TTFT ≈ total latency.
- **No KV-cache pressure.** Единственное состояние — weights. Batch size ограничен activation memory, а не cache.
- **Trivial continuous batching.** Поскольку каждый request требует одинаковых fixed FLOPs, static batch при целевой загрузке server обычно оптимален. In-flight scheduler не нужен.

Поэтому GAN distillation (SDXL-Turbo, SD3-Turbo, ADD, LCM) — доминирующая техника для fast text-to-image в 2026 году: она схлопывает diffusion pipeline из 20-50 шагов в 1-4 GAN-style forward passes, сохраняя distribution diffusion base. Adversarial loss выживает как training-time knob для превращения медленных generators в быстрые.

## Дополнительное чтение

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — оригинальная статья о GAN.
- [Radford et al. (2015). Unsupervised Representation Learning with DCGAN](https://arxiv.org/abs/1511.06434) — первая стабильная архитектура.
- [Arjovsky, Chintala, Bottou (2017). Wasserstein GAN](https://arxiv.org/abs/1701.07875) — WGAN.
- [Miyato et al. (2018). Spectral Normalization for GANs](https://arxiv.org/abs/1802.05957) — SN.
- [Karras et al. (2020). Analyzing and Improving the Image Quality of StyleGAN](https://arxiv.org/abs/1912.04958) — StyleGAN2.
- [Karras et al. (2021). Alias-Free Generative Adversarial Networks](https://arxiv.org/abs/2106.12423) — StyleGAN3.
- [Sauer et al. (2023). Adversarial Diffusion Distillation](https://arxiv.org/abs/2311.17042) — SDXL-Turbo.
