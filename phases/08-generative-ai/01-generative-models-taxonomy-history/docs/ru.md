# Генеративные модели — таксономия и история

> Каждая модель изображений, текста, видео и 3D попадает в одну из пяти корзин. Выберите не ту корзину — и неделями будете бороться с математикой. Выберите правильную — и прогресс последних двенадцати лет аккуратно уложится в голове.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 2 (ML Fundamentals), Фаза 3 (Deep Learning Core), Фаза 7 · 14 (Transformers)
**Время:** ~45 минут

## Проблема

Генеративная модель делает одну вещь: получив обучающие примеры из неизвестного распределения `p_data(x)`, она выдает новые примеры, похожие на взятые из того же распределения. Лица, предложения, MIDI-файлы, структуры белков — если прищуриться, это одна и та же задача.

Проблема в том, что `p_data` живет в пространстве с миллионами измерений (RGB-изображение 512x512 — это ~786k измерений), примеры лежат на тонком многообразии внутри этого пространства, а данных у вас, возможно, всего 10M. Перебирать плотность напрямую безнадежно. Любая генеративная модель — это компромисс: она меняет одну сложную задачу на чуть менее сложную.

За последние двенадцать лет выжили пять семейств. Понимание компромисса каждого семейства объясняет, почему оно выигрывает на одних задачах и ломается на других.

## Концепция

![Five families of generative models — taxonomy by what they model](../assets/taxonomy.svg)

**1. Явная плотность, вычислимая.** Записать `log p(x)` как сумму, которую реально можно посчитать. Авторегрессионные модели (PixelCNN, WaveNet, GPT) факторизуют `p(x) = ∏ p(x_i | x_<i)`. Нормализующие потоки (RealNVP, Glow) строят `p(x)` как обратимое преобразование простой базовой плотности. Плюс: точное правдоподобие, чистая функция потерь. Минус: авторегрессионный inference последовательный (медленный для длинных последовательностей), потокам нужны обратимые архитектуры (это жестко ограничивает дизайн).

**2. Явная плотность, приближенная.** Ограничить `log p(x)` снизу (ELBO) и оптимизировать эту границу. VAE (Kingma 2013) используют encoder-decoder с вариационным posterior. Диффузионные модели (DDPM, Ho 2020) обучают denoiser, который неявно оптимизирует взвешенный ELBO. В 2026 году diffusion — доминирующая основа для изображений, видео и 3D.

**3. Неявная плотность.** Полностью пропустить плотность; выучить генератор `G(z)`, который производит примеры, и discriminator `D(x)`, который отличает реальные от поддельных. Это GANs (Goodfellow 2014). Быстрые на inference (один forward pass), но печально известны нестабильностью обучения. StyleGAN 1/2/3 даже в 2026 году остаются state of the art для фотореализма в фиксированных доменах (лица, спальни).

**4. Score-based / непрерывное время.** Напрямую выучить градиент log-density `∇_x log p(x)` (score). Song & Ermon (2019) показали, что score matching обобщает diffusion до SDE. Flow matching (Lipman 2023) — горячая тема 2024-2026: обучение без симуляции, более прямые траектории, sampling в 4-10x быстрее DDPM. Stable Diffusion 3, Flux, AudioCraft 2 используют flow matching.

**5. Token-based autoregressive по дискретным кодам.** Сжать высокоразмерные данные с помощью VQ-VAE или residual quantizer в короткую последовательность дискретных токенов, затем использовать Transformer для моделирования последовательности токенов. Parti, MuseNet, AudioLM, VALL-E, patch tokenizer у Sora — все используют этот подход. Это корзина 1 плюс обученный tokenizer.

## Краткая история

| Год | Модель | Почему это было важно |
|------|-------|-----------------|
| 2013 | VAE (Kingma) | Первая глубокая генеративная модель с пригодной функцией потерь. |
| 2014 | GAN (Goodfellow) | Неявная плотность, без likelihood — поразительно четкие примеры. |
| 2015 | DRAW, PixelCNN | Последовательная генерация изображений. |
| 2017 | Glow, RealNVP | Обратимые потоки; точное likelihood с глубиной. |
| 2017 | Progressive GAN | Первые мегапиксельные лица. |
| 2019 | StyleGAN / StyleGAN2 | Фотореалистичные лица все еще трудно превзойти в этом одном домене. |
| 2020 | DDPM (Ho) | Diffusion становится практичной. |
| 2021 | CLIP, DALL-E 1, VQGAN | Text-to-image выходит в мейнстрим. |
| 2022 | Imagen, Stable Diffusion 1, DALL-E 2 | Latent diffusion + text conditioning = массовая технология. |
| 2022 | ControlNet, LoRA | Точный контроль над pretrained diffusion. |
| 2023 | SDXL, Midjourney v5, Flow matching | Масштаб + лучшая динамика обучения. |
| 2024 | Sora, Stable Diffusion 3, Flux.1 | Video diffusion; flow matching выигрывает. |
| 2025 | Veo 2, Kling 1.5, Runway Gen-3, Nano Banana | Видео production-grade. |
| 2026 | Consistency + Rectified Flow | Одношаговый sampling из diffusion backbones. |

## Пять вопросов для triage

Когда выходит новая статья о генеративной модели, ответьте на эти пять вопросов до чтения раздела с методом.

1. **Что моделируется?** Пиксели, latents, дискретные токены, 3D Gaussians, meshes, waveforms?
2. **Плотность явная или неявная?** Выписывают ли авторы `log p(x)`?
3. **Sampling: one-shot или iterative?** Iterative означает более медленный inference; one-shot обычно означает adversarial или distilled.
4. **Conditioning: unconditional, class, text, image, pose?** Это определяет loss и архитектурный каркас.
5. **Evaluation: FID, CLIP score, IS, human preference, task accuracy?** У каждого есть известные режимы отказа (см. Lesson 14).

Вы будете заново отвечать на эти пять вопросов в каждом уроке этой фазы. К концу это станет рефлексом.

## Практика

Код этого урока — легкая визуализация: подогнать 1-D mixture-of-Gaussians по сэмплам тремя игрушечными подходами (kernel density, дискретная histogram и ближайший к примеру "GAN-ish" generator), чтобы увидеть разницу между explicit и implicit density на задаче, которую можно напечатать на одном экране.

Запустите `code/main.py`. Он берет 2000 samples из двухмодовой Gaussian mixture, затем печатает:

```
explicit density (histogram): p(x in [-0.5, 0.5]) ≈ 0.38
approximate density (KDE):     p(x in [-0.5, 0.5]) ≈ 0.41
implicit (nearest-sample gen): 20 new samples printed, no p(x)
```

Заметьте: первые два позволяют спросить "насколько вероятна эта точка?" Третий — нет. Это различие *explicit vs implicit*, которое будет важно во всех следующих уроках.

## Применение

Какое семейство для какой задачи в 2026 году?

| Задача | Лучшее семейство | Почему |
|------|-------------|-----|
| Фотореалистичные лица, узкий домен | StyleGAN 2/3 | Все еще самые четкие, самый быстрый inference. |
| General text-to-image | Latent diffusion + flow matching | SD3, Flux.1, DALL-E 3. |
| Быстрый text-to-image | Rectified flow + distillation | SDXL-Turbo, SD3-Turbo, LCM. |
| Text-to-video | Diffusion Transformer + flow matching | Sora, Veo 2, Kling. |
| Речь + музыка | Token-based AR (AudioLM, VALL-E, MusicGen) или flow matching (AudioCraft 2) | Дискретные токены дешево масштабируются. |
| 3D-сцены | Gaussian Splatting fit, diffusion prior | 3D-GS для реконструкции, diffusion для novel-view. |
| Оценка плотности (без sampling) | Flows | Единственное семейство с точным `log p(x)`. |
| Симуляция / физика | Flow matching, score SDE | Прямолинейные траектории, гладкие векторные поля. |

## Запуск в продукт

Сохраните как `outputs/skill-model-chooser.md`.

Навык принимает описание задачи и выдает: (1) какое семейство использовать, (2) ранжированный список из трех open и трех hosted вариантов, (3) вероятный режим отказа, за которым нужно следить, и (4) бюджет compute/time.

## Упражнения

1. **Легко.** Для каждого из пяти продуктов определите семейство и backbone: ChatGPT image, Midjourney v7, Sora, Runway Gen-3, ElevenLabs. Доказательства должны быть из публичных технических отчетов.
2. **Средне.** Статья, которую вы собираетесь читать завтра, заявляет sampling в 100x быстрее diffusion. Запишите три вопроса, чтобы проверить, сохраняется ли ускорение при conditioning и высоком разрешении.
3. **Сложно.** Возьмите важный для вас домен (например, protein structure, CAD, molecules, trajectories). Ответьте на пять triage-вопросов для текущей SOTA-модели в этом домене и набросайте, что изменила бы лучшая модель.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Generative model | "Она делает новые штуки" | Учит sampler для `p_data(x)`, опционально предоставляет `log p(x)`. |
| Explicit density | "Ее можно вычислить" | Модель дает closed-form или tractable `log p(x)`. |
| Implicit density | "GAN-style" | Только sampler — нет способа вычислить `p(x)` для заданной точки. |
| ELBO | "Evidence lower bound" | Вычислимая нижняя граница для `log p(x)`; VAE и diffusion оптимизируют ее. |
| Score | "Gradient of log-density" | `∇_x log p(x)`; diffusion и SDE-модели учат это поле. |
| Manifold hypothesis | "Данные живут на поверхности" | Высокоразмерные данные концентрируются на низкоразмерном многообразии; поэтому работает dimensionality reduction. |
| Autoregressive | "Предсказывает следующий кусок" | Факторизует совместное распределение как произведение условных. |
| Latent | "Сжатый код" | Низкоразмерное представление, из которого decoder может восстановить вход. |

## Production note: пять семейств, пять форм inference

Каждое семейство соответствует другой кривой стоимости inference-server. Литература по production-inference описывает LLM inference как prefill + decode; та же декомпозиция применима здесь:

- **Autoregressive (корзины 1 и 5).** Последовательный decode доминирует latency; KV-cache, continuous batching и speculative decoding применимы напрямую.
- **VAE / diffusion / flow-matching (корзины 2 и 4).** Decode в смысле LLM здесь отсутствует. Стоимость = `num_steps × step_cost`, а `step_cost` — forward transformer или U-Net на полном latent-разрешении. Production-ручки: число шагов (DDIM / DPM-Solver / distillation), batch size и precision (bf16 / fp8 / int4).
- **GAN (корзина 3).** Один forward pass. Нет schedule, нет KV-cache. TTFT ≈ total latency. Поэтому StyleGAN все еще выигрывает в UX узкого домена.

Когда видите в abstract статьи "faster than diffusion", переводите это как "fewer steps × same step cost" или "same steps × cheaper step cost". Все остальное — маркетинг.

## Дополнительное чтение

- [Goodfellow et al. (2014). Generative Adversarial Nets](https://arxiv.org/abs/1406.2661) — статья о GAN.
- [Kingma & Welling (2013). Auto-Encoding Variational Bayes](https://arxiv.org/abs/1312.6114) — статья о VAE.
- [Ho, Jain, Abbeel (2020). Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2006.11239) — статья о DDPM.
- [Song et al. (2021). Score-Based Generative Modeling through SDEs](https://arxiv.org/abs/2011.13456) — diffusion как SDE.
- [Lipman et al. (2023). Flow Matching for Generative Modeling](https://arxiv.org/abs/2210.02747) — статья о flow matching.
- [Esser et al. (2024). Scaling Rectified Flow Transformers for High-Resolution Image Synthesis](https://arxiv.org/abs/2403.03206) — Stable Diffusion 3.
