# Inpainting, Outpainting и редактирование изображений

> Text-to-image создает новое. Inpainting исправляет старое. В production 70% оплачиваемой работы с изображениями — это editing: заменить background, удалить logo, расширить canvas, перегенерировать hand. Inpainting — место, где diffusion окупает себя.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 07 (Latent Diffusion), Фаза 8 · 08 (ControlNet & LoRA)
**Время:** ~75 минут

## Проблема

Клиент присылает идеальное product photo с отвлекающей вывеской на background. Вы хотите стереть вывеску и оставить все остальное pixel-identical. Нельзя запускать text-to-image с нуля — получится другой color, другое lighting, другой product angle. Нужно regenerate *только* masked region, и regeneration должен уважать surrounding context.

Это inpainting. Варианты:

- **Inpainting.** Regenerate внутри mask, keep outside pixels.
- **Outpainting.** Regenerate outside a mask (или за пределами canvas), keep inside.
- **Image editing.** Regenerate whole image, но сохранить semantic или structural fidelity к original (SDEdit, InstructPix2Pix).

Каждая diffusion pipeline в 2026 году поставляет inpainting mode. Flux.1-Fill, Stable Diffusion Inpaint, SDXL-Inpaint, DALL-E 3 Edit. Все работают по одному принципу.

## Концепция

![Inpainting: mask-aware denoising with context-preserving reinjection](../assets/inpainting.svg)

### Наивный подход (и почему он неверен)

Запустить standard text-to-image с mask. На каждом sampling step заменить unmasked region noisy latent на forward-diffused clean image. Работает... плохо. Boundary artifacts просачиваются, потому что у модели нет информации о том, что находится в masked region.

### Правильная inpainting model

Обучите modified U-Net, который принимает 9 input channels вместо 4:

```
input = concat([ noisy_latent (4ch), encoded_image (4ch), mask (1ch) ], dim=channel)
```

Extra channels — копия VAE-encoded source image плюс single-channel mask. На training случайно mask regions of image и обучайте модель denoise-ить только masked region, пока unmasked region дан как clean conditioning signal. На inference модель "видит", что окружает masked region, и производит coherent completions.

SD-Inpaint, SDXL-Inpaint, Flux-Fill используют такой 9-channel (или аналогичный) input. Diffusers `StableDiffusionInpaintPipeline`, `FluxFillPipeline`.

### SDEdit (Meng et al., 2022) — free editing

Добавьте noise к source image до intermediate `t`, затем запустите reverse chain от `t` до 0 с новым prompt. Без retraining. Выбор starting `t` меняет fidelity на creative freedom:

- `t/T = 0.3` → почти identical to source, малые stylistic changes
- `t/T = 0.6` → moderate edits, preserves coarse structure
- `t/T = 0.9` → generated from near-noise, minimal source preservation

### InstructPix2Pix (Brooks et al., 2023)

Fine-tune diffusion model на `(input_image, instruction, output_image)` triples. На inference condition и на input image, и на text instruction ("make it sunset", "add a dragon"). Две CFG scales: image scale и text scale.

### RePaint (Lugmayr et al., 2022)

Оставьте standard unconditional diffusion model. На каждом reverse step делайте resample — иногда прыгайте обратно в noisier state и regenerate. Это снижает boundary artifacts. Используется, когда trained inpainting model нет.

## Практика

`code/main.py` реализует toy 1-D inpainting scheme на 5-dimensional data. Мы обучаем DDPM на 5-D mixture data, где каждый sample — 5 floats из одного из двух clusters. На inference "mask" 2 из 5 dimensions, inject noisy-forward version трех unmasked на каждом step и regenerate только masked dimensions.

### Шаг 1: 5-D DDPM data

```python
def sample_data(rng):
    cluster = rng.choice([0, 1])
    center = [-1.0] * 5 if cluster == 0 else [1.0] * 5
    return [c + rng.gauss(0, 0.2) for c in center], cluster
```

### Шаг 2: train denoiser over all 5 dims

Standard DDPM. Net outputs 5-D noise prediction for 5-D noisy input.

### Шаг 3: at inference, mask-aware reverse

```python
def inpaint_step(x_t, mask, clean_image, alpha_bars, t, rng):
    # replace unmasked dims with a freshly noised version of the clean source
    a_bar = alpha_bars[t]
    for i in range(len(x_t)):
        if not mask[i]:
            x_t[i] = math.sqrt(a_bar) * clean_image[i] + math.sqrt(1 - a_bar) * rng.gauss(0, 1)
    # ...then run the normal reverse step on x_t
```

Это naive approach, и он работает на toy 1-D data. Real image inpainting использует 9-channel input, потому что texture coherence важнее.

### Шаг 4: outpainting

Outpainting — это inpainting с inverted mask: mask new (previously non-existent) canvas, fill rest with original. Training objective идентична.

## Подводные камни

- **Seams.** Naive approach оставляет visible boundaries, потому что gradient info не проходит через mask. Исправление: dilate mask на 8-16 pixels или используйте proper inpainting model.
- **Mask leakage.** Если unmasked region conditioning image низкого качества или noisy, он загрязняет generation inside mask. Denoise или слегка blur.
- **CFG interacts with mask size.** High CFG на small mask = saturated patch. Уменьшайте CFG для small edits.
- **SDEdit fidelity cliff.** Переход от `t/T = 0.5` к `t/T = 0.6` может потерять identity subject. Sweep and checkpoint.
- **Prompt mismatch.** Prompt должен описывать *whole* image, а не только new content. "A cat sitting on a chair", не "a cat".

## Применение

| Task | Pipeline |
|------|----------|
| Remove object, small mask | SD-Inpaint or Flux-Fill, standard prompt |
| Replace sky | SD-Inpaint + "blue sky at sunset" |
| Extend canvas | SDXL outpaint mode (8px feather) or Flux-Fill with outpaint mask |
| Regenerate hand / face | SD-Inpaint with prompt re-describing the subject + ControlNet-Openpose |
| Change style of one region | SDEdit at `t/T=0.5` on masked region |
| "Make it sunset" | InstructPix2Pix or Flux-Kontext |
| Background replacement | SAM mask → SD-Inpaint |
| Ultra-high-fidelity | Flux-Fill or GPT-Image (hosted) for hardest cases |

SAM (Meta's Segment Anything, 2023) + diffusion inpaint — background-removal pipeline 2026 года. SAM 2 (2024) работает на video.

## Запуск в продукт

Сохраните `outputs/skill-editing-pipeline.md`. Навык принимает original image + edit description + optional mask (or SAM prompt) и выдает: mask-generation approach, base model, CFG scales (image + text), SDEdit-t или inpainting mode, and QA checklist.

## Упражнения

1. **Легко.** В `code/main.py` меняйте fraction of dimensions masked от 0.2 до 0.8. При какой fraction inpaint quality (residual in masked dims) равна unconditional generation?
2. **Средне.** Реализуйте RePaint: на каждом 10-м reverse step прыгайте назад на 5 steps (add noise) и re-denoise. Измерьте, уменьшает ли это boundary residual на mask edge.
3. **Сложно.** Используйте Hugging Face diffusers, чтобы сравнить: SD 1.5 Inpaint + ControlNet-Openpose vs Flux.1-Fill на 20 face-regeneration tasks. Оценивайте pose adherence и identity preservation отдельно.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Inpainting | "Fill the hole" | Regenerate внутри mask; keep outside pixels. |
| Outpainting | "Extend the canvas" | Regenerate outside canvas; keep inside. |
| 9-channel U-Net | "Proper inpainting model" | U-Net с `noisy | encoded-source | mask` как input. |
| SDEdit | "Img2img with noise level" | Noise to time `t`, denoise with new prompt. |
| InstructPix2Pix | "Text-only edits" | Fine-tuned diffusion на (image, instruction, output) triples. |
| RePaint | "No retraining" | Периодически re-noise во время reverse, чтобы снизить seams. |
| SAM | "Segment Anything" | Mask generator по clicks или boxes; pairs with inpaint. |
| Flux-Kontext | "Edit with context" | Flux variant, принимающий reference image + instruction for edits. |

## Production note: edit pipelines latency-sensitive

Пользователи, редактирующие image, ожидают sub-5-second round trips. 30-step SDXL-Inpaint на 1024² — 3-4 s на L4, плюс SAM mask generation (~200 ms) и VAE encode/decode (~500 ms combined). В production framing это TTFT-bound, а не throughput-bound — batch 1, low concurrency, minimize every stage:

- **SAM-H is the slow one.** SAM-H на 1024² — ~200 ms; SAM-ViT-B — ~40 ms с minor quality loss. SAM 2 (video) добавляет temporal overhead; не используйте его для single-image edits.
- **Skip the encode when possible.** `pipe.image_processor.preprocess(img)` encodes to latents. Если latents есть от previous generation (типично в iterative-edit UIs), передайте их напрямую через `latents=...`, чтобы пропустить один VAE encode.
- **Mask dilation matters for throughput too.** Small mask значит, что большая часть U-Net forward pass wasted (unmasked pixels все равно clamped). `diffusers` `StableDiffusionInpaintPipeline` запускает full U-Net независимо; только 9-channel proper-inpaint variants используют masked compute.
- **Flux-Kontext is the 2025 answer.** Single forward pass over `(source_image, instruction)` — без separate mask, без SDEdit noise sweep. На H100 edit поставляется за ~1.5 s. Architectural lesson: collapse the stages.

## Дополнительное чтение

- [Lugmayr et al. (2022). RePaint: Inpainting using Denoising Diffusion Probabilistic Models](https://arxiv.org/abs/2201.09865) — training-free inpainting.
- [Meng et al. (2022). SDEdit: Guided Image Synthesis and Editing with Stochastic Differential Equations](https://arxiv.org/abs/2108.01073) — SDEdit.
- [Brooks, Holynski, Efros (2023). InstructPix2Pix](https://arxiv.org/abs/2211.09800) — text-instruction editing.
- [Kirillov et al. (2023). Segment Anything](https://arxiv.org/abs/2304.02643) — SAM, mask source.
- [Ravi et al. (2024). SAM 2: Segment Anything in Images and Videos](https://arxiv.org/abs/2408.00714) — video SAM.
- [Hertz et al. (2022). Prompt-to-Prompt Image Editing with Cross-Attention Control](https://arxiv.org/abs/2208.01626) — attention-level editing.
- [Black Forest Labs (2024). Flux.1-Fill and Flux.1-Kontext](https://blackforestlabs.ai/flux-1-tools/) — tooling 2024 года.
