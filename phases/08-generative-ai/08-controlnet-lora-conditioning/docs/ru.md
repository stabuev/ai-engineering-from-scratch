# ControlNet, LoRA и conditioning

> Один text — неуклюжий control signal. ControlNet позволяет клонировать pretrained diffusion model и направлять ее depth map, pose skeleton, scribble или edge image. LoRA позволяет fine-tune модель с 2B parameters, обучая 10 million parameters. Вместе они превратили Stable Diffusion из игрушки в image pipeline 2026 года, которую поставляют в каждом agency.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 8 · 07 (Latent Diffusion), Фаза 10 (LLMs from Scratch — for LoRA foundation)
**Время:** ~75 минут

## Проблема

Prompt вроде "a woman in a red dress walking a dog on a busy street" не дает модели информации о том, *где* собака, *в какой pose* женщина или *какая perspective* у улицы. Text фиксирует около 10% того, что нужно задать для изображения. Остальное визуально и неэффективно описывается словами.

Обучать новую conditional model с нуля для каждого сигнала (pose, depth, canny, segmentation) слишком дорого. Нужно оставить 2.6B-param SDXL backbone frozen, прикрепить малую side-network, которая читает conditioning, и позволить ей слегка двигать intermediate features backbone. Это ControlNet.

Также нужно обучить модель новым concepts (ваше лицо, ваш продукт, ваш стиль) без retraining всей модели. Нужна delta в 100x меньше. Это LoRA — low-rank adapters, подключаемые к существующим attention weights.

ControlNet + LoRA + text = toolkit практика в 2026 году. Большинство production image pipelines накладывают 2-5 LoRAs, 1-3 ControlNets и IP-Adapter поверх SDXL / SD3 / Flux base.

## Концепция

![ControlNet clones the encoder; LoRA adds low-rank deltas](../assets/controlnet-lora.svg)

### ControlNet (Zhang et al., 2023)

Возьмите pretrained SD. *Клонируйте* encoder half U-Net. Заморозьте original. Обучите clone принимать extra conditioning input (edges, depth, pose). Подключите clone обратно к decoder half original через *zero-convolution* skip connections (1×1 convs, initialized to zero — стартуют как no-op, учат delta).

```
SD U-Net decoder:   ... ← orig_enc_features + zero_conv(controlnet_enc(condition))
```

Zero-conv init означает, что ControlNet начинается как identity — вреда нет даже до training. Обучайте на 1M triples (prompt, condition, image) со standard diffusion loss.

Per-modality ControlNets поставляются как small side models (~360M для SDXL, ~70M для SD 1.5). Их можно compose на inference:

```
features += weight_a * control_a(depth) + weight_b * control_b(pose)
```

### LoRA (Hu et al., 2021)

Для любого linear layer `W ∈ R^{d×d}` в модели заморозьте `W` и добавьте low-rank delta:

```
W' = W + ΔW,  ΔW = B @ A,  A ∈ R^{r×d},  B ∈ R^{d×r}
```

где `r << d`. Rank 4-16 стандартен для attention, rank 64-128 — для heavy fine-tunes. Новых параметров: `2 · d · r` вместо `d²`. Для SDXL attention при `d=640`, `r=16`: 20k params на adapter вместо 410k — reduction 20x. По всей модели LoRA обычно 20-200MB против base 5GB.

На inference LoRA можно scale-ить: `W' = W + α · B @ A`. `α = 0.5-1.5` нормально. Несколько LoRAs складываются аддитивно (с обычной оговоркой, что взаимодействуют нелинейно).

### IP-Adapter (Ye et al., 2023)

Крошечный adapter, который принимает *image* как conditioning (вместе с text). Использует CLIP image encoder для получения image tokens и внедряет их в cross-attention рядом с text tokens. ~20MB на base model. Позволяет делать "generate an image in the style of this reference" без LoRA.

## Матрица совместимости

| Tool | What it controls | Size | When to use |
|------|------------------|------|-------------|
| ControlNet | Spatial structure (pose, depth, edges) | 70-360MB | Exact layout, composition |
| LoRA | Style, subject, concept | 20-200MB | Personalization, style |
| IP-Adapter | Style or subject from reference image | 20MB | No text can describe the look |
| Textual Inversion | Single concept as a new token | 10KB | Legacy, mostly replaced by LoRA |
| DreamBooth | Full fine-tune on a subject | 2-5GB | Strong identity, high compute |
| T2I-Adapter | Lighter ControlNet alternative | 70MB | Edge devices, inference budget |

ControlNet ≈ spatial. LoRA ≈ semantic. Используйте оба.

## Практика

`code/main.py` симулирует два механизма в 1-D:

1. **LoRA.** Pretrained linear layer `W`. Freeze it. Обучить low-rank `B @ A`, чтобы `W + BA` совпал с target linear layer. Показать, что `r = 1` достаточно для идеального rank-1 correction.

2. **ControlNet-lite.** "Frozen base" predictor и "side network", читающая extra signal. Output side network gated learnable scalar, initialized to zero (наша версия zero-conv). Обучите и наблюдайте, как gate растет.

### Шаг 1: LoRA math

```python
def lora(W, A, B, x, alpha=1.0):
    # W is frozen; A, B are the trainable low-rank factors.
    return [W[i][j] * x[j] for i, j in ...] + alpha * (B @ (A @ x))
```

### Шаг 2: zero-init side network

```python
side_out = control_net(x, condition)
gated = gate * side_out  # gate initialized to 0
h = base(x) + gated
```

На step 0 output идентичен base. Early training медленно обновляет `gate` — без catastrophic drift.

## Подводные камни

- **Over-scaling LoRAs.** `α = 2` или `α = 3` — частый hack "make it stronger", который дает over-stylized / broken outputs. Держите `α ≤ 1.5`.
- **ControlNet weight conflict.** Pose ControlNet с weight 1.0 и Depth ControlNet с weight 1.0 обычно overshoots. Sum of weights ≈ 1.0 — safe default.
- **LoRA on the wrong base.** SDXL LoRAs silently no-op на SD 1.5, потому что attention dimensions не совпадают. Diffusers предупредит в 0.30+.
- **Textual Inversion drift.** Tokens, обученные на одном checkpoint, сильно drift на другом. LoRA более portable.
- **LoRA weight-merging and storage.** Можно bake LoRA в base model weights для faster inference (нет runtime addition), но вы теряете возможность scale `α` at runtime. Держите обе версии.

## Применение

| Goal | 2026 pipeline |
|------|---------------|
| Воспроизвести art style бренда | LoRA trained on ~30 curated images at rank 32 |
| Поместить мое лицо в generated image | DreamBooth or LoRA + IP-Adapter-FaceID |
| Specific pose + prompt | ControlNet-Openpose + SDXL + text |
| Depth-aware composition | ControlNet-Depth + SD3 |
| Reference + prompt | IP-Adapter + text |
| Exact layout | ControlNet-Scribble or ControlNet-Canny |
| Background replace | ControlNet-Seg + Inpainting (Lesson 09) |
| Fast 1-step style | LCM-LoRA on SDXL-Turbo |

## Запуск в продукт

Сохраните `outputs/skill-sd-toolkit-composer.md`. Навык принимает task (input assets: prompt, optional reference image, optional pose, optional depth, optional scribble) и выдает tool stack, weights и reproducible seed protocol.

## Упражнения

1. **Легко.** В `code/main.py` меняйте LoRA rank `r` от 1 до 4. На каком rank LoRA точно совпадает с rank-2 target delta?
2. **Средне.** Обучите две отдельные LoRAs на двух target transforms. Загрузите их вместе и покажите additive interaction. Когда interaction ломает linearity?
3. **Сложно.** Используйте diffusers, чтобы сложить: SDXL-base + Canny-ControlNet (weight 0.8) + style LoRA (α 0.8) + IP-Adapter (weight 0.6). Измерьте trade-off FID-vs-prompt-adherence при изменении stack weights.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| ControlNet | "Spatial control" | Cloned encoder + zero-conv skips; читает conditioning image. |
| Zero convolution | "Starts as identity" | 1×1 conv initialized to zero; ControlNet starts as no-op. |
| LoRA | "Low-rank adapter" | `W + B @ A`, `r << d`; в 100x меньше params, чем full fine-tune. |
| rank r | "The knob" | LoRA compression; 4-16 typical, 64+ для heavy personalization. |
| α | "LoRA strength" | Runtime scaling of the LoRA delta. |
| IP-Adapter | "Reference image" | Small image-conditioning adapter через CLIP-image tokens. |
| DreamBooth | "Full subject fine-tune" | Train full model на ~30 images of a subject. |
| Textual Inversion | "New token" | Learn only new word embedding; legacy, mostly replaced. |

## Production note: LoRA swaps, ControlNet lanes, multi-tenant serving

Реальный text-to-image SaaS обслуживает сотни LoRAs и десяток ControlNets поверх одного base checkpoint. Serving problem похожа на LLM multi-tenancy (production literature покрывает LLM case через continuous batching и LoRAX / S-LoRA):

- **Hot-swap LoRAs, do not merge.** Merging `W' = W + α·B·A` into base дает ~3-5% faster per-step inference, но замораживает `α` и base. Держите LoRAs hot in VRAM как rank-r deltas; diffusers предоставляет `pipe.load_lora_weights()` + `pipe.set_adapters([...], adapter_weights=[...])` для per-request activation. Swap cost — weights `2 · d · r · num_layers`: MB-scale, sub-second.
- **ControlNet as a second attention lane.** Cloned encoder работает параллельно с base. Два ControlNets at weight 1.0 each = два extra forward passes per step, а не один merged pass. Batch-size headroom падает квадратично. Бюджетируйте ~1.5× step cost на active ControlNet.
- **Quantized LoRAs too.** Если base quantized (см. Lesson 07, Flux on 8GB), LoRA delta тоже чисто quantizes to 8-bit или 4-bit. QLoRA-style loading позволяет stack 5-10 LoRAs поверх 4-bit Flux base без взрыва memory.

Flux-specific: notebook Niels Flux-on-8GB quantizes base to 4-bit; stacking style LoRA (`pipe.load_lora_weights("user/style-lora")`) на этом quantized base с `weight_name="pytorch_lora_weights.safetensors"` все еще работает. Это рецепт, который most SaaS agencies ship in 2026.

## Дополнительное чтение

- [Zhang, Rao, Agrawala (2023). Adding Conditional Control to Text-to-Image Diffusion Models](https://arxiv.org/abs/2302.05543) — ControlNet.
- [Hu et al. (2021). LoRA: Low-Rank Adaptation of Large Language Models](https://arxiv.org/abs/2106.09685) — LoRA (изначально для LLMs; портирована в diffusion).
- [Ye et al. (2023). IP-Adapter: Text Compatible Image Prompt Adapter](https://arxiv.org/abs/2308.06721) — IP-Adapter.
- [Mou et al. (2023). T2I-Adapter: Learning Adapters to Dig Out More Controllable Ability](https://arxiv.org/abs/2302.08453) — более легкая альтернатива ControlNet.
- [Ruiz et al. (2023). DreamBooth: Fine Tuning Text-to-Image Diffusion Models for Subject-Driven Generation](https://arxiv.org/abs/2208.12242) — DreamBooth.
- [HuggingFace Diffusers — ControlNet / LoRA / IP-Adapter docs](https://huggingface.co/docs/diffusers/training/controlnet) — reference pipelines.
