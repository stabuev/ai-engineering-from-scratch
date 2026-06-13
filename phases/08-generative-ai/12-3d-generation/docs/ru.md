# Генерация 3D

> 3D — modality, где leverage 2D-to-3D сильнее всего. Прорыв 2023 года — 3D Gaussian Splatting. Генеративный push 2024-2026 накладывает multi-view diffusion + 3D reconstruction сверху, чтобы получать objects и scenes из одного prompt или photo.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 4 (Vision), Фаза 8 · 07 (Latent Diffusion)
**Время:** ~45 минут

## Цели обучения

- Реализовывать 2D Gaussian splat, рендерить суммированием сплатов и подгонять градиентным спуском.
- Объяснять, как multi-view diffusion плюс 3D-реконструкция превращают промпт в объект.
- Сравнивать 3D Gaussian Splatting, NeRF и triplane-представления.

## Проблема

3D content болезнен:

- **Representation.** Meshes, point clouds, voxel grids, signed distance fields (SDFs), neural radiance fields (NeRFs), 3D Gaussians. У каждого trade-offs.
- **Data scarcity.** У ImageNet 14M images. Крупнейший clean 3D dataset (Objaverse-XL, 2023) имеет ~10M objects, большинство low quality.
- **Memory.** Voxel grid 512³ — 128M voxels; полезная scene NeRF требует 1M samples/ray. Generation сложнее reconstruction.
- **Supervision.** Для 2D image есть pixels. Для 3D обычно есть несколько 2D views, и их надо lift to 3D.

Stack 2026 года разделяет две задачи. Сначала генерировать *2D multi-view images* diffusion model. Затем подогнать *3D representation* (обычно Gaussian splatting) к этим images.

## Концепция

![3D generation: multi-view diffusion + 3D reconstruction](../assets/3d-generation.svg)

### Representation: 3D Gaussian Splatting (Kerbl et al., 2023)

Представьте scene как cloud of ~1M 3D Gaussians. У каждой 59 parameters: position (3), covariance (6 или quaternion 4 + scale 3), opacity (1), spherical-harmonics color (48 at degree 3, 3 at degree 0).

Rendering = projection + alpha-compositing. Fast (~100 fps at 1080p on a 4090). Differentiable. Fit by gradient descent against ground-truth photos. Scene fit за 5-30 minutes на consumer GPU.

Две инновации 2023-2024 поверх:
- **Generative Gaussian splats.** Models like LGM, LRM, InstantMesh predict Gaussian cloud directly from one or a few images.
- **4D Gaussian Splatting.** Gaussians with per-frame offsets for dynamic scenes.

### Multi-view diffusion

Fine-tune pretrained image diffusion model, чтобы генерировать несколько consistent views одного object из text prompt или single image. Zero123 (Liu et al., 2023), MVDream (Shi et al., 2023), SV3D (Stability, 2024), CAT3D (Google, 2024). Обычно output 4-16 views around object, затем lifted to 3D via Gaussian splatting or NeRF.

### Text-to-3D pipelines

| Model | Input | Output | Time |
|-------|-------|--------|------|
| DreamFusion (2022) | text | NeRF via SDS | ~1 hour per asset |
| Magic3D | text | mesh + texture | ~40 min |
| Shap-E (OpenAI, 2023) | text | implicit 3D | ~1 min |
| SJC / ProlificDreamer | text | NeRF / mesh | ~30 min |
| LRM (Meta, 2023) | image | triplane | ~5 s |
| InstantMesh (2024) | image | mesh | ~10 s |
| SV3D (Stability, 2024) | image | novel views | ~2 min |
| CAT3D (Google, 2024) | 1-64 images | 3D NeRF | ~1 min |
| TripoSR (2024) | image | mesh | ~1 s |
| Meshy 4 (2025) | text + image | PBR mesh | ~30 s |
| Rodin Gen-1.5 (2025) | text + image | PBR mesh | ~60 s |
| Tencent Hunyuan3D 2.0 (2025) | image | mesh | ~30 s |

Направление 2025-2026: direct text-to-mesh models with PBR materials, пригодные для game engines. Multi-view diffusion intermediate step все еще лучший рецепт для general objects.

### NeRF (для контекста)

Neural Radiance Field (Mildenhall et al., 2020). Tiny MLP принимает `(x, y, z, view direction)` и выдает `(color, density)`. Render by integrating along rays. Качественнее mesh-based novel-view synthesis, но render в 100-1000x slower. Для большинства real-time use заменен Gaussian splatting, но в research все еще dominant.

## Практика

`code/main.py` реализует toy 2D "Gaussian splatting" fit: представить synthetic target image (smooth gradient) как сумму 2D Gaussian splats. Optimize positions, colors и covariances by gradient descent to match target. Вы увидите две core operations: forward render (splat + alpha-composite) и fit by gradient descent.

### Шаг 1: 2D Gaussian splat

```python
def gaussian_at(x, y, gaussian):
    px, py = gaussian["pos"]
    sigma = gaussian["sigma"]
    d2 = (x - px) ** 2 + (y - py) ** 2
    return math.exp(-d2 / (2 * sigma * sigma))
```

### Шаг 2: render by summing splats

```python
def render(image_size, gaussians):
    img = [[0.0] * image_size for _ in range(image_size)]
    for g in gaussians:
        for y in range(image_size):
            for x in range(image_size):
                img[y][x] += g["color"] * gaussian_at(x, y, g)
    return img
```

Real 3D Gaussian splatting sorts Gaussians by depth and alpha-composites in order. Our 2D toy just sums.

### Шаг 3: fit by gradient descent

```python
for step in range(steps):
    pred = render(size, gaussians)
    loss = mse(pred, target)
    gradients = compute_grads(pred, target, gaussians)
    update(gaussians, gradients, lr)
```

## Подводные камни

- **View inconsistency.** Если generate 4 views independently, и они спорят о structure объекта, 3D fit становится blurry. Исправление: multi-view diffusion with shared attention.
- **Back-side hallucination.** Single-image → 3D должен invent unseen side. Quality varies wildly.
- **Gaussian splat explosion.** Unconstrained training растет до 10M splats и overfits. Densification + pruning heuristics (из original 3D-GS paper) essential.
- **Topology issues.** Meshes from implicit fields (SDFs) часто имеют holes или self-intersections. Run remesher (e.g. blender's voxel remesh) before shipping.
- **License of training data.** Objaverse has mixed licenses; commercial use varies per model.

## Применение

| Task | 2026 pick |
|------|-----------|
| Scene reconstruction from photos | Gaussian splatting (3DGS, Gsplat, Scaniverse) |
| Text-to-3D object for games | Meshy 4 or Rodin Gen-1.5 (PBR output) |
| Image-to-3D | Hunyuan3D 2.0, TripoSR, InstantMesh |
| Novel-view synthesis from few images | CAT3D, SV3D |
| Dynamic scene reconstruction | 4D Gaussian Splatting |
| Avatar / clothed human | Gaussian Avatar, HUGS |
| Research / SOTA | Whatever dropped last week |

Для production 3D in a game or e-commerce pipeline: Meshy 4 или Rodin Gen-1.5 output PBR meshes, которые идут прямо в Unity / Unreal.

## Запуск в продукт

Сохраните `outputs/skill-3d-pipeline.md`. Навык принимает 3D brief (input: text / one image / few images; output: mesh / splat / NeRF; usage: render / game / VR) и выдает: pipeline (multi-view diffusion + fit или direct mesh model), base model, iteration budget, topology post-processing, material channels needed.

## Упражнения

1. **Легко.** Запустите `code/main.py` с 4, 16, 64 Gaussians. Сообщите final MSE vs target.
2. **Средне.** Расширьте до color Gaussians (RGB). Подтвердите, что reconstruction matches target color pattern.
3. **Сложно.** Используя gsplat или Nerfstudio, reconstruct real object from a 50-photo capture. Сообщите fit time и final SSIM on held-out views.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| 3D Gaussian Splatting | "3DGS" | Scene as cloud of 3D Gaussians; differentiable alpha-composite render. |
| NeRF | "Neural radiance field" | MLP outputs color + density at 3D point; render by ray integration. |
| Triplane | "Three 2-D planes" | Factor 3D into three 2-D axis-aligned feature grids; cheaper than volumetric. |
| SDS | "Score distillation sampling" | Train 3D model using 2D-diffusion score as pseudo-gradient. |
| Multi-view diffusion | "Many views at once" | Diffusion model outputs batch of consistent camera views. |
| PBR | "Physically-based rendering" | Material with albedo, roughness, metallic, normal channels. |
| Densification | "Grow splats" | 3DGS training heuristic: split / clone splats in high-gradient regions. |

## Production note: у 3D пока нет shared substrate

В отличие от image (latent diffusion + DiT) и video (spatiotemporal DiT), у 3D в 2026 году нет единого dominant runtime. Production decision tree ветвится по representation:

- **NeRF / triplane.** Inference — ray-marching + MLP forward per sample. Render 512² требует millions of MLP forwards. Batch ray samples aggressively; SDPA/xformers applies.
- **Multi-view diffusion + LRM reconstruction.** Two-stage pipeline. Stage 1 (multi-view DiT) — diffusion server как в Lesson 07. Stage 2 (LRM transformer) — one-shot forward pass over views. Overall latency profile — "diffusion + one-shot"; выбирайте per-stage serving primitives.
- **SDS / DreamFusion.** Per-asset optimization, не inference. Build jobs, not request handlers.

Для большинства products 2026 года правильный ответ: "run multi-view diffusion model on request, reconstruct to 3DGS asynchronously, serve 3DGS for real-time viewing". Это чисто делит workload между GPU-inference server (fast) и offline optimizer (slow).

## Дополнительное чтение

- [Mildenhall et al. (2020). NeRF: Representing Scenes as Neural Radiance Fields](https://arxiv.org/abs/2003.08934) — NeRF.
- [Kerbl et al. (2023). 3D Gaussian Splatting for Real-Time Radiance Field Rendering](https://arxiv.org/abs/2308.04079) — 3DGS.
- [Poole et al. (2022). DreamFusion: Text-to-3D using 2D Diffusion](https://arxiv.org/abs/2209.14988) — SDS.
- [Liu et al. (2023). Zero-1-to-3: Zero-shot One Image to 3D Object](https://arxiv.org/abs/2303.11328) — Zero123.
- [Shi et al. (2023). MVDream](https://arxiv.org/abs/2308.16512) — multi-view diffusion.
- [Hong et al. (2023). LRM: Large Reconstruction Model for Single Image to 3D](https://arxiv.org/abs/2311.04400) — LRM.
- [Gao et al. (2024). CAT3D: Create Anything in 3D with Multi-View Diffusion Models](https://arxiv.org/abs/2405.10314) — CAT3D.
- [Stability AI (2024). Stable Video 3D (SV3D)](https://stability.ai/research/sv3d) — SV3D.
