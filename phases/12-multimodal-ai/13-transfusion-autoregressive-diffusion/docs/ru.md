# Transfusion: авторегрессионный текст + diffusion-изображение в одном трансформере

> Chameleon и Emu3 ставят все на дискретные токены. Они работают, но bottleneck квантования заметен — качество изображений выходит на плато ниже continuous-space diffusion models. Transfusion (Meta, Zhou et al., август 2024) делает противоположную ставку: оставить изображения непрерывными, полностью убрать VQ-VAE и обучать один трансформер с двумя функциями потерь. Текстовые токены получают next-token-prediction. Image patches получают flow-matching / diffusion loss. Обе цели оптимизируют одни и те же веса. Архитектура, лежащая в основе Stable Diffusion 3 (MMDiT), является близким родственником. Этот урок разбирает тезис Transfusion, строит игрушечный trainer с двумя loss и прослеживает attention mask, который позволяет одному трансформеру делать обе задачи.

**Тип:** Практика
**Языки:** Python (stdlib, two-loss trainer on MNIST-scale toy)
**Предварительные требования:** Phase 12 · 11 (Chameleon), Phase 8 (Generative AI)
**Время:** ~180 минут

## Цели обучения

- Собрать трансформер, который запускает две функции потерь (NTP на текстовых токенах, diffusion MSE на image patches) на одном backbone.
- Объяснить, почему bidirectional attention по image patches плюс causal attention по текстовым токенам — правильный выбор mask.
- Сравнить Transfusion-style (continuous images, diffusion loss) с Chameleon-style (discrete images, NTP) по compute, quality и code complexity.
- Назвать вклад MMDiT: modality-specific weights в каждом блоке, joint attention на residual stream.

## Проблема

Спор о дискретных и непрерывных токенах изображения старше LLM. Непрерывные представления (raw pixels, VAE latents) сохраняют детали. Дискретные токены (VQ indices) подходят родному словарю трансформера, но теряют детали на этапе квантования.

Chameleon / Emu3 пошли по дискретному пути: одна loss, одна архитектура, но fidelity изображений ограничена качеством токенизатора.

Diffusion models пошли по непрерывному пути: исключительное качество изображений, но отдельная от LLM модель, сложная инженерия noise-schedule и отсутствие чистой интеграции с генерацией текста.

Transfusion спрашивает: можем ли мы получить оба преимущества? Оставить изображения непрерывными, все равно обучать одну модель и использовать две loss, сшитые в один gradient step.

## Концепция

### Архитектура с двумя loss

Один decoder-only transformer обрабатывает последовательность, содержащую:

- Текстовые токены (дискретные, из BPE vocab).
- Image patches (непрерывные, 16x16 pixel blocks, спроецированные в hidden dim через linear embedding — как вход ViT encoder).
- Теги `<image>` и `</image>`, отмечающие, где находятся continuous patches.

Forward pass выполняется один раз. Loss выбирает одну из двух heads для каждого токена:

- Для текстовых токенов: стандартная cross-entropy на vocab-logits head.
- Для image patches: diffusion loss на continuous patches — предсказать шум, добавленный к каждому patch.

Градиент проходит через общий transformer body. Обе loss одновременно улучшают общие веса.

### Attention mask: causal text + bidirectional image

Текстовые токены должны быть causal — нельзя позволять текстовому токену смотреть на будущий текст, иначе teacher forcing ломается. Image patches, однако, представляют один snapshot; они должны bidirectionally attend друг к другу внутри одного image block.

Маска:

```
M[i, j] = 1 if:
  (i is text and j is text and j <= i)   # causal for text
  OR (i is image and j is image and same_image_block(i, j))   # bidirectional within image
  OR (i is text and j is image and j < i_image_end)   # text attends to previous images
  OR (i is image and j is text and j < i_image_start)   # image attends to preceding text
```

Реализуется как block-triangular mask при обучении и инференсе.

### Diffusion loss внутри трансформера

Diffusion loss стандартная: добавить шум к image patch и попросить модель предсказать шум (или clean patch, эквивалентно). Версия Transfusion использует flow matching — предсказывает velocity field от noisy к clean.

Во время обучения:
1. Для каждого image patch x0, sample a random timestep t.
2. Sample noise ε, compute xt = (1-t) * x0 + t * ε (linear interpolation for flow matching).
3. The transformer predicts v_theta(xt, t); loss = MSE(v_theta(xt, t), ε - x0).
4. Backprop alongside text NTP losses from the same sequence.

На инференсе генерация:
- Текстовые токены: стандартный autoregressive sampling.
- Image patches: diffusion sampling loop (обычно 10-30 steps), conditioned on the prior text tokens.

### MMDiT: вариант Stable Diffusion 3

Stable Diffusion 3 (Esser et al., март 2024) выпустил MMDiT (Multimodal Diffusion Transformer) примерно в то же время, что и Transfusion. Эти архитектуры — родственные.

Ключевые отличия MMDiT:

- Modality-specific weights per block. У каждого transformer block есть отдельные Q, K, V и MLP weights для текстовых токенов и image patches. Attention общий (cross-modality); все остальное modality-specific.
- Rectified flow training. Специальный вариант flow-matching с известным sampling и более простой математикой, чем DDPM.
- Масштаб. MMDiT — backbone для SD3 (варианты 2B и 8B параметров). Статья Transfusion масштабируется до 7B.

Оба сходятся к одной основной идее: один трансформер запускает NTP на тексте и diffusion на continuous image representations.

### Почему это превосходит Chameleon-style

Разрыв качества между continuous-diffusion и discrete-NTP в генерации изображений измерим. Статья Transfusion сообщает:

- При 7B params превосходит same-size Chameleon-style model на FID на 3-5 пунктов.
- Не требуется обучение токенизатора — image encoder проще (Linear projection to hidden, как input layer у ViT).
- Инференс может parallelize image patch denoising, в отличие от autoregressive image tokens.

Минус: Transfusion — dual-loss model, что делает динамику обучения сложнее. Loss weights нужно настраивать. Schedule mismatch между NTP и diffusion может привести к доминированию одной head.

### Что находится downstream

Janus-Pro (Lesson 12.15) уточняет идею Transfusion, decoupling vision encoder для понимания и генерации — SigLIP для одного, VQ для другого — при общем transformer body. Show-o (Lesson 12.14) заменяет diffusion на discrete-diffusion (masked prediction). Семейство unified-generation быстро ветвится после Transfusion.

Production VLM 2026 года, которые выдают изображения — Gemini 3 Pro, GPT-5, путь генерации изображений Claude Opus 4.7 — почти наверняка используют какого-то потомка этого семейства. Детали proprietary.

## Использование

`code/main.py` строит игрушечный Transfusion на маленькой MNIST-like задаче:

- Text captions — короткие integer sequences, описывающие digit (0-9).
- Images — 4x4 grids of bytes.
- Пара shared-weight linear projections выступает как stand-in трансформера; NTP loss на тексте, MSE loss на noisy patches.
- Training loop чередует две loss, attention mask явно задана.
- Generation создает text caption и 4x4 image за один forward pass.

Трансформер игрушечный. Реальные артефакты — two-loss plumbing, построение attention mask и inference loop.

## Результат

Этот урок создает `outputs/skill-two-loss-trainer-designer.md`. Для новой multimodal training task (text + image, text + audio, text + video) он проектирует two-loss schedule (loss weights, mask shape, shared vs modality-specific blocks) и отмечает implementation risks.

## Упражнения

1. Модель Transfusion-style обучается на 70% text tokens и 30% image patches. Image diffusion loss примерно в 10x больше text NTP loss по величине. Какие loss weights их сбалансируют?

2. Реализуйте block-triangular mask для последовательности: `[T, T, <image>, P, P, P, P, </image>, T]`. Отметьте каждый entry 0 или 1.

3. У MMDiT есть modality-specific QKV weights. Какой overhead по числу параметров это добавляет по сравнению с fully-shared transformer у Transfusion? При 7B params это стоит того?

4. Generation: по text prompt модель запускает NTP на 50 tokens, затем встречает `<image>`, затем запускает diffusion на 256 patches за 20 denoise steps. Сколько всего forward passes?

5. Прочитайте SD3 paper Section 3. Опишите rectified flow и почему он сходится за меньшее число inference steps, чем DDPM.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Two-loss training | "NTP + diffusion" | Один трансформер оптимизирует и cross-entropy на текстовых токенах, и MSE на continuous image patches в одном gradient step |
| Flow matching | "Rectified flow" | Вариант diffusion, который предсказывает velocity field от шума к clean data; математика проще, чем DDPM |
| MMDiT | "Multimodal DiT" | Архитектура Stable Diffusion 3: joint attention, modality-specific MLPs и norms |
| Block-triangular mask | "Causal text + bidirectional image" | Attention mask, causal по тексту, но bidirectional внутри image regions |
| Continuous image representation | "No VQ" | Image patches как real-valued vectors, а не integer codebook indices |
| Velocity prediction | "v-parameterization" | Выход сети — velocity field между шумом и данными, а не сам шум |

## Дополнительное чтение

- [Zhou et al. — Transfusion (arXiv:2408.11039)](https://arxiv.org/abs/2408.11039)
- [Esser et al. — Stable Diffusion 3 / MMDiT (arXiv:2403.03206)](https://arxiv.org/abs/2403.03206)
- [Peebles & Xie — DiT (arXiv:2212.09748)](https://arxiv.org/abs/2212.09748)
- [Zhao et al. — MonoFormer (arXiv:2409.16280)](https://arxiv.org/abs/2409.16280)
- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
