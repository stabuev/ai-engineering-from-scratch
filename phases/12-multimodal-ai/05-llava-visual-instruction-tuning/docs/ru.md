# LLaVA и visual instruction tuning

> LLaVA (April 2023) — самая копируемая multimodal architecture на планете. Она заменила Q-Former из BLIP-2 на 2-layer MLP, заменила gated cross-attention из Flamingo на naive token concatenation и обучалась на 158k visual-instruction turns, сгенерированных GPT-4 из text-only captions. Любой practitioner, строивший VLM между 2023 и 2026, строил какой-то вариант LLaVA. LLaVA-1.5 добавила AnyRes. LLaVA-NeXT повысила resolution. LLaVA-OneVision унифицировала image, multi-image и video в одном recipe. Этот урок разбирает recipe, реализует projector и объясняет, почему "simpler won."

**Тип:** Практика
**Языки:** Python (stdlib, projector + instruction-template builder)
**Предварительные требования:** Phase 12 · 02 (CLIP), Phase 11 (LLM Engineering — instruction tuning)
**Время:** ~180 минут

## Цели обучения

- Построить 2-layer MLP projector, который maps ViT patch embeddings (dim 1024) to an LLM's embedding dim (dim 4096).
- Пройти LLaVA two-stage recipe: (1) projector alignment on 558k caption pairs, (2) visual instruction tuning on 158k GPT-4-generated turns.
- Сконструировать LLaVA-format prompt with the image token placeholder, system prompt, and user/assistant turns.
- Объяснить, почему community moved from Q-Former to MLP despite Q-Former's token-budget win.

## Проблема

Q-Former из BLIP-2 (Lesson 12.03) сжимает image до 32 tokens. Чисто, эффективно, хорошо для benchmarks. Но у него две проблемы.

Во-первых, Q-Former trainable, но его loss не является final task. Stage 1 trains ITC+ITM+ITG. Stage 2 trains LM loss. Queries учат некоторую intermediate representation, которую LLM затем должна декодировать. Информация теряется в bottleneck.

Во-вторых, Q-Former занимает 188M params, и на масштабе LLaVA 2023 года его нужно было co-design with your target LLM. Change the LLM, retrain the Q-Former. Change the vision encoder, retrain. Каждая комбинация была отдельным R&D project.

Ответ LLaVA был до неловкости простым: взять 576 patch tokens ViT, пропустить каждый через 2-layer MLP (`1024 → 4096 → 4096`) и отправить все 576 в input sequence LLM. No bottleneck. No stage 1 pretraining on weird objectives. Просто train the MLP on a direct LM loss.

Откуда берутся data? Второй insight LLaVA: использовать GPT-4 (text-only), чтобы generate instruction data. Подать GPT-4 COCO caption и bounding-box data для image, попросить его produce conversations, descriptions, and complex reasoning questions. 158k instruction-response turns бесплатно. No human annotation.

Результат: VLM, которая ran on 8 A100s for one day, beat Flamingo on MMMU и выпустила open checkpoint, который community могла расширять. К концу 2023 года она породила 50+ forks.

## Концепция

### The architecture

LLaVA-1.5 at 13B:
- Vision encoder: CLIP ViT-L/14 @ 336 (frozen during stage 1, optionally unfrozen stage 2).
- Projector: 2-layer MLP with GELU activation, `1024 → 4096 → 4096`.
- LLM: Vicuna-13B (later Llama-3.1-8B).

Forward pass on an image + text prompt:

```
img -> ViT -> 576 patches of dim 1024
patches -> MLP -> 576 tokens of dim 4096
prompt: system + "<image>" placeholder + user question
replace <image> token with the 576 projected tokens
feed the full sequence to the LLM
decode response
```

Image занимает 576 tokens context LLM. При context 2048 остается 1472 tokens for text. При 32k context это rounding error.

### Stage 1: projector alignment

Freeze ViT. Freeze LLM. Train only the 2-layer MLP. Dataset: 558k image-caption pairs (LAION-CC-SBU). Loss: language modeling on the caption, conditioned on the projected image tokens.

За single epoch at batch 128 это делается за few hours. Projector учится map ViT-space to LLM-space. No task-specific supervision.

### Stage 2: visual instruction tuning

Unfreeze the projector (still trainable). Unfreeze the LLM (usually fully, sometimes LoRA). Train on 158k visual-instruction turns.

Instruction data — главный прием. Liu et al. generated it by:
1. Take a COCO image.
2. Extract the text description (5 human captions + bounding-box list).
3. Send to GPT-4 with three prompt templates:
   - Conversation: "Generate a back-and-forth dialogue between a user and assistant about this image."
   - Detailed description: "Give a rich, detailed description of the image."
   - Complex reasoning: "Ask a question that requires reasoning about the image, then answer it."
4. Parse GPT-4's output into (instruction, response) pairs.

Ничего из этого не touches the image directly — только text description. GPT-4 hallucinates plausible image content. Some noise, but it worked: 158k turns was enough to unlock dialogue.

### Why the community copied this

- No stage-1-specific losses to tune. LM loss throughout.
- Projector trains in hours, not days.
- LLM can be swapped (LLaVA-Llama2, LLaVA-Mistral, LLaVA-Llama3) by retraining just the projector.
- Visual-instruction data pipeline uses GPT-4 and is cheap to regenerate for a new domain.

### LLaVA-1.5 and LLaVA-NeXT

LLaVA-1.5 (October 2023) added:
- Academic-task data (VQA, OKVQA, RefCOCO) mixed into instruction tuning.
- Better system prompt.
- 2048 → 32k context.

LLaVA-NeXT (January 2024) added:
- AnyRes: split high-res images into a 2x2 or 1x3 grid of 336x336 crops, plus one global low-res thumbnail. Each crop becomes 576 tokens; total around 2880 visual tokens per image. OCR and chart tasks jumped.
- Better instruction data mixture with ShareGPT4V (high-quality GPT-4V captions).
- Stronger base LLMs (Mistral-7B, Yi-34B).

### LLaVA-OneVision

Lesson 12.08 covers OneVision in depth. Short version: same projector, but trained with a curriculum that covers single-image, multi-image, and video in one model with shared visual-token budget.

### The comparison to Q-Former

| | Q-Former (BLIP-2) | MLP (LLaVA) |
|---|---|---|
| Visual tokens per image | 32 | 576 (base) or 2880 (AnyRes) |
| Trainable params | 188M + LM | 40M + LM |
| Stage 1 loss | ITC+ITM+ITG | LM only |
| LLM drop-in | Requires retrain | Swap with minimal retrain |
| Multi-image | Awkward | Natural (concat) |
| Video | Awkward | Natural (per-frame concat) |
| Token budget | Small | Large |

MLP выигрывает по simplicity и token flexibility. Q-Former выигрывает по token budget. К концу 2023 года token budget уже не был binding constraint (LLM contexts grew to 32k-128k+), и simplicity dominated.

### The prompt format

```
A chat between a curious human and an artificial intelligence assistant. The assistant gives helpful, detailed, and polite answers to the human's questions. USER: <image> Describe this image in detail. ASSISTANT: The image shows ...
```

`<image>` — placeholder token. До tokenization он replaces with 576 visual tokens (or 2880 with AnyRes). Tokenizer sees a slightly longer sequence than it was trained on, but the LLM handles the novel input because stage 1 taught it to.

### Parameter economy

LLaVA-1.5-7B breakdown:
- CLIP ViT-L/14 @ 336: 303M (frozen stage 1, often unfrozen stage 2).
- Projector (2x linear): ~22M trainable.
- Llama-7B: 7B.
- Total: 7.3B params. Trainable during stage 2: full 7B + 22M projector.

Training cost for stage 2: ~20 hours on 8xA100. Это key number — one day, one node, reproducible. Поэтому LLaVA spread.

## Использование

`code/main.py` implements:

1. The 2-layer MLP projector (dim 16 → 32 → 32 for toy scale) in pure Python.
2. The prompt-building pipeline: system prompt + `<image>` replaced with N projected tokens + user turn + assistant generation placeholder.
3. A visualizer for what the 576-token visual block looks like in LLM context (percentage of 2k / 32k / 128k context consumed).

## Результат

Этот урок создает `outputs/skill-llava-vibes-eval.md`. Given a LLaVA-family checkpoint, it runs a 10-prompt vibes-eval suite (3 captioning, 3 VQA, 2 reasoning, 2 refusal) and reports a human-readable scorecard. Not a benchmark; a smoke test to confirm the projector and LLM are connecting well.

## Упражнения

1. Вычислите trainable-parameter count для 2-layer MLP projector at `1024 → 4096 → 4096`. With GELU and bias, what fraction of LLaVA-13B does it represent?

2. Construct a LLaVA prompt for a "refusal" case — image contains a private individual. Напишите expected assistant response. Why should LLaVA refuse this zero-shot and what training data would be needed to reinforce the refusal?

3. Прочитайте AnyRes section of the LLaVA-NeXT blog. Вычислите visual token count for a 1344x672 image at AnyRes. Compare to base 576 tokens at 336x336.

4. LLaVA stage-1 projector trained with LM loss on captions. Что произойдет, если skip stage 1 and go straight to stage 2 (visual instruction tuning)? Cite the Prismatic VLMs ablation (arXiv:2402.07865) for the answer.

5. LLaVA-Instruct-150k uses GPT-4 with COCO captions to generate instructions. Для нового domain (medical X-rays, satellite imagery) опишите four-step data pipeline to generate domain instructions. What could go wrong at each step?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Projector | "MLP bridge" | 2-layer MLP with GELU mapping ViT dim to LLM dim |
| Image token | "<image> placeholder" | Prompt marker, заменяемый N projected visual tokens before inference |
| Visual instruction tuning | "LLaVA stage 2" | Training on GPT-4-generated (image, instruction, response) triplets |
| Stage 1 alignment | "Projector pretraining" | Freeze ViT and LLM, train projector with LM loss on captions |
| AnyRes | "Multi-crop tiling" | Split high-res image into a tile grid and concatenate each tile's visual tokens |
| LLaVA-Instruct | "GPT-4-generated" | 158k instruction-response pairs synthesized from COCO captions + GPT-4 |
| Vision encoder freeze | "Backbone locked" | CLIP weights do not update in stage 1, sometimes not in stage 2 either |
| ShareGPT4V | "Better captions" | 1M dense captions generated by GPT-4V, used for higher-quality alignment |
| VQA | "Visual question answering" | Task of answering a free-form question about an image |
| Prismatic VLMs | "Design-space paper" | Karamcheti 2024 ablation systematically testing projector and data choices |

## Дополнительное чтение

- [Liu et al. — Visual Instruction Tuning (arXiv:2304.08485)](https://arxiv.org/abs/2304.08485) — the LLaVA paper.
- [Liu et al. — Improved Baselines with Visual Instruction Tuning (arXiv:2310.03744)](https://arxiv.org/abs/2310.03744) — LLaVA-1.5.
- [Chen et al. — ShareGPT4V (arXiv:2311.12793)](https://arxiv.org/abs/2311.12793) — dense captions dataset.
- [Karamcheti et al. — Prismatic VLMs (arXiv:2402.07865)](https://arxiv.org/abs/2402.07865) — design-space ablations.
- [Li et al. — LLaVA-OneVision (arXiv:2408.03326)](https://arxiv.org/abs/2408.03326) — unified single-image, multi-image, video.
