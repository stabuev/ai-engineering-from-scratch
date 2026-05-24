# Show-o и unified-модели с дискретной diffusion

> Transfusion смешивает непрерывные и дискретные представления. Show-o (Xie et al., август 2024) идет другим путем: текстовые токены используют causal next-token prediction, токены изображения используют masked discrete diffusion в духе MaskGIT. Оба находятся внутри одного трансформера с hybrid attention mask. Результат объединяет VQA, text-to-image, inpainting и генерацию смешанной модальности на одном backbone, одном токенизаторе на модальность и одной формулировке loss (next-token, расширенный до masked prediction). Этот урок проходит по дизайну Show-o — почему masked discrete diffusion является параллельным, few-step генератором изображений — и сравнивает его с Transfusion и Emu3.

**Тип:** Изучение
**Языки:** Python (stdlib, masked-discrete-diffusion sampler)
**Предварительные требования:** Phase 12 · 13 (Transfusion)
**Время:** ~120 минут

## Цели обучения

- Объяснить masked discrete diffusion: schedule, который равномерно маскирует токены, а затем просит трансформер восстановить их.
- Сравнить parallel image decoding (Show-o, MaskGIT) с autoregressive image decoding (Chameleon, Emu3) по скорости и качеству.
- Назвать три задачи, которые Show-o решает в одном checkpoint: T2I, VQA, image inpainting.
- Выбрать masking schedule (cosine, linear, truncated) и рассуждать о его влиянии на качество samples.

## Проблема

Two-loss training в Transfusion работает, но имеет более сложную динамику — continuous diffusion loss живет в другой числовой шкале, чем discrete NTP loss. Балансировка loss weights превращается в hyperparameter search. Архитектура эффективна, но сложна.

Ответ Show-o: оставить обе модальности дискретными (как Chameleon), но генерировать изображения параллельно через masked discrete diffusion вместо последовательной генерации. Training objective становится единым masked-token-prediction, которое естественно обобщает next-token-prediction.

## Концепция

### Masked discrete diffusion (MaskGIT)

Оригинальный прием Chang et al. (2022) MaskGIT элегантен. Начните с полностью замаскированного изображения (каждый токен — специальный `<MASK>` id). На каждом шаге предскажите все masked tokens параллельно, затем сохраните top-K самых confident predictions и снова замаскируйте остальные. После ~8-16 iterations все токены заполнены. Schedule того, сколько токенов unmask на каждом шаге, настраивается — cosine schedules работают хорошо.

Обучение простое: sample a masking ratio uniformly from [0, 1], применить его к VQ tokens изображения, обучить трансформер восстанавливать замаскированные. Ровно то, что BERT делал для текста, масштабированное на генерацию изображений.

### Show-o: один трансформер, hybrid mask

Show-o помещает MaskGIT внутрь causal-language-model transformer. Attention mask:

- Текстовые токены: causal (стандартная LLM).
- Токены изображения: full bidirectional внутри image block (чтобы masked tokens могли видеть все остальные токены изображения во время prediction).
- Text-to-image: текст attends to prior images, image attends to prior text.

Обучение чередует:
1. Standard NTP on text sequences.
2. T2I samples: text → image with masked image tokens, masked-token-prediction loss.
3. VQA samples: image → text with masked text tokens (really just NTP).

Unified loss — это cross-entropy на `<MASK>` tokens, которая покрывает и text NTP (только последний токен "masked"), и image masked-diffusion (случайное подмножество masked).

### Parallel sampling

Show-o генерирует изображение примерно за ~16 steps вместо ~1000 (autoregressive per token) или ~20 (diffusion). На каждом шаге предсказывает все masked tokens параллельно; фиксирует top-K confident; повторяет.

Сравнение:
- Chameleon / Emu3 (autoregressive over tokens): N_tokens forward passes, обычно 1024-4096 на изображение.
- Transfusion (continuous diffusion): ~20 steps, каждый — полный transformer pass.
- Show-o (masked discrete diffusion): ~16 steps, каждый — полный transformer pass.

Show-o быстрее Chameleon при моделях похожего масштаба и примерно соответствует Transfusion по числу шагов с меньшей стоимостью каждого шага (discrete vocab logits vs continuous MSE loss).

### Задачи в одном checkpoint

Show-o поддерживает четыре задачи на инференсе, выбранные форматом prompt:

- Text generation: стандартный autoregressive text output.
- VQA: image in, text out.
- T2I: text in, image out через masked discrete diffusion.
- Inpainting: image с некоторыми masked tokens, fill in.

Возможность inpainting появляется бесплатно из masked-prediction training. Замаскируйте область VQ-token grid, подайте остальное плюс text prompt, предскажите masked tokens.

### Masking schedule

Schedule того, сколько токенов unmask на каждом шаге, формирует качество. Show-o рекомендует cosine:

```
mask_ratio(t) = cos(pi * t / (2 * T))   # t = 0..T
```

На step 0 все токены masked (ratio 1.0). На step T ни один не masked. Cosine концентрирует массу на mid-range ratios, где prediction наиболее информативен. Linear schedules тоже работают, но быстрее выходят на плато.

### Show-o2

Show-o2 (follow-up 2025, arXiv 2506.15564) масштабирует Show-o: larger LLM base, better tokenizer, improved mask schedule. Тот же architectural pattern.

### Где находится Show-o

В taxonomy 2026 года:

- Discrete tokens + NTP: Chameleon, Emu3. Просто, но медленный инференс.
- Discrete tokens + masked diffusion: Show-o, MaskGIT, LlamaGen, Muse. Parallel sampling, все еще lossy из-за токенизатора.
- Continuous + diffusion: Transfusion, MMDiT, DiT. Максимальное качество, более сложное обучение.
- Continuous + flow matching in a VLM: JanusFlow, InternVL-U. Самые новые.

Выбирайте по задаче: Show-o, когда нужны T2I + inpainting + VQA в одной open model с разумной скоростью; Transfusion, когда качество первично и вы можете позволить себе two-loss plumbing.

## Использование

`code/main.py` симулирует Show-o sampling:

- Игрушечная grid из 16 VQ tokens.
- Mock "transformer", который предсказывает logits на основе prompt и currently-unmasked tokens.
- Parallel masked sampling за 8 steps с cosine schedule.
- Печатает intermediate states (эволюцию mask pattern) и final tokens.

Запустите его и наблюдайте, как mask растворяется шаг за шагом.

## Результат

Этот урок создает `outputs/skill-unified-gen-model-picker.md`. Для продукта, которому нужны и understanding (VQA, captioning), и generation (T2I, inpainting) с ограничением open-weights, он выбирает между Show-o family, Transfusion/MMDiT family и Emu3 / Chameleon family с конкретными trade-offs.

## Упражнения

1. Masked discrete diffusion делает samples примерно за ~16 steps. Почему не за 1? Что ломается, если unmask everything на step 0?

2. Inpainting бесплатен с masked diffusion. Предложите product use case (реальный или гипотетический), где inpainting Show-o превосходит specialist model.

3. Cosine schedule vs linear schedule: проследите число unmasked tokens на каждом step для T=8. Какой более сбалансирован?

4. 512x512 Show-o image — это 1024 tokens. При vocab K=16384 модель выдает 1024 * log2(16384) = 14,336 bits (~1.75 KiB) данных. Stable Diffusion выдает 512*512*24 bits = 6,291,456 bits (~768 KiB) сырых пикселей. Какова compression ratio и какое качество она покупает?

5. Прочитайте LlamaGen (arXiv:2406.06525). Чем class-conditional autoregressive image model LlamaGen отличается от masked approach Show-o?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Masked discrete diffusion | "MaskGIT-style" | Обучение предсказывать masked tokens; на инференсе итеративно unmask самые confident predictions |
| Cosine schedule | "Unmask schedule" | Убывание mask ratio по inference steps; концентрирует рост уверенности в mid-range |
| Parallel decoding | "All tokens at once" | Каждый step предсказывает полную последовательность masked tokens за один forward pass, затем фиксирует top-K |
| Hybrid attention | "Causal + bidirectional" | Mask, causal по текстовым токенам и bidirectional внутри image blocks |
| Inpainting | "Fill-in generation" | Condition on an image with some tokens masked, predict the missing ones; появляется бесплатно из training objective |
| Commitment rate | "Top-K per step" | Сколько токенов объявляется "done" за iteration; управляет trade-off между inference и quality |

## Дополнительное чтение

- [Xie et al. — Show-o (arXiv:2408.12528)](https://arxiv.org/abs/2408.12528)
- [Show-o2 (arXiv:2506.15564)](https://arxiv.org/abs/2506.15564)
- [Chang et al. — MaskGIT (arXiv:2202.04200)](https://arxiv.org/abs/2202.04200)
- [Sun et al. — LlamaGen (arXiv:2406.06525)](https://arxiv.org/abs/2406.06525)
- [Chang et al. — Muse (arXiv:2301.00704)](https://arxiv.org/abs/2301.00704)
