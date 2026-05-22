# Multi-Token Prediction (MTP)

> Каждая autoregressive LLM от GPT-2 до Llama 3 обучается на одном loss на позицию: предсказать следующий token. DeepSeek-V3 добавила второй loss на позицию: предсказать token после следующего. Дополнительные 14B parameters (на модели 671B) были distilled обратно в основную модель через gradient flow, а обученные MTP heads на inference переиспользовали как speculative-decoding drafters с acceptance выше 80%. 1.8× generation throughput достался почти бесплатно. Этот урок строит sequential MTP module из технического отчета DeepSeek, считает loss и shared-head parameter layout и объясняет, почему MTP сохраняет causal chain, тогда как исходный parallel MTP Gloeckle et al. ломал его.

**Type:** Build
**Languages:** Python (stdlib)
**Prerequisites:** Phase 10 · 04 (pre-training a mini GPT), Phase 10 · 15 (speculative decoding)
**Time:** ~60 minutes

## Learning Objectives

- Сформулировать MTP training objective и вывести joint loss по prediction depths.
- Объяснить различие между parallel MTP heads Gloeckle et al. (2024) и sequential MTP modules DeepSeek-V3, а также почему sequential design сохраняет causal chain.
- Посчитать parameter и memory overhead от добавления MTP modules в pre-training run.
- Реализовать один MTP module с нуля: shared embedding, per-depth transformer block, projection и shared output head.

## The Problem

Next-token prediction — стандартная training objective для LLM. Каждый hidden state supervised предсказывать ровно одну вещь: непосредственно следующий token. Это удивительно слабый signal. Большая часть информации в sequence выходит за один token — структура, связность, factuality, arithmetic flow. Модель должна выучить это, накапливая много однотокенных сигналов на триллионах tokens.

MTP спрашивает: что если каждый hidden state supervised предсказывать несколько будущих tokens сразу? Gloeckle et al. (Meta, 2024) показали, что это помогает. Их implementation ставила несколько независимых output heads поверх backbone, каждая предсказывала другой offset. Параллельно, просто, но heads видели один и тот же hidden state без иерархического уточнения — и predictions не связывались каузально, поэтому их нельзя было использовать для speculative decoding.

DeepSeek-V3 (декабрь 2024) переработала MTP как sequential modules, сохраняющие causal chain на каждом prediction depth. Модель предсказывает `t+1` из `h_i^(0)`, затем предсказывает `t+2` из нового hidden state `h_i^(1)`, который объединяет `h_i^(0)` с embedding `E(t+1)`, и так далее. Каждый depth — собственный небольшой transformer block. Shared embedding и shared output head удерживают parameter overhead умеренным. В масштабе DeepSeek-V3 это 14B дополнительных parameters в MTP modules поверх 671B main-model weights. Эти 2% overhead дали более плотные training signals и готовый speculative-decoding draft на inference.

Этот урок строит single MTP module и D-depth loss с нуля. Математика аккуратная. Implementation занимает 150 строк.

## The Concept

### The sequential MTP recipe

DeepSeek-V3 добавляет `D` MTP modules поверх main model. Каждый module `k` (для `k = 1..D`) предсказывает token на depth `k` — то есть `t_{i+k}` по prefix до position `i`.

Module `k` состоит из:

- Transformer block `T_k` со своим attention и MLP.
- Projection matrix `M_k`, объединяющей hidden state предыдущего depth с embedding ground-truth token следующего depth.
- Shared embedding `E` (та же, что у main model).
- Shared output head `Out` (та же, что у main model).

На training для prefix до position `i` per-depth hidden state:

```
h_i^(0) = main model backbone at position i
h_i^(k) = T_k( M_k * concat(RMSNorm(h_i^(k-1)), RMSNorm(E(t_{i+k}))) )   for k >= 1
```

Per-depth prediction:

```
logits_{i+k} = Out(h_i^(k-1))   for k = 1..D
```

Per-depth loss — cross-entropy против ground-truth `t_{i+k}`:

```
L_k = CE(logits_{i+k}, t_{i+k})
```

Joint loss по depths:

```
L_MTP = (lambda / D) * sum_{k=1..D} L_k
```

`lambda` — небольшой weighting factor: DeepSeek-V3 использует 0.3 для первых 10% training и 0.1 после этого. Total training loss равен `L_main + L_MTP`.

### Why sequential, not parallel

Исходный parallel MTP Gloeckle имел D output heads, каждая напрямую применялась к `h_i^(0)`. Каждая head предсказывает `t_{i+k}` из одного и того же backbone hidden state. Это обучается нормально, но predictions не conditioned друг на друга. Нельзя использовать output `head_1`, чтобы помочь `head_2` — heads срабатывают параллельно.

Sequential design DeepSeek-V3 строит `h_i^(k)` из `h_i^(k-1)` плюс actual next-token embedding `E(t_{i+k})`. Это сохраняет causal chain: чтобы предсказать `t_{i+k+1}`, module на depth `k+1` видит, что было в `t_{i+k}`. Структурно это идентично тому, как autoregressive decoder потребляет собственный output, что делает MTP modules напрямую пригодными как speculative-decoding drafters.

На inference: подайте `h_i^(k-1)` и drafted `t_{i+k}` в module `k+1`, получите prediction для `t_{i+k+1}`. Повторите. Это ровно EAGLE-style draft, использующий обученный MTP module как draft network. DeepSeek-V3 сообщает acceptance 80%+ на первом MTP module и speedup ~1.8×.

### Parameter accounting

Для модели с hidden `h` и vocabulary `V`:

- Main model: миллиарды parameters плюс один output head размера `V * h`.
- Shared output head: reuse main model's head. Нет extra params.
- Shared embedding: reuse main model's embedding. Нет extra params.
- Per-MTP module:
  - Projection `M_k`: `(2h) * h = 2h^2`.
  - Transformer block `T_k`: attention (`4h^2` для MHA) плюс MLP (обычно `8h^2` для SwiGLU с ratio 8/3). Около `12h^2` на block.

Итого extra per module: `~14h^2`. Для DeepSeek-V3 с `h = 7168`, D = 1 module: `~14 * 7168^2 = ~720M` parameters на бумаге. DeepSeek-V3 сообщает 14B — разница главным образом в том, что expert layers в MTP module тоже MoE.

### The speculative-decoding payoff

Во время pre-training MTP modules замедляют training примерно на 10% (больше forward compute, extra loss). Выигрыш двойной:

1. Более плотный training signal. Каждый hidden state видит D+1 supervision targets. Измеренный эффект на MMLU, GSM8K, MATH, HumanEval: устойчивые улучшения на несколько percentage points в ablations DeepSeek-V3.

2. Бесплатный speculative decoding draft на inference. MTP module уже обучен предсказывать следующие несколько tokens. Переиспользованный как draft network, он дает acceptance rates 80%+. На таком уровне N=3 или N=5 spec decoding дает 1.8× throughput. 10% training-time cost окупается при первом серьезном inference.

### Relation to EAGLE

EAGLE обучает небольшой draft model ОТДЕЛЬНО после pre-training. MTP встраивает draft в pre-training. Два подхода сходятся к похожим accept rates, но через разные pipelines:

| Dimension | EAGLE-3 | MTP (DeepSeek-V3) |
|-----------|---------|------------------|
| When trained | Post-pre-training | During pre-training |
| Backward-compatible with existing weights | Yes | No (need to re-train) |
| Draft params | 1-2 transformer layers | 1 transformer block + projection |
| Acceptance rate | 0.88-0.92 | 0.80+ at depth 1 |
| Benefit beyond speedup | Speculative decoding only | Denser training signal + speedup |

## Build It

`code/main.py` строит single MTP module end to end: shared embedding, projection, transformer block, shared output head. Затем считает per-depth cross-entropy loss на короткой synthetic sequence и печатает parameter count по components. Toy vocabulary из 32 tokens делает числа читаемыми.

### Step 1: shared embedding table

Одна таблица `vocab_size x hidden` используется main model И каждым MTP module на каждом depth. Не вторая копия — буквально тот же tensor.

### Step 2: the per-depth combination

```python
def combine(prev_hidden, next_token_embed, M_k):
    # concat along feature dim, then project down to hidden
    concat = rms_norm(prev_hidden) + rms_norm(next_token_embed)  # vector addition stand-in
    projected = matvec(M_k, concat)
    return projected
```

Настоящий DeepSeek-V3 конкатенирует два RMSNormed vectors в `[2h]` и проецирует матрицей `h x 2h`. Toy использует vector addition ради краткости stdlib.

### Step 3: the transformer block at depth k

Self-attention plus MLP. В toy one-layer linear attention block и SwiGLU MLP сохраняют видимой структуру без numpy.

### Step 4: the shared output head

Reuse main model's output projection. Logits по vocabulary.

### Step 5: per-depth loss

Cross-entropy of softmax(logits) против ground-truth token с offset `k`. Aggregate по depths с scaling factor `lambda / D`.

### Step 6: parameter accounting

Напечатайте total parameter count, shared (embedding, head) count и per-module extra count. Покажите ratio MTP extra к main-model size.

## Use It

MTP интегрирован в DeepSeek-V3 (декабрь 2024) и series DeepSeek-R1. На inference:

- Собственный serving stack DeepSeek использует MTP modules как speculative decoders out of the box.
- vLLM и SGLang имеют integration paths для DeepSeek-V3 MTP по состоянию на апрель 2026.
- AMD ROCm SGLang tutorial показывает конкретный MTP speculative-decoding config с измеренным speedup 1.8× на V3 checkpoint.

Когда использовать MTP в новом pre-training run:

- Вы контролируете полный pre-training pipeline и хотите получить более плотный training signal.
- Вы знаете, что будете serving model at scale, и хотите speculative decoding почти бесплатно.
- Hidden size не меньше 4096. В масштабе 1B overhead болит сильнее, чем помогает gain.

Когда не использовать:

- Fine-tuning существующей pre-trained dense model. MTP module не обучен.
- Research models, где нужен чистый baseline для сравнения. MTP меняет architecture.

## Ship It

Этот урок создает `outputs/skill-mtp-planner.md`. По спецификации pre-training run (model size, data, compute) он возвращает план интеграции MTP: число depths D, schedule для `lambda`, memory overhead и inference-time speculative-decoding wiring.

## Exercises

1. Запустите `code/main.py`. Покажите, что per-depth loss монотонно уменьшается по мере усиления synthetic signal. Измените synthetic так, чтобы использовать fixed pattern, и проверьте, что и depth-1, и depth-2 losses сходятся.

2. Посчитайте parameter overhead для dense 70B model (hidden 8192, 80 layers) с D=1 MTP module. Сравните с заявленным DeepSeek-V3 overhead 14B. Объясните, почему число DeepSeek выше: MTP transformer block наследует ту же MoE structure, раздувая per-module parameter count.

3. Реализуйте D=2 в toy: добавьте второй MTP module, который принимает h^(1) и предсказывает `t_{i+2}`. Проверьте, что joint loss и parameter accounting совпадают с equations 19-21 статьи DeepSeek.

4. Переключите toy на parallel MTP (Gloeckle-style): добавьте D output heads поверх main hidden state, каждая предсказывает другой offset. Измерьте, как losses по depth сравниваются с sequential version на том же synthetic signal. Sequential version должна давать меньший depth-k loss для k > 1, потому что conditioned on intermediate predictions.

5. Используйте обученный MTP module как EAGLE-style draft: вызовите module k, чтобы предложить `t_{i+k}` на inference. Измерьте acceptance rate этих draft tokens против predictions main model на held-out sequence. Если на toy вы получите 50%+, вы воспроизвели эмпирическое свойство MTP-as-draft.

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|------------------------|
| MTP module | "Extra loss block" | Небольшой transformer block плюс projection, предсказывающий token на `k` позиций вперед от main model |
| Prediction depth | "Which offset" | Целое `k`, такое что module `k` предсказывает `t_{i+k}` по prefix до position `i` |
| Parallel MTP | "Gloeckle-style" | D независимых heads на одном backbone hidden state, без conditional chain |
| Sequential MTP | "DeepSeek-V3 style" | Каждый module conditioned on hidden state предыдущего depth плюс embedding следующего token; сохраняет causal chain |
| Shared output head | "Reuse the main head" | MTP modules вызывают LM head main model, а не отдельную output projection |
| Shared embedding | "Reuse the main table" | Одна vocabulary embedding table используется везде; duplicate parameters нет |
| Projection matrix M_k | "Combine hidden + next-token" | Linear layer `h x 2h`, который сворачивает previous hidden state и target-token embedding во input следующего depth |
| Joint loss L_MTP | "Averaged extra losses" | Arithmetic mean per-depth cross-entropy losses, scaled by `lambda` |
| Acceptance rate at depth 1 | "How often MTP draft is right" | Доля случаев, где top-1 prediction D=1 MTP module равна top-1 prediction main model; 80%+ на DeepSeek-V3 |
| Lambda weighting | "Extra-loss importance" | Per-depth scaling factor; 0.3 в начале training, 0.1 позже на DeepSeek-V3 |

## Further Reading

- [DeepSeek-AI — DeepSeek-V3 Technical Report (arXiv:2412.19437)](https://arxiv.org/abs/2412.19437) — полное описание sequential MTP (Section 2.2), включая joint-loss equations и speedup 1.8× на inference
- [Gloeckle et al. — Better & Faster Large Language Models via Multi-token Prediction (arXiv:2404.19737)](https://arxiv.org/abs/2404.19737) — parallel MTP baseline, который улучшает дизайн DeepSeek
- [DeepSeek-V3 model card on Hugging Face](https://huggingface.co/deepseek-ai/DeepSeek-V3) — 685B total (671B main + 14B MTP), deployment notes
- [Leviathan et al. — Fast Inference from Transformers via Speculative Decoding (arXiv:2211.17192)](https://arxiv.org/abs/2211.17192) — speculative-decoding framework, куда вписывается MTP
- [Li et al. — EAGLE-3 (arXiv:2503.01840)](https://arxiv.org/abs/2503.01840) — draft architecture EAGLE 2025 года, counterpart MTP
