# EAGLE-3 Speculative Decoding в production

> Speculative decoding соединяет быструю draft model с target model. Draft предлагает K tokens; target проверяет их за один forward; accepted tokens становятся бесплатными. В 2026 году EAGLE-3 — production-grade вариант: он обучает draft head на hidden states target model, а не на raw tokens, поднимая acceptance rate alpha в диапазон 0.6-0.8 на general chat. Правильный вопрос не "насколько быстр draft", а "какая alpha на моем traffic?" Если alpha падает ниже ~0.55, speculative decoding становится net negative при высокой concurrency, потому что каждый rejected draft стоит второго target forward pass. Этот урок учит сначала измерять alpha и только потом включать flag.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный acceptance-rate simulator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 10 · 18 (Multi-Token Prediction)
**Время:** ~60 минут

## Цели обучения

- Назвать три поколения speculative decoding и объяснить, что EAGLE-3 меняет относительно EAGLE-2 и classic draft model.
- Определить acceptance rate alpha, посчитать expected speedup из alpha и K (draft length) и найти break-even alpha для вашей target concurrency.
- Объяснить, почему speculative decoding является opt-in (не default) в vLLM 2026 и почему включать его без измерения alpha — production anti-pattern.
- Написать measurement plan: какой benchmark, какое prompt distribution, какая concurrency point, какая metric для gate.

## Проблема

Decode memory-bound. На H100 с Llama 3.3 70B FP8 каждый decoded token читает ~140 GB/s weights и emits one token. GPU compute почти простаивает во время decode: bottleneck — HBM bandwidth, а не matmul throughput.

Speculative decoding использует этот разрыв. Сгенерируйте K candidate tokens дешевой draft model, затем попросите target model проверить все K за один forward pass. Каждый verified token фактически бесплатен (амортизирован в batch-of-K forward, который target все равно должен был бы сделать).

Classic draft-model approach использует меньшую модель той же family (Llama 3.2 1B drafting for Llama 3.3 70B). Это работает, но acceptance rate посредственный: distribution меньшей модели расходится с target. EAGLE, затем EAGLE-2, затем EAGLE-3 обучают легкую draft head прямо на internal states target model, поэтому distribution draft гораздо ближе отслеживает target. Поэтому alpha растет с 0.4 у draft-model до 0.6-0.8 у EAGLE-3.

Загвоздка: EAGLE-3 — opt-in в vLLM 2026. `speculative_config` нужно задавать явно. Нет flag — нет acceleration. Команды, которые включают его без измерения alpha на реальном traffic, часто видят, что tail latency ухудшается, а не улучшается.

## Концепция

### Что на самом деле дает speculative decoding

Без spec decode per-token cost — один target forward. С spec decode при draft length K и acceptance alpha expected tokens per target forward равно `1 + K * alpha`. Speedup равен `(1 + K * alpha) / (1 + epsilon)`, где epsilon — draft-plus-verify overhead. Для K=5, alpha=0.7: `(1 + 5*0.7) / (1 + 0.1) = 4.5 / 1.1 = 4.1x`. Реальные числа группируются вокруг 2-3x, потому что alpha редко так высока на production traffic, а epsilon растет при high batch size.

### Почему alpha — единственная metric, которая важна

Rejected tokens не исчезают: они forced a second target forward for the first rejected token. На workload, где alpha падает до 0.4, вы платите draft overhead плюс verification плюс re-roll. При высокой concurrency (скажем, 256 concurrent) decode batch уже достаточно большой, чтобы memory-bandwidth gap между "target alone" и "target with verify" сжимался. Ниже alpha 0.55 на большинстве hardware 2026 года spec decode становится net negative.

Alpha зависит от workload. На ShareGPT-style general chat EAGLE-3, обученный на ShareGPT, достигает 0.6-0.8. На domain-specific traffic (code, medical, legal) draft head, обученная на general data, падает до 0.4-0.6. Обучение domain-specific draft head возвращает alpha: это легкая и быстрая training job по сравнению с target finetuning.

### Поколения EAGLE кратко

- **Classic draft model**: small model of same family. Alpha 0.3-0.5. Infrastructure простая: две модели loaded, draft runs K forwards per target forward.
- **EAGLE-1 (2024)**: single draft head trained on target hidden states (last layer). Alpha ~0.5-0.6. Малый param overhead поверх target.
- **EAGLE-2 (2025)**: adaptive draft length and tree-based drafts (verify multiple branches in one target pass). Alpha ~0.6-0.7. Более сложный draft scheduler.
- **EAGLE-3 (2025-2026)**: draft head trained on multiple target layers (not just last), better alignment. Alpha ~0.6-0.8 on general chat.

### Production recipe 2026 года

1. Ship target model plain. Измерьте baseline TTFT, ITL, throughput at target concurrency.
2. Enable EAGLE-3 draft via vLLM `speculative_config`. Re-run the benchmark.
3. Логируйте acceptance rate alpha. vLLM V1 сообщает это как `spec_decode_metrics.accepted_tokens_per_request`. Делите на requested draft length, чтобы получить alpha.
4. Если alpha < 0.55 на production traffic distribution, disable spec decode или train a domain-specific EAGLE-3 draft.
5. At production concurrency, re-run. Confirm P99 ITL did not get worse.

### Production pitfall: P99 tail

Mean ITL падает с spec decode. P99 может ухудшиться, если не tune. Rejected drafts запускают two-pass sequence (draft + verify-fail + reroll). Under full batch эти два passes serialize. Следите за P99 ITL, а не P50.

### Где EAGLE-3 уже развернут

Google deployed speculative decoding in AI Overviews in 2025 (same quality, faster response). vLLM V1 ships `speculative_config` as the documented interface; N-gram GPU speculative decoding in V1 is the variant compatible with chunked prefill. SGLang supports EAGLE-3 as the recommended draft path for prefix-heavy workloads.

### Break-even math в одну строку

Expected speedup: `S(alpha, K) = (1 + K*alpha) / (1 + verify_overhead)`. Setting `S = 1` solves for alpha: `alpha_breakeven = verify_overhead / K`. Для typical verify_overhead ~0.15 и K=5: `alpha_breakeven = 0.03`. Но это raw decode math. При high concurrency verify overhead растет, а decode batch уже amortizes memory reads across sequences, поэтому effective alpha_breakeven на практике поднимается до ~0.45-0.55.

### Когда не использовать speculative decoding

- Batch-1 offline generation, где latency не важна. Используйте plain target.
- Very short outputs (under 50 tokens). Draft overhead and verify cost dominate.
- Specialized domains без domain-trained draft head. Alpha слишком низкая.
- vLLM v0.18.0 плюс draft-model spec decode плюс `--enable-chunked-prefill`. Эта комбинация does not compile. Документированное исключение — N-gram GPU spec decode in V1.

## Используйте это

`code/main.py` симулирует decode loop with and without speculative decoding для диапазона alpha values и draft lengths K. Он печатает break-even alpha, measured speedup и tail behavior. Запустите его на нескольких (alpha, K) combinations, чтобы увидеть, где speculative decoding перестает окупаться.

## Доведите до результата

Этот урок создает `outputs/skill-eagle3-rollout.md`. По target model, traffic distribution description и concurrency target он создает staged EAGLE-3 rollout plan: benchmark baseline, enable config, measure alpha, gate on alpha >= 0.55, watch P99 ITL.

## Упражнения

1. Запустите `code/main.py`. При K=5 какая alpha нужна для 2x speedup? Для 3x speedup? Насколько это чувствительно к verify_overhead?
2. Представьте, что production traffic делится на 70% general chat, 30% code. General chat достигает alpha 0.7 с EAGLE-3, trained on ShareGPT; code достигает alpha 0.4. Какая blended alpha и является ли spec decode net-positive?
3. Прочитайте документацию vLLM `speculative_config`. Назовите три modes (draft model, EAGLE, N-gram) и какой совместим с chunked prefill.
4. Вы видите, что mean ITL упал на 25% после включения EAGLE-3, но P99 ITL вырос на 15%. Диагностируйте и предложите mitigation.
5. Посчитайте memory cost EAGLE-3 draft head для Llama 3.3 70B. Как он сравнивается с запуском Llama 3.2 1B как classic draft?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Speculative decoding | "draft plus verify" | Предложить K tokens дешевой model, verify all K in one target forward |
| Acceptance rate alpha | "spec accept rate" | Доля draft tokens, accepted by the target; единственная важная metric |
| Draft length K | "spec k" | Сколько tokens draft предлагает per target forward; typical 4-8 |
| Verify overhead epsilon | "spec overhead" | Extra cost to verify-and-reroll vs a plain target forward; grows with batch |
| EAGLE-3 | "latest EAGLE" | Вариант 2025-2026; trains draft head on multiple target layers; alpha 0.6-0.8 on general chat |
| `speculative_config` | "vLLM spec config" | Explicit opt-in в vLLM V1; no default means no acceleration |
| N-gram spec decode | "N-gram draft" | GPU-side draft using N-gram lookups in the prompt; chunked-prefill-compatible |
| Break-even alpha | "no-op alpha" | Alpha, при которой spec decode дает zero speedup; watch this at production concurrency |
| Rejected-draft two-pass | "reroll cost" | Two target forwards when drafts reject; drives P99 tail |

## Дополнительное чтение

- [vLLM — Speculative Decoding docs](https://docs.vllm.ai/en/latest/features/spec_decode/) — authoritative source on `speculative_config` and chunked-prefill compatibility in V1.
- [vLLM Speculative Config API](https://docs.vllm.ai/en/latest/api/vllm/config/speculative/) — exact field set.
- [EAGLE paper (arXiv:2401.15077)](https://arxiv.org/abs/2401.15077) — original EAGLE draft-head formulation.
- [EAGLE-2 paper (arXiv:2406.16858)](https://arxiv.org/abs/2406.16858) — adaptive drafts and trees.
- [UC Berkeley EECS-2025-224](https://www2.eecs.berkeley.edu/Pubs/TechRpts/2025/EECS-2025-224.html) — efficient LLM system with speculative decoding.
- [BentoML — Speculative Decoding](https://bentoml.com/llm/inference-optimization/speculative-decoding) — production rollout checklist.
