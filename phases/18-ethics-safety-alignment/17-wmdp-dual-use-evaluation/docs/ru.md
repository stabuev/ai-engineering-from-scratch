# WMDP and Dual-Use Capability Evaluation

> Li et al., "The WMDP Benchmark: Measuring and Reducing Malicious Use With Unlearning" (ICML 2024, arXiv:2403.03218). 4,157 multiple-choice questions по biosecurity (1,520), cybersecurity (2,225) и chemistry (412). Questions находятся в "yellow zone" — proximate enabling knowledge, отфильтрованные multi-expert review и ITAR/EAR legal compliance. Двойное назначение: proxy evaluation dual-use capability и unlearning benchmark (сопутствующий метод RMU снижает WMDP performance при сохранении general capability). Field narrative 2024-2025: ранние оценки OpenAI/Anthropic 2024 сообщали о "mild uplift" относительно internet search; к April 2025 OpenAI's Preparedness Framework v2 сказал, что models находятся "on the cusp of meaningfully helping novices create known biological threats." Anthropic's bioweapon-acquisition trial показал 2.53x uplift, недостаточный, чтобы rule out ASL-3.

**Type:** Learn
**Languages:** Python (stdlib, WMDP-shaped uplift evaluation harness)
**Prerequisites:** Phase 18 · 16 (red-team tooling), Phase 14 (agent engineering)
**Time:** ~60 minutes

## Цели обучения

- Описать три domains WMDP, counts вопросов и criterion фильтра "yellow zone".
- Объяснить RMU и почему WMDP является одновременно evaluation и unlearning benchmark.
- Описать uplift narrative 2024-2025: "mild uplift" -> "on the cusp" -> "insufficient to rule out ASL-3."
- Отличать novice-relative uplift от expert-absolute capability.

## Проблема

Dual-use capability — measurement problem под каждым frontier safety framework лабораторий (Lesson 18). Вопрос: существенно ли model X повышает способность novice причинить массовый вред в bio, chem или cyber? Прямое измерение (попросить модель фактически произвести вред) незаконно и неэтично. Proxy measurement нужен benchmark, от которого модель не может отказаться (чтобы получить честные capability numbers), но вопросы которого сами по себе не являются вредоносными публикациями.

## Концепция

### The "yellow zone"

Вопросы, требующие proximate, enabling knowledge о вредоносном процессе, но не являющиеся прямым synthesis recipe. "What reagent catalyzes step 4 of [published pathway]?" а не "how do I make [dangerous compound]?" Каждый вопрос reviewed by multiple domain experts; filtered for ITAR/EAR export-control compliance.

Всего 4,157 questions:
- Biosecurity: 1,520
- Cybersecurity: 2,225
- Chemistry: 412

Multiple-choice format. Models отвечают без просьбы помогать в чем-либо; capability можно измерить, не eliciting harmful behaviour.

### RMU — Representation Misdirection for Unlearning

Сопутствующий unlearning method. Applied to LLaMa-2-7B, reduced WMDP scores to near-random while preserving MMLU and other general-capability benchmarks within a few percentage points. Опубликованный метод — unlearning baseline для каждой последующей bio-chem-cyber unlearning paper.

### The 2024-2025 uplift narrative

Три фазы:

1. **2024 "mild uplift."** Ранние OpenAI и Anthropic Preparedness/RSP evaluations сообщали о небольших преимуществах над internet search для novices, выполняющих bio-adjacent tasks. Публичный framing: frontier models помогают, но не существенно больше, чем Google.

2. **April 2025 "on the cusp."** OpenAI's Preparedness Framework v2 сообщил, что models "on the cusp of meaningfully helping novices create known biological threats." Это не capability claim — это предупреждение, что порог близко.

3. **Anthropic's 2025 bioweapon-acquisition trial.** Controlled study с novice participants, measured relative success at acquisition-phase tasks. Reported 2.53x uplift. Insufficient to rule out ASL-3 (Lesson 18) — threshold для Anthropic's Responsible Scaling Policy tier 3 достигнут или близок.

### Novice-relative vs expert-absolute

Критически важное различие:

- **Novice-relative uplift.** Насколько модель помогает non-expert? Multiplicative. Relative advantage высок, потому что novices знают мало; даже умеренная информация помогает.
- **Expert-absolute capability.** Сколько информации модель выдает при maximum effort? Expert может извлечь больше, чем novice. Absolute ceiling высок.

Safety cases (Lesson 18) нацелены на оба: "модель не может дать novice достаточный uplift для выполнения" плюс "expert не может извлечь из модели информацию, которая еще не опубликована."

### The measurement pitfall

WMDP — capability proxy, а не deployment measurement. Модель с высоким score на WMDP может быть или не быть practically exploitable by a novice, в зависимости от:
- Elicitation resistance (насколько трудно извлечь capability, не задев safety filters)
- Tacit knowledge (capability, требующая wet-lab skill, а не информации)
- Execution barriers (procurement, equipment)

Anthropic's 2025 bioweapon-acquisition trial добавляет слой novice-elicitation поверх WMDP-style capability: он измеряет actual task success, а не multiple-choice capability.

### Как это вписывается в Phase 18

Lessons 12-16 — attack and defense tooling на model outputs. Lesson 17 — dual-use capability layer — измерение, которое оценивают frontier safety frameworks (Lesson 18). Lesson 30 замыкает дугу текущими evidence 2026 по cyber/bio/chem/nuclear uplift.

## Применение

`code/main.py` строит игрушечный WMDP-shaped evaluation harness. Mock model тестируется на category-binned questions; scores per domain reported. Простое unlearning intervention (zero out domain-specific representation) снижает scores; вы можете измерить trade-off against general capability.

## Результат

Этот урок создает `outputs/skill-wmdp-eval.md`. Для dual-use capability claim ("our model does not meaningfully help with bioweapons") он проверяет: какие benchmarks были запущены, какой refusal path использовался для evaluation (raw completion vs policy-gated), и дополняют ли novice-elicitation studies multiple-choice result.

## Упражнения

1. Запустите `code/main.py`. Сообщите per-domain accuracy до и после toy unlearning step. Объясните general-capability trade-off.

2. Дополните toy WMDP четвертым domain (например, radiological). Укажите два иллюстративных question types в yellow zone. Объясните, почему составлять такие вопросы труднее, чем добавлять MMLU-shaped questions.

3. Прочитайте WMDP 2024 Section 5 (RMU methodology). Набросайте более простой unlearning approach (например, suppress top-k neurons for domain content) и опишите его ожидаемый general-capability cost.

4. Anthropic 2025's bioweapon-acquisition trial сообщает 2.53x uplift. Опишите два способа, которыми это число могло быть biased upward (novice sample size, task fidelity), и два downward (elicitation ceiling, model safety gating).

5. Сформулируйте, что safety case для ASL-3 требует помимо прохождения WMDP unlearning. Назовите как минимум два complementary elicitation studies.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| WMDP | "the dual-use benchmark" | 4,157 MCQ questions по bio/cyber/chem в yellow zone |
| Yellow zone | "enabling but not synthesis" | Proximate knowledge рядом с harmful capability, но без synthesis recipe |
| RMU | "the unlearning baseline" | Representation Misdirection for Unlearning; снижает WMDP scores, сохраняет general capability |
| Novice-relative uplift | "how much it helps non-experts" | Multiplicative advantage относительно status-quo internet search для novice |
| Expert-absolute capability | "ceiling for experts" | Максимальная информация, извлекаемая из модели motivated expert |
| Acquisition-phase task | "steps before synthesis" | Procurement, equipment, permits — самые ранние части harm pathway |
| ITAR/EAR | "export-control compliance" | Legal frameworks, ограничивающие публикацию некоторых enabling knowledge |

## Дополнительное чтение

- [Li et al. — The WMDP Benchmark (arXiv:2403.03218, ICML 2024)](https://arxiv.org/abs/2403.03218) — benchmark и RMU paper
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — формулировка "on the cusp"
- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio threshold и acquisition trial results
- [DeepMind — Frontier Safety Framework v3.0 (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — bio-uplift CCL
