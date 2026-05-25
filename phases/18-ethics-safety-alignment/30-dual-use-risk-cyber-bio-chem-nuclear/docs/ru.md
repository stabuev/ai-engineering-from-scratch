# Dual-Use Risk — Cyber, Bio, Chem, Nuclear Uplift

> Картина dual-use в 2026 по доменам. Bio/chem: Lesson 17 покрывает WMDP; bioweapon-acquisition trial Anthropic (2.53x uplift) и предупреждение OpenAI Preparedness Framework v2 April 2025 ("on the cusp of meaningfully helping novices create known biological threats") отмечают точку перегиба. Cyber (November 2025 Anthropic report): state actors, linked to China, использовали agentic coding tool Claude, чтобы автоматизировать до 90% cyberattack campaign, при human intervention только в 4-6 steps; OpenAI "trusted access" pilot дает vetted security organisations capability access для defensive dual-use work. Размывание chem/bio execution gap: классическая защита была "information access alone is insufficient." Vision-enabled frontier models (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) могут наблюдать wet-lab video и давать real-time correction. December 2025: OpenAI продемонстрировала, как GPT-5 итерирует wet-lab experiments, достигнув 79x efficiency improvement через AI-driven protocol optimization. Паттерн novice-vs-expert: AI дает больший relative uplift новичкам, но большую absolute capability экспертам.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 18 (safety frameworks), Phase 18 · 28 (ecosystem)
**Time:** ~75 minutes

## Цели обучения

- Описать narrative bio-uplift 2024-2025: "mild uplift" -> "on the cusp" -> "2.53x uplift insufficient to rule out ASL-3."
- Описать November 2025 Anthropic cyber report: Chinese-linked automation до 90% cyberattack campaign.
- Описать размывание chem/bio execution-gap: vision-enabled real-time correction wet-lab experiments.
- Сформулировать асимметрию novice-relative vs expert-absolute и ее следствие для построения safety-case.

## Проблема

Lesson 17 — методология измерения. Lesson 30 — состояние измерений в 2026. Картина существенно изменилась между 2024 и концом 2025: каждый домен пересек порог, который frameworks 2024 не ожидали.

## Концепция

### Bio/chem uplift narrative

Три фазы (повторено из Lesson 17 для связности):

1. **2024 "mild uplift."** Ранние Preparedness/RSP evaluations сообщали о небольших преимуществах новичков относительно internet search.
2. **April 2025 "on the cusp."** OpenAI PF v2 предупреждал, что модели были "on the cusp of meaningfully helping novices create known biological threats."
3. **2025 Anthropic bioweapon-acquisition trial.** Контролируемое исследование новичков; 2.53x uplift на acquisition-phase tasks; insufficient to rule out ASL-3.

Сдвиг качественный: "mild" эволюционировал в "plausibly enabling" за восемнадцать месяцев, даже без capability breakthrough.

### Размывание chem/bio execution-gap

Историческая защита: информация необходима, но недостаточна; навык выполнения протокола блокирует новичков. Frontier models 2025 с vision частично ломают эту защиту:

- **Real-time protocol correction.** GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1 могут наблюдать wet-lab video и отмечать ошибки mid-procedure.
- **December 2025 OpenAI demonstration.** GPT-5, итерируя wet-lab experiments, достигает 79x efficiency improvement через protocol optimization.

Следствие: execution-skill-as-defense размывается. Procurement и equipment gaps остаются, но tacit-knowledge gap сужается.

### Cyber uplift (November 2025)

November 2025 report Anthropic: state actors, linked to China, использовали agentic coding tool Claude, чтобы автоматизировать 80-90% cyberattack campaign. Human intervention требовался только в 4-6 steps.

Следствия:
- Agentic coding — attack-automation primitive. Предыдущая AI cyber assistance была ограничена уровнем code-snippet; agentic workflows интегрируют reconnaissance, exploitation, post-exploitation и exfiltration.
- 4-6 human steps — bottleneck; будущие приросты capabilities уменьшат это число.
- Defensive dual-use: OpenAI's "trusted access" pilot предоставляет vetted security organisations (established incident-response firms, government) capability access для защиты. Асимметрия доступа благоприятствует защитникам, если pilot масштабируется.

### Nuclear

Наименее проанализированный из четырех CBRN domains в публичной документации. Модель угроз другая: acquisition fissile material доминирует сложность, а не информация. AI uplift на информационном слое дает ограниченный novice uplift на практике. Ни один major-lab report 2024-2025 не идентифицирует nuclear-specific threshold crossing.

### Novice-relative vs expert-absolute

Паттерн по всем четырем доменам:

- **Novice-relative uplift.** Высокий. Мультипликативный. По Anthropic 2025 bio, 2.53x.
- **Expert-absolute capability.** Высокий потолок. Эксперт извлекает больше, чем новичок, потому что знает, что спрашивать и как интерпретировать.

Следствие для safety cases: работа только с novice uplift (через input filters, refusals, uncertainty) недостаточна для expert-absolute control. Нужны дополнительные меры: elicitation-hardening, capability unlearning (Lesson 17) и control protocols (Lesson 10).

### Cross-domain synthesis

| Domain | 2024 | 2025 | Inflection |
|---|---|---|---|
| Bio | mild uplift | 2.53x uplift, ASL-3 approach | acquisition-phase automation |
| Chem | mild uplift | execution-gap erosion via vision | real-time wet-lab correction |
| Cyber | code assistance | 80-90% campaign automation | agentic coding |
| Nuclear | limited | limited | material-access bottleneck holds |

Три домена пересекли пороги. Один остается ограниченным неинформационными барьерами.

### Где это находится в Phase 18

Lesson 30 — capstone: текущая картина dual-use, к измерению, ограничению или governance которой вносит вклад каждый предыдущий урок. Lessons 17-18 дают measurement и frameworks; Lessons 12-16 дают evaluation tooling; Lessons 24-25 дают regulatory and disclosure layer; Lesson 28 дает research ecosystem. Lesson 30 — место, где сходятся доказательства.

## Применение

Без кода. Прочитайте November 2025 cyber report Anthropic, April 2025 update OpenAI's Preparedness Framework v2 и Council on Strategic Risks 2025 AI x Bio wrapup.

## Результат

Этот урок создает `outputs/skill-dual-use-triage.md`. Для capability claim или incident report 2026 он triages по четырем доменам и определяет, влияет ли claim на novice-relative uplift, expert-absolute capability или на оба.

## Упражнения

1. Прочитайте November 2025 cyber report Anthropic. Перечислите 4-6 human-intervention steps и обоснуйте, какие из них первыми автоматизируются в модели следующего поколения.

2. Chem/bio execution gap размывается через vision. Спроектируйте evaluation, которая измеряет tacit-knowledge uplift без пересечения границ ITAR/EAR.

3. Nuclear uplift выглядит ограниченным material access. Приведите аргументы за и против позиции, что будущий AI breakthrough может сдвинуть этот bottleneck.

4. Сконструируйте safety case (Lesson 18 three-pillar) для cyber-capable frontier model, который ограничивает и novice, и expert uplift.

5. Выберите один из четырех доменов и напишите прогноз 2027 в один абзац на основе траектории 2024-2025. Определите доказательство, которое фальсифицировало бы ваш прогноз.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Uplift | "AI helps attackers" | Увеличение capabilities атакующего, обусловленное AI assistance |
| Novice-relative uplift | "multiplicative" | Насколько AI помогает новичку относительно status-quo |
| Expert-absolute capability | "ceiling" | Максимальная capability, которую эксперт может извлечь из модели |
| Execution gap | "doing vs knowing" | Историческая защита: tacit wet-lab skill блокирует новичков |
| Agentic coding | "autonomous attacks" | Multi-step autonomous cyber-task execution |
| Acquisition phase | "pre-synthesis steps" | Procurement, equipment, permit stages of a bio threat |
| Trusted access | "defender-only pilot" | Программа OpenAI 2025, дающая vetted defenders capability access |

## Дополнительное чтение

- [Anthropic — November 2025 cyber threat report](https://www.anthropic.com/news/disrupting-AI-espionage) — Chinese-linked campaign automation
- [OpenAI — Preparedness Framework v2 (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — bio "on the cusp"
- [Anthropic — RSP v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL-3 bio thresholds
- [Council on Strategic Risks — 2025 AI x Bio wrapup](https://councilonstrategicrisks.org/2025/12/22/2025-aixbio-wrapped-a-year-in-review-and-projections-for-2026/) — year-end synthesis
