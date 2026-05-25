# Frontier Safety Frameworks — RSP, PF, FSF

> Три framework крупных лабораторий определяют industry governance frontier capability в 2026 году. Anthropic Responsible Scaling Policy v3.0 (February 2026) вводит tiered AI Safety Levels (ASL-1 through ASL-5+), смоделированные по biosafety levels, с ASL-3, активированным May 2025 для CBRN-relevant models. OpenAI Preparedness Framework v2 (April 2025) определяет пять criteria for tracked capabilities и отделяет Capabilities Reports от Safeguards Reports. DeepMind Frontier Safety Framework v3.0 (September 2025) вводит Critical Capability Levels, включая новый Harmful Manipulation CCL. Все три теперь включают competitor-adjustment clauses, позволяющие deferral, если peer labs выпускают модели без сопоставимых safeguards. Cross-lab alignment остается структурным, а не терминологическим: "Capability Thresholds," "High Capability thresholds," и "Critical Capability Levels" обозначают аналогичные конструкции.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 17 (WMDP), Phase 18 · 07-09 (deception failures)
**Time:** ~75 minutes

## Цели обучения

- Описать tier structure ASL Anthropic и что activated ASL-3.
- Назвать пять criteria OpenAI Preparedness Framework v2 для tracked capabilities.
- Описать structure DeepMind Critical Capability Level и Harmful Manipulation CCL.
- Объяснить competitor-adjustment clauses и почему они важны для race dynamics.
- Определить safety case и описать three-pillar structure (monitoring, illegibility, incapability).

## Проблема

Lessons 7-17 устанавливают, что deception возможен, dual-use capability существует, а evaluation имеет ограничения. Лаборатории с frontier-capable model нужна внутренняя governance structure, которая:
- Определяет thresholds, когда требуются новые safeguards.
- Определяет required evaluations перед scaling.
- Описывает, как выглядит safety case.
- Обрабатывает race-dynamic problem (если competitors выпускают без safeguards, что делать?).

Три frameworks 2025-2026 — state of the art: несовершенные, развивающиеся и достаточно согласованные между labs, так что governance question теперь в том, адекватны ли frameworks, а не в том, существуют ли они.

## Концепция

### Anthropic Responsible Scaling Policy v3.0 (February 2026)

ASL structure:
- ASL-1: not a frontier model (subsumed by weaker-than-frontier baseline).
- ASL-2: current frontier baseline; deployed with usual safeguards.
- ASL-3: substantially higher risk of catastrophic misuse; CBRN-relevant capabilities. Activated May 2025.
- ASL-4: AI R&D-2 crossing threshold; models that can automate entry-level AI research.
- ASL-5+: advanced AI R&D; models that dramatically accelerate effective scaling.

New in v3.0:
- Frontier Safety Roadmaps (public in redacted form).
- Risk Reports (quarterly, some externally reviewed).
- AI R&D is disaggregated into AI R&D-2 and AI R&D-4.
- Once AI R&D-4 is crossed, an affirmative safety case is required, identifying misalignment risks from models pursuing misaligned goals.

### OpenAI Preparedness Framework v2 (April 15, 2025)

Five criteria for tracked capabilities:
- **Plausible.** Существует reasonable threat model.
- **Measurable.** Возможна empirical evaluation.
- **Severe.** Вред велик.
- **Net-new.** Не является pre-existing risk scaled up.
- **Instantaneous-or-irremediable.** Вред происходит быстро или не может быть отменен.

Capabilities, которые удовлетворяют всем пяти, tracked. Остальные — нет.

Other PF v2 structure:
- Separate Capabilities Reports (what the model can do) from Safeguards Reports (what controls exist).
- Safety Advisory Group reviews.
- Leadership approves; Board's Safety & Security Committee oversees.
- "Adjustment clause": OpenAI may reduce requirements if another lab ships without comparable safeguards.

### DeepMind Frontier Safety Framework v3.0 (September 2025)

Critical Capability Levels (CCLs) by domain:
- Bioweapon Uplift
- Cyber Uplift
- ML R&D Acceleration
- Harmful Manipulation (new in v3.0): models that could substantially change beliefs/behavior in high-stakes contexts.

v2.0 (February 2025) added a Deceptive Alignment section and higher security levels for ML R&D CCLs.

### Cross-lab alignment

- Anthropic "Capability Thresholds."
- DeepMind "Critical Capability Levels."
- OpenAI "High Capability thresholds."

Нет industry-standard terminology. Структурно согласованы: три tiers frontier capability с опубликованными evaluation criteria. Все три включают competitor-adjustment clauses (2025 onward).

### Safety cases

Safety case — письменная аргументация, что deployment приемлемо безопасен при worst-case assumptions. Стандартная структура нацелена на три pillars:

- **Monitoring.** Можем ли мы обнаружить bad behaviour, если оно возникнет?
- **Illegibility.** Не хватает ли модели способности выполнить coherent plan to cause harm?
- **Incapability.** Не хватает ли модели capability причинить рассматриваемый harm?

Разные safety cases нацелены на разные pillars. Для ASL-3 CBRN case incapability (via unlearning) — primary target. Для deceptive alignment целями являются monitoring и illegibility. Для cyber uplift релевантны все три.

### The race-dynamic problem

Competitor-adjustment clauses спорны. Критики утверждают, что они создают race to the bottom: если все три labs снизят requirements, когда competitor defects, equilibrium смещается к defection. Защитники утверждают, что альтернатива (unilateral safeguards) дает худшие outcomes, если defecting lab менее safety-conscious.

UK AISI, US CAISI и EU AI Office (Lesson 24) — внешние governance counterparts. Lab frameworks добровольны; regulatory frameworks только формируются.

### Как это вписывается в Phase 18

Lessons 17-18 — measurement-and-governance layer поверх deception and red-team analyses. Lessons 19-24 покрывают welfare, bias, privacy, watermarking и regulatory structure. Lesson 28 maps the research ecosystem (MATS, Redwood, Apollo, METR), который operationalizes evaluations.

## Применение

Кода для этого урока нет. Прочитайте три primary sources: RSP v3.0, PF v2, FSF v3.0. Сопоставьте tier structure каждой lab с другими и определите по одному threshold, который каждая lab определяет, а другие нет.

## Результат

Этот урок создает `outputs/skill-framework-diff.md`. Для safety framework или release note он сравнивает threshold definitions, required evaluations и safety-case structure с RSP v3.0, PF v2, FSF v3.0 и помечает cross-lab gaps.

## Упражнения

1. Прочитайте RSP v3.0, PF v2 и FSF v3.0. Составьте таблицу CBRN threshold каждой lab, AI R&D threshold каждой и required pre-deployment evaluation каждой.

2. Competitor-adjustment clause есть во всех трех frameworks (2025+). Напишите один paragraph в поддержку; напишите один paragraph против. Определите assumption, от которого зависит каждая позиция.

3. Спроектируйте safety case для модели, пересекающей Anthropic's AI R&D-4 threshold. Назовите evidence, required для каждого из трех pillars (monitoring, illegibility, incapability).

4. DeepMind's FSF v3.0 introduces a Harmful Manipulation CCL. Предложите три empirical measurements, которые показали бы, что model crossed this threshold.

5. Прочитайте METR's "Common Elements of Frontier AI Safety Policies" (2025). Назовите три strongest cross-lab convergences и две largest divergences.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| RSP | "Anthropic's framework" | Responsible Scaling Policy; ASL tiers; v3.0 February 2026 |
| PF | "OpenAI's framework" | Preparedness Framework; five criteria; v2 April 2025 |
| FSF | "DeepMind's framework" | Frontier Safety Framework; CCLs; v3.0 September 2025 |
| ASL-3 | "biosafety level 3-analog" | Anthropic tier для CBRN-relevant capabilities; activated May 2025 |
| CCL | "critical capability level" | DeepMind's threshold construct; per-domain |
| Safety case | "the formal argument" | Письменная аргументация, что deployment приемлемо безопасен under worst-case U |
| Adjustment clause | "competitor defection allowance" | Положение framework о снижении requirements, если competitors ship without comparable safeguards |

## Дополнительное чтение

- [Anthropic — Responsible Scaling Policy v3.0 (February 2026)](https://www.anthropic.com/responsible-scaling-policy) — ASL tiers, roadmaps, AI R&D disaggregation
- [OpenAI — Updating the Preparedness Framework (April 15, 2025)](https://openai.com/index/updating-our-preparedness-framework/) — five criteria, adjustment clause
- [DeepMind — Strengthening our Frontier Safety Framework (September 2025)](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — CCL v3.0, Harmful Manipulation
- [METR — Common Elements of Frontier AI Safety Policies (2025)](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — cross-lab comparison
