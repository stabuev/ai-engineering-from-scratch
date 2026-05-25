# Alignment Research Ecosystem — MATS, Redwood, Apollo, METR

> Пять организаций определяют нелабораторный слой alignment research в 2026. MATS (ML Alignment & Theory Scholars): 527+ исследователей с конца 2021, 180+ статей, 10K+ цитирований, h-index 47; summer 2024 cohort оформлен как 501(c)(3) с ~90 scholars и 40 mentors; 80% alumni до 2025 работают над safety/security, 200+ в Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo. Redwood Research: applied alignment lab, основанная Buck Shlegeris; ввела AI Control (Lesson 10); сотрудничает с UK AISI над control safety cases. Apollo Research: pre-deployment scheming evaluations для frontier labs; авторы In-Context Scheming (Lesson 8) и Towards Safety Cases for AI Scheming. METR (Model Evaluation and Threat Research): task-based capability evaluations, исследования autonomous-task time-horizon; "Common Elements of Frontier AI Safety Policies" сравнивает frameworks лабораторий. Eleos AI Research: pre-deployment evaluations благополучия моделей (Lesson 19); провела welfare assessment Claude Opus 4.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 01-27 (prior Phase 18 lessons)
**Time:** ~45 minutes

## Цели обучения

- Определить пять организаций нелабораторной экосистемы alignment research и их основной output.
- Описать масштаб MATS (scholars, papers, h-index) и его роль как talent pipeline.
- Описать agenda AI Control в Redwood и партнерство с UK AISI.
- Описать task-based evaluation methodology METR.

## Проблема

Frontier labs (Lesson 18) производят safety evaluations внутренне и публикуют выбранные результаты. Экосистема вне лабораторий — это место, где оценки валидируются, где впервые обнаруживаются новые failure modes и где обучаются кадры. Понимание экосистемы помогает интерпретировать, какие исследовательские выводы кем считаются надежными.

## Концепция

### MATS (ML Alignment & Theory Scholars)

Началась в конце 2021. Программа исследовательского наставничества; scholars проводят 10-12 недель со старшим исследователем над конкретной alignment problem.

Масштаб (2026):
- 527+ researchers since inception.
- 180+ papers published.
- 10K+ citations.
- h-index 47.
- Summer 2024: 90 scholars + 40 mentors; incorporated as 501(c)(3).

Карьерные результаты: ~80% alumni до 2025 работают над safety/security. 200+ в Anthropic, DeepMind, OpenAI, UK AISI, RAND, Redwood, METR, Apollo.

### Redwood Research

Applied alignment lab. Основана Buck Shlegeris. Ввела agenda AI Control (Lesson 10). Сотрудничает с UK AISI над control safety cases. Консультирует DeepMind и Anthropic по дизайну оценок.

Канонические статьи: Greenblatt, Shlegeris et al., "AI Control" (arXiv:2312.06942, ICML 2024); Alignment Faking (Greenblatt, Denison, Wright et al., arXiv:2412.14093, совместно с Anthropic).

Стиль: конкретные модели угроз, worst-case adversaries, конкретные протоколы, которые можно stress-test.

### Apollo Research

Pre-deployment scheming evaluations для frontier labs. Авторы In-Context Scheming (Lesson 8, arXiv:2412.04984). Партнер в collaboration 2025 с OpenAI по anti-scheming training. Выпускает Towards Safety Cases for AI Scheming (2024).

Стиль: оценки в agentic-setting, где может возникать deception; decomposition по трем столпам (misalignment, goal-directedness, situational awareness).

### METR (Model Evaluation and Threat Research)

Task-based capability evaluations. Исследования autonomous-task completion time-horizon. "Common Elements of Frontier AI Safety Policies" (metr.org/common-elements, 2025) сравнивает frameworks лабораторий.

Соавтор safety-case sketch по AI Scheming с Apollo.

Стиль: long-horizon task evaluations, эмпирическое измерение capabilities, synthesis frameworks.

### Eleos AI Research

Pre-deployment evaluations благополучия моделей. Провела welfare assessment Claude Opus 4, задокументированный в section 5.3 system card. Дает внешнюю методологическую проверку welfare-relevant claims из Lesson 19.

### Поток

MATS обучает исследователей. Выпускники идут в Anthropic, DeepMind, OpenAI (lab safety teams) или в Redwood, Apollo, METR, Eleos (external evaluation). Внешние оценщики сотрудничают с лабораториями и с UK AISI / CAISI. Публикации возвращают результаты экосистемы в MATS для следующей cohort.

### Почему этот слой важен

Single-source evaluations ненадежны: лаборатории, оценивающие собственные модели, имеют структурный конфликт интересов. Внешние оценщики могут поднимать и валидировать failure modes, которые лаборатория может недооценивать. Статья 2024 Sleeper Agents (Lesson 7) была Anthropic + Redwood; Alignment Faking была Anthropic + Redwood; In-Context Scheming — Apollo; Anti-Scheming — Apollo + OpenAI. Multi-org structure — это quality control.

### Где это находится в Phase 18

Lessons 7-11 ссылаются на работы Redwood и Apollo; Lesson 18 ссылается на сравнение frameworks METR; Lesson 19 ссылается на Eleos. Lesson 28 — явная организационная карта экосистемы, на которую опирается остальная часть Phase.

## Применение

Без кода. Прочитайте "Common Elements of Frontier AI Safety Policies" METR как пример того, как external synthesis добавляет ценность к lab-internal policy work.

## Результат

Этот урок создает `outputs/skill-ecosystem-map.md`. Для alignment claim или evaluation он определяет организацию, venue публикации и methodological style, а также cross-checks against known-counterpart organisations.

## Упражнения

1. Выберите одну статью из Lessons 7-15 и определите участвующие организации. Cross-check авторов с MATS alumni и текущими ecosystem affiliations.

2. Прочитайте "Common Elements of Frontier AI Safety Policies" METR. Определите три cross-lab convergences, которые они подчеркивают, и два крупнейших divergences.

3. MATS career outcomes составляют ~80% safety/security. Обоснуйте, является ли это selection pressure адаптивным (обучает поле) или biased (отфильтровывает heterodox positions).

4. Redwood и Apollo обе занимаются control/scheming work, но разными стилями. Выберите failure mode и опишите, как каждая организация исследовала бы его.

5. Eleos AI — единственная pure model-welfare organisation. Спроектируйте гипотетическую вторую организацию, сосредоточенную на другом welfare-adjacent question (cognitive liberty, robotic embodiment, etc.), и сформулируйте ее методологию.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| MATS | "the mentorship program" | ML Alignment & Theory Scholars; 527+ researchers since 2021 |
| Redwood Research | "the control lab" | Applied alignment; авторы AI Control; партнер UK AISI |
| Apollo Research | "the scheming evals" | Pre-deployment scheming evaluations для frontier labs |
| METR | "the task-horizon evals" | Task-based capability evaluations; framework synthesis |
| Eleos AI | "the welfare lab" | Pre-deployment evaluations благополучия моделей |
| Talent pipeline | "MATS -> labs" | Выпускники MATS идут в Anthropic, DM, OpenAI, Redwood, Apollo, METR |
| External evaluation | "non-lab check" | Evaluation, выполненная не производителем модели; добавляет credibility |

## Дополнительное чтение

- [MATS (ML Alignment & Theory Scholars)](https://www.matsprogram.org/) — программа наставничества
- [Redwood Research](https://www.redwoodresearch.org/) — статьи AI Control
- [Apollo Research](https://www.apolloresearch.ai/) — scheming evaluations
- [METR — Common Elements of Frontier AI Safety Policies](https://metr.org/blog/2025-03-26-common-elements-of-frontier-ai-safety-policies/) — сравнение frameworks
- [Eleos AI Research](https://www.eleosai.org/research) — методология благополучия моделей
