# OpenAI Preparedness Framework и DeepMind Frontier Safety Framework

> OpenAI Preparedness Framework v2 (апрель 2025) вводит Research Categories — Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards — отдельно от Tracked Categories. Tracked Categories запускают Capabilities Reports и Safeguards Reports, проверяемые Safety Advisory Group. FSF v3 от DeepMind (сентябрь 2025, Tracked Capability Levels добавлены 17 апреля 2026) встраивает autonomy в домены ML R&D и Cyber (ML R&D autonomy level 1 = полностью автоматизировать AI R&D pipeline при конкурентной стоимости относительно human + AI tools). FSF v3 явно рассматривает deceptive alignment через automated monitoring для misuse на основе instrumental reasoning. Честная оговорка: Research Categories в PF v2 (включая Long-range Autonomy) не запускают mitigations автоматически; язык политики - "potential." Сама DeepMind говорит, что automated monitoring "will not remain sufficient long-term", если instrumental reasoning усилится.

**Тип:** Изучение
**Языки:** Python (stdlib, инструмент diff decision-table для трех frameworks)
**Предварительные требования:** Phase 15 · 19 (Anthropic RSP)
**Время:** ~45 минут

## Цели обучения

- Сравнивать исследовательские категории OpenAI Preparedness Framework v2 с Frontier Safety Framework от DeepMind.
- Объяснять пороги возможностей и смягчения, которые они запускают.
- Рассуждать о согласованности frontier-safety политик между лабораториями.

## Проблема

Lesson 19 внимательно прочитал scaling policy Anthropic. Этот урок завершает картину чтением документов OpenAI и DeepMind. Три документа - родственные артефакты, отвечающие на один вопрос: когда frontier lab должна pause или gate модель. Они сходятся на небольшом наборе категорий и расходятся в конкретных местах, которые важны.

Схождение: все три называют long-range autonomy классом capabilities, который стоит отслеживать. Все три признают deceptive behavior (alignment faking, sandbagging) как отдельный класс риска. У всех трех есть internal review body. Расхождение: OpenAI делит категории на "Tracked" (обязательная mitigation) и "Research" (без автоматического trigger). DeepMind встраивает autonomy в два домена, а не называет ее отдельно. Лаборатории называют это Tracked vs Research, Critical vs Moderate или Tier-1 vs Tier-2; операционное последствие того, в каком bucket находится capability, различается между лабораториями.

Полезное упражнение - читать их вместе. Одна и та же capability может быть "mandatory mitigation" у Anthropic, "monitored but not triggering" у OpenAI и "tracked in a specific domain" у DeepMind. Policy posture имеет значение.

## Концепция

### OpenAI Preparedness Framework v2 (апрель 2025)

Структура:

- **Tracked Categories**: запускают Capabilities Reports (что модель умеет) и Safeguards Reports (какие mitigations установлены). Проверяются Safety Advisory Group перед deployment.
- **Research Categories**: capabilities модели, которые лаборатория отслеживает, но пока не обязуется к конкретным mitigations. Включают Long-range Autonomy, Sandbagging, Autonomous Replication and Adaptation, Undermining Safeguards.

Ключевой framing в v2: Research Categories не запускают mitigations автоматически. Язык политики - "potential" mitigations. Это конкретная delta, за которой нужно следить: расположение категории в Tracked или Research - шарнир между "gated" и "observed."

### DeepMind Frontier Safety Framework v3 (сентябрь 2025; Tracked Capability Levels добавлены 17 апреля 2026)

Структура:

- **Critical Capability Levels (CCLs)**: capability thresholds по пяти доменам: Cyber, Bio, ML R&D, CBRN, Autonomy (встроена в ML R&D и Cyber).
- **Tracked Capability Levels**: дополнительная гранулярность, добавленная в апреле 2026. Конкретный пример: ML R&D autonomy level 1 = полностью автоматизировать AI R&D pipeline при конкурентной стоимости относительно human + AI tools.
- **Deceptive alignment monitoring**: явное обязательство к automated monitoring для misuse на основе instrumental reasoning.

Framing autonomy отличается от OpenAI. DeepMind не оставляет "Autonomy" top-level domain; она встроена в домены, где autonomy может причинить вред (ML R&D и Cyber). Аргумент: autonomy без домена - capability без риска; контраргумент: autonomy across domains - meta-risk, который framework должен называть.

### В чем сходятся все три

- Internal Safety Advisory Group (Anthropic SAG, OpenAI SAG, DeepMind internal committee). Review перед deployment для high-capability models.
- Явное упоминание deceptive alignment / alignment faking как класса риска.
- Постоянные артефакты по заявленной cadence (Anthropic: Frontier Safety Roadmap, Risk Report; OpenAI: Capabilities and Safeguards Reports; DeepMind: FSF update cycle).
- Признание, что monitoring-only defenses имеют потолок. DeepMind говорит явно: "automated monitoring will not remain sufficient long-term."

### Где они расходятся

- **Anthropic**: pause commitment удалено в v3.0; AI R&D-4 threshold - named next gate.
- **OpenAI**: разделение Tracked vs Research; Research Categories (включая Long-range Autonomy) не gate автоматически.
- **DeepMind**: autonomy встроена в другие домены; Tracked Capability Levels добавляют гранулярность в апреле 2026.

### Sandbagging: конкретная capability, усложняющая все три

Sandbagging (стратегическое занижение результатов моделью на evaluations) находится в Research Categories OpenAI. RSP v3.0 Anthropic рассматривает его через evaluation-context gap (Lesson 1). DeepMind рассматривает его через deceptive alignment monitoring в FSF v3.

Если модель sandbags на evaluations, capability thresholds любого framework занижаются. Framework работает только если measurement работает. Поэтому external measurement (Lesson 21, METR) и adversarial evaluation необходимы в дополнение к lab self-assessment.

### Навык чтения policy

- Locate: каждая capability, которая вам важна, должна находиться в policy. Если ее нельзя найти, policy ее не покрывает.
- Classify: является ли она Tracked (запускает mitigation) или Research (отслеживается, но не запускает)? OpenAI называет это явно; у Anthropic и DeepMind свои эквиваленты.
- Cadence: обновляется ли policy по заявленному расписанию или только после конкретных событий? Заявленная cadence сильнее.
- Independence: является ли external review обязательным или optional? Anthropic сотрудничает с Apollo и US AI Safety Institute; OpenAI с METR; DeepMind главным образом с internal SAG.

## Использование

`code/main.py` реализует небольшой decision-table diff tool. Для заданной capability (autonomy, deceptive alignment, R&D automation, cyber uplift и т. д.) он выводит, как каждая из трех policies классифицирует capability и какие mitigations запускаются. Это вспомогательный инструмент для чтения, а не policy tool.

## Практический результат

`outputs/skill-cross-policy-diff.md` создает cross-policy comparison для конкретной capability, используя три frameworks как reference.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что output diff tool совпадает с policies как минимум для двух capabilities, которые вы можете проверить по source documents.

2. Прочитайте OpenAI Preparedness Framework v2 полностью. Найдите каждую Research Category. Для каждой напишите одно предложение о том, почему она находится в Research, а не Tracked.

3. Прочитайте DeepMind FSF v3 полностью, плюс April 2026 Tracked Capability Levels update. Найдите конкретные evaluation criteria для ML R&D autonomy level 1. Как бы вы измерили это externally?

4. Sandbagging находится в Research Categories OpenAI. Спроектируйте evaluation, которая заставила бы sandbagging model раскрыть actual capability. Сошлитесь на обсуждение eval-context-gaming из Lesson 1.

5. Сравните три policies по конкретной capability (на ваш выбор). Назовите, классификацию какой policy вы считаете наиболее строгой, а какой - наименее строгой. Обоснуйте source text.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| Preparedness Framework | "Scaling policy OpenAI" | PF v2 (апрель 2025); категории Tracked и Research |
| Tracked Category | "Обязательная mitigation" | Запускает Capabilities + Safeguards Reports; review SAG |
| Research Category | "Только мониторинг" | Отслеживается, но без automatic mitigation; включает Long-range Autonomy |
| Frontier Safety Framework | "Scaling policy DeepMind" | FSF v3 (сентябрь 2025) + Tracked Capability Levels (апрель 2026) |
| CCL | "Critical Capability Level" | Порог DeepMind по домену (Cyber, Bio, ML R&D, CBRN) |
| ML R&D autonomy level 1 | "Автоматизация R&D" | Полная автоматизация AI R&D pipeline при конкурентной стоимости |
| Sandbagging | "Стратегическое занижение результатов" | Модель занижает результаты на evals; в Research Categories OpenAI |
| Instrumental reasoning | "Рассуждение о средствах и целях" | Рассуждение о том, как достигать целей; target мониторинга DeepMind |

## Дополнительное чтение

- [OpenAI — Updating our Preparedness Framework](https://openai.com/index/updating-our-preparedness-framework/) — announcement v2.
- [OpenAI — Preparedness Framework v2 PDF](https://cdn.openai.com/pdf/18a02b5d-6b67-4cec-ab64-68cdfbddebcd/preparedness-framework-v2.pdf) — полный документ.
- [DeepMind — Strengthening our Frontier Safety Framework](https://deepmind.google/blog/strengthening-our-frontier-safety-framework/) — announcement FSF v3.
- [DeepMind — Updating the Frontier Safety Framework (April 2026)](https://deepmind.google/blog/updating-the-frontier-safety-framework/) — добавление Tracked Capability Levels.
- [Gemini 3 Pro FSF Report](https://storage.googleapis.com/deepmind-media/gemini/gemini_3_pro_fsf_report.pdf) — пример Risk Report в формате FSF.
