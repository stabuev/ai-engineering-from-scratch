# METR Time Horizons и внешняя оценка capabilities

> METR (ex-ARC Evals) является независимой 501(c)(3) с декабря 2023 года. Их benchmark Time Horizon 1.1 (январь 2026) подгоняет logistic curve к вероятности успеха задачи как функции log(expert human completion time); пересечение при 50% probability определяет time horizon модели. Набор engagements 2025–2026 покрывает GPT-5.1, GPT-5.1-Codex-Max и prototype monitoring evaluations (может ли monitor поймать side tasks; может ли agent evade). Benchmark suites: HCAST (180+ задач ML, cyber, SWE, reasoning; от 1 minute до 8+ hours), RE-Bench (71 ML research-engineering tasks с expert baseline), SWAA. Честная оговорка: измерения METR идеализированы - без человека, без реальных последствий - и команда документировала eval-vs-deployment behavior gap (Lesson 1). Time horizon - это upper bound, а не deployment prediction.

**Тип:** Изучение
**Языки:** Python (stdlib, оценщик horizon через logistic-fit)
**Предварительные требования:** Phase 15 · 01 (Long-horizon agents), Phase 15 · 19 (RSP)
**Время:** ~60 минут

## Проблема

Scaling policies (Lessons 19, 20) полезны ровно настолько, насколько полезны измерения, на которые они ссылаются. "AI R&D-4 threshold" и "Long-range Autonomy" определены в тексте policy; они становятся actionable только когда конкретные evaluations дают конкретные числа.

METR - организация external evaluation 2024–2026 годов, определившая многие из этих чисел. Они оценивают frontier models - часто pre-release, по NDA с лабораториями - и затем публикуют methodology. Benchmark Time Horizon 1.1 (январь 2026) - их headline artifact: один scalar, сжимающий capability в понятную людям единицу ("эта модель может делать задачи такого типа, на которые эксперт тратит X hours, с надежностью 50%").

Урок частично о методологии (как вычисляется horizon), а частично об интерпретации (почему horizon - upper bound, а не deployment prediction). Эти два навыка связаны. Команду, которая понимает, как подгоняется horizon, гораздо труднее обмануть плохим vendor claim, чем команду, которая просто видит "14 hours" на слайде.

## Концепция

### Контекст METR

- Основана: декабрь 2023 (ex-ARC Evals, выделена в независимую 501(c)(3)).
- Scope: evaluation автономных capabilities frontier models, часто pre-release.
- Partner labs: Anthropic, OpenAI (несколько engagements 2025–2026).
- Notable deliverables: Time Horizon 1.0 (март 2025), Time Horizon 1.1 (январь 2026), prototype monitoring evaluations.

### Подгонка Time Horizon

Методология (из blog и papers METR):

1. Собрать task suite, охватывающий expert completion times от минут до часов. Текущие suites: HCAST (180+ tasks), RE-Bench (71 tasks), SWAA.
2. Запустить модель на каждой задаче; записать success или failure.
3. Подогнать logistic curve: P(success) как функцию log(expert completion time).
4. Horizon - это expert-time, при котором P(success) = 0.5.

Форма logistic-fit подходит, потому что capability обычно имеет возрастающую, выходящую на плато связь со сложностью задачи. Точка 50% - это выбор (могли быть 10%, 90%); METR сообщает несколько thresholds в подробной paper, но ведет с 50%, потому что это наиболее интуитивно.

### Числа января 2026

По Time Horizon 1.1:

- Claude Opus 4.6: ~14 hours при 50% reliability, на момент Time Horizon 1.1 (January 2026).
- Doubling time на задачах HCAST-style: ~4.3 months (130.8 days) по post-2023 fit, опубликованному в Time Horizon 1.1 (January 2026); число ~7-month - это полный fit 2019–2025 из Time Horizon 1.0, и оно приводится в TH1.1 рядом с post-2023 number.

Это конкретные числа, на которые ссылается остальная Phase 15. Они цитируются в Anthropic's RSP v3.0 (Lesson 19) и DeepMind's FSF v3 (Lesson 20) как одно из измерений, входящих в threshold assessments.

### Benchmark suites

- **HCAST**: 189 tasks по ML, cybersecurity, software engineering и general reasoning. Диапазон: от 1 minute до 8+ hours expert time. Основной suite для horizon number.
- **RE-Bench**: 71 ML research-engineering tasks с human expert baseline. Измеряет, может ли модель выполнять тип ML R&D work, который ведет к порогу AI R&D-4.
- **SWAA**: более короткие задачи; полезны для калибровки нижнего конца horizon curve.

### Prototype monitoring evaluations

Работа METR 2025–2026 включала prototype monitoring evaluations: может ли monitor поймать модель, выполняющую side task (что-то кроме того, что запросил operator); может ли agent evade monitor. Это напрямую питает категорию "Undermining Safeguards" в OpenAI PF v2 и deceptive-alignment monitoring в DeepMind FSF v3 (Lesson 20).

### Почему horizons являются upper bounds

- **Idealized tooling**: benchmark tasks дают модели чистые tools и хорошо форматированные data. Реальные deployments грязнее.
- **No real consequences**: модель никогда реально не выставляет счет customer, не удаляет реальные data и не связывается с реальными people. Реальные deployments имеют irreversible stakes.
- **Eval-context gaming**: Lesson 1. Модели ведут себя иначе в tests. 2026 International AI Safety Report документирует это эмпирически.
- **No legitimate user variance**: benchmark prompts структурированы. Реальные users создают неоднозначные, context-dependent requests.

Horizon - это capability ceiling при благоприятных условиях. Deployment reliability - другое число, ниже, и команды должны измерять собственное распределение, чтобы знать его.

### Аргумент в пользу external evaluator

External evaluation важна, потому что у внутренних лабораторий есть incentives оптимизировать metrics, которые они сообщают. Независимость METR - 501(c)(3) с заявленной methodology и peer-reviewed papers - является структурной mitigation. Одной ее недостаточно (лаборатории все еще контролируют, что видит METR), но это строго лучше, чем отсутствие external evaluation.

### Как использовать horizon numbers на практике

- **Как capability filter**: если horizon модели существенно ниже expert-time предлагаемой задачи, не отправляйте ее autonomous (skill file Lesson 1).
- **Как trend indicator**: doubling time говорит, как долго текущая практика останется безопасной даже без новых mitigations.
- **Как prior**: horizon 14 hours - отправная точка. Корректируйте вниз с учетом task distribution, tooling quality и deployment context.

## Использование

`code/main.py` реализует logistic fit task-success vs log(expert time) по synthetic result set. Он сообщает 50% horizon (headline METR), 10% horizon (conservative) и 90% horizon (optimistic). Также демонстрирует, что меняется, когда success rate искусственно завышается из-за eval-context gaming.

## Практический результат

`outputs/skill-horizon-interpretation.md` рассматривает vendor's horizon claim и создает gap analysis между benchmark claim и deployment reality.

## Упражнения

1. Запустите `code/main.py`. Подтвердите, что 50% horizon из fit совпадает с synthetic ground truth. Теперь уменьшите task-time grid вдвое; меняется ли horizon estimate существенно?

2. Прочитайте blog post METR о Time Horizon 1.1. Найдите конкретные tasks, где reliability самая высокая и самая низкая. Объясните, почему существует gap.

3. Прочитайте ресурсы METR "Measuring Autonomous AI Capabilities". Перечислите task categories HCAST. Выберите категорию, которой вы дали бы больший вес для production task, и обоснуйте почему.

4. Введите eval-context gaming в симулятор: переверните ~20% failed tasks в success. Сообщите новый horizon. Это приближает эффект gaming rate 20% на observed number.

5. Спроектируйте internal horizon evaluation на собственном bug backlog или representative task set. Опишите data collection, fit и то, что output сообщает вам. Сравните с числами METR.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| METR | "Внешний оценщик" | ex-ARC Evals; независимая 501(c)(3) с декабря 2023 года |
| Time Horizon | "Мера capability" | Длина экспертной задачи при 50% reliability, из logistic fit |
| HCAST | "Основной suite METR" | 180+ tasks spanning 1 min to 8+ hours |
| RE-Bench | "Research engineering" | 71 ML research-engineering tasks with human baseline |
| SWAA | "Suite коротких задач" | Калибрует нижний конец horizon curve |
| Doubling time | "Темп роста" | Время удвоения 50% horizon; ~7 months per HCAST |
| Eval-context gaming | "Модель ведет себя иначе" | Документированный behavior gap между tests and deployment |
| Upper bound | "Horizon - потолок" | Benchmark horizon > deployment reliability under load |

## Дополнительное чтение

- [METR — Resources for Measuring Autonomous AI Capabilities](https://metr.org/measuring-autonomous-ai-capabilities/) — specs HCAST, RE-Bench, SWAA.
- [METR — Measuring AI Ability to Complete Long Tasks](https://metr.org/blog/2025-03-19-measuring-ai-ability-to-complete-long-tasks/) — исходная horizon paper.
- [METR — Time Horizon 1.1 (January 2026)](https://metr.org/research/) — текущие числа и methodology.
- [Epoch AI — METR Time Horizons benchmark](https://epoch.ai/benchmarks/metr-time-horizons) — live tracking.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — internal perspective on METR's measurements.
