# Anthropic Responsible Scaling Policy v3.0

> RSP v3.0 вступила в силу 24 февраля 2026 года, заменив политику 2023 года. Двухуровневая mitigation: что Anthropic будет делать самостоятельно и что оформлено как industry-wide recommendation (включая стандарты безопасности RAND SL-4). Добавляет Frontier Safety Roadmaps и Risk Reports как постоянные документы, а не разовые deliverables. Убирает pause commitment 2023 года. Вводит порог AI R&D-4: после его пересечения Anthropic должна опубликовать affirmative case с указанием misalignment risks и mitigations. Claude Opus 4.6 его не пересекает. В объявлении v3.0 Anthropic утверждает, что "confidently ruling this out is becoming difficult." SaferAI оценивала RSP 2023 года на 2.2; v3.0 они понизили до 1.9, поместив Anthropic в категорию "weak" RSP вместе с OpenAI и DeepMind. Качественные пороги заменили количественные обязательства 2023 года; удаление pause clause - самая резкая регрессия.

**Тип:** Изучение
**Языки:** Python (stdlib, движок принятия решений по порогам RSP)
**Предварительные требования:** Phase 15 · 06 (AAR), Phase 15 · 07 (RSI)
**Время:** ~45 минут

## Проблема

Frontier labs публикуют scaling policies, которые одновременно являются техническими документами, governance-документами и сигналами регуляторам. RSP v3.0 - текущий документ Anthropic. Читать его внимательно важно не потому, что соблюдение обязательно (это не так), а потому что framing формирует то, как лаборатория понимает catastrophic risk и как она объясняет trade-offs обществу.

Diff v3.0 vs v2.0 - полезная единица анализа. Что добавлено: Frontier Safety Roadmaps, Risk Reports, порог AI R&D-4. Что удалено: pause commitment 2023 года. Что переоформлено: двухуровневый график mitigation, разделенный на Anthropic-unilateral и industry-recommendation. Внешняя оценка - SaferAI - снизила score с 2.2 (v2) до 1.9 (v3.0). Так scaling policy может стать менее строгой, выглядя при этом более отполированной.

## Концепция

### Двухуровневый график mitigation

- **Anthropic unilateral actions**: что Anthropic будет делать независимо от того, что делают другие лаборатории. Остановки обучения выше порога, конкретные меры безопасности, конкретные deployment gates.
- **Industry-wide recommendations**: что Anthropic считает необходимым делать индустрии коллективно. Включает стандарты безопасности RAND SL-4. Это не обязательства Anthropic; это policy advocacy.

Двухуровневой структуры не было в v2. Она означает, что читателю нужно смотреть, в какой колонке находится каждое обязательство. Мера безопасности в колонке "industry-wide recommendation" - не обещание Anthropic; это надежда Anthropic.

### Порог AI R&D-4

Это уровень capabilities, который RSP v3.0 называет важным следующим порогом. Конкретно: модель, способная автоматизировать значительную долю AI research при конкурентной стоимости. Как только Anthropic считает, что модель его пересекает, компания должна опубликовать affirmative case, указывающий misalignment risks и mitigations, до продолжения scaling.

Claude Opus 4.6 не пересекает его согласно объявлению v3.0. Документ добавляет: "confidently ruling this out is becoming difficult." Эта формулировка важна; она признает, что порог достаточно близок, чтобы быть актуальной заботой, а не спекулятивным пределом.

Lesson 6 (Automated Alignment Research) и Lesson 7 (Recursive Self-Improvement) напрямую ведут к этому порогу. Automated alignment researchers, пересекающие research-quality bars, - evidence того, что порог AI R&D-4 приближается.

### Frontier Safety Roadmaps и Risk Reports

v3.0 поднимает два типа артефактов до статуса постоянных документов:

- **Frontier Safety Roadmap**: forward-looking document, описывающий планируемую safety work, ожидания по capabilities и mitigation research.
- **Risk Report**: retrospective document по конкретным моделям после release, описывающий наблюдаемые capabilities и residual risk.

Оба публичны. Оба обновляются по заявленной cadence. Практическая польза: читатель может отслеживать, как то, что Anthropic обещала сделать в Roadmap, соотносится с тем, что она сообщает в Risk Report.

### Удаление pause clause

RSP 2023 года включала явное pause commitment: если модель пересекает конкретные capability thresholds, training останавливается до появления mitigations. v3.0 заменяет явную паузу более мягкой формулировкой (опубликовать affirmative case, продолжать, если mitigations адекватны). SaferAI и другие аналитики прямо назвали это самой сильной регрессией нового документа.

Policy-аргумент в пользу изменения: количественные пороги 2023 года оказались недостижимыми для capability benchmarks эпохи 2026 года, потому что сами benchmarks были переотмасштабированы. Контраргумент: pause clause в scaling policy - это commitment device; его удаление подрывает credibility политики.

### Понижение оценки SaferAI

SaferAI - независимая организация, оценивающая документы в стиле RSP. Их публичная оценка: Anthropic RSP 2023 года получила 2.2 (по шкале, где 4.0 - лучшая текущая RSP, а 1.0 - nominal). v3.0 получила 1.9. Это переместило Anthropic из "moderate" в "weak", вместе с OpenAI и DeepMind в слабой категории.

Факторы понижения по SaferAI:
- Качественные пороги заменили количественные.
- Pause commitment удалено.
- Mitigations для порога AI R&D-4 описаны как "affirmative case", а не как конкретные меры.
- Review mechanisms зависят от Anthropic's Safety Advisory Group, с ограниченным independent oversight.

### Чем этот урок не является

Это не урок по compliance. RSP v3.0 не является regulation; ничто не заставляет Anthropic ей следовать. Урок состоит в чтении документа с той конкретностью и скепсисом, которых он заслуживает. Scaling policies - главный публичный сигнал frontier labs о позиции по catastrophic risk. Умение хорошо их читать - практический навык для всех, чья работа зависит от frontier capabilities.

## Использование

`code/main.py` реализует небольшой decision engine, отражающий форму threshold evaluation в RSP: по candidate model и набору capability measurements вернуть, пересечен ли порог AI R&D-4, какие разделы affirmative case требуются и может ли deployment продолжаться. Он намеренно простой; цель - сделать логику документа явной.

## Практический результат

`outputs/skill-scaling-policy-review.md` рассматривает scaling policy (Anthropic, OpenAI, DeepMind или internal) относительно reference v3.0: двухуровневая структура, пороги, pause commitments, independent review.

## Упражнения

1. Запустите `code/main.py`. Передайте три synthetic models с разными capability levels. Подтвердите, что threshold evaluator ведет себя ожидаемо и создает правильный шаблон affirmative case.

2. Прочитайте RSP v3.0 полностью (32 pages). Найдите каждое обязательство, находящееся на уровне "industry-wide recommendation". Какие из этих обязательств в v2 были бы "Anthropic unilateral"?

3. Прочитайте методологию оценки RSP от SaferAI. Воспроизведите их score 1.9 для v3.0, применив rubric к документу. Какая строка rubric сильнее всего повлияла на downgrade?

4. Pause commitment 2023 года было удалено. Предложите replacement commitment, которое сохраняет credibility политики, признавая проблему benchmark-rescaling 2026 года.

5. Сравните RSP v3.0 с OpenAI Preparedness Framework v2 (Lesson 20). Выберите одну область, где v3.0 сильнее. Выберите одну область, где Preparedness Framework сильнее.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|---|---|---|
| RSP | "Scaling policy Anthropic" | Responsible Scaling Policy; v3.0 вступила в силу 24 февраля 2026 года |
| AI R&D-4 | "Порог автоматизации исследований" | Способность автоматизировать существенную часть AI research при конкурентной стоимости |
| Affirmative case | "Обоснование безопасности" | Опубликованный аргумент, что risks выявлены, а mitigations адекватны |
| Frontier Safety Roadmap | "План на будущее" | Постоянный документ о планируемой safety work и expected capabilities |
| Risk Report | "Ретроспектива по модели" | Постоянный документ о observed capability и residual risk после release |
| Two-tier mitigation | "Anthropic vs индустрия" | Разделенные обязательства Anthropic и industry recommendations |
| Pause commitment | "Пункт 2023 года" | Явное обещание pause training; удалено в v3.0 |
| SaferAI rating | "Независимая оценка RSP" | Third-party rubric; v3.0 получила 1.9 (v2 получила 2.2) |

## Дополнительное чтение

- [Anthropic — Responsible Scaling Policy v3.0](https://anthropic.com/responsible-scaling-policy/rsp-v3-0) — полная политика на 32 страницы.
- [Anthropic — RSP v3.0 announcement](https://www.anthropic.com/news/responsible-scaling-policy-v3) — summary изменений относительно v2.
- [Anthropic — Frontier Safety Roadmap](https://www.anthropic.com/research/frontier-safety) — постоянный документ, linked from RSP v3.0.
- [Anthropic — Risk Report: Claude Opus 4.6](https://www.anthropic.com/research/risk-report-claude-opus-4-6) — retrospective по текущей frontier model.
- [Anthropic — Measuring agent autonomy in practice](https://www.anthropic.com/research/measuring-agent-autonomy) — связывает AI R&D-4 с measured autonomy.
