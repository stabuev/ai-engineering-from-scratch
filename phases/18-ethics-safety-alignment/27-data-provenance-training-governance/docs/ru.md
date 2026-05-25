# Data Provenance and Training-Data Governance

> EU AI Act требует машиночитаемые стандарты opt-out для GPAI к августу 2025 (через TDM exception в EU Copyright Directive). California AB 2013 (подписан в 2024) — прозрачность обучающих данных Generative AI требует от разработчиков публиковать summary датасетов с 12 обязательными полями. Согласование DPA в 2025 по legitimate interest: Irish DPC (21 May 2025) принимает обучение LLM Meta на first-party public EU/EEA adult content с safeguards после мнения EDPB; Cologne Higher Regional Court (23 May 2025) отклоняет injunction; Hamburg DPA снимает срочность; UK ICO (23 September 2025) выпускает положительный regulatory response на safeguards LinkedIn для AI-training (прозрачность, упрощенный opt-out, расширенные окна objection) и продолжает monitoring — это не formal clearance. Brazilian ANPD (2 July 2024) приостановил обработку Meta из-за недостаточной information transparency; preventive measure была снята 30 August 2024 после подачи Meta compliance plan. Ключевая проблема необратимости: cookie-consent frameworks разработаны для real-time, reversible tracking; когда данные уже в весах модели, surgical erasure невозможно — нет практического GDPR right-to-erasure для обученных нейронных сетей. Compliance window находится во время сбора. Data Provenance Initiative (dataprovenance.org, Longpre, Mahari, Lee et al., "Consent in Crisis", July 2024): крупномасштабный аудит показывает быстрый спад AI data commons по мере того, как издатели добавляют ограничения robots.txt.

**Type:** Learn
**Languages:** Python (stdlib, 12-field California AB 2013 scaffolding generator)
**Prerequisites:** Phase 18 · 24 (regulatory), Phase 18 · 26 (cards)
**Time:** ~60 minutes

## Цели обучения

- Описать 12 обязательных полей California AB 2013 для прозрачности обучающих данных Generative AI.
- Сформулировать позицию DPA 2025 по обучению LLM на основании legitimate-interest (Irish DPC, UK ICO, Hamburg, Cologne).
- Описать проблему необратимости: почему GDPR right-to-erasure не имеет практического эквивалента для обученных нейронных сетей.
- Сформулировать вывод Data Provenance Initiative "Consent in Crisis".

## Проблема

Управление обучающими данными — upstream для каждой model card (Lesson 26) и регуляторного обязательства (Lesson 24). В 2024-2025 регуляторный ландшафт консолидировался вокруг трех принципов: opt-out infrastructure, per-dataset disclosure и legitimate-interest accommodations для публично доступных данных. Провайдеры, которые не соблюдают требования на этапе сбора, не могут исправить это downstream.

## Концепция

### California AB 2013

Подписан в 2024. Документация должна быть опубликована не позднее January 1, 2026 для систем, выпущенных on or after January 1, 2022. Section 3111(a) требует от разработчиков публиковать high-level summary датасетов, использованных в обучении, с 12 statutory items:
1. Источники или владельцы датасетов.
2. Описание того, как датасеты продвигают предполагаемую цель AI-системы.
3. Количество data points в датасетах (допустимы общие диапазоны; оценки для динамических датасетов).
4. Описание типов data points (типы меток для размеченных датасетов; общие характеристики для неразмеченных).
5. Содержат ли датасеты какие-либо данные, защищенные copyright, trademark или patent, либо полностью находятся в public domain.
6. Были ли датасеты куплены или лицензированы.
7. Содержат ли датасеты personal information (per Cal. Civ. Code §1798.140(v)).
8. Содержат ли датасеты aggregate consumer information (per Cal. Civ. Code §1798.140(b)).
9. Cleaning, processing или другая модификация разработчиком с предполагаемой целью.
10. Период времени, в течение которого данные собирались, с уведомлением, если сбор продолжается.
11. Даты, когда датасеты впервые использовались во время разработки.
12. Использует ли система или непрерывно использует synthetic data generation.

Item 12 (synthetic data) является новым относительно Gebru et al. 2018 datasheets. Item 7 (personal information) запускает обязательства Privacy Rights Act (CPRA). Закон освобождает security/integrity, aircraft-operation и federal-only national-security systems (Section 3111(b)).

### EU AI Act (Lesson 24) and TDM opt-out

Исключение text-and-data-mining в EU Copyright Directive разрешает обучение на публично доступном контенте, если правообладатель не сделал opt out. EU AI Act GPAI Code of Practice Copyright chapter требует от GPAI providers уважать машиночитаемые opt-out signals (robots.txt, C2PA "No AI Training" claim, etc.).

### Сближение DPA 2025 по legitimate interest

Irish DPC (21 May 2025): план Meta обучаться на first-party public EU/EEA adult-user content принят с safeguards после мнения EDPB. Cologne Higher Regional Court (23 May 2025) отклоняет injunction против Meta: opt-out достаточен. Hamburg DPA снимает urgency procedure ради согласованности по ЕС. UK ICO (23 September 2025) выпустил положительный regulatory response — не formal clearance — на возобновление AI training LinkedIn с похожими safeguards и ongoing monitoring.

Сходящийся принцип: legitimate interest может обосновывать обучение на публично доступном first-party content с opt-out. Consent не требуется.

### Brazilian ANPD (June 2024)

Приостановил обработку данных бразильских пользователей Meta для AI training из-за недостаточной information transparency. Результат отличается от EU DPAs — ANPD поставил transparency выше допустимости legitimate-interest.

### Проблема необратимости

Cookie-consent был разработан для real-time, reversible tracking. Обучающие данные отличаются: когда данные попадают в веса модели, surgical erasure невозможно. Переобучение с нуля — единственное полное исправление, и оно запредельно дорого.

Частичные исправления:
- **Unlearning.** Приближенное удаление; измеряется MIA (Lesson 22).
- **Influence function-based localization.** Выявить веса, на которые данные повлияли сильнее всего; выборочно обновить.
- **Fine-tune-suppression.** Обучить модель отказываться от выводов, производных от этих данных.

Ни одно не решает проблему полностью. Compliance window находится на этапе сбора.

### Data Provenance Initiative

dataprovenance.org. Longpre, Mahari, Lee et al. "Consent in Crisis" (July 2024): крупномасштабный аудит commons обучающих данных AI. Вывод: издатели добавляют ограничения robots.txt с ускоряющимся темпом. Commons, открытый для обучения, быстро сжимается. В 2023 -> 2024 около 25% top training sources добавили какое-либо ограничение. Следствие: будущая доступность обучающих данных зависит от новых парадигм получения (лицензирование, synthetic generation, incentivized participation).

### Где это находится в Phase 18

Lesson 26 — документация уровня модели. Lesson 27 — управление уровня датасетов. Вместе они определяют слой прозрачности. Lesson 28 картирует исследовательскую экосистему, работающую над этими вопросами.

## Применение

`code/main.py` генерирует AB 2013-compliant scaffold 12-полевого summary датасета для игрушечного датасета. Вы можете заполнить поля и увидеть, какие из них запускают последующие обязательства privacy или copyright.

## Результат

Этот урок создает `outputs/skill-provenance-check.md`. Для датасета, использованного в обучении, он проверяет покрытие 12 полей AB 2013, соответствие opt-out infrastructure, согласованность с DPA и оценку irreversibility-risk.

## Упражнения

1. Запустите `code/main.py`. Создайте 12-полевое summary для игрушечного датасета и определите, какие поля недостаточно специфицированы.

2. EU Copyright Directive TDM opt-out является машиночитаемым. Предложите стандартный формат для opt-out signal и сравните его с robots.txt и C2PA "No AI Training."

3. Прочитайте "Consent in Crisis" Data Provenance Initiative (July 2024). Опишите три категории контента с самым быстрым ростом ограничений и обоснуйте одно экономическое последствие.

4. Согласование DPA 2025 принимает legitimate interest для обучения на public-content. Сконструируйте сценарий, в котором legitimate interest был бы недостаточен, и определите legal basis, который провайдеру понадобился бы вместо него.

5. Набросайте training-data-provenance manifest, который компонуется с полями AB 2013 и C2PA-signed provenance chain для каждого датасета. Укажите один технический и один юридический барьер.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| AB 2013 | "the California law" | Прозрачность обучающих данных Generative AI; 12 обязательных полей |
| TDM exception | "text-and-data-mining" | Исключение EU Copyright Directive для обучающих данных с opt-out |
| Legitimate interest | "the EU basis" | Основание GDPR Article 6, которое может оправдать обучение на публичном контенте |
| Opt-out signal | "machine-readable no-train" | robots.txt, C2PA "No AI Training," TDM.Reservation |
| Irreversibility | "cannot un-train" | Данные в весах модели нельзя хирургически удалить |
| Unlearning | "approximate removal" | Post-training interventions для снижения зависимости модели от конкретных данных |
| Consent in Crisis | "the DPI audit" | Вывод July 2024 об ускоряющихся ограничениях robots.txt |

## Дополнительное чтение

- [California AB 2013](https://leginfo.legislature.ca.gov/faces/billNavClient.xhtml?bill_id=202320240AB2013) — закон о прозрачности обучающих данных Generative AI
- [EU AI Act + GPAI Code of Practice (Lesson 24)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — Copyright chapter
- [Longpre, Mahari, Lee et al. — Consent in Crisis (dataprovenance.org, July 2024)](https://www.dataprovenance.org/consent-in-crisis-paper) — аудит DPI
- [IAPP — EU Digital Omnibus GDPR amendments (2025)](https://iapp.org/news/a/eu-digital-omnibus-amendments-to-gdpr-to-facilitate-ai-training-miss-the-mark) — регуляторный контекст
