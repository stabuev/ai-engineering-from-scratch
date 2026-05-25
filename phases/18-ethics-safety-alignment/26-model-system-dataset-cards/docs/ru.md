# Model, System, and Dataset Cards

> Три формата документации структурируют прозрачность AI. Model Cards (Mitchell et al. 2019) — "этикетки пищевой ценности" для моделей: обучающие данные, количественные дезагрегированные анализы, этические соображения, оговорки; только 0.3% model cards на Hugging Face документируют этические соображения (Oreamuno et al. 2023). Datasheets for Datasets (Gebru et al. 2018, CACM) — мотивация, состав, процесс сбора, разметка, распространение, сопровождение; аналогия с datasheet в электронике. Data Cards (Pushkarna et al., Google 2022) — модульная многоуровневая детализация (telescopic, periscopic, microscopic) как boundary objects для разных читателей. Разработки 2024-2025: автоматическая генерация через LLMs (CardGen, Liu et al. 2024); детализация model-card коррелирует с ростом загрузок на HF до 29% (Liang et al. 2024); проверяемые attestations (Laminator, Duddu et al. 2024); дополнения к sustainability reporting для carbon/water (Jouneaux et al. July 2025); появляются регуляторные карты EU/ISO. System Cards (Sidhpurwala 2024; системная прозрачность Meta; "Blueprints of Trust" arXiv:2509.20394) — end-to-end документация AI-системы, покрывающая security capabilities, защиту от prompt-injection, обнаружение data-exfiltration, alignment with human values.

**Type:** Build
**Languages:** Python (stdlib, model-card + datasheet + system-card generator)
**Prerequisites:** Phase 18 · 18 (safety frameworks), Phase 18 · 24 (regulatory)
**Time:** ~60 minutes

## Цели обучения

- Описать исходную model card Mitchell et al. 2019 и datasheet Gebru et al. 2018.
- Описать telescopic/periscopic/microscopic слои Data Cards.
- Описать System Cards и их end-to-end покрытие.
- Назвать три разработки 2024-2025 (автоматическая генерация, проверяемые attestations, sustainability reporting).

## Проблема

Регуляторные фреймворки (Lesson 24) и политики безопасности лабораторий (Lesson 18) требуют документации. Форматы документации эволюционировали от специфичных для модели (model cards) к специфичным для датасета (datasheets) и затем к специфичным для системы (system cards). Каждый охватывает разную область прозрачности. Работы 2024-2025 по автоматизации и проверяемым attestations решают давнюю проблему внедрения.

## Концепция

### Model Cards (Mitchell et al. 2019)

Разделы:
- Детали модели.
- Предполагаемое использование.
- Факторы (релевантные демографические или средовые факторы для оценки).
- Метрики.
- Оценочные данные.
- Обучающие данные.
- Количественные анализы (дезагрегированные по факторам).
- Этические соображения.
- Оговорки и рекомендации.

Проблема внедрения: аудит Oreamuno et al. 2023 model cards на Hugging Face показал, что только 0.3% документируют этические соображения.

### Datasheets for Datasets (Gebru et al. 2018)

Аналогия с datasheet в электронике. Разделы:
- Мотивация (зачем был создан датасет).
- Состав (что в нем находится).
- Процесс сбора (как он был собран).
- Разметка (если применимо).
- Использования (предполагаемые, запрещенные, риски).
- Распространение.
- Сопровождение.

Опубликовано в CACM 2021. Datasheet — это upstream-документация; model card зависит от точности datasheet.

### Data Cards (Pushkarna et al., Google 2022)

Модульная многоуровневая детализация. Три уровня приближения:
- **Telescopic.** Высокоуровневое резюме для неспециалистов.
- **Periscopic.** Среднеуровневый обзор для ML-практиков.
- **Microscopic.** Подробная документация на уровне признаков для аудиторов.

Фрейминг boundary object: разные читатели извлекают разную информацию из одного и того же документа.

### System Cards

Область: end-to-end AI-система, включая model + safety stack + deployment context. Разделы обычно включают:
- Security capabilities.
- Защиту от prompt-injection.
- Обнаружение data-exfiltration.
- Alignment with stated human values.
- Incident response.

Sidhpurwala 2024 и работа Meta по системной прозрачности. "Blueprints of Trust" (arXiv:2509.20394) формализует System Card как deployment-layer дополнение к Model Cards.

### Разработки 2024-2025

- **CardGen (Liu et al. 2024).** Автоматическая генерация model-card через LLMs; сообщает о более высокой объективности, чем у многих human-authored cards, по стандартизированным полям Mitchell 2019.
- **Download correlation (Liang et al. 2024).** Подробные model cards коррелируют с повышением загрузок на HF до 29% — давление на внедрение теперь рыночное, а не только compliance-driven.
- **Laminator (Duddu et al. 2024).** Проверяемые attestations через hardware TEE / cryptographic signatures — позволяет model card нести proof-of-claim, а не только claim.
- **Sustainability (Jouneaux et al. July 2025).** Дополнения для carbon, water и compute-energy footprint; формирующиеся стандарты ISO.
- **Regulatory cards.** EU AI Act (Lesson 24) GPAI Code of Practice Transparency chapter требует model cards как compliance artifact.

### Где это находится в Phase 18

Lessons 24-25 — регуляторный слой и слой CVE. Lesson 26 — слой документации. Lesson 27 — управление обучающими данными, upstream для datasheet. Lesson 28 — исследовательская экосистема, которая производит оценки, упоминаемые в cards.

## Применение

`code/main.py` генерирует минимальные model card, datasheet и system card для игрушечного развертывания. Каждая следует канонической структуре разделов. Вы можете изучить формат и сравнить три области.

## Результат

Этот урок создает `outputs/skill-card-audit.md`. Для model card, datasheet или system card он аудитирует покрытие разделов, числовую дезагрегацию и наличие проверяемых attestations.

## Упражнения

1. Запустите `code/main.py`. Изучите сгенерированные cards. Определите слабые разделы (только placeholder) и укажите, какие доказательства их усилили бы.

2. Расширьте model card количественным дезагрегированным анализом по двум демографическим группам (Lesson 20).

3. Прочитайте Oreamuno et al. 2023 о 0.3% adoption rate. Предложите одно структурное изменение спецификации model card, которое повысило бы внедрение ethical-considerations.

4. Laminator (Duddu et al. 2024) использует TEEs для проверяемых attestations. Спроектируйте поле model-card, которое несет cryptographic attestation результата оценки, и опишите роль verifier.

5. Напишите System Card (System Card, не Model Card) для одного из ваших прошлых проектов или гипотетического развертывания. Определите самый ценный раздел для сторонних аудиторов.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Model Card | "the Mitchell card" | Стандартная документация Mitchell et al. 2019 для ML-моделей |
| Datasheet | "the Gebru datasheet" | Стандартная документация Gebru et al. 2018 для датасетов |
| Data Card | "the Pushkarna card" | Модульная многоуровневая документация данных Google 2022 |
| System Card | "the deployment card" | End-to-end документация AI-системы, включая safety stack |
| Boundary object | "different readers, one doc" | Фрейминг Data Cards: один документ обслуживает разные аудитории |
| Verifiable attestation | "the Laminator attestation" | Криптографическое или TEE-доказательство, привязанное к утверждению документации |
| Sustainability field | "carbon / water footprint" | Формирующееся дополнение 2025 для экологического учета |

## Дополнительное чтение

- [Mitchell et al. — Model Cards for Model Reporting (arXiv:1810.03993, FAT* 2019)](https://arxiv.org/abs/1810.03993) — каноническая model card
- [Gebru et al. — Datasheets for Datasets (CACM 2021, arXiv:1803.09010)](https://arxiv.org/abs/1803.09010) — статья о datasheet
- [Pushkarna et al. — Data Cards (Google 2022)](https://arxiv.org/abs/2204.01075) — многоуровневая документация данных
- [Sidhpurwala et al. — Blueprints of Trust (arXiv:2509.20394)](https://arxiv.org/abs/2509.20394) — формализация System Card
