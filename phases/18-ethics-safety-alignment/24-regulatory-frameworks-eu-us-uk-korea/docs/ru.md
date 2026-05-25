# Регуляторные рамки — ЕС, США, Великобритания, Корея

> Четыре основных регуляторных режима определяют ландшафт управления ИИ в 2026 году. EU AI Act (в силе с 1 августа 2024) — запрещенные практики и AI literacy с 2 февраля 2025; обязательства GPAI с 2 августа 2025; полная применимость и прозрачность Article 50 с 2 августа 2026; legacy GPAI и embedded high-risk systems с 2 августа 2027; штрафы до 15M EUR или 3% глобального оборота. GPAI Code of Practice (10 июля 2025): три главы — Transparency, Copyright, Safety and Security — 12 обязательств; enforcement начинается в августе 2026. UK AISI -> AI Security Institute (февраль 2025): переименование сигнализирует более узкую область. US AISI -> CAISI (июнь 2025): Center for AI Standards and Innovation при NIST; сдвиг к pro-growth позиции. Korean AI Framework Act (принят в декабре 2024, вступает в силу в январе 2026): Article 12 учреждает AISI при MSIT; требует local representatives для иностранных AI companies, risk assessment, safety measures для high-impact и generative AI.

**Type:** Learn
**Languages:** none
**Prerequisites:** Phase 18 · 18 (frontier frameworks), Phase 18 · 27 (data governance)
**Time:** ~75 minutes

## Цели обучения

- Описать уровни риска EU AI Act (prohibited, high-risk, general-purpose, limited-risk) и timeline August 2025 / August 2026 / August 2027.
- Описать три главы GPAI Code of Practice и каких providers связывает каждая.
- Описать ребрендинги 2025 года: UK AISI -> AI Security Institute; US AISI -> CAISI; что каждый ребрендинг означает для направления политики.
- Сформулировать ключевое положение Korea's AI Framework Act.

## Проблема

Лабораторные рамки (Lesson 18) добровольны. Регуляторные рамки обязательны. Период 2024-2026 принес первую волну комплексного AI-регулирования, вступившего в силу. Deployers должны сопоставлять технические контроли с регуляторными обязанностями; это сопоставление различается по юрисдикциям.

## Концепция

### EU AI Act

**В силе с 1 августа 2024.** Структура уровней риска:

- **Prohibited practices** (Article 5). Social scoring, real-time remote biometric identification in public (with law-enforcement exceptions), exploitative manipulation of vulnerable groups. Применяются с 2 февраля 2025.
- **High-risk systems** (Annex III). Employment, education, credit, law enforcement, justice, migration. Требуют conformity assessment, risk management, logging, transparency.
- **General-Purpose AI (GPAI) models**. Применяются с 2 августа 2025. Все GPAI providers имеют обязательства; systemic-risk GPAI (>1e25 FLOP training compute) имеют дополнительные обязательства.
- **Limited-risk systems**. Transparency obligations under Article 50 (AI-generated content labelling). Применяются с 2 августа 2026.

Timeline:
- 2 Feb 2025: prohibited practices + AI literacy.
- 2 Aug 2025: GPAI + governance.
- 2 Aug 2026: full applicability + Article 50 transparency + penalties up to 15M EUR / 3% global turnover.
- 2 Aug 2027: legacy GPAI + embedded high-risk.

Commission предложила скорректировать high-risk timeline до 16 месяцев в конце 2025.

### GPAI Code of Practice

Опубликован 10 июля 2025. Три главы:

- **Transparency.** Все GPAI providers.
- **Copyright.** Все GPAI providers.
- **Safety and Security.** Systemic-risk GPAI providers (estimated 5-15 companies).

Всего 12 обязательств. Signatory Taskforce под председательством AI Office управляет реализацией. Enforcement начинается 2 августа 2026; до тех пор принимается good-faith compliance.

### Transparency Code for Article 50

Первый черновик 17 декабря 2025. Второй черновик март 2026. Финальная версия июнь 2026. Покрывает маркировку AI-generated content, включая deepfakes — регуляторный слой, который требует технологию watermarking из Lesson 23.

### UK AI Security Institute (февраль 2025)

Переименован из AI Safety Institute. Ребрендинг сужает область: убирает рамки algorithmic bias и free-speech; фокусируется на frontier capability security. Open-sourced инструмент оценки Inspect (май 2024). Сотрудничает с Redwood (Lesson 10) по control safety cases.

### US CAISI (июнь 2025)

Администрация Trump преобразует NIST's AI Safety Institute в Center for AI Standards and Innovation. Сдвиг к "pro-growth AI policies" согласно remarks VP Vance на Paris AI Action Summit. Меньший акцент на pre-deployment evaluation; акцент на стандартах и поддержке инноваций. Внутренний противовес регуляторной позиции EU AI Act.

### Korean AI Framework Act

Принят в декабре 2024. Обнародован в январе 2025. Вступает в силу в январе 2026. Консолидирует 19 отдельных AI bills.

Article 12 учреждает AISI при Ministry of Science and ICT (MSIT). Требует:
- Local representatives для иностранных AI companies, работающих в Корее.
- Risk assessment для "high-impact" AI systems.
- Safety measures для generative AI и high-impact AI.

Первая азиатская юрисдикция с комплексным горизонтальным AI-регулированием.

### Межъюрисдикционная динамика

- ЕС: строгий, risk-tiered, тяжелые штрафы. Ориентир для privacy-adjacent regulation.
- США: благоприятствует инновациям, децентрализован, штаты (например, California AB 2013 — Lesson 27) закрывают федеральные пробелы.
- Великобритания: узкий фокус на security, сильная инфраструктура оценки.
- Корея: ведется MSIT, фокус на foreign-provider.

Конкурирующие регуляторные философии. Deployers в нескольких юрисдикциях должны соответствовать самой строгой, которой в 2026 году обычно является EU AI Act.

### Как это вписывается в Phase 18

Lesson 18 — добровольное лабораторное управление; Lesson 24 — регуляторное; Lesson 25 — возникающий класс CVEs для AI systems; Lessons 26-27 покрывают документацию (cards) и управление обучающими данными.

## Применение

Кода нет. Прочитайте первоисточники EU AI Act: текст регулирования, GPAI Code of Practice, UK AISI Inspect framework. Сопоставьте свое развертывание с применимыми обязательствами для каждой юрисдикции.

## Результат

Этот урок создает `outputs/skill-regulatory-map.md`. Для заданного описания развертывания он сопоставляет применимые юрисдикции, классификации уровней в каждой, обязанности по юрисдикциям и структуру дедлайнов.

## Упражнения

1. Прочитайте EU AI Act (regulation 2024/1689) и GPAI Code of Practice (10 July 2025). Определите три обязательства, применимые к каждому GPAI provider, и три, применимые только к systemic-risk GPAI.

2. Развертывание создано US company, работает на EU infrastructure и обслуживает Korean users. Правила каких трех юрисдикций применяются и какое правило связывает по каждому существенному вопросу?

3. Переименование UK AI Security Institute сужает область. Аргументируйте за и против более узкой рамки. Укажите политическое предположение, от которого зависит каждая позиция.

4. Рамка CAISI "pro-growth" отходит от модели AI safety institute 2022-2024. Определите два измеримых политических сдвига, которые следовали бы из этой рамки.

5. Korea's AI Framework Act требует local representatives для иностранных providers. Опишите операционные последствия для Bay Area company, обслуживающей Korean users.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| EU AI Act | "the regulation" | Горизонтальное AI-регулирование на основе уровней риска; в силе с Aug 2024 |
| GPAI | "general-purpose AI" | Крупные foundation models; subset systemic-risk имеет дополнительные обязательства |
| Article 50 | "transparency obligations" | Маркировка AI-generated content; применяется Aug 2026 |
| UK AISI | "AI Security Institute" | Переименован Feb 2025; более узкий фокус на frontier-security |
| CAISI | "US center for AI standards" | Переименован Jun 2025 из AI Safety Institute; pro-growth posture |
| Korean AI Framework Act | "MSIT horizontal regulation" | Первый комплексный азиатский AI law; effective Jan 2026 |
| Systemic-risk GPAI | "the 1e25 FLOP threshold" | Уровень дополнительных обязательств; estimated 5-15 companies bound |

## Дополнительное чтение

- [EU AI Act text (Regulation 2024/1689)](https://digital-strategy.ec.europa.eu/en/policies/regulatory-framework-ai) — регулирование и timeline
- [GPAI Code of Practice (10 July 2025)](https://digital-strategy.ec.europa.eu/en/library/final-version-general-purpose-ai-code-practice) — код из трех глав
- [UK AI Security Institute (renamed Feb 2025)](https://www.gov.uk/government/organisations/ai-security-institute) — официальная страница
- [CSET — South Korea AI Framework Act Analysis (2025)](https://cset.georgetown.edu/publication/south-korea-ai-law-2025/) — анализ корейской рамки
