# Управляемые LLM-платформы — Bedrock, Vertex AI, Azure OpenAI

> Три hyperscaler-провайдера, три разные стратегии. AWS Bedrock — маркетплейс моделей: Claude, Llama, Titan, Stability, Cohere за одним API. Azure OpenAI — эксклюзивное партнерство с OpenAI плюс Provisioned Throughput Units (PTUs) для выделенной емкости. Vertex AI в первую очередь строится вокруг Gemini, с лучшей историей про длинный контекст и мультимодальность. В 2026 году Artificial Analysis измеряет Azure OpenAI примерно на уровне ~50 ms медианной задержки, а Bedrock — около ~75 ms на эквивалентах Llama 3.1 405B. Разрыв объясняют PTU: выделенная емкость выигрывает у общего on-demand. Правило выбора не "что быстрее", а "какой каталог моделей и какая FinOps-поверхность подходят моему продукту". Этот урок учит выбирать с явно записанными компромиссами, а не по ощущениям.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный компаратор стоимости и задержки)
**Предварительные требования:** Фаза 11 (LLM Engineering), Фаза 13 (Tools & Protocols)
**Время:** ~60 минут

## Цели обучения

- Назвать три платформенные стратегии (marketplace vs exclusive vs Gemini-first) и сопоставить каждую с продуктовым сценарием.
- Объяснить, что дают Provisioned Throughput Units (PTUs) в Azure OpenAI и почему on-demand Bedrock обычно выглядит примерно на ~25 ms медленнее на масштабе 405B.
- Нарисовать FinOps-поверхность атрибуции для каждой платформы (Bedrock Application Inference Profiles vs Vertex project-per-team vs Azure scopes + PTU reservations).
- Записать политику "минимум два провайдера" и объяснить, почему single-vendor lock-in — дорогая ошибка в 2026 году.

## Проблема

Вы выбрали Claude 3.7 Sonnet для продукта. Теперь его нужно обслуживать. Можно вызывать Anthropic API напрямую, можно вызывать его через AWS Bedrock, а можно идти через gateway. Прямой API проще всего; Bedrock добавляет BAAs, VPC endpoints, IAM и атрибуцию в CloudWatch. Gateway добавляет failover, единую биллинговую модель и rate limits между провайдерами.

Более глубокий вопрос — каталог. Если в одном продукте нужны Claude, Llama и Gemini, вы не сможете купить их все в одном месте, если только это место не является одновременно Bedrock плюс Vertex плюс Azure OpenAI. Hyperscaler-платформы не взаимозаменяемы: каждая сделала свою ставку на то, кому принадлежит слой моделей.

Этот урок сопоставляет три ставки, разрыв в задержке, разрыв в FinOps и риск lock-in.

## Концепция

### Три стратегии

**AWS Bedrock** — marketplace. Claude (Anthropic), Llama (Meta), Titan (AWS first-party), Stability (image), Cohere (embeddings), Mistral, плюс подкаталоги изображений и embeddings. Один API, одна IAM-поверхность, один CloudWatch export. Ставка Bedrock в том, что клиентам важнее опциональность, чем одна конкретная модель.

**Azure OpenAI** — эксклюзивное партнерство. Вы получаете GPT-4 / 4o / 5 / o-series, DALL·E, Whisper и fine-tuning моделей OpenAI в дата-центрах Azure. Не-OpenAI моделей в каталоге "Azure OpenAI Service" нет — они уходят в Azure AI Foundry (отдельный продукт). Ставка Azure в том, что OpenAI остается frontier-игроком, а клиентам нужны enterprise controls именно для этих отношений.

**Vertex AI** — сначала Gemini, все остальное потом. Gemini 1.5 / 2.0 / 2.5 Flash и Pro, плюс Model Garden (third-party). Ставка Vertex — мультимодальный длинный контекст: 1M-token Gemini context является дифференциатором.

### Разрыв задержки на масштабе

Artificial Analysis запускает непрерывные бенчмарки. На эквивалентных развертываниях Llama 3.1 405B (shared on-demand) медианная first-token latency у Azure OpenAI около 50 ms; у Bedrock около 75 ms. Разрыв не означает провал AWS — это разница в модели емкости. Azure продает PTUs (Provisioned Throughput Units), которые резервируют GPU-емкость для вашего tenant. Эквивалент Bedrock (Provisioned Throughput) существует, но начинается примерно с $21/hour за unit, и большинство клиентов остаются на shared on-demand.

On-demand shared capacity конкурирует с трафиком всех остальных клиентов. Dedicated capacity — нет. Если SLA продукта требует TTFT < 100 ms на P99, вы либо покупаете PTUs в Azure, либо покупаете Bedrock Provisioned Throughput, либо принимаете variance по умолчанию.

### Экономика Provisioned Throughput

Azure PTUs: зарезервированный блок inference compute. До ~70% экономии против on-demand для предсказуемых нагрузок. Стоимость фиксирована за час независимо от трафика — вы платите за резерв даже в простое. Break-even обычно около 40-60% устойчивой утилизации.

Bedrock Provisioned Throughput: $21-$50 в час в зависимости от модели и региона. Математика похожая — break-even около половины пиковой утилизации. Требуется месячный commitment.

Vertex provisioned capacity продается по Gemini SKU; цены зависят от модели и региона и менее публично рекламируются.

### FinOps-поверхность — настоящий дифференциатор

**Bedrock Application Inference Profiles** — самая чистая атрибуция в marketplace. Пометьте profile тегами `team`, `product`, `feature`; направляйте через него все model invocations; CloudWatch разбивает стоимость по profile без постобработки. Добавлено в 2025 году, все еще самая гранулярная нативная hyperscaler-возможность.

**Vertex** — это project-per-team плюс labels-everywhere. Вы моделируете каждую команду как GCP project, ставите labels на каждый ресурс и используете BigQuery Billing Export + DataStudio для rollups. Работы больше, но BigQuery дает произвольный SQL по cost data.

**Azure** опирается на subscription/resource-group scopes плюс tags, при этом PTU reservations являются cost object первого класса. Tags наследуются от resource groups, а не от requests, поэтому per-request attribution требует Application Insights custom metrics или gateway, который проставляет headers.

Паттерн: Bedrock чище всего нативно, Vertex наиболее гибок через BigQuery, Azure наиболее непрозрачен без instrumenting.

### Lock-in — риск 2026 года

Single-hyperscaler commitment был нормален, когда доминировала одна модель. В 2026 году frontier движется ежемесячно: Claude 3.7 в одном квартале, Gemini 2.5 в следующем, GPT-5 в квартале после этого. Привязка к одной платформе отрезает вас от двух третей frontier.

Паттерн, который используют рабочие команды: минимум два провайдера для любого product-critical LLM call. Bedrock плюс Azure OpenAI — распространенная пара: Claude от одного, GPT от другого, failover между ними, общий gateway. Cost uplift незначителен, потому что gateway маршрутизирует оптимально; availability uplift во время outages (например, Azure OpenAI January 2025 incident, AWS us-east-1 outage) решающий.

### Data residency, BAAs и регулируемые отрасли

Bedrock: BAAs в большинстве регионов; VPC endpoints; guardrails. Частый default для fintech.
Azure OpenAI: HIPAA, SOC 2, ISO 27001; EU data residency; default для enterprise-regulated сценариев.
Vertex: HIPAA, GDPR, data residency per region; compliance stack Google Cloud.

Все три закрывают базовый checkbox. Различия — в политиках data retention, обработке logs и в том, читает ли abuse-monitoring ваш трафик (по умолчанию opt-in у большинства; opt-out доступен для enterprise).

### Числа, которые нужно помнить

- Azure OpenAI median TTFT на эквивалентах Llama 3.1 405B: ~50 ms (с PTUs).
- Bedrock median TTFT on-demand: ~75 ms.
- Bedrock Provisioned Throughput: $21-$50/hr per unit.
- Azure PTU break-even: ~40-60% sustained utilization.
- Экономия PTU против on-demand при высокой утилизации: до 70%.

## Используйте это

`code/main.py` сравнивает три платформы на синтетической нагрузке: моделирует экономику on-demand vs PTU, variance TTFT и качество cost attribution. Запустите его, чтобы увидеть, где PTUs окупаются и где широта model marketplace перевешивает разрыв TTFT.

## Доведите до результата

Этот урок создает `outputs/skill-managed-platform-picker.md`. По профилю нагрузки (нужные модели, TTFT SLA, дневной объем, compliance requirements) он рекомендует primary platform, fallback и план FinOps-инструментирования.

## Упражнения

1. Запустите `code/main.py`. При какой sustained utilization Azure PTU выигрывает у on-demand для модели класса 70B? Посчитайте break-even и сравните с заявленным диапазоном 40-60%.
2. Вашему продукту нужны Claude 3.7 Sonnet и GPT-4o. Спроектируйте two-provider deployment: что идет к какому hyperscaler, какой gateway стоит перед ними, какая failover policy?
3. Регулируемый healthcare-клиент требует BAAs, US-East data residency и sub-100ms P99 TTFT. Выберите платформу и обоснуйте тремя конкретными features.
4. Вы обнаружили, что счет Bedrock вырос в 4 раза за месяц без изменения трафика. Без Application Inference Profiles как бы вы нашли виновника? С profiles сколько это займет?
5. Прочитайте pricing pages Azure OpenAI и Bedrock. Для Claude-нагрузки 100M-token/month что дешевле: direct Anthropic API, Bedrock on-demand или Bedrock Provisioned Throughput?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Bedrock | "AWS LLM service" | Marketplace моделей Claude, Llama, Titan, Mistral, Cohere |
| Azure OpenAI | "Azure's ChatGPT" | Эксклюзивные модели OpenAI в дата-центрах Azure с enterprise controls |
| Vertex AI | "Google's LLM" | Gemini-first платформа с Model Garden для third-party моделей |
| PTU | "dedicated capacity" | Provisioned Throughput Unit — зарезервированные inference GPUs, цена за час |
| Application Inference Profile | "Bedrock tagging" | Per-product cost/usage profile с tags, CloudWatch-native |
| Model Garden | "Vertex catalog" | Раздел third-party моделей Vertex AI, отдельный от Gemini |
| Two-provider minimum | "LLM redundancy" | Политика запуска каждого critical LLM path через ≥2 hyperscalers |
| BAA | "HIPAA paperwork" | Business Associate Agreement; требуется для PHI; предоставляется всеми тремя |
| Abuse monitoring | "the log watcher" | Provider-side safety scan prompts/outputs; opt-out в enterprise |

## Дополнительное чтение

- [AWS Bedrock Pricing](https://aws.amazon.com/bedrock/pricing/) — authoritative rate card и цены Provisioned Throughput.
- [Azure OpenAI Service Pricing](https://azure.microsoft.com/en-us/pricing/details/cognitive-services/openai-service/) — экономика PTU и rate cards.
- [Vertex AI Generative AI Pricing](https://cloud.google.com/vertex-ai/generative-ai/pricing) — Gemini tiers и surcharges Model Garden.
- [Artificial Analysis LLM Leaderboard](https://artificialanalysis.ai/) — непрерывные latency и throughput benchmarks между провайдерами.
- [The AI Journal — AWS Bedrock vs Azure OpenAI CTO Guide 2026](https://theaijournal.co/2026/03/aws-bedrock-vs-azure-openai/) — enterprise decision framework.
- [Finout — Bedrock vs Vertex vs Azure FinOps](https://www.finout.io/blog/bedrock-vs.-vertex-vs.-azure-cognitive-a-finops-comparison-for-ai-spend) — механики attribution side-by-side.
