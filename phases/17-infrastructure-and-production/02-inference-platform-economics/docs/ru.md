# Экономика inference-платформ — Fireworks, Together, Baseten, Modal, Replicate, Anyscale

> Рынок inference в 2026 году больше не является арендой GPU-времени. Он разделился на custom silicon (Groq, Cerebras, SambaNova), GPU platforms (Baseten, Together, Fireworks, Modal) и API-first marketplaces (Replicate, DeepInfra). Fireworks поднял цену на $1/hr per GPU с 1 мая 2026 года, а valuation $4B при 10T+ tokens/day показывает, что модель, driven by volume, работает. Baseten закрыл $300M Series E при $5B в январе 2026 года. Правило конкурентного позиционирования простое: Fireworks оптимизирует latency, Together — catalog breadth, Baseten — enterprise polish, Modal — Python-native DX, Replicate — multimodal reach, Anyscale — distributed Python. Этот урок дает матрицу, которую можно отдать founder.

**Тип:** Изучение
**Языки:** Python (stdlib, учебный компаратор экономики per-call)
**Предварительные требования:** Phase 17 · 01 (Managed LLM Platforms), Phase 17 · 04 (vLLM Serving Internals)
**Время:** ~60 минут

## Цели обучения

- Назвать три сегмента рынка (custom silicon, GPU platforms, API-first) и сопоставить каждого vendor с сегментом.
- Объяснить, почему pricing model "per-token" сжимается к cost curve serving engine, а не hardware.
- Посчитать effective cost per request минимум для трех vendors и объяснить, когда per-minute (Baseten, Modal) выигрывает у per-token.
- Определить правильную default-платформу для заданной нагрузки (serverless bursty, steady high-throughput, fine-tuned variants, multimodal).

## Проблема

Вы оценили managed hyperscaler platforms. Вы решили, что нужен более узкий и быстрый провайдер: Fireworks для latency, Together для breadth, Baseten для fine-tuned custom model. Теперь у вас шесть реальных вариантов, а pricing pages не сопоставляются. Fireworks показывает $/M tokens; Baseten показывает $/minute; Modal показывает $/second; Replicate показывает $/prediction. Нельзя сравнить их head-to-head без моделирования нагрузки.

Хуже того, бизнес-модель за каждой pricing page разная. Fireworks запускает собственный custom engine (FireAttention) на shared GPUs; per-token rate отражает их utilization curve. Baseten дает Truss + dedicated GPUs; per-minute отражает эксклюзивность. Modal — настоящий Python serverless: per-second billing с sub-second cold starts. Один и тот же выход (LLM response), три разные cost functions.

Этот урок моделирует все шесть вариантов и показывает, когда каждый выигрывает.

## Концепция

### Три сегмента

**Custom silicon** — Groq (LPU), Cerebras (WSE), SambaNova (RDU). Обычно 5-10x быстрее decode, чем GPU-based cluster на той же модели. Более высокая per-token price (Groq был примерно ~$0.99/M на Llama-70B в конце 2025), но непревзойден для latency-sensitive use cases. Groq — production choice для voice agents и real-time translation.

**GPU platforms** — Baseten, Together, Fireworks, Modal, Anyscale. Работают на NVIDIA (H100, H200, B200 в 2026) или иногда AMD. Экономический слой между "raw GPU rental" (RunPod, Lambda) и "hyperscaler managed service" (Bedrock).

**API-first marketplaces** — Replicate, DeepInfra, OpenRouter, Fal. Широкий каталог, pay-per-prediction или pay-per-second, акцент на time-to-first-call.

### Fireworks — GPU-платформа, оптимизированная под latency

- FireAttention engine (custom); marketed as 4x lower latency than vLLM на эквивалентных configs.
- Batch tier примерно за ~50% от serverless rate для non-interactive workloads.
- Fine-tuned model обслуживается по той же ставке, что и base model — реальный дифференциатор относительно провайдеров, которые берут premium за ваш LoRA.
- Mid-2026: on-demand GPU rental поднят на $1/hour effective May 1, 2026. Volume pricing negotiable at scale.
- Финансовый сигнал: $4B valuation, обработка 10T+ tokens/day.

### Together — оптимизирован под breadth

- 200+ models, включая open-source releases в течение нескольких дней после upstream publication.
- На 50-70% дешевле Replicate на эквивалентных LLM models — позиционирование "AI Native Cloud" основано на volume и catalog.
- Inference + fine-tuning + training в одном API.

### Baseten — оптимизирован под enterprise polish

- Truss framework: упаковка модели с dependencies, secrets и serving config в одном manifest.
- Диапазон GPU от T4 до B200. Per-minute billing с разумной cold-start mitigation.
- SOC 2 Type II, HIPAA-ready. Частый выбор для fintech и healthcare.
- $5B valuation, January 2026 Series E ($300M от CapitalG, IVP, NVIDIA).

### Modal — оптимизирован под Python-native

- Infrastructure-as-code на чистом Python. Декорируете function через `@modal.function(gpu="A100")` и деплоите одной командой.
- Per-second billing. Cold starts 2-4s с pre-warming; <1s для малых моделей.
- $87M Series B при $1.1B valuation (2025). Самый сильный developer experience score в независимых surveys.

### Replicate — мультимодальная широта

- Pay-per-prediction. Default platform для image, video и audio models.
- Integration ecosystem (Zapier, Vercel, CMS plugins).
- Менее конкурентен по LLM per-token rates, но выигрывает по multimodal variety.

### Anyscale — Ray-native

- Построен на Ray; RayTurbo — proprietary inference engine Anyscale (конкурирует с vLLM).
- Лучший вариант для distributed Python workloads, где inference step — один node в большем graph.
- Managed Ray clusters; тесная интеграция с Ray AIR и Ray Serve.

### Per-token versus per-minute — когда что выигрывает

Per-token имеет смысл, когда workload latency-insensitive и bursty: вы платите только за использование. Per-minute имеет смысл, когда utilization высокая и предсказуемая: вы выигрываете у per-token, когда насыщаете GPU.

Грубое правило: для workloads выше ~30% sustained utilization выделенного GPU per-minute (Baseten, Modal) начинает выигрывать у per-token (Fireworks, Together). Ниже этого выигрывает per-token, потому что вы не платите за idle.

### Custom engine — настоящий moat

Каждая платформа выше vLLM и SGLang заявляет custom engine. FireAttention, RayTurbo, inference stack Baseten. Заявления о custom engine часто окрашены marketing; честная формулировка такая: vLLM + SGLang представляют около 80% production open-source inference, а дифференциаторы platform layer — это DX, attribution и SLAs.

### Числа, которые нужно помнить

- Fireworks GPU rental: повышение на $1/hr effective May 1, 2026.
- Fireworks claim: 4x lower latency than vLLM на эквивалентных configs.
- Together: на 50-70% дешевле Replicate на LLMs.
- Baseten valuation: $5B (Series E, Jan 2026, $300M round).
- Modal valuation: $1.1B (Series B, 2025).
- Per-minute выигрывает у per-token выше ~30% sustained utilization.

## Используйте это

`code/main.py` сравнивает шесть vendors на синтетической нагрузке между pricing models. Выводит $/day и effective $/M tokens. Запустите его, чтобы найти break-even между per-token и per-minute.

## Доведите до результата

Этот урок создает `outputs/skill-inference-platform-picker.md`. По workload profile, SLA и budget выбирает primary inference platform и называет runner-up.

## Упражнения

1. Запустите `code/main.py`. При какой sustained utilization Baseten (per-minute) выигрывает у Fireworks (per-token) для модели 70B на одном H100? Выведите crossover самостоятельно и сравните с rule of thumb.
2. Ваш продукт обслуживает image generation плюс chat плюс speech-to-text. Выберите platforms для каждой modality и назовите gateway pattern, который их объединяет.
3. Fireworks повышает prices на $1/hr для вашей primary model. Смоделируйте blended cost impact, если 40% вашего traffic переходит в batch tier (50% off).
4. Regulated customer требует SOC 2 Type II + HIPAA + dedicated GPUs. Какие три platforms жизнеспособны и какая выигрывает по FinOps?
5. Сравните cost per 1,000 predictions для Llama 3.1 70B на Fireworks serverless, Together on-demand, Baseten dedicated и Replicate API. Что дешевле при 10 predictions/day? При 10,000?

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Custom silicon | "non-GPU chips" | Groq LPU, Cerebras WSE, SambaNova RDU — оптимизированы под decode |
| FireAttention | "Fireworks engine" | Custom attention kernel; marketed at 4x lower latency than vLLM |
| Truss | "Baseten's format" | Manifest упаковки модели; dependencies + secrets + serving config |
| Per-token | "API pricing" | Оплата по consumed tokens; без платы за idle |
| Per-minute | "dedicated pricing" | Оплата wall-clock GPU time; выигрывает при высокой utilization |
| Per-prediction | "Replicate pricing" | Оплата per model invocation; часто для image/video |
| RayTurbo | "Anyscale engine" | Proprietary inference on Ray; конкурирует с vLLM на Ray clusters |
| Batch tier | "50% off" | Non-interactive queue по сниженной ставке; common on Fireworks, OpenAI |
| Fine-tuned at base rate | "Fireworks LoRA" | LoRA-served requests тарифицируются по rate base model (дифференциатор) |

## Дополнительное чтение

- [Fireworks Pricing](https://fireworks.ai/pricing) — per-token rates, batch tier, GPU rental.
- [Baseten Pricing](https://www.baseten.co/pricing/) — per-minute rates, committed capacity, enterprise tiers.
- [Modal Pricing](https://modal.com/pricing) — per-second GPU rates and free tier.
- [Together AI Pricing](https://www.together.ai/pricing) — model catalog and per-token rates.
- [Anyscale Pricing](https://www.anyscale.com/pricing) — RayTurbo and managed Ray pricing.
- [Northflank — Fireworks AI Alternatives](https://northflank.com/blog/7-best-fireworks-ai-alternatives-for-inference) — comparative assessment.
- [Infrabase — AI Inference API Providers 2026](https://infrabase.ai/blog/ai-inference-api-providers-compared) — vendor landscape.
