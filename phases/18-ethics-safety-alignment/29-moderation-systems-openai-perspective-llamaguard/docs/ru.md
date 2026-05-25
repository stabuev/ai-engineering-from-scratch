# Moderation Systems — OpenAI, Perspective, Llama Guard

> Production moderation systems операционализируют safety policies, определенные в Lessons 12-16. OpenAI Moderation API: `omni-moderation-latest` (2024), построенный на GPT-4o, классифицирует text + images одним вызовом; на 42% лучше на multilingual test set, чем предыдущая версия; response schema возвращает 13 category booleans — harassment, harassment/threatening, hate, hate/threatening, illicit, illicit/violent, self-harm, self-harm/intent, self-harm/instructions, sexual, sexual/minors, violence, violence/graphic; бесплатен для большинства разработчиков. Layered patterns: Input moderation (pre-generation), Output moderation (post-generation), Custom moderation (domain rules). Async parallel calls скрывают latency; placeholder responses при flag. Llama Guard 3/4 (Lesson 16): 14 MLCommons hazards, Code Interpreter Abuse, 8 languages (v3), multi-image (v4). Perspective API (Google Jigsaw): toxicity scoring, предшествующий волне LLM-as-moderator; в основном single-dimension toxicity с вариантами severe-toxicity/insult/profanity; baseline для исследований content-moderation. Deprecations: Azure Content Moderator deprecated February 2024, retired February 2027, replaced by Azure AI Content Safety.

**Type:** Build
**Languages:** Python (stdlib, three-layer moderation harness)
**Prerequisites:** Phase 18 · 16 (Llama Guard / Garak / PyRIT)
**Time:** ~60 minutes

## Цели обучения

- Описать category taxonomy OpenAI Moderation API и ее отличие от MLCommons set в Llama Guard 3.
- Описать three moderation-layer pattern (input, output, custom) и назвать один failure mode каждого слоя.
- Описать положение Perspective API как baseline до эпохи LLM и почему он остается используемым в исследованиях.
- Сформулировать timeline deprecation Azure.

## Проблема

Lessons 12-16 описывают атаки и defense tooling. Lesson 29 покрывает развернутые moderation systems, которые операционализируют защиты на поверхности, где пользователи взаимодействуют с продуктом. Three-layer pattern — конфигурация по умолчанию в 2026.

## Концепция

### OpenAI Moderation API

`omni-moderation-latest` (2024). Построен на GPT-4o. Классифицирует text + images одним вызовом. Бесплатен для большинства разработчиков.

Categories (13 booleans in the response schema):
- harassment, harassment/threatening
- hate, hate/threatening
- self-harm, self-harm/intent, self-harm/instructions
- sexual, sexual/minors
- violence, violence/graphic
- illicit, illicit/violent

Multimodal support применяется к `violence`, `self-harm` и `sexual`, но не к `sexual/minors`; остальные категории text-only.

Для code harness в `code/main.py` мы сворачиваем sub-categories `/threatening`, `/intent`, `/instructions` и `/graphic` в их top-level parents ради педагогической простоты. Production code должен использовать полную 13-category schema.

На 42% лучше на multilingual test set, чем moderation endpoint предыдущего поколения. Per-category scores; приложения задают thresholds.

### Llama Guard 3/4

Покрыт в Lesson 16. 14 MLCommons hazard categories (организованы иначе, чем 13 response-schema booleans OpenAI). Поддерживает 8 languages (v3). Llama Guard 4 (April 2025) — нативно multimodal, 12B.

Таксономии OpenAI и Llama Guard пересекаются, но расходятся. OpenAI имеет "illicit" как широкую категорию; Llama Guard разделяет "violent crimes" и "non-violent crimes". Развертывания выбирают по соответствию своей policy taxonomy.

### Perspective API (Google Jigsaw)

Система toxicity scoring, предшествующая волне LLM-as-moderator (pre-2020). Categories: TOXICITY, SEVERE_TOXICITY, INSULT, PROFANITY, THREAT, IDENTITY_ATTACK. Single-dimension primary score (TOXICITY) с sub-dimension variants.

Широко используется как baseline в исследованиях content-moderation, потому что API стабилен, документирован и имеет годы calibration data. Для современных LLM-adjacent use cases Llama Guard или OpenAI Moderation обычно подходят лучше.

### Three-layer pattern

1. **Input moderation.** Классифицировать prompt пользователя до generation. Отклонить, если flagged. Latency: один вызов classifier.
2. **Output moderation.** Классифицировать output модели до доставки. Заменить refusal, если flagged. Latency: один вызов classifier после generation.
3. **Custom moderation.** Domain-specific rules (regex, allowlists, business policy). Запускается либо на input, либо на output.

Три слоя последовательны по замыслу: input moderation должен завершиться до generation, а output moderation запускается после generation. Parallelism применяется внутри слоя — одновременный запуск нескольких classifiers (e.g., OpenAI Moderation + Llama Guard + Perspective) на одном и том же тексте скрывает per-classifier latency. Как необязательная оптимизация, placeholder response ("one moment, checking...") может показываться, пока завершается input moderation и token-1 streaming откладывается. Flag behaviour настраивается: refuse, sanitize, escalate to human review.

### Failure modes

- **Input only.** Не ловит output hallucinations (Lesson 12-14 encoding attacks bypass input classifiers).
- **Output only.** Позволяет любому input попасть в модель; увеличивает cost; раскрывает internal reasoning атакующему.
- **Custom only.** Нестойко по категориям; regexes хрупкие.

Layered — default. Belt-and-suspenders.

### Azure deprecation

Azure Content Moderator: deprecated February 2024, retired February 2027. Заменен Azure AI Content Safety, который основан на LLM и интегрируется с Azure OpenAI. Migration — field-level project 2024-2027 для Azure deployments.

### Где это находится в Phase 18

Lesson 16 покрывает moderation tooling в red-team context. Lesson 29 покрывает deployed moderation. Lesson 30 завершает текущими доказательствами dual-use capabilities.

## Применение

`code/main.py` строит three-layer moderation harness: input moderator (keyword + category score), output moderator (тот же classifier на output), custom moderator (domain rules). Вы можете прогонять inputs и наблюдать, какой слой что ловит.

## Результат

Этот урок создает `outputs/skill-moderation-stack.md`. Для развертывания он рекомендует configuration moderation stack: какой classifier на input, какой на output, какие custom rules и какой judge для edge cases.

## Упражнения

1. Запустите `code/main.py`. Прогоните benign, borderline и harmful input через все три слоя. Сообщите, какой слой срабатывает для каждого.

2. Расширьте harness Perspective-API-style toxicity scoring на конкретную категорию. Сравните его threshold behaviour с category score.

3. Прочитайте docs OpenAI Moderation API и category list Llama Guard 3. Сопоставьте каждую категорию OpenAI с ближайшими категориями Llama Guard. Определите три категории, которые не сопоставляются чисто.

4. Спроектируйте moderation stack для code-assistant deployment (e.g., GitHub Copilot). Определите наиболее и наименее релевантные категории и предложите custom rules.

5. Azure Content Moderator retires February 2027. Спланируйте migration на Azure AI Content Safety. Определите самый рискованный элемент migration.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| OpenAI Moderation | "omni-moderation-latest" | GPT-4o-based 13-category (text) classifier с частичной multimodal support |
| Perspective API | "Google Jigsaw toxicity" | Baseline toxicity scoring до эпохи LLM |
| Llama Guard | "MLCommons 14-category" | Hazard classifier Meta (v3: 8B text, 8 langs; v4: 12B multimodal) |
| Input moderation | "pre-generation filter" | Classifier на prompt пользователя до model call |
| Output moderation | "post-generation filter" | Classifier на output модели до delivery |
| Custom moderation | "domain rules" | Deployment-specific rules (regex, allowlist, policy) |
| Layered moderation | "all three layers" | Стандартный pattern production deployment |

## Дополнительное чтение

- [OpenAI Moderation API docs](https://platform.openai.com/docs/api-reference/moderations) — omni-moderation endpoint
- [Meta PurpleLlama + Llama Guard](https://github.com/meta-llama/PurpleLlama) — repo Llama Guard
- [Google Jigsaw Perspective API](https://perspectiveapi.com/) — toxicity scoring
- [Azure AI Content Safety](https://learn.microsoft.com/en-us/azure/ai-services/content-safety/) — замена Azure
