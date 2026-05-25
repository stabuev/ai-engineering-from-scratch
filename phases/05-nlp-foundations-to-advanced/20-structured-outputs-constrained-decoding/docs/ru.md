# Структурированные выводы и ограниченное декодирование

> Попросите LLM вернуть JSON. Получите JSON почти всегда. В продакшене «почти» и есть проблема. Ограниченное декодирование (constrained decoding) превращает «почти» в «всегда», редактируя логиты перед сэмплированием.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Phase 5 · 17 (Chatbots), Phase 5 · 19 (Subword Tokenization)
**Время:** ~60 минут

## Проблема

Классификатор отправляет LLM промпт: "Return one of {positive, negative, neutral}." Модель возвращает "The sentiment is positive — this review is overwhelmingly favorable because the customer explicitly states that they ...". Ваш парсер падает. F1 вашего классификатора равен 0.0.

Свободная генерация не является контрактом. Это предложение. Продакшен-системе нужен контракт.

В 2026 году есть три слоя.

1. **Промптинг.** Попросить вежливо. "Return only the JSON object." Работает примерно в 80% случаев на frontier-моделях, хуже на меньших моделях.
2. **Нативные API структурированного вывода.** OpenAI `response_format`, Anthropic tool use, Gemini JSON mode. Надежны для поддерживаемых схем. Привязаны к вендору.
3. **Ограниченное декодирование (constrained decoding).** Изменять логиты на каждом шаге генерации так, чтобы модель *не могла* выдать недопустимые токены. 100% валидность по построению. Работает на любой локальной модели.

Этот урок формирует интуицию для всех трех подходов и объясняет, когда выбирать каждый из них.

## Концепция

![Ограниченное декодирование маскирует недопустимые токены на каждом шаге](../assets/constrained-decoding.svg)

**Как работает ограниченное декодирование.** На каждом шаге генерации LLM выдает вектор логитов по всему словарю (~100k токенов). *Обработчик логитов* (logit processor) находится между моделью и сэмплером. Он вычисляет, какие токены допустимы с учетом текущей позиции в целевой грамматике — JSON Schema, regex, контекстно-свободная грамматика — и устанавливает логиты всех недопустимых токенов в минус бесконечность. Softmax по оставшимся логитам распределяет вероятностную массу только по допустимым продолжениям.

Реализации в 2026 году:

- **Outlines.** Компилирует JSON Schema или regex в конечный автомат (finite-state machine). Каждый токен получает O(1)-поиск допустимого следующего токена. Основано на FSM, поэтому рекурсивные схемы нужно уплощать.
- **XGrammar / llguidance.** Движки контекстно-свободных грамматик (context-free grammar). Обрабатывают рекурсивную JSON Schema. Почти нулевой overhead декодирования. OpenAI упоминала llguidance в своей реализации структурированного вывода 2025 года.
- **vLLM guided decoding.** Встроенные `guided_json`, `guided_regex`, `guided_choice`, `guided_grammar` через backend'ы Outlines, XGrammar или lm-format-enforcer.
- **Instructor.** Обертка на основе Pydantic поверх любой LLM. Делает повторы при ошибке валидации. Кросс-провайдерный, но не изменяет логиты — полагается на повторы + промпты, учитывающие структурированный вывод.

### Контринтуитивный результат

Ограниченное декодирование часто *быстрее*, чем неограниченная генерация. Две причины. Во-первых, оно сужает пространство поиска следующего токена. Во-вторых, умные реализации полностью пропускают генерацию токенов для принудительных токенов (scaffolding вроде `{"name": "` — каждый байт уже определен).

### Ловушка, которая дорого обходится

Порядок полей важен. Поставьте `answer` перед `reasoning`, и модель зафиксирует ответ до того, как подумает. JSON валиден. Ответ неверен. Никакая валидация это не поймает.

```json
// BAD
{"answer": "yes", "reasoning": "because ..."}

// GOOD
{"reasoning": "... therefore ...", "answer": "yes"}
```

Порядок полей схемы — это логика, а не форматирование.

## Соберите это

### Шаг 1: regex-ограниченная генерация с нуля

См. `code/main.py` для самостоятельной реализации FSM. Основная идея в 30 строках:

```python
def mask_logits(logits, valid_token_ids):
    mask = [float("-inf")] * len(logits)
    for tid in valid_token_ids:
        mask[tid] = logits[tid]
    return mask


def generate_constrained(model, tokenizer, prompt, fsm):
    ids = tokenizer.encode(prompt)
    state = fsm.initial_state
    while not fsm.is_accept(state):
        logits = model.next_token_logits(ids)
        valid = fsm.valid_tokens(state, tokenizer)
        logits = mask_logits(logits, valid)
        tok = sample(logits)
        ids.append(tok)
        state = fsm.transition(state, tok)
    return tokenizer.decode(ids)
```

FSM отслеживает, какие части грамматики мы уже удовлетворили. `valid_tokens(state, tokenizer)` вычисляет, какие токены словаря могут продвинуть FSM, не выходя из принимающего пути.

### Шаг 2: Outlines для JSON Schema

```python
from pydantic import BaseModel
from typing import Literal
import outlines


class Review(BaseModel):
    sentiment: Literal["positive", "negative", "neutral"]
    confidence: float
    evidence_span: str


model = outlines.models.transformers("meta-llama/Llama-3.2-3B-Instruct")
generator = outlines.generate.json(model, Review)

result = generator("Classify: 'The wait staff was attentive and the food arrived hot.'")
print(result)
# Review(sentiment='positive', confidence=0.93, evidence_span='attentive ... hot')
```

Ноль ошибок валидации. Всегда. FSM делает недопустимый вывод недостижимым.

### Шаг 3: Instructor для провайдер-независимого Pydantic

```python
import instructor
from anthropic import Anthropic
from pydantic import BaseModel, Field


class Invoice(BaseModel):
    vendor: str
    total_usd: float = Field(ge=0)
    line_items: list[str]


client = instructor.from_anthropic(Anthropic())
invoice = client.messages.create(
    model="claude-opus-4-7",
    max_tokens=1024,
    response_model=Invoice,
    messages=[{"role": "user", "content": "Extract from: 'Acme Corp $420. Widget, Gizmo.'"}],
)
```

Другой механизм. Instructor не трогает логиты. Он форматирует схему в промпт, парсит вывод и повторяет запрос при ошибке валидации (по умолчанию 3 раза). Работает с любым провайдером. Повторы добавляют задержку и стоимость. Кросс-провайдерная переносимость — его главный плюс.

### Шаг 4: нативные API вендоров

```python
from openai import OpenAI

client = OpenAI()
response = client.responses.create(
    model="gpt-5",
    input=[{"role": "user", "content": "Classify: 'The food was cold.'"}],
    text={"format": {"type": "json_schema", "name": "sentiment",
          "schema": {"type": "object", "required": ["sentiment"],
                     "properties": {"sentiment": {"type": "string",
                                                  "enum": ["positive", "negative", "neutral"]}}}}},
)
print(response.output_parsed)
```

Серверное ограниченное декодирование. Паритет надежности с Outlines для поддерживаемых схем. Не нужно управлять локальной моделью. Привязывает вас к вендору.

## Ловушки

- **Рекурсивные схемы.** Outlines уплощает рекурсию до фиксированной глубины. Древовидные выводы (вложенные комментарии, AST) требуют XGrammar или llguidance (на основе CFG).
- **Огромные enum.** enum на 10 000 вариантов компилируется медленно или уходит в timeout. Переключитесь на retriever: сначала предскажите top-k кандидатов, затем ограничьте вывод ими.
- **Слишком строгая грамматика.** Принудительный regex `date: "YYYY-MM-DD"` не дает модели вывести `"unknown"` для отсутствующих дат. Модель компенсирует это, выдумывая дату. Разрешите `null` или sentinel.
- **Преждевременная фиксация.** См. ловушку порядка полей выше. Всегда ставьте reasoning первым.
- **Вендорский JSON mode без схемы.** Чистый JSON mode гарантирует только валидный JSON, а не валидность *для вашего сценария*. Всегда предоставляйте полную схему.

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Модель OpenAI/Anthropic/Google, простая схема | Нативный структурированный вывод вендора |
| Любой провайдер, Pydantic workflow, можно терпеть повторы | Instructor |
| Локальная модель, нужна 100% валидность, плоская схема | Outlines (FSM) |
| Локальная модель, рекурсивная схема | XGrammar или llguidance |
| Self-hosted inference server | Guided decoding в vLLM |
| Batch processing с допустимыми повторами | Instructor + самая дешевая модель |

## Доведите до продакшена

Сохраните как `outputs/skill-structured-output-picker.md`:

```markdown
---
name: structured-output-picker
description: Choose a structured output approach, schema design, and validation plan.
version: 1.0.0
phase: 5
lesson: 20
tags: [nlp, llm, structured-output]
---

Given a use case (provider, latency budget, schema complexity, failure tolerance), output:

1. Mechanism. Native vendor structured output, Instructor retries, Outlines FSM, or XGrammar CFG. One-sentence reason.
2. Schema design. Field order (reasoning first, answer last), nullable fields for "unknown", enum vs regex, required fields.
3. Failure strategy. Max retries, fallback model, graceful `null` handling, out-of-distribution refusal.
4. Validation plan. Schema compliance rate (target 100%), semantic validity (LLM-judge), field-coverage rate, latency p50/p99.

Refuse any design that puts `answer` or `decision` before reasoning fields. Refuse to use bare JSON mode without a schema. Flag recursive schemas behind an FSM-only library.
```

## Упражнения

1. **Легко.** Запустите промпт к небольшой open-weights модели (например, Llama-3.2-3B) без ограниченного декодирования для `Review(sentiment, confidence, evidence_span)`. Измерьте долю выводов, которые парсятся как валидный JSON, на 100 отзывах.
2. **Средне.** Тот же корпус с JSON mode в Outlines. Сравните compliance rate, задержку и семантическую точность.
3. **Сложно.** Реализуйте regex-ограниченный декодер с нуля для телефонных номеров (`\d{3}-\d{3}-\d{4}`). Проверьте 0 недопустимых выводов на 1000 сэмплах.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| Ограниченное декодирование (constrained decoding) | Заставить вывод быть валидным | Маскировать логиты недопустимых токенов на каждом шаге генерации. |
| Обработчик логитов (logit processor) | То, что ограничивает | Функция: `(logits, state) -> masked_logits`. |
| FSM | Конечный автомат | Скомпилированное представление грамматики; O(1)-поиск допустимого следующего токена. |
| CFG | Контекстно-свободная грамматика | Грамматика, которая обрабатывает рекурсию; медленнее, но выразительнее FSM. |
| Порядок полей схемы | Это важно? | Да — первое поле фиксирует решение; всегда ставьте reasoning перед answer. |
| Guided decoding | Название этого в vLLM | Та же концепция, интегрированная в inference server. |
| JSON mode | Ранняя версия OpenAI | Гарантирует синтаксис JSON; НЕ гарантирует совпадение со схемой. |

## Дополнительное чтение

- [Willard, Louf (2023). Efficient Guided Generation for LLMs](https://arxiv.org/abs/2307.09702) — статья Outlines.
- [XGrammar paper (2024)](https://arxiv.org/abs/2411.15100) — быстрое ограниченное декодирование на основе CFG.
- [vLLM — Structured Outputs](https://docs.vllm.ai/en/latest/features/structured_outputs.html) — интеграция с inference server.
- [OpenAI — Structured Outputs guide](https://platform.openai.com/docs/guides/structured-outputs) — справочник API + подводные камни.
- [Instructor library](https://python.useinstructor.com/) — Pydantic + повторы между провайдерами.
- [JSONSchemaBench (2025)](https://arxiv.org/abs/2501.10868) — бенчмарк 6 фреймворков ограниченного декодирования.
