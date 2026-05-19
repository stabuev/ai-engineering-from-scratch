# API и ключи

> Любой AI API работает одинаково: отправляете запрос — получаете ответ. Меняются детали, но не сам паттерн.

**Тип:** Практика  
**Языки:** Python, TypeScript  
**Предварительные требования:** Фаза 0, Урок 01  
**Время:** ~30 минут

## Цели обучения

- Безопасно хранить API-ключи с помощью переменных окружения и `.env` файлов
- Выполнять запросы к LLM API через Python SDK Anthropic и через «сырой» HTTP
- Сравнивать форматы запросов/ответов SDK и HTTP для отладки
- Распознавать и обрабатывать типичные API-ошибки: аутентификацию и rate limits

## Проблема

Начиная с фазы 11 вы будете вызывать LLM API (Anthropic, OpenAI, Google). В фазах 13–16 вы будете строить агентов, использующих эти API в цикле. Вам нужно понимать, как работают API-ключи, как безопасно их хранить и как сделать свой первый API-запрос.

## Концепция

Каждый API-запрос включает:

1. Endpoint (URL)
2. API-ключ (аутентификация)
3. Тело запроса (что вы хотите)
4. Тело ответа (что вы получаете)

## Практика

### Шаг 1: безопасное хранение API-ключей

Никогда не храните API-ключи прямо в коде. Используйте переменные окружения.

```bash
export ANTHROPIC_API_KEY="sk-ant-..."
export OPENAI_API_KEY="sk-..."
```

Или используйте `.env` файл (добавьте его в `.gitignore`):

```text
ANTHROPIC_API_KEY=sk-ant-...
OPENAI_API_KEY=sk-...
```

### Шаг 2: первый API-запрос (Python)

```python
import anthropic

client = anthropic.Anthropic()

response = client.messages.create(
    model="claude-sonnet-4-20250514",
    max_tokens=256,
    messages=[{"role": "user", "content": "What is a neural network in one sentence?"}]
)

print(response.content[0].text)
```

### Шаг 3: первый API-запрос (TypeScript)

```typescript
import Anthropic from "@anthropic-ai/sdk";

const client = new Anthropic();

const response = await client.messages.create({
  model: "claude-sonnet-4-20250514",
  max_tokens: 256,
  messages: [{ role: "user", content: "What is a neural network in one sentence?" }],
});

console.log(response.content[0].text);
```

### Шаг 4: «Сырой» HTTP (без SDK)

```python
import os
import urllib.request
import json

url = "https://api.anthropic.com/v1/messages"
headers = {
    "Content-Type": "application/json",
    "x-api-key": os.environ["ANTHROPIC_API_KEY"],
    "anthropic-version": "2023-06-01",
}
body = json.dumps({
    "model": "claude-sonnet-4-20250514",
    "max_tokens": 256,
    "messages": [{"role": "user", "content": "What is a neural network in one sentence?"}],
}).encode()
```

Понимание HTTP-запросов помогает при отладке SDK.

## Использование

| API | Когда нужен | Бесплатный тариф |
|-----|-------------|-----------------|
| Anthropic (Claude) | Фазы 11–16 | $5 после регистрации |
| OpenAI | Фаза 11 | $5 после регистрации |
| Hugging Face | Фазы 4–10 | Бесплатно |

## Упражнения

1. Получите API-ключ Anthropic и сделайте первый API-запрос
2. Попробуйте вариант с HTTP и сравните формат ответа с SDK
3. Намеренно используйте неверный API-ключ и изучите сообщение об ошибке

## Ключевые термины

| Термин | Как обычно говорят | Что это означает на самом деле |
|------|----------------|----------------------|
| API key | «Пароль для API» | Уникальная строка, идентифицирующая ваш аккаунт и авторизующая запросы |
| Rate limit | «Меня ограничивают» | Максимальное число запросов в минуту/час |
| Token | «Слово» | Единица тарификации входных и выходных данных |
| Streaming | «Ответ в реальном времени» | Получение ответа по частям, а не ожидание полного результата |
