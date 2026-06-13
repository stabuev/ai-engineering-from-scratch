# Связывание и разрешение неоднозначности сущностей

> NER нашел "Paris." Связывание сущностей решает: Paris, France? Paris Hilton? Paris, Texas? Paris (the Trojan prince)? Без связывания ваш граф знаний остается неоднозначным.

**Тип:** Практика
**Языки:** Python
**Предварительные требования:** Фаза 5 · 06 (NER), Фаза 5 · 24 (Разрешение кореференции)
**Время:** ~60 минут

## Цели обучения

- Строить индекс алиасов из редиректов Википедии и разрешать неоднозначность по контексту.
- Добавлять linking на эмбеддингах (в стиле BLINK) и генеративный.
- Обрабатывать NIL (нет сущности) и измерять генерацию кандидатов против разрешения.

## Проблема

Предложение гласит: "Jordan beat the press." Ваш NER помечает "Jordan" как PERSON. Хорошо. Но *какой* Jordan?

- Michael Jordan (баскетбол)?
- Michael B. Jordan (актер)?
- Michael I. Jordan (профессор ML в Berkeley — да, такая путаница реальна в ML-статьях)?
- Jordan (страна)?
- Jordan (еврейское имя)?

Связывание сущностей (Entity linking, EL) разрешает каждое упоминание в уникальную запись базы знаний: Wikidata, Wikipedia, DBpedia или вашу доменную KB. Две подзадачи:

1. **Генерация кандидатов.** Для "Jordan," какие записи KB правдоподобны?
2. **Разрешение неоднозначности.** С учетом контекста, какой кандидат правильный?

Оба шага обучаемы. Оба имеют бенчмарки. Объединенный конвейер стабилен уже десятилетие — меняется качество дизамбигуатора.

## Концепция

![Конвейер связывания сущностей: упоминание → кандидаты → разрешенная сущность](../assets/entity-linking.svg)

**Генерация кандидатов.** По поверхностной форме упоминания ("Jordan") найдите кандидатов в индексе алиасов. Словари алиасов Wikipedia покрывают большинство именованных сущностей: "JFK" → John F. Kennedy, Jacqueline Kennedy, JFK airport, JFK (movie). Типичный индекс возвращает 10-30 кандидатов на упоминание.

**Разрешение неоднозначности: три подхода.**

1. **Априорная вероятность + контекст (Milne & Witten, 2008).** `P(entity | mention) × context-similarity(entity, text)`. Хорошо работает, быстро, без обучения.
2. **На эмбеддингах (ESS / REL / Blink).** Закодировать упоминание + контекст. Закодировать описание каждого кандидата. Выбрать максимум cosine. Вариант по умолчанию в 2020-2024.
3. **Генеративный (GENRE, 2021; на основе LLM, 2023+).** Декодировать каноническое имя сущности токен за токеном. Ограничить trie допустимых имен сущностей, чтобы выход гарантированно был валидным KB id.

**End-to-end против конвейера.** Современные модели (ELQ, BLINK, ExtEnD, GENRE) выполняют NER + генерацию кандидатов + разрешение неоднозначности за один проход. Конвейерные системы по-прежнему доминируют в продакшене, потому что компоненты можно заменять.

### Две меры

- **Recall упоминаний (candidate gen).** Доля золотых упоминаний, где правильная запись KB появляется в списке кандидатов. Нижняя граница для всего конвейера.
- **Accuracy / F1 разрешения неоднозначности.** При правильных кандидатах, как часто top-1 верен.

Всегда сообщайте оба показателя. Система с 99% дизамбигуации при 80% candidate recall — это 80% конвейер.

## Соберите это

### Шаг 1: постройте индекс алиасов из редиректов Wikipedia

```python
alias_to_entities = {
    "jordan": ["Q41421 (Michael Jordan)", "Q810 (Jordan, country)", "Q254110 (Michael B. Jordan)"],
    "paris":  ["Q90 (Paris, France)", "Q663094 (Paris, Texas)", "Q55411 (Paris Hilton)"],
    "apple":  ["Q312 (Apple Inc.)", "Q89 (apple, fruit)"],
}
```

Данные алиасов Wikipedia: ~18M пар (alias, entity). Скачайте из дампов Wikidata. Храните как инвертированный индекс.

### Шаг 2: разрешение неоднозначности на основе контекста

```python
def disambiguate(mention, context, alias_index, entity_desc):
    candidates = alias_index.get(mention.lower(), [])
    if not candidates:
        return None, 0.0
    context_words = set(tokenize(context))
    best, best_score = None, -1
    for entity_id in candidates:
        desc_words = set(tokenize(entity_desc[entity_id]))
        union = len(context_words | desc_words)
        score = len(context_words & desc_words) / union if union else 0.0
        if score > best_score:
            best, best_score = entity_id, score
    return best, best_score
```

Jaccard overlap — игрушечный вариант. Замените на cosine similarity по эмбеддингам (см. `code/main.py` step-2 для transformer-версии).

### Шаг 3: на эмбеддингах (в стиле BLINK)

```python
from sentence_transformers import SentenceTransformer
encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

def embed_mention(text, mention_span):
    start, end = mention_span
    marked = f"{text[:start]} [MENTION] {text[start:end]} [/MENTION] {text[end:]}"
    return encoder.encode([marked], normalize_embeddings=True)[0]

def embed_entity(entity_id, description):
    return encoder.encode([f"{entity_id}: {description}"], normalize_embeddings=True)[0]
```

Во время индексации один раз создайте embedding каждой сущности KB. Во время запроса один раз создайте embedding упоминания + контекста, выполните dot-product с пулом кандидатов, выберите максимум.

### Шаг 4: генеративное связывание сущностей (концепция)

GENRE декодирует заголовок Wikipedia сущности посимвольно. Ограниченное декодирование (см. урок 20) гарантирует, что можно вывести только допустимые заголовки. Тесная интеграция с trie, подкрепленным KB. Современный наследник — REL-GEN и LLM-prompted EL со structured output.

```python
prompt = f"""Text: {text}
Mention: {mention}
List the best Wikipedia title for this mention.
Respond with JSON: {{"title": "..."}}"""
```

В сочетании с whitelist (Outlines `choice`) это самый простой EL-конвейер для поставки в 2026 году.

### Шаг 5: оцените на AIDA-CoNLL

AIDA-CoNLL — стандартный EL-бенчмарк: 1,393 статьи Reuters, 34k упоминаний, сущности Wikipedia. Сообщайте in-KB accuracy (`P@1`) и out-of-KB NIL-detection rate.

## Подводные камни

- **Обработка NIL.** Некоторые упоминания отсутствуют в KB (новые сущности, малоизвестные люди). Системы должны предсказывать NIL вместо угадывания неправильной сущности. Измеряется отдельно.
- **Ошибки границ упоминаний.** Вышестоящий NER пропускает частичные span ("Bank of America" помечен только как "Bank"). EL recall падает.
- **Смещение популярности.** Обученные системы чрезмерно предсказывают частые сущности. Упоминание "Michael I. Jordan" в ML-статье часто связывается с баскетбольным Jordan.
- **Кросс-языковой EL.** Сопоставление упоминаний в китайском тексте с сущностями English Wikipedia. Требует многоязычного энкодера или шага перевода.
- **Устаревание KB.** Новых компаний, событий, людей нет в прошлогоднем дампе Wikipedia. Продакшен-конвейерам нужен цикл обновления.

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| General-purpose English + Wikipedia | BLINK или REL |
| Кросс-языковой, KB = Wikipedia | mGENRE |
| LLM-friendly, несколько упоминаний/день | Prompt Claude/GPT-4 со списком кандидатов + constrained JSON |
| Доменная KB (медицина, право) | Custom BERT с KB-aware retrieval + fine-tune на доменном AIDA-style наборе |
| Экстремально низкая задержка | Только exact-match prior (Milne-Witten baseline) |
| Исследовательский SOTA | GENRE / ExtEnD / generative LLM-EL |

Продакшен-паттерн, который поставляют в 2026 году: NER → coref → EL на каждом упоминании → сворачивание кластеров к одной канонической сущности на кластер. Выход: один KB id на сущность в документе, а не один на упоминание.

## Доведите до поставки

Сохраните как `outputs/skill-entity-linker.md`:

```markdown
---
name: entity-linker
description: Design an entity linking pipeline — KB, candidate generator, disambiguator, evaluation.
version: 1.0.0
phase: 5
lesson: 25
tags: [nlp, entity-linking, knowledge-graph]
---

Given a use case (domain KB, language, volume, latency budget), output:

1. Knowledge base. Wikidata / Wikipedia / custom KB. Version date. Refresh cadence.
2. Candidate generator. Alias-index, embedding, or hybrid. Target mention recall @ K.
3. Disambiguator. Prior + context, embedding-based, generative, or LLM-prompted.
4. NIL strategy. Threshold on top score, classifier, or explicit NIL candidate.
5. Evaluation. Mention recall @ 30, top-1 accuracy, NIL-detection F1 on held-out set.

Refuse any EL pipeline without a mention-recall baseline (you cannot evaluate a disambiguator without knowing candidate gen surfaced the right entity). Refuse any pipeline using LLM-prompted EL without constrained output to valid KB ids. Flag systems where popularity bias affects minority entities (e.g. name-clashes) without domain fine-tuning.
```

## Упражнения

1. **Легко.** Реализуйте дизамбигуатор prior+context в `code/main.py` на 10 неоднозначных упоминаниях (Paris, Jordan, Apple). Вручную разметьте правильную сущность. Измерьте accuracy.
2. **Средне.** Закодируйте 50 неоднозначных упоминаний с помощью sentence transformer. Создайте embedding описания каждого кандидата. Сравните разрешение неоднозначности на эмбеддингах с Jaccard context overlap.
3. **Сложно.** Постройте доменную KB на 1k сущностей (например, сотрудники + продукты в вашей компании). Реализуйте NER + EL end-to-end. Измерьте precision и recall на 100 held-out предложениях.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-------------------|--------------------------------|
| Связывание сущностей (EL) | Ссылка на Wikipedia | Сопоставить упоминание с уникальной записью KB. |
| Генерация кандидатов | Кто это может быть? | Вернуть короткий список правдоподобных записей KB для упоминания. |
| Разрешение неоднозначности | Выбрать правильное | Оценить кандидатов по контексту, выбрать победителя. |
| Индекс алиасов | Таблица поиска | Отображение поверхностная форма → кандидатные сущности. |
| NIL | Нет в KB | Явное предсказание, что ни одна запись KB не подходит. |
| KB | База знаний | Wikidata, Wikipedia, DBpedia или ваша доменная KB. |
| AIDA-CoNLL | Бенчмарк | 1,393 статьи Reuters с золотыми ссылками сущностей. |

## Дополнительное чтение

- [Milne, Witten (2008). Learning to Link with Wikipedia](https://www.cs.waikato.ac.nz/~ihw/papers/08-DM-IHW-LearningToLinkWithWikipedia.pdf) — основополагающий подход prior+context.
- [Wu et al. (2020). Zero-shot Entity Linking with Dense Entity Retrieval (BLINK)](https://arxiv.org/abs/1911.03814) — рабочая лошадка на эмбеддингах.
- [De Cao et al. (2021). Autoregressive Entity Retrieval (GENRE)](https://arxiv.org/abs/2010.00904) — генеративный EL с ограниченным декодированием.
- [Hoffart et al. (2011). Robust Disambiguation of Named Entities in Text (AIDA)](https://www.aclweb.org/anthology/D11-1072.pdf) — статья о бенчмарке.
- [REL: An Entity Linker Standing on the Shoulders of Giants (2020)](https://arxiv.org/abs/2006.01969) — открытый продакшен-стек.
