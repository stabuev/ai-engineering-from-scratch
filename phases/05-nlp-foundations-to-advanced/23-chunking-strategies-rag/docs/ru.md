# Стратегии chunking для RAG

> Конфигурация chunking влияет на качество retrieval не меньше, чем выбор модели эмбеддингов (Vectara NAACL 2025). Ошибитесь с chunking — и никакой reranking вас не спасет.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 5 · 14 (Information Retrieval), Фаза 5 · 22 (Embedding Models)
**Время:** ~60 минут

## Цели обучения

- Реализовывать фиксированный, рекурсивный, семантический, parent-document и контекстный чанкинг.
- Объяснять, почему чанкинг влияет на качество поиска не меньше выбора модели эмбеддингов.
- Применять паттерн contextual retrieval от Anthropic.

## Проблема

Вы загружаете 50-страничный контракт в RAG-систему. Пользователь спрашивает: "What is the termination clause?" Retriever возвращает титульную страницу. Почему? Потому что модель обучалась на чанках по 512 токенов, а положение о расторжении находится на 20 страницах дальше, разорвано переносом страницы и не содержит локальных ключевых слов, связывающих его с запросом.

Решение — не "купить модель эмбеддингов получше". Решение — chunking. Насколько крупные чанки? Перекрытие? Где разбивать? С окружающим контекстом?

Бенчмарки февраля 2026 года показывают неожиданные результаты:

- Исследование Vectara 2026: recursive 512-token chunking превзошел semantic chunking по accuracy 69% → 54%.
- SPLADE + Mistral-8B на Natural Questions: overlap не дал измеримой пользы.
- Context cliff: качество ответов резко падает примерно на 2,500 токенах контекста.

"Очевидный" ответ (semantic chunking, 20% overlap, 1000 токенов) часто неверен. Этот урок развивает интуицию для шести стратегий и объясняет, когда какую выбирать.

## Концепция

![Шесть стратегий chunking, визуализированные на одном фрагменте](../assets/chunking.svg)

**Фиксированный chunking (fixed chunking).** Разбивать каждые N символов или токенов. Самый простой baseline. Рвет текст посреди предложения. Хорошее сжатие, плохая связность.

**Recursive.** `RecursiveCharacterTextSplitter` из LangChain. Сначала пробует разбивать по `\n\n`, затем по `\n`, затем по `.`, затем по пробелу. Аккуратно откатывается к более грубым правилам. Дефолт 2026 года.

**Семантический (semantic).** Встраивает каждое предложение. Вычисляет косинусное сходство между соседними предложениями. Разбивает там, где сходство падает ниже порога. Сохраняет тематическую связность. Медленнее; иногда создает крошечные фрагменты по 40 токенов, которые вредят retrieval.

**По предложениям (sentence).** Разбивать по границам предложений. Одно предложение на чанк или окно из N предложений. Соответствует semantic chunking примерно до ~5k токенов за долю стоимости.

**Parent-document.** Хранить маленькие дочерние чанки для retrieval *и* более крупный родительский чанк для контекста. Извлекать по дочернему; возвращать родительский. Деградирует плавно: плохие дочерние чанки все равно возвращают разумных родителей.

**Late chunking (2024).** Сначала встроить весь документ на уровне токенов, затем агрегировать токеновые эмбеддинги в эмбеддинги чанков. Сохраняет межчанковый контекст. Работает с long-context embedders (BGE-M3, Jina v3). Выше вычислительная стоимость.

**Contextual retrieval (Anthropic, 2024).** Добавлять перед каждым чанком LLM-сгенерированное резюме его положения в документе ("This chunk is section 3.2 of the termination clauses..."). Улучшение retrieval на 35-50% в собственном бенчмарке Anthropic. Дорого индексировать.

### Правило, которое бьет любой дефолт

Сопоставляйте размер чанка с типом запроса:

| Тип запроса | Размер чанка |
|------------|-----------|
| Фактоидный ("what is the CEO's name?") | 256-512 токенов |
| Аналитический / multi-hop | 512-1024 токена |
| Понимание целого раздела | 1024-2048 токенов |

Бенчмарк NVIDIA 2026 года. Чанк должен быть достаточно большим, чтобы содержать ответ плюс локальный контекст, и достаточно маленьким, чтобы top-K retriever возвращал фокус на ответе, а не шум контекста.

## Соберите это

### Шаг 1: fixed и recursive chunking

```python
def chunk_fixed(text, size=512, overlap=0):
    step = size - overlap
    return [text[i:i + size] for i in range(0, len(text), step)]


def chunk_recursive(text, size=512, seps=("\n\n", "\n", ". ", " ")):
    if len(text) <= size:
        return [text]
    for sep in seps:
        if sep not in text:
            continue
        parts = text.split(sep)
        chunks = []
        buf = ""
        for p in parts:
            if len(p) > size:
                if buf:
                    chunks.append(buf)
                    buf = ""
                chunks.extend(chunk_recursive(p, size=size, seps=seps[1:] or (" ",)))
                continue
            candidate = buf + sep + p if buf else p
            if len(candidate) <= size:
                buf = candidate
            else:
                if buf:
                    chunks.append(buf)
                buf = p
        if buf:
            chunks.append(buf)
        return [c for c in chunks if c.strip()]
    return chunk_fixed(text, size)
```

### Шаг 2: semantic chunking

```python
def chunk_semantic(text, encoder, threshold=0.6, min_chars=200, max_chars=2048):
    sentences = split_sentences(text)
    if not sentences:
        return []
    embs = encoder.encode(sentences, normalize_embeddings=True)
    chunks = [[sentences[0]]]
    for i in range(1, len(sentences)):
        sim = float(embs[i] @ embs[i - 1])
        current_len = sum(len(s) for s in chunks[-1])
        if sim < threshold and current_len >= min_chars:
            chunks.append([sentences[i]])
        else:
            chunks[-1].append(sentences[i])

    result = []
    for group in chunks:
        text_group = " ".join(group)
        if len(text_group) > max_chars:
            result.extend(chunk_recursive(text_group, size=max_chars))
        else:
            result.append(text_group)
    return result
```

Настраивайте `threshold` на своем домене. Слишком высокий → фрагменты. Слишком низкий → один гигантский чанк.

### Шаг 3: parent-document

```python
def chunk_parent_child(text, parent_size=2048, child_size=256):
    parents = chunk_recursive(text, size=parent_size)
    mapping = []
    for p_idx, parent in enumerate(parents):
        children = chunk_recursive(parent, size=child_size)
        for child in children:
            mapping.append({"child": child, "parent_idx": p_idx, "parent": parent})
    return mapping


def retrieve_parent(child_query, mapping, encoder, top_k=3):
    child_embs = encoder.encode([m["child"] for m in mapping], normalize_embeddings=True)
    q_emb = encoder.encode([child_query], normalize_embeddings=True)[0]
    scores = child_embs @ q_emb
    top = np.argsort(-scores)[:top_k]
    seen, parents = set(), []
    for i in top:
        if mapping[i]["parent_idx"] not in seen:
            parents.append(mapping[i]["parent"])
            seen.add(mapping[i]["parent_idx"])
    return parents
```

Ключевая мысль: дедуплицируйте родителей. Несколько дочерних чанков могут ссылаться на одного родителя; возвращать все означало бы тратить контекст впустую.

### Шаг 4: contextual retrieval (паттерн Anthropic)

```python
def contextualize_chunks(document, chunks, llm):
    context_prompts = [
        f"""<document>{document}</document>
Here is the chunk to situate: <chunk>{c}</chunk>
Write 50-100 words placing this chunk in the document's context."""
        for c in chunks
    ]
    contexts = llm.batch(context_prompts)
    return [f"{ctx}\n\n{c}" for ctx, c in zip(contexts, chunks)]
```

Индексируйте контекстуализированные чанки. Во время запроса retrieval выигрывает от дополнительного окружающего сигнала.

### Шаг 5: оценка

```python
def recall_at_k(queries, corpus_chunks, encoder, k=5):
    chunk_embs = encoder.encode(corpus_chunks, normalize_embeddings=True)
    hits = 0
    for q_text, gold_idxs in queries:
        q_emb = encoder.encode([q_text], normalize_embeddings=True)[0]
        top = np.argsort(-(chunk_embs @ q_emb))[:k]
        if any(i in gold_idxs for i in top):
            hits += 1
    return hits / len(queries)
```

Всегда бенчмаркайте. "Лучшая" стратегия для вашего корпуса может не совпасть ни с одним постом в блоге.

## Подводные камни

- **Chunking оценивается только на фактоидных запросах.** Multi-hop запросы выявляют совсем других победителей. Используйте eval-набор, стратифицированный по типам запросов.
- **Semantic chunking без минимального размера.** Создает фрагменты по 40 токенов, которые вредят retrieval. Всегда задавайте `min_tokens`.
- **Overlap как cargo cult.** Исследования 2026 года показывают, что overlap часто не дает пользы и удваивает стоимость индекса. Измеряйте, не предполагайте.
- **Нет контроля min/max.** Чанки по 5 токенов или 5000 токенов одинаково ломают retrieval. Зажимайте размер.
- **Cross-doc chunking.** Никогда не позволяйте чанку охватывать два документа. Всегда разбивайте per-doc, затем объединяйте.

## Используйте это

Стек 2026 года:

| Ситуация | Стратегия |
|-----------|----------|
| Первая сборка, неизвестный корпус | Recursive, 512 токенов, без overlap |
| Фактоидный QA | Recursive, 256-512 токенов |
| Аналитический / multi-hop | Recursive, 512-1024 токена + parent-document |
| Много cross-reference (контракты, статьи) | Late chunking или contextual retrieval |
| Разговорный / диалоговый корпус | Чанки на уровне реплик + speaker metadata |
| Короткие высказывания (tweets, reviews) | Один документ = один чанк |

Начните с recursive 512. Измерьте recall@5 на eval-наборе из 50 запросов. Настраивайте дальше оттуда.

## Доведите до поставки

Сохраните как `outputs/skill-chunker.md`:

```markdown
---
name: chunker
description: Pick a chunking strategy, size, and overlap for a given corpus and query distribution.
version: 1.0.0
phase: 5
lesson: 23
tags: [nlp, rag, chunking]
---

Given a corpus (document types, avg length, domain) and query distribution (factoid / analytical / multi-hop), output:

1. Strategy. Recursive / sentence / semantic / parent-document / late / contextual. Reason.
2. Chunk size. Token count. Reason tied to query type.
3. Overlap. Default 0; justify if >0.
4. Min/max enforcement. `min_tokens`, `max_tokens` guards.
5. Evaluation plan. Recall@5 on 50-query stratified eval set (factoid, analytical, multi-hop).

Refuse any chunking strategy without min/max chunk size enforcement. Refuse overlap above 20% without an ablation showing it helps. Flag semantic chunking recommendations without a min-token floor.
```

## Упражнения

1. **Легко.** Разбейте один 20-страничный документ с fixed(512, 0), recursive(512, 0) и recursive(512, 100). Сравните количество чанков и качество границ.
2. **Средне.** Соберите eval-набор из 30 запросов по 5 документам. Измерьте recall@5 для recursive, semantic и parent-document. Что выигрывает? Совпадает ли это с постами в блогах?
3. **Сложно.** Реализуйте contextual retrieval. Измерьте улучшение MRR относительно baseline recursive. Сообщите стоимость индекса (LLM calls) vs прирост accuracy.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| Чанк (chunk) | Кусок документа | Поддокументная единица, которую встраивают, индексируют и извлекают. |
| Overlap | Safety margin | N токенов, общих между соседними чанками; часто бесполезно в бенчмарках 2026 года. |
| Semantic chunking | Умный chunking | Разбивать там, где падает similarity эмбеддингов соседних предложений. |
| Parent-document | Двухуровневый retrieval | Извлекать маленьких children, возвращать более крупных parents. |
| Late chunking | Chunk after embedding | Встроить весь документ на уровне токенов, агрегировать в chunk vectors. |
| Contextual retrieval | Трюк Anthropic | LLM-сгенерированное резюме добавляется перед каждым чанком до индексирования. |
| Context cliff | 2500-token wall | Падение качества, наблюдаемое примерно на 2.5k токенов контекста в RAG (Jan 2026). |

## Дополнительное чтение

- [Yepes et al. / LangChain — Recursive Character Splitting docs](https://python.langchain.com/docs/how_to/recursive_text_splitter/) — дефолт в production.
- [Vectara (2024, NAACL 2025). Chunking configurations analysis](https://arxiv.org/abs/2410.13070) — chunking важен не меньше, чем выбор эмбеддингов.
- [Jina AI — Late Chunking in Long-Context Embedding Models (2024)](https://jina.ai/news/late-chunking-in-long-context-embedding-models/) — статья о late chunking.
- [Anthropic — Contextual Retrieval](https://www.anthropic.com/news/contextual-retrieval) — улучшение retrieval на 35-50% с LLM-сгенерированными context prefixes.
- [NVIDIA 2026 chunk-size benchmark — Premai summary](https://blog.premai.io/rag-chunking-strategies-the-2026-benchmark-guide/) — размер чанков по типам запросов.
