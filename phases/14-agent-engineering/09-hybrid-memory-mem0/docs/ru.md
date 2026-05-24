# Гибридная память: Vector + Graph + KV (Mem0)

> Mem0 (Chhikara et al., 2025) рассматривает память как три параллельных хранилища — vector для семантического сходства, KV для быстрого поиска фактов, graph для reasoning по сущностям и отношениям. Слой скоринга объединяет все три при retrieval. Это продакшен-стандарт 2026 года для внешней памяти.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 07 (MemGPT), Фаза 14 · 08 (Letta Blocks)
**Время:** ~75 минут

## Цели обучения

- Объяснить, почему одного хранилища (только vector, только graph, только KV) недостаточно для памяти агента.
- Назвать три параллельных хранилища Mem0 и то, под что оптимизировано каждое.
- Описать fusion scoring Mem0 — relevance, importance, recency — и почему это взвешенная сумма, а не иерархия.
- Реализовать игрушечную память из трех хранилищ на stdlib с `add()`, который пишет во все три, и `search()`, который объединяет результаты.

## Проблема

Одно хранилище ошибочно для одного из трех классов запросов:

- **Semantic similarity** — "что мы обсуждали про agent drift на прошлой неделе?" Vector побеждает; KV и graph промахиваются.
- **Fact lookup** — "какой номер телефона пользователя?" KV побеждает; vector расточителен, graph избыточен.
- **Relationship reasoning** — "у каких клиентов одна и та же billing entity?" Graph побеждает; vector и KV не могут ответить.

Продакшен-агенты задают все три типа в одной сессии. Память с одним хранилищем всегда ошибается на двух из них. Вклад Mem0 — подключить все три за единой поверхностью `add`/`search` со scoring-функцией, которая их объединяет.

## Концепция

### Три хранилища параллельно

Mem0 (arXiv:2504.19413, April 2025) при `add(text, user_id, metadata)`:

1. Извлекает candidate facts из текста (LLM-driven шаг).
2. Записывает каждый факт в vector store (embedding) для semantic search.
3. Записывает каждый факт в KV store с ключом (user_id, fact_type, entity) для O(1) lookup.
4. Записывает каждый факт в graph store (Mem0g) как типизированные ребра для relationship queries.

При `search(query, user_id)`:

1. Vector store возвращает top-k по embedding cosine.
2. KV store возвращает прямые попадания по query-derived (user_id, type, entity).
3. Graph store возвращает подграф, достижимый из сущностей запроса.
4. Слой scoring объединяет все три.

### Fusion scoring

```
score = w_relevance * relevance(q, record)
      + w_importance * importance(record)
      + w_recency * recency(record)
```

- **Relevance** — vector cosine, точное совпадение KV, вес пути graph.
- **Importance** — проставляется при записи или обучается (некоторые факты важнее: имена, ID, политики).
- **Recency** — экспоненциальное затухание по времени с последней записи или чтения.

Веса настраиваются под продукт. Более высокий `w_recency` для chat agents; более высокий `w_importance` для compliance agents; более высокий `w_relevance` для retrieval agents.

### Mem0g и temporal reasoning

Mem0g добавляет conflict detector. Когда новый факт противоречит существующему ребру, существующее ребро помечается invalid, но не удаляется. Temporal queries ("какой город был у пользователя в марте?") обходят подграф valid-at-time.

Это compliance-grade поведение, которое обобщает паттерн инвалидации Letta.

### Числа benchmark

Статья Mem0 сообщает (2025):

- **LoCoMo** (long-form conversation memory): 91.6
- **LongMemEval** (long-horizon episodic memory): 93.4
- **BEAM 1M** (1M-token memory benchmark): 64.1

Базовые сравнения (full-context 128k LLM, flat vector store, flat KV) проигрывают на 10+ пунктов. Одни benchmarks не оправдывают выбор — важна эксплуатационная форма, — но числа показывают, что fusion-дизайн не является погрешностью округления.

### Таксономия scope

Mem0 разделяет память по scope:

- **User memory** — сохраняется между сессиями, ключуется по `user_id`.
- **Session memory** — сохраняется внутри одного thread.
- **Agent memory** — состояние отдельного экземпляра агента.

Каждая запись выбирает один scope. Retrieval может запрашивать несколько scope с весами на scope. Бездумное смешивание scope приводит к инцидентам вида "ассистент рассказал Alice о проекте Bob."

### Где этот паттерн ломается

- **Embedding drift.** Vector-результаты, которые выглядят правильно на первой сотне запросов, деградируют по мере роста корпуса. Добавьте периодический re-embedding top-N-used записей.
- **KV schema creep.** `(user_id, type, entity)` выглядит просто, пока каждая команда не добавляет свой `type`. Аудируйте набор типов ежеквартально.
- **Graph explosion.** Один шумный extractor добавляет 50 ребер на сообщение. Ограничьте graph-записи на один вызов `add`; отбрасывайте low-confidence edges.

## Соберите это

`code/main.py` реализует паттерн трех хранилищ на stdlib:

- `VectorStore` — наивное token-overlap similarity как замена embedding.
- `KVStore` — dict с ключом `(user_id, fact_type, entity)`.
- `GraphStore` — типизированные ребра (subject, relation, object, valid).
- `Mem0` — верхнеуровневый facade с `add()`, `search()`, fusion scoring и scope-aware retrieval.
- Проработанную трассу на multi-user, multi-session разговоре.

Запустите:

```
python3 code/main.py
```

Вывод показывает три отдельных пути recall плюс объединенный top-k. Поменяйте scoring weights вверху `main()` и посмотрите, как меняется ranking.

## Используйте это

- **Mem0 (Apache 2.0)** — production-ready. Self-host с Postgres + Qdrant + Neo4j или управляемое облако.
- **Letta** — трехуровневая схема core/recall/archival; приносите свои vector и graph backends.
- **Zep** — коммерческая альтернатива с temporal KG и fact extraction.
- **Custom builds** — когда нужен точный контроль над extractor (compliance) или fusion weights (voice agents, где recency доминирует).

## Доведите до продакшена

`outputs/skill-hybrid-memory.md` генерирует scaffold треххранилищной памяти с fusion scorer, scope taxonomy и подключенной temporal invalidation.

## Упражнения

1. Замените игрушечное vector similarity реальной embedding-моделью (sentence-transformers, Ollama, OpenAI embeddings). Измерьте recall@10 на синтетическом длинном разговоре. Дрейфует ли ranking после 1000 записей?
2. Добавьте temporal query: `search(query, as_of=timestamp)`. Возвращайте только записи, valid at or before этого времени. Какому хранилищу потребуется больше всего работы?
3. Реализуйте conflict detector: если входящий факт противоречит graph edge, инвалидируйте старое ребро и залогируйте оба. Протестируйте на "user lives in Berlin" -> "user lives in Lisbon."
4. Перенесите fusion scorer так, чтобы он включал измерение `user_feedback` (thumbs-up на retrieved records). Как предотвратить gaming (агент возвращает только записи, которые ему уже понравились)?
5. Прочитайте документацию Mem0 (`docs.mem0.ai`). Перенесите игрушку на вызовы клиента `mem0`. Сравните качество retrieval на тех же 20 test queries.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Hybrid memory | "Vector плюс graph плюс KV" | Три хранилища, записываемые параллельно и объединяемые при retrieval |
| Fact extraction | "Memory ingestion" | LLM-шаг, который разбивает текст на кортежи (entity, relation, fact) |
| Fusion scoring | "Relevance ranking" | Взвешенная сумма relevance, importance, recency |
| Scope | "Memory namespace" | user / session / agent — определяет, кто что видит |
| Mem0g | "Memory graph" | Типизированные ребра с temporal validity для relationship queries |
| Temporal invalidation | "Soft delete" | Пометить противоречивые ребра invalid; никогда не удалять |
| Embedding drift | "Retrieval rot" | Vector-качество деградирует по мере роста корпуса; периодически делать re-embed |

## Дополнительное чтение

- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — исходная статья
- [Mem0 docs](https://docs.mem0.ai/platform/overview) — production API, SDKs, managed cloud
- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — предшественник virtual-context
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — родственный трехуровневый дизайн
