# Информационный поиск и Search

> BM25 точен, но хрупок. Dense охватывает широко, но пропускает ключевые слова. Hybrid — стандарт 2026 года. Все остальное — настройка.

**Тип:** Build
**Языки:** Python
**Предварительные требования:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 04 (GloVe, FastText, Subword)
**Время:** ~75 минут

## Проблема

Пользователь вводит "what happens if someone lies to get money" и ожидает найти норму, которая действительно это покрывает: "Section 420 IPC." Поиск по ключевым словам полностью ее пропускает (нет общей лексики). Семантический поиск пропускает ее, если embeddings не обучались на юридическом тексте. Реальный поиск должен справляться с обоими случаями.

IR — это pipeline под каждой RAG-системой, каждой поисковой строкой, каждым fuzzy lookup на сайте документации. Архитектура 2026 года, работающая в продакшене, — не один метод. Это цепочка взаимодополняющих методов, где каждый ловит отказы предыдущего.

В этом уроке мы строим каждую часть и называем, какие сбои она покрывает.

## Концепция

![Гибридный retrieval: BM25 + dense + RRF + cross-encoder rerank](../assets/retrieval.svg)

Четыре слоя. Выберите те, которые нужны.

1. **Sparse retrieval (BM25).** Быстрый, точный на exact matches, плохой на семантике. Работает поверх инвертированного индекса. Менее 10 мс на запрос по миллионам документов. Хорошо находит ссылки на нормы, коды продуктов, сообщения об ошибках, именованные сущности.
2. **Dense retrieval.** Кодирует запрос и документы в векторы. Поиск ближайших соседей. Улавливает перефразирования и семантическую близость. Пропускает точные совпадения ключевых слов, отличающиеся на один символ. 50-200 мс на запрос с FAISS или vector DB.
3. **Fusion.** Объединяет ранжированные списки из sparse и dense. Reciprocal Rank Fusion (RRF) — простой вариант по умолчанию, потому что игнорирует сырые scores (они живут в разных шкалах) и использует только позиции в ранжировании. Weighted fusion — вариант, когда вы знаете, что один сигнал доминирует в вашем домене.
4. **Cross-encoder rerank.** Возьмите top-30 после fusion. Запустите cross-encoder (query + document вместе, оценка каждой пары). Оставьте top-5. Cross-encoders медленнее на пару, чем bi-encoders, но гораздо точнее. Вы амортизируете стоимость, запуская их только на top-30.

Three-way retrieval (BM25 + dense + learned-sparse вроде SPLADE) превосходит two-way в бенчмарках 2026 года, но требует инфраструктуры для learned-sparse индексов. Для большинства команд two-way плюс cross-encoder rerank — оптимальная точка.

## Собираем

### Шаг 1: BM25 с нуля

```python
import math
import re
from collections import Counter

TOKEN_RE = re.compile(r"[a-z0-9]+")


def tokenize(text):
    return TOKEN_RE.findall(text.lower())


class BM25:
    def __init__(self, corpus, k1=1.5, b=0.75):
        if not corpus:
            raise ValueError("corpus must not be empty")
        self.corpus = [tokenize(d) for d in corpus]
        self.k1 = k1
        self.b = b
        self.n_docs = len(self.corpus)
        self.avg_dl = sum(len(d) for d in self.corpus) / self.n_docs
        self.df = Counter()
        for doc in self.corpus:
            for term in set(doc):
                self.df[term] += 1

    def idf(self, term):
        n = self.df.get(term, 0)
        return math.log(1 + (self.n_docs - n + 0.5) / (n + 0.5))

    def score(self, query, doc_idx):
        q_tokens = tokenize(query)
        doc = self.corpus[doc_idx]
        dl = len(doc)
        freq = Counter(doc)
        score = 0.0
        for term in q_tokens:
            f = freq.get(term, 0)
            if f == 0:
                continue
            numerator = f * (self.k1 + 1)
            denominator = f + self.k1 * (1 - self.b + self.b * dl / self.avg_dl)
            score += self.idf(term) * numerator / denominator
        return score

    def rank(self, query, top_k=10):
        scored = [(self.score(query, i), i) for i in range(self.n_docs)]
        scored.sort(reverse=True)
        return scored[:top_k]
```

Два параметра стоит знать. `k1=1.5` управляет насыщением частоты термина; большее значение означает больший вес повторения термина. `b=0.75` управляет нормализацией длины; 0 игнорирует длину документа, 1 полностью нормализует. Значения по умолчанию — рекомендации Robertson из оригинальной статьи, и их редко нужно настраивать.

### Шаг 2: dense retrieval с bi-encoder

```python
from sentence_transformers import SentenceTransformer
import numpy as np


def build_dense_index(corpus, model_id="sentence-transformers/all-MiniLM-L6-v2"):
    encoder = SentenceTransformer(model_id)
    embeddings = encoder.encode(corpus, normalize_embeddings=True)
    return encoder, embeddings


def dense_search(encoder, embeddings, query, top_k=10):
    q_emb = encoder.encode([query], normalize_embeddings=True)
    sims = (embeddings @ q_emb.T).flatten()
    order = np.argsort(-sims)[:top_k]
    return [(float(sims[i]), int(i)) for i in order]
```

L2-нормализуйте embeddings, чтобы скалярное произведение равнялось cosine. `all-MiniLM-L6-v2` — 384-мерная, быстрая и достаточно сильная модель для большинства задач поиска на английском. Для многоязычной работы используйте `paraphrase-multilingual-MiniLM-L12-v2`. Для максимальной точности — `bge-large-en-v1.5` или `e5-large-v2`.

### Шаг 3: Reciprocal Rank Fusion

```python
def reciprocal_rank_fusion(rankings, k=60):
    scores = {}
    for ranking in rankings:
        for rank, (_, doc_idx) in enumerate(ranking):
            scores[doc_idx] = scores.get(doc_idx, 0.0) + 1.0 / (k + rank + 1)
    fused = sorted(scores.items(), key=lambda x: x[1], reverse=True)
    return [(score, doc_idx) for doc_idx, score in fused]
```

Константа `k=60` взята из оригинальной статьи RRF. Более высокое `k` сглаживает вклад различий в рангах; более низкое `k` делает верхние ранги доминирующими. 60 — опубликованное значение по умолчанию, и его редко нужно настраивать.

### Шаг 4: hybrid search + rerank

```python
from sentence_transformers import CrossEncoder

reranker = CrossEncoder("cross-encoder/ms-marco-MiniLM-L-6-v2")


def hybrid_search(query, bm25, encoder, dense_embeddings, corpus, top_k=5, pool_size=30, reranker=reranker):
    sparse_ranking = bm25.rank(query, top_k=pool_size)
    dense_ranking = dense_search(encoder, dense_embeddings, query, top_k=pool_size)
    fused = reciprocal_rank_fusion([sparse_ranking, dense_ranking])[:pool_size]

    pairs = [(query, corpus[doc_idx]) for _, doc_idx in fused]
    scores = reranker.predict(pairs)
    reranked = sorted(zip(scores, [doc_idx for _, doc_idx in fused]), reverse=True)
    return reranked[:top_k]
```

Три стадии соединены вместе. BM25 находит лексические совпадения. Dense находит семантические совпадения. RRF объединяет два ранжирования без калибровки scores. Cross-encoder заново оценивает top-30, используя пары query-document вместе, что улавливает тонкую релевантность, которую пропустил bi-encoder. Оставляйте top-5.

### Шаг 5: оценка

| Метрика | Значение |
|--------|---------|
| Recall@k | Среди запросов, где правильный документ существует, как часто он попадает в top-k? |
| MRR (Mean Reciprocal Rank) | Среднее значение 1/rank первого релевантного документа. |
| nDCG@k | Учитывает градации релевантности, а не только бинарное relevant/not. |

Для RAG конкретно **Recall@k** retriever — самое важное число. Reader не сможет ответить, если нужный фрагмент не попал в retrieved set.

Совет по отладке: для неудачных запросов сравните sparse и dense rankings. Если один находит правильный документ, а другой нет, у вас либо vocabulary mismatch (исправление: добавить недостающую половину), либо semantic ambiguity (исправление: лучшие embeddings или reranker).

## Применение

Стек 2026 года:

| Масштаб | Стек |
|-------|-------|
| 1k-100k docs | In-memory BM25 + `all-MiniLM-L6-v2` embeddings + RRF. Без отдельной DB. |
| 100k-10M docs | FAISS или pgvector для dense + Elasticsearch / OpenSearch для BM25. Запускать параллельно. |
| 10M+ docs | Qdrant / Weaviate / Vespa / Milvus с hybrid support. Cross-encoder rerank на top-30. |
| Максимальное качество на frontier-уровне | Трехкомпонентный стек (BM25 + dense + SPLADE) + ColBERT late-interaction reranking |

Что бы вы ни выбрали, заложите бюджет на оценку. Сначала бенчмарк retrieval recall, потом бенчмарк end-to-end RAG accuracy. Reader не исправит то, что retriever пропустил.

### Тяжелые уроки из production RAG 2026 года

- **80% отказов RAG идут от ingestion и chunking, а не от модели.** Команды неделями меняют LLM и настраивают prompts, пока retrieval тихо возвращает неправильный контекст на каждый третий запрос. Сначала исправьте chunking.
- **Стратегия chunking важнее, чем размер chunk.** Fixed-size splits ломают таблицы, код и вложенные заголовки. Sentence-aware — вариант по умолчанию; semantic или LLM-based chunking окупается для технической документации и руководств продуктов.
- **Parent-doc pattern.** Извлекайте маленькие "child" chunks для точности. Когда появляется несколько children из одной parent section, подставляйте parent block, чтобы сохранить контекст. Это стабильно повышает качество ответов без дообучения.
- **k_rerank=3 обычно оптимален.** Каждый дополнительный chunk после этого добавляет token cost и generation latency без повышения качества ответа. Если k=8 у вас все еще лучше, чем k=3, reranker работает недостаточно хорошо.
- **HyDE / query expansion.** Сгенерируйте гипотетический ответ из запроса, embed его, затем retrieve. Это закрывает разрыв формулировок между короткими вопросами и длинными документами. Бесплатный прирост precision без обучения.
- **Context budget меньше 8K токенов.** Стабильные попадания в этот лимит означают, что threshold reranker слишком свободный.
- **Версионируйте все.** Prompts, правила chunking, embedding model, reranker. Любой drift незаметно ломает качество ответов. CI gates по faithfulness, context precision и unanswered-question rate блокируют регрессии до того, как их увидят пользователи.
- **Three-way retrieval (BM25 + dense + learned-sparse вроде SPLADE) превосходит two-way** в бенчмарках 2026 года, особенно для запросов, смешивающих имена собственные с семантикой. Ship it, когда инфраструктура поддерживает SPLADE indexes.

Правильный retrieval design снижает hallucinations на 70-90% по отраслевым измерениям 2026 года. Большинство приростов RAG performance приходит от лучшего retrieval, а не от fine-tuning модели.

## Доставка

Сохраните как `outputs/skill-retrieval-picker.md`:

```markdown
---
name: retrieval-picker
description: Pick a retrieval stack for a given corpus and query pattern.
version: 1.0.0
phase: 5
lesson: 14
tags: [nlp, retrieval, rag, search]
---

Given requirements (corpus size, query pattern, latency budget, quality bar, infra constraints), output:

1. Stack. BM25 only, dense only, hybrid (BM25 + dense + RRF), hybrid + cross-encoder rerank, or three-way (BM25 + dense + learned-sparse).
2. Dense encoder. Name the specific model. Match to language(s), domain, and context length.
3. Reranker. Name the specific cross-encoder model if used. Flag that rerank adds 30-100ms latency on top-30.
4. Evaluation plan. Recall@10 is the primary retriever metric. MRR for multi-answer. Baseline first, incremental improvements measured against it.

Refuse to recommend dense-only for corpora with named entities, error codes, or product SKUs unless the user has evidence dense handles exact matches. Refuse to skip reranking for high-stakes retrieval (legal, medical) where the final top-5 decides the user's answer.
```

## Упражнения

1. **Легко.** Реализуйте `hybrid_search` выше на корпусе из 500 документов. Протестируйте 20 запросов. Сравните recall at 5 между BM25-only, dense-only и hybrid.
2. **Средне.** Добавьте расчет MRR. Для каждого тестового запроса с известным правильным документом найдите rank правильного doc в ранжированиях BM25, dense и hybrid. Сообщите MRR для каждого.
3. **Сложно.** Fine-tune dense encoder на вашем домене с MultipleNegativesRankingLoss (Sentence Transformers). Соберите training set из 500 пар query-document. Сравните recall до и после fine-tune.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| BM25 | Keyword search | Okapi BM25. Оценивает документы по term frequency, IDF и длине. |
| Dense retrieval | Vector search | Кодирует query + doc в векторы, ищет ближайших соседей. |
| Bi-encoder | Embedding model | Кодирует query и doc независимо. Быстр во время запроса. |
| Cross-encoder | Reranker model | Кодирует query + doc вместе. Медленный, но точный. |
| RRF | Rank fusion | Объединяет два ранжирования суммированием `1/(k + rank)`. |
| Recall@k | Retrieval metric | Доля запросов, где релевантный doc находится в top-k. |

## Дополнительное чтение

- [Robertson and Zaragoza (2009). The Probabilistic Relevance Framework: BM25 and Beyond](https://www.staff.city.ac.uk/~sbrp622/papers/foundations_bm25_review.pdf) — исчерпывающее изложение BM25.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, канонический bi-encoder.
- [Formal et al. (2021). SPLADE: Sparse Lexical and Expansion Model](https://arxiv.org/abs/2107.05720) — learned-sparse retriever, закрывающий разрыв с dense.
- [Cormack, Clarke, Büttcher (2009). Reciprocal Rank Fusion outperforms Condorcet and individual Rank Learning Methods](https://plg.uwaterloo.ca/~gvcormac/cormacksigir09-rrf.pdf) — статья RRF.
- [Khattab and Zaharia (2020). ColBERT: Efficient and Effective Passage Search](https://arxiv.org/abs/2004.12832) — retrieval с late interaction.
