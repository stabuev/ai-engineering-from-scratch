# Модели эмбеддингов — глубокий разбор 2026 года

> Word2Vec давал вам вектор на слово. Современные модели эмбеддингов дают вам вектор на фрагмент текста, кросс-языковой, с разреженным, плотным и мультивекторным представлениями, подобранный под размер вашего индекса. Ошибетесь с выбором — и ваш RAG будет извлекать не то.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 5 · 03 (Word2Vec), Фаза 5 · 14 (Information Retrieval)
**Время:** ~60 минут

## Проблема

Ваша RAG-система извлекает неправильный фрагмент в 40% случаев. Причина редко в векторной базе данных или промпте. Причина — модель эмбеддингов.

Выбор эмбеддинга в 2026 году означает выбор по пяти осям:

1. **Плотные (dense) vs разреженные (sparse) vs мультивекторные (multi-vector).** Один вектор на фрагмент, или один на токен, или разреженный взвешенный мешок слов.
2. **Языковое покрытие.** Одноязычные английские модели все еще выигрывают на задачах только на английском. Многоязычные модели выигрывают, когда корпуса смешанные.
3. **Длина контекста.** 512 токенов vs 8,192 vs 32,768 — а реальная эффективная емкость часто составляет 60-70% от заявленного максимума.
4. **Бюджет размерности.** 3,072 числа с плавающей точкой при полной точности = 12 KB на вектор. При 100M векторов хранение стоит $1,300/месяц. Усечение Matryoshka сокращает это в 4 раза.
5. **Открытые веса vs hosted.** Открытые веса означают, что вы контролируете стек и данные. Hosted означает, что вы меняете контроль на always-latest.

Этот урок называет компромиссы, чтобы вы выбирали на основе доказательств, а не того, что было популярно в прошлом квартале.

## Концепция

![Плотные, разреженные и мультивекторные эмбеддинги](../assets/embedding-modes.svg)

**Плотные эмбеддинги (dense embeddings).** Один вектор на фрагмент (обычно 384-3,072 размерности). Косинусное сходство ранжирует фрагменты по семантической близости. OpenAI `text-embedding-3-large`, плотный режим BGE-M3, Voyage-3. Выбор по умолчанию.

**Разреженные эмбеддинги (sparse embeddings).** В стиле SPLADE. Трансформер предсказывает вес для каждого токена словаря, затем обнуляет большую часть весов. Результат — разреженный вектор размера |vocab|. Улавливает лексическое совпадение (как BM25), но с обученными весами терминов. Силен на запросах с большим количеством ключевых слов.

**Мультивекторные представления (multi-vector, late interaction).** ColBERTv2, Jina-ColBERT. Один вектор на токен. Скоринг через MaxSim: для каждого токена запроса найти самый похожий токен документа и суммировать оценки. Дороже в хранении и скоринге, но выигрывает на длинных запросах и доменных корпусах.

**BGE-M3: все три сразу.** Одна модель одновременно выдает плотные, разреженные и мультивекторные представления. Каждое можно запрашивать независимо; оценки объединяются взвешенной суммой. Дефолт 2026 года, когда нужна гибкость из одного checkpoint.

**Matryoshka Representation Learning.** Обучается так, чтобы первые N размерностей вектора образовывали полезный самостоятельный эмбеддинг. Усеките 1,536-мерный вектор до 256 размерностей и заплатите ~1% точности за 6-кратную экономию хранения. Поддерживается OpenAI text-3, Cohere v4, Voyage-4, Jina v5, Gemini Embedding 2, Nomic v1.5+.

### Лидерборд MTEB рассказывает только часть истории

Massive Text Embedding Benchmark — 56 задач по 8 типам задач на запуске (2022), расширен до 100+ задач в MTEB v2. В начале 2026 года Gemini Embedding 2 лидирует в retrieval (67.71 MTEB-R). Cohere embed-v4 лидирует в общем зачете (65.2 MTEB). BGE-M3 лидирует среди многоязычных моделей с открытыми весами (63.0). Лидерборд необходим, но недостаточен — всегда бенчмаркайте на своем домене.

### Трехуровневый паттерн

| Сценарий | Паттерн |
|----------|---------|
| Быстрый первый проход | Плотный bi-encoder (BGE-M3, text-3-small) |
| Увеличение recall | Разреженный (SPLADE, BGE-M3 sparse) + RRF fusion |
| Точность на top-50 | Мультивекторный (ColBERTv2) или cross-encoder reranker |

Большинство production-стеков используют все три.

## Соберите это

### Шаг 1: baseline — плотные эмбеддинги с Sentence-BERT

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")
corpus = [
    "The first iPhone launched in 2007.",
    "Apple released the iPod in 2001.",
    "Android is an operating system from Google.",
]
emb = encoder.encode(corpus, normalize_embeddings=True)

query = "When was the iPhone released?"
q_emb = encoder.encode([query], normalize_embeddings=True)[0]
scores = emb @ q_emb
print(sorted(enumerate(scores), key=lambda x: -x[1]))
```

`normalize_embeddings=True` делает скалярное произведение равным косинусному сходству. Всегда устанавливайте это.

### Шаг 2: усечение Matryoshka

```python
def truncate(vectors, dim):
    out = vectors[:, :dim]
    return out / np.linalg.norm(out, axis=1, keepdims=True)

emb_256 = truncate(emb, 256)
emb_128 = truncate(emb, 128)
```

Повторно нормализуйте после усечения. Nomic v1.5, OpenAI text-3 и Voyage-4 обучены так, что это почти без потерь для первых нескольких уровней. Не-Matryoshka модели (оригинальный Sentence-BERT) резко деградируют при усечении.

### Шаг 3: многофункциональность BGE-M3

```python
from FlagEmbedding import BGEM3FlagModel

model = BGEM3FlagModel("BAAI/bge-m3", use_fp16=True)

output = model.encode(
    corpus,
    return_dense=True,
    return_sparse=True,
    return_colbert_vecs=True,
)
# output["dense_vecs"]:    (n_docs, 1024)
# output["lexical_weights"]: list of dict {token_id: weight}
# output["colbert_vecs"]:  list of (n_tokens, 1024) arrays
```

Три индекса, один вызов инференса. Объединение оценок:

```python
dense_score = ... # cosine over dense_vecs
sparse_score = model.compute_lexical_matching_score(q_lex, d_lex)
colbert_score = model.colbert_score(q_col, d_col)
final = 0.4 * dense_score + 0.2 * sparse_score + 0.4 * colbert_score
```

Настраивайте веса на своем домене.

### Шаг 4: MTEB eval на пользовательской задаче

```python
from mteb import MTEB

tasks = ["ArguAna", "SciFact", "NFCorpus"]
evaluation = MTEB(tasks=tasks)
results = evaluation.run(encoder, output_folder="./mteb-results")
```

Запускайте модели-кандидаты на *репрезентативном* подмножестве. Не доверяйте одному только месту в лидерборде — ваш домен имеет значение.

### Шаг 5: косинус вручную с нуля

См. `code/main.py`. Эмбеддинги на усредненном Hashing Trick (только stdlib). Неконкурентоспособны с трансформерными эмбеддингами, но показывают форму: токенизация → вектор → нормализация → скалярное произведение.

## Подводные камни

- **Одна и та же модель для запроса и документа.** Некоторые модели (Voyage, Jina-ColBERT) используют асимметричное кодирование — запрос и документ проходят по разным путям. Всегда проверяйте model card.
- **Отсутствующий prefix.** Моделям `bge-*` нужно добавлять `"Represent this sentence for searching relevant passages: "` в начало запросов. Если забыть, recall падает на 3-5 пунктов.
- **Чрезмерное усечение Matryoshka.** 1,536 → 256 обычно безопасно. 1,536 → 64 — нет. Валидируйте на своем eval-наборе.
- **Усечение контекста.** Большинство моделей молча обрезают входы сверх максимальной длины. Длинным документам нужен chunking (см. урок 23).
- **Игнорирование хвоста latency.** Оценки MTEB скрывают p99 latency. Модель на 600M может обгонять модель на 335M на 2 пункта, но стоить в 3 раза дороже на запрос.

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| Только английский, быстро, API | `text-embedding-3-large` или `voyage-3-large` |
| Открытые веса, английский | `BAAI/bge-large-en-v1.5` |
| Открытые веса, многоязычность | `BAAI/bge-m3` или `Qwen3-Embedding-8B` |
| Длинный контекст (32k+) | Voyage-3-large, Cohere embed-v4, Qwen3-Embedding-8B |
| CPU-only deployment | Nomic Embed v2 (137M params, MoE) |
| Ограничено хранение | Matryoshka-truncated + int8 quantization |
| Запросы с большим количеством ключевых слов | Добавьте SPLADE sparse, RRF-fuse with dense |

Паттерн 2026 года: начните с BGE-M3 или text-3-large, оцените на своем домене с MTEB, замените, если доменная модель выигрывает больше чем на 3 пункта.

## Доведите до поставки

Сохраните как `outputs/skill-embedding-picker.md`:

```markdown
---
name: embedding-picker
description: Pick embedding model, dimension, and retrieval mode for a given corpus and deployment.
version: 1.0.0
phase: 5
lesson: 22
tags: [nlp, embeddings, retrieval]
---

Given a corpus (size, languages, domain, avg length), deployment target (cloud / edge / on-prem), latency budget, and storage budget, output:

1. Model. Named checkpoint or API. One-sentence reason.
2. Dimension. Full / Matryoshka-truncated / int8-quantized. Reason tied to storage budget.
3. Mode. Dense / sparse / multi-vector / hybrid. Reason.
4. Query prefix / template if required by the model card.
5. Evaluation plan. MTEB tasks relevant to domain + held-out domain eval with nDCG@10.

Refuse recommendations that truncate Matryoshka to <64 dims without domain validation. Refuse ColBERTv2 for corpora under 10k passages (overhead not justified). Flag long-document corpora (>8k tokens) routed to models with 512-token windows.
```

## Упражнения

1. **Легко.** Закодируйте 100 предложений с `bge-small-en-v1.5` в полной размерности (384), затем с Matryoshka 128. Измерьте падение MRR на 10 запросах.
2. **Средне.** Сравните BGE-M3 dense, sparse и colbert на 500 фрагментах из вашего домена. Что выигрывает по recall@10? Превосходит ли RRF fusion лучший одиночный режим?
3. **Сложно.** Запустите MTEB на трех моделях-кандидатах по двум главным доменным задачам. Сообщите MTEB score, p99 latency на батче из 100 запросов и $/1M queries. Выберите Pareto-optimal вариант.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Плотный эмбеддинг (dense embedding) | Вектор | Один вектор фиксированного размера на текст. Косинусное сходство для ранжирования. |
| Разреженный эмбеддинг (sparse embedding) | Обученный BM25 | Один вес на токен словаря; в основном нули; обучается end-to-end. |
| Мультивекторный (multi-vector) | В стиле ColBERT | Один вектор на токен; скоринг MaxSim; индекс больше, recall лучше. |
| Matryoshka | Трюк с матрешкой | Первые N размерностей сами по себе являются валидным меньшим эмбеддингом. |
| MTEB | Бенчмарк | Massive Text Embedding Benchmark — 56 задач на запуске, 100+ в v2. |
| BEIR | Retrieval-бенчмарк | 18 zero-shot retrieval задач; часто цитируется для cross-domain устойчивости. |
| Асимметричное кодирование (asymmetric encoding) | Query ≠ doc path | Модель использует разные проекции для запросов и документов. |

## Дополнительное чтение

- [Reimers, Gurevych (2019). Sentence-BERT](https://arxiv.org/abs/1908.10084) — статья о bi-encoder.
- [Muennighoff et al. (2022). MTEB: Massive Text Embedding Benchmark](https://arxiv.org/abs/2210.07316) — статья о лидерборде.
- [Chen et al. (2024). BGE-M3: Multi-lingual, Multi-functionality, Multi-granularity](https://arxiv.org/abs/2402.03216) — единая трехрежимная модель.
- [Kusupati et al. (2022). Matryoshka Representation Learning](https://arxiv.org/abs/2205.13147) — training objective лестницы размерностей.
- [Santhanam et al. (2022). ColBERTv2: Effective and Efficient Retrieval via Lightweight Late Interaction](https://arxiv.org/abs/2112.01488) — late interaction в production.
- [MTEB leaderboard on Hugging Face](https://huggingface.co/spaces/mteb/leaderboard) — живой рейтинг.
