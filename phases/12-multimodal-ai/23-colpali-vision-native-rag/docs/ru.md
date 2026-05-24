# ColPali и vision-native document RAG

> Традиционный RAG разбирает PDF в текст, делит на chunks, встраивает chunks, хранит vectors. Каждый шаг теряет сигнал: OCR теряет данные диаграмм, chunking ломает строки таблиц, text embeddings игнорируют figures. ColPali (Faysse et al., июль 2024) задал более простой вопрос: зачем вообще извлекать текст? Встраивайте изображение страницы напрямую через PaliGemma, используйте late interaction в стиле ColBERT для retrieval и сохраняйте весь сигнал layout, figures, fonts и formatting, который несет документ. Опубликованные бенчмарки: на visually-rich documents end-to-end accuracy лучше text-RAG на 20-40%. ColQwen2, ColSmol и VisRAG расширили этот паттерн. Этот урок разбирает тезис vision-native RAG и строит маленький индексатор в стиле ColPali.

**Тип:** Практика
**Языки:** Python (stdlib, multi-vector indexer + MaxSim scorer)
**Предварительные требования:** Phase 11 (LLM Engineering — RAG basics), Phase 12 · 05 (LLaVA)
**Время:** ~180 минут

## Цели обучения

- Объяснить разницу между bi-encoder retrieval (один vector на document) и late-interaction retrieval (много vectors на document).
- Описать операцию MaxSim в ColBERT и то, как ColPali обобщает ее с text tokens на image patches.
- Построить маленький индексатор в стиле ColPali: page → patch embeddings → MaxSim over query-term embeddings → top-k pages.
- Сравнить ColPali + Qwen2.5-VL generator с text-RAG + GPT-4 на use case invoices / financial reports.

## Проблема

Text-RAG на PDF выбрасывает большую часть документа. Рост выручки Q3 в финансовом отчете обычно находится на chart; findings в medical report — в annotated images; signature block в legal contract — это layout fact, а не text fact.

Pipeline text-RAG:

1. PDF → text via OCR / pdftotext.
2. Text → 300-500 token chunks.
3. Chunk → bi-encoder embedding (one vector).
4. User query → embedding → cosine similarity → top-k chunks.
5. Chunks + query → LLM.

Пять шагов с потерями. Charts не захвачены. Tables разорваны между chunks. Multi-column layout сплющивается. Figure annotations исчезают.

Исправление ColPali: пропустить OCR, встроить изображение страницы напрямую. Использовать late interaction в стиле ColBERT для retrieval, чтобы модель могла учитывать fine-grained patches во время query.

## Концепция

### ColBERT (2020)

ColBERT (Khattab & Zaharia, arXiv:2004.12832) — метод text retrieval. Вместо одного vector на document он создает один vector на token. Во время query:

- Query tokens получают собственные embeddings (N_q vectors).
- Document tokens получают embeddings (N_d vectors, usually cached).
- Score = sum over query tokens of max over document tokens of cosine similarity: Σ_i max_j cos(q_i, d_j).

Это операция MaxSim. Каждый query token "выбирает" лучший matching document token. Финальный score — сумма.

Плюсы: сильный recall, обрабатывает term-level semantics. Минусы: N_d vectors per document, storage expensive.

### ColPali

ColPali (Faysse et al., arXiv:2407.01449) применяет паттерн ColBERT к изображениям.

- Каждая page кодируется PaliGemma (ViT + language) в patch embeddings: N_p vectors per page.
- Каждый user query (text) кодируется в query-token embeddings: N_q vectors.
- Score = Σ_i max_j cos(q_i, p_j), то есть MaxSim по query-text-tokens и page-image-patches.
- Top-k pages извлекаются по total score.

Во время document-ingestion: встроить каждую page через PaliGemma, сохранить все patch embeddings. Во время query: встроить query tokens, вычислить MaxSim против всех сохраненных page embeddings, вернуть top-k pages.

Плюсы: end-to-end превосходит text-RAG на 20-40% на visually rich documents. Каждый patch-vector захватывает локальный layout и content.

Минусы: N_p patches × 4-byte floats × D-dim vectors per page = storage быстро растет. Смягчается через PQ / OPQ quantization.

### ColQwen2 и ColSmol

ColQwen2 (illuin-tech, 2024-2025) заменяет PaliGemma на Qwen2-VL. Лучше base encoder, лучше retrieval.

ColSmol — smaller-scale variant для local / edge use. ColSmol retriever на ~1B params работает на consumer GPU.

### VisRAG

VisRAG (Yu et al., arXiv:2410.10594) — другой вариант: вместо MaxSim на patches он сворачивает каждую page в single vector через VLM, затем выполняет bi-encoder retrieve. Быстрее indexing + меньше storage, слабее recall.

Компромисс quality-vs-cost: ColPali для качества, VisRAG для масштаба.

### M3DocRAG

M3DocRAG (Cho et al., arXiv:2411.04952) расширяет multi-modal retrieval до multi-page multi-document reasoning. Извлекает pages across documents, составляет multi-page context для VLM.

### ViDoRe — бенчмарк

Сопутствующий бенчмарк ColPali. Visual Document Retrieval Evaluation. Задачи включают financial reports, scientific papers, administrative documents, medical records, manuals. Метрика: nDCG@5.

ColPali-v1 набирает ~80% nDCG@5 на ViDoRe; text-RAG на тех же документах набирает ~50-60%.

### End-to-end RAG pipeline

Для vision-native RAG:

1. Ingest: PDF → page images → PaliGemma encoding → store all patch embeddings.
2. Query: user text → query-token embeddings → MaxSim against all indexed pages → top-k pages.
3. Generate: top-k page images + query → VLM (Qwen2.5-VL or Claude) → answer.

Никакого OCR нигде. Figures, charts, fonts, layout — все попадает в ответ.

### Математика хранения

Финансовый отчет на 50 страниц с 729 patches per page и 128-dim embeddings:

- ColPali: 50 * 729 * 128 * 4 bytes = ~18 MB raw, ~4 MB after PQ.
- Text-RAG: 50 chunks * 768-dim * 4 bytes = ~150 kB.

ColPali требует примерно ~30x больше storage per document. В масштабе OPQ / PQ снижает это до ~5-10x, обычно терпимо.

### Когда text-RAG все еще выигрывает

- Pure-text documents без layout signal (wiki articles, chat logs). Text-RAG проще и дешевле по storage.
- Multi-million-page archives, где storage доминирует в стоимости.
- Строгие regulatory requirements, требующие extractable OCR text alongside the retrieval.

Для всего остального в 2026 — financial reports, scientific papers, legal contracts, medical records, UX documentation — vision-native RAG выигрывает.

## Применение

`code/main.py`:

- Toy patch encoder: отображает "page" (small grid of feature vectors) в array of patch embeddings.
- MaxSim scorer: вычисляет score в стиле ColBERT между query token embedding set и page patch set.
- Индексирует 5 toy pages, запускает 3 queries, возвращает top-k with scores.

## Результат

Этот урок создает `outputs/skill-vision-rag-designer.md`. По document-RAG project выбирает ColPali / ColQwen2 / VisRAG / text-RAG и рассчитывает storage.

## Упражнения

1. Annual report на 200 страниц при 729 patches per page, 128-dim emb, 4-byte floats. Посчитайте raw storage и PQ-compressed (8x) storage.

2. MaxSim — это Σ_i max_j cos(q_i, p_j). Что эта сумма захватывает такого, чего не захватывает simple mean similarity?

3. ColPali индексирует pages как patch sets. Что изменится, если вместо этого индексировать на word level (как ColBERT)? Trade-offs?

4. Спроектируйте end-to-end pipeline для корпуса на 1M-page с latency budget 500ms per query. Выберите ColQwen2 / VisRAG и обоснуйте.

5. Прочитайте M3DocRAG (arXiv:2411.04952). Опишите multi-page attention pattern и чем он отличается от single-page ColPali retrieval.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Late interaction | "ColBERT-style" | Retrieval с per-token или per-patch embeddings + MaxSim, а не single doc vector |
| MaxSim | "Max-over-patches" | Для каждого query token выбрать document token с максимальной similarity; суммировать по query |
| Bi-encoder | "Single-vector" | Один vector на document; быстрее, но теряет granularity |
| Multi-vector | "Many-vectors-per-doc" | Хранить N_p vectors per document / page; storage cost растет, но recall улучшается |
| Patch embedding | "Page feature" | Один vector на image patch из VLM encoder, cached per page |
| ViDoRe | "Vision doc bench" | Benchmark suite ColPali для visual document retrieval |
| PQ quantization | "Product quantization" | Сжатие, сохраняющее vector similarity при уменьшении storage примерно ~8x |

## Дополнительное чтение

- [Faysse et al. — ColPali (arXiv:2407.01449)](https://arxiv.org/abs/2407.01449)
- [Khattab & Zaharia — ColBERT (arXiv:2004.12832)](https://arxiv.org/abs/2004.12832)
- [Yu et al. — VisRAG (arXiv:2410.10594)](https://arxiv.org/abs/2410.10594)
- [Cho et al. — M3DocRAG (arXiv:2411.04952)](https://arxiv.org/abs/2411.04952)
- [illuin-tech/colpali GitHub](https://github.com/illuin-tech/colpali)
