# Capstone 08 — Production RAG Chatbot for a Regulated Vertical

> Harvey, Glean, Mendable и LlamaCloud в 2026 году используют одну и ту же production-форму. Ingest через docling or Unstructured и ColPali для visuals. Hybrid search. Re-rank через bge-reranker-v2-gemma. Synthesize с Claude Sonnet 4.7, используя prompt caching при 60-80% hit rate. Guard через Llama Guard 4 и NeMo Guardrails. Watch через Langfuse и Phoenix. Grade через RAGAS на 200-question golden set. Постройте такой chatbot в regulated domain (legal, clinical, insurance), и capstone — пройти golden set, red team и drift dashboard.

**Тип:** Capstone
**Языки:** Python (pipeline + API), TypeScript (chat UI)
**Предварительные требования:** Phase 5 (NLP), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure), Phase 18 (safety)
**Задействованные фазы:** P5 · P7 · P11 · P12 · P17 · P18
**Время:** 30 hours

## Цели обучения

- Построить продакшен RAG-чатбот для регулируемой вертикали с приёмом данных, гибридным поиском и переранжированием.
- Добавить ссылки, ограждения и набор для оценки.
- Обрабатывать визуальные документы поиском в стиле ColPali.

## Проблема

Regulated-domain RAG (legal contracts, clinical trial protocols, insurance policies) — самая часто поставляемая production-форма 2026 года, потому что ROI очевиден, а stakes конкретны. Harvey (Allen & Overy) построил это для legal. Mendable поставляет вариант для developer-docs. Glean покрывает enterprise search. Pattern таков: ingest high-fidelity, retrieve hybrid with rerank, synthesize with citation enforcement and prompt caching, guard with multiple safety layers и monitor drift continuously.

Сложные части — не model. Это jurisdiction-aware compliance (HIPAA, GDPR, SOC2), auditability на уровне citations, cost control (prompt caching дает 60-90% discount при высоком hit rate), hallucination detection через RAGAS faithfulness и drift detection, когда source documents обновляются, а index не успевает. Этот capstone требует отгрузить все это на 200-question golden set с red-team suite рядом.

## Концепция

У pipeline две стороны. **Ingestion**: docling or Unstructured разбирает structured documents; ColPali обрабатывает visually rich ones; chunks получают summaries, tags и role-based access labels. Vectors идут в pgvector + pgvectorscale (under 50M vectors) or Qdrant Cloud; sparse BM25 работает рядом. **Conversation**: LangGraph ведет memory and multi-turn; каждый query запускает hybrid retrieval, reranks через bge-reranker-v2-gemma-2b, synthesizes with Claude Sonnet 4.7 (prompt-cached), пропускает output через Llama Guard 4 and NeMo Guardrails и выдает citation-anchored response.

Eval stack имеет четыре слоя. **Golden set** (200 labeled Q/A with citations) для correctness. **Red team** (jailbreaks, PII extraction attempts, off-domain questions) для safety. **RAGAS** для faithfulness / answer relevance / context precision автоматически per-turn. **Drift dashboard** (Arize Phoenix), который еженедельно следит за retrieval quality and hallucination score.

Prompt caching — главный cost lever. Claude 4.5+ and GPT-5+ поддерживают caching system prompts + retrieved context. При 60-80% hit rate per-query cost падает в 3-5x. Pipeline должен быть спроектирован под stable prefixes (system prompt + reranked context first), чтобы достигать высокого cache hit rate.

## Архитектура

```
documents (contracts, protocols, policies)
      |
      v
docling / Unstructured parse + ColPali for visuals
      |
      v
chunks + summaries + role-labels + jurisdiction tags
      |
      v
pgvector + pgvectorscale  +  BM25 (Tantivy)
      |
query + role + jurisdiction
      |
      v
LangGraph conversational agent
   +--- retrieve (hybrid)
   +--- filter by role + jurisdiction
   +--- rerank (bge-reranker-v2-gemma-2b or Voyage rerank-2)
   +--- synthesize (Claude Sonnet 4.7, prompt cached)
   +--- guard (Llama Guard 4 + NeMo Guardrails + Presidio output PII scrub)
   +--- cite + return
      |
      v
eval:
  RAGAS faithfulness / answer_relevance / context_precision (online)
  Langfuse annotation queue (sampled)
  Arize Phoenix drift (weekly)
  red team suite (pre-release)
```

## Стек

- Ingestion: Unstructured.io or docling for structured documents; ColPali for visually-rich PDFs
- Vector DB: pgvector + pgvectorscale under 50M vectors; Qdrant Cloud otherwise
- Sparse: Tantivy BM25 with field weights
- Orchestration: LlamaIndex Workflows (ingestion) + LangGraph (conversation)
- Re-ranker: bge-reranker-v2-gemma-2b self-hosted or Voyage rerank-2 hosted
- LLM: Claude Sonnet 4.7 with prompt caching; fallback Llama 3.3 70B self-hosted
- Eval: RAGAS 0.2 online, DeepEval for hallucination and jailbreak suites
- Observability: Langfuse self-hosted with annotation queue; Arize Phoenix for drift
- Guardrails: Llama Guard 4 input/output classifier, NeMo Guardrails v0.12 policy, Presidio PII scrub
- Compliance: role-based access labels on chunks; jurisdiction tags for GDPR/HIPAA

## Постройте это

1. **Ingestion.** Разберите свой corpus (1000-10000 documents для серьезной сборки) через Unstructured or docling. Для scanned / visual-heavy pages направляйте через ColPali. Производите chunks with summaries, role-labels, jurisdiction tags.

2. **Index.** Dense embeddings (Voyage-3 or Nomic-embed-v2) в pgvector + pgvectorscale. BM25 side-index через Tantivy. Role and jurisdiction filters as payload.

3. **Hybrid retrieve.** Сначала filter by role+jurisdiction; затем parallel dense + BM25; merge with reciprocal rank fusion; top-20 to reranker; top-5 to synth.

4. **Synthesize with prompt caching.** System prompt + static policies в cache header; reranked context как cache extension; user question как uncached suffix. Цель — 60-80% cache hit rate in steady state.

5. **Guardrails.** Llama Guard 4 на input; NeMo Guardrails rails блокируют off-domain questions or policy-forbidden topics; Presidio очищает accidental PII in the output; citation enforcement post-filter.

6. **Golden set.** 200 Q/A pairs, размеченные domain expert с (answer, citations). Оценивайте агента по exact-citation match, answer correctness, faithfulness (RAGAS).

7. **Red team.** 50 adversarial prompts: jailbreaks (PAIR, TAP), PII exfiltration attempts, off-domain, cross-jurisdiction leaks. Оценивайте pass/fail and severity.

8. **Drift dashboard.** Arize Phoenix отслеживает retrieval quality (nDCG, citation faithfulness) еженедельно. Alert on 5% drop.

9. **Cost report.** Langfuse: prompt-caching hit rate, tokens per query, $/query breakdown by stage.

## Используйте это

```
$ chat --role=analyst --jurisdiction=GDPR
> what is the data-retention obligation for EU user profiles under our contract?
[retrieve]  hybrid top-20 filtered to GDPR + analyst-role
[rerank]    top-5 kept
[synth]     claude-sonnet-4.7, cache hit 74%, 0.8s
answer:
  The contract (Section 12.4, Master Services Agreement dated 2024-03-11)
  obligates EU user profile deletion within 30 days of termination per GDPR
  Article 17. The DPA amendment (DPA-v2.1, Section 5) extends this to 14 days
  for "restricted" category data.
  citations: [MSA-2024-03-11 s12.4, DPA-v2.1 s5]
```

## Отгрузите это

`outputs/skill-production-rag.md` описывает deliverable. Regulated-domain chatbot, deployed with compliance labels, passed through the rubric, observed with live drift monitoring.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | RAGAS faithfulness + answer relevance | Online scores on the golden set (200 Q/A) |
| 20 | Корректность citations | Fraction of answers with verifiable source anchors |
| 20 | Guardrail coverage | Llama Guard 4 pass rate + jailbreak suite results |
| 20 | Cost / latency engineering | Prompt-cache hit rate, p95 latency, $/query |
| 15 | Drift monitoring dashboard | Phoenix live dashboard with weekly retrieval-quality trend |
| **100** | | |

## Упражнения

1. Постройте второй corpus slice под другую jurisdiction, например HIPAA alongside GDPR. Продемонстрируйте, что role+jurisdiction filtering предотвращает cross-leak на 20-question cross-jurisdiction probe.

2. Измерьте prompt-cache hit rate за неделю production traffic. Определите, какие queries ломают cache prefix. Переструктурируйте.

3. Добавьте multi-turn memory with a 10k-token summary buffer. Измерьте, падает ли faithfulness по мере роста conversation.

4. Замените Claude Sonnet 4.7 на Llama 3.3 70B self-hosted. Измерьте $/query and faithfulness delta.

5. Добавьте "unsure" mode: если top reranked scores ниже threshold, агент говорит "I do not have confident citations" вместо ответа. Измерьте false-confidence reduction.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Prompt caching | "Cached system + context" | Claude/OpenAI feature: cached prefix tokens discounted 60-90% on hit |
| RAGAS | "RAG evaluator" | Automated scoring of faithfulness, answer relevance, context precision |
| Golden set | "Labeled eval" | 200+ expert-labeled Q/A with citations; the ground truth |
| Jurisdiction tag | "Compliance label" | GDPR/HIPAA/SOC2 scope attached to chunks; enforced by retrieval filter |
| Citation faithfulness | "Grounded answer rate" | Fraction of claims backed by retrievable source spans |
| Drift | "Retrieval quality decay" | Weekly change in nDCG or citation score; alert threshold 5% |
| Red team | "Adversarial eval" | Pre-release jailbreak, PII extraction, off-domain probes |

## Дополнительное чтение

- [Harvey AI](https://www.harvey.ai) — эталонный legal production stack
- [Glean enterprise search](https://www.glean.com) — эталонный RAG at enterprise scale
- [Mendable documentation](https://mendable.ai) — developer-docs RAG reference
- [LlamaCloud Parse + Index](https://docs.llamaindex.ai/en/stable/examples/llama_cloud/llama_parse/) — managed ingestion
- [Anthropic prompt caching](https://docs.anthropic.com/en/docs/build-with-claude/prompt-caching) — cost-lever reference
- [RAGAS 0.2 documentation](https://docs.ragas.io/) — canonical RAG eval framework
- [Arize Phoenix](https://github.com/Arize-ai/phoenix) — эталонная drift observability
- [Llama Guard 4](https://ai.meta.com/research/publications/llama-guard-4/) — 2026 safety classifier
- [NeMo Guardrails v0.12](https://docs.nvidia.com/nemo-guardrails/) — policy rail framework
