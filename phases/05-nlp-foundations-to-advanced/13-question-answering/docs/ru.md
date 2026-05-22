# Question Answering Systems

> Три системы сформировали modern QA. Extractive находила spans. Retrieval-augmented grounded them in documents. Generative производила answers. Каждый современный AI assistant - смесь этих трех.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Phase 5 · 11 (Machine Translation), Phase 5 · 10 (Attention Mechanism)
**Время:** ~75 минут

## Проблема

Пользователь вводит "When did the first iPhone launch?" и ожидает "June 29, 2007." Не "Apple's history is long and varied." Не "2007", висящее в изоляции без sentence. Нужен direct, grounded, correct answer.

За последнее десятилетие в QA доминировали три architectures.

- **Extractive QA.** Даны question и passage, про который известно, что он содержит answer; нужно найти start и end indices answer span в passage. SQuAD - canonical benchmark.
- **Open-domain QA.** Passage не дан. Сначала retrieve relevant passage, затем extract или generate answer. Это основа каждого RAG pipeline сегодня.
- **Generative / Closed-book QA.** Large language model отвечает из своей parametric memory. Без retrieval. Самый быстрый inference, наименее надежный для facts.

Тренд 2026 года - hybrid: retrieve несколько лучших passages, затем prompt generative model ответить grounded in those passages. Это RAG, и урок 14 подробно покрывает retrieval half. Этот урок строит QA half.

## Концепция

![QA architectures: extractive, retrieval-augmented, generative](../assets/qa.svg)

**Extractive.** Encode question и passage вместе с transformer (семейство BERT). Обучите две heads, которые predict start и end token indices of the answer. Loss - cross-entropy по valid positions. Output - span из passage. Никогда не hallucinate (by construction), никогда не обрабатывает questions, на которые passage не может ответить (by construction).

**Retrieval-augmented (RAG).** Две стадии. Сначала retriever находит top-`k` passages из corpus. Затем reader (extractive или generative) производит answer, используя эти passages. Разделение retriever-reader позволяет обучать и оценивать каждый независимо. Modern RAG часто добавляет reranker между ними.

**Generative.** Decoder-only LLM (GPT, Claude, Llama) отвечает из learned weights. Без retrieval step. Отлично работает на common knowledge, катастрофично на rare или recent facts. Hallucination rate обратно коррелирует с fact frequency в pretraining data.

## Соберите это

### Шаг 1: extractive QA с pretrained model

```python
from transformers import pipeline

qa = pipeline("question-answering", model="deepset/roberta-base-squad2")

passage = (
    "Apple Inc. released the first iPhone on June 29, 2007. "
    "The device was announced by Steve Jobs at Macworld in January 2007."
)
question = "When was the first iPhone released?"

answer = qa(question=question, context=passage)
print(answer)
```

```python
{'score': 0.98, 'start': 57, 'end': 70, 'answer': 'June 29, 2007'}
```

`deepset/roberta-base-squad2` обучен на SQuAD 2.0, который включает unanswerable questions. По умолчанию pipeline `question-answering` возвращает highest-scoring span даже тогда, когда null score выигрывает - он *не* возвращает automatically empty answer. Чтобы получить явное поведение "no answer", передайте `handle_impossible_answer=True` в pipeline call: тогда pipeline возвращает empty answer только когда null score превышает каждый span score. В любом случае всегда проверяйте поле `score`.

### Шаг 2: retrieval-augmented pipeline (sketch)

```python
from sentence_transformers import SentenceTransformer
import numpy as np

encoder = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")

corpus = [
    "Apple Inc. released the first iPhone on June 29, 2007.",
    "Macworld 2007 featured the iPhone announcement by Steve Jobs.",
    "Android launched in 2008 as Google's mobile operating system.",
    "The first iPod was released in 2001.",
]
corpus_embeddings = encoder.encode(corpus, normalize_embeddings=True)


def retrieve(question, top_k=2):
    q_emb = encoder.encode([question], normalize_embeddings=True)
    sims = (corpus_embeddings @ q_emb.T).squeeze()
    order = np.argsort(-sims)[:top_k]
    return [corpus[i] for i in order]


def answer(question):
    passages = retrieve(question, top_k=2)
    combined = " ".join(passages)
    return qa(question=question, context=combined)


print(answer("When was the first iPhone released?"))
```

Двухстадийный pipeline. Dense retriever (Sentence-BERT) находит relevant passages по semantic similarity. Extractive reader (RoBERTa-SQuAD) вытаскивает answer span из объединенных top passages. Работает на small corpora. Для million-document corpus используйте FAISS или vector database.

### Шаг 3: generative with RAG

```python
def rag_generate(question, llm):
    passages = retrieve(question, top_k=3)
    prompt = f"""Context:
{chr(10).join('- ' + p for p in passages)}

Question: {question}

Answer using only the context above. If the context does not contain the answer, say "I don't know."
"""
    return llm(prompt)
```

Prompt pattern важен. Явная инструкция model ground in the context и возвращать "I don't know", когда context недостаточен, снижает hallucination rates на 40-60% по сравнению с naive prompting. Более elaborate patterns добавляют citations, confidence scores и structured extraction.

### Шаг 4: evaluation, отражающий real world

SQuAD использует **Exact Match (EM)** и **token-level F1**. EM - strict match после normalization (lowercase, strip punctuation, remove articles): prediction либо совпадает exactly, либо получает 0. F1 вычисляется по token overlap между prediction и reference и дает partial credit. Обе метрики недооценивают paraphrases: "June 29, 2007" vs "June 29th, 2007" обычно получает 0 EM (ordinal ломает normalization), но все еще получает substantial F1 за overlapping tokens.

Для production QA:

- **Answer accuracy** (LLM-judged или human-judged, потому что metrics не ловят semantic equivalence).
- **Citation accuracy.** Действительно ли cited passage поддерживает answer? Это тривиально проверять automatically через string match между generated citations и retrieved passages.
- **Refusal calibration.** Когда answer нет в retrieved passages, правильно ли system говорит "I don't know"? Измеряйте false confidence rate.
- **Retrieval recall.** Перед evaluation reader измерьте, попадает ли right passage в top-`k` retriever. Reader не может исправить missing passage.

### RAGAS: production eval framework 2026 года

`RAGAS` специально создан для RAG systems и является shipping default в 2026 году. Он оценивает четыре dimensions без requiring gold references:

- **Faithfulness.** Происходит ли каждое claim в answer из retrieved context? Измеряется NLI-based entailment. Ваша основная hallucination metric.
- **Answer relevance.** Отвечает ли answer на question? Измеряется генерацией hypothetical questions из answer и сравнением с real question.
- **Context precision.** Какая доля retrieved chunks действительно была relevant? Low precision = noise in prompt.
- **Context recall.** Содержал ли retrieved set всю needed information? Low recall = reader cannot succeed.

Reference-free scoring позволяет evaluate на live production traffic без curated gold answers. Добавьте LLM-as-judge сверху для open-ended questions, где exact-match metrics бесполезны.

`pip install ragas`. Подключите ваш retriever + reader. Получите четыре scalars per query. Alert on regressions.

## Используйте это

Stack 2026 года.

| Use case | Recommended |
|---------|-------------|
| Given passage, find answer span | `deepset/roberta-base-squad2` |
| Over a fixed corpus, closed-book not acceptable | RAG: dense retriever + LLM reader |
| Real-time over a document store | RAG with hybrid (BM25 + dense) retriever + reranker (lesson 14) |
| Conversational QA (follow-up questions) | LLM with conversation history + RAG on each turn |
| Highly factual, regulated domains | Extractive over an authoritative corpus; never generative alone |

Extractive QA немоден в 2026 году, потому что RAG with LLMs покрывает больше cases. Он все еще shipped там, где требуется literal quotation: legal research, regulatory compliance, audit tools.

## Отгрузите это

Сохраните как `outputs/skill-qa-architect.md`:

```markdown
---
name: qa-architect
description: Choose QA architecture, retrieval strategy, and evaluation plan.
version: 1.0.0
phase: 5
lesson: 13
tags: [nlp, qa, rag]
---

Given requirements (corpus size, question type, factuality constraint, latency budget), output:

1. Architecture. Extractive, RAG with extractive reader, RAG with generative reader, or closed-book LLM. One-sentence reason.
2. Retriever. None, BM25, dense (name the encoder), or hybrid.
3. Reader. SQuAD-tuned model, LLM by name, or "domain-fine-tuned DistilBERT."
4. Evaluation. EM + F1 for extractive benchmarks; answer accuracy + citation accuracy + refusal calibration for production. Name what you are measuring and how you are measuring it.

Refuse closed-book LLM answers for regulatory or compliance-sensitive questions. Refuse any QA system without a retrieval-recall baseline (you cannot evaluate the reader without knowing the retriever surfaced the right passage). Flag questions that require multi-hop reasoning as needing specialized multi-hop retrievers like HotpotQA-trained systems.
```

## Упражнения

1. **Easy.** Настройте SQuAD extractive pipeline выше на 10 Wikipedia passages. Hand-craft 10 questions. Измерьте, как часто answer correct. Вы должны увидеть 7-9 correct, если passages и questions clean.
2. **Medium.** Добавьте refusal classifier. Когда top retrieval score ниже threshold (скажем, 0.3 cosine), возвращайте "I don't know" вместо вызова reader. Tune threshold на held-out set.
3. **Hard.** Постройте RAG pipeline по corpus из 10,000 documents на ваш выбор. Реализуйте hybrid retrieval (BM25 + dense) с RRF fusion (см. урок 14). Измерьте answer accuracy with and without hybrid step. Document, какие question types выигрывают больше всего.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Extractive QA | Find the answer span | Predict start и end indices answer внутри given passage. |
| Open-domain QA | QA over a corpus | Passage не дан; нужно сначала retrieve, затем answer. |
| RAG | Retrieve then generate | Retrieval-augmented generation. Pipeline retriever + reader. |
| SQuAD | Canonical benchmark | Stanford Question Answering Dataset. Metrics EM + F1. |
| Hallucination | Made-up answer | Reader output, не поддержанный retrieved context. |
| Refusal calibration | Know when to shut up | System correctly says "I don't know" when unable to answer. |

## Дополнительное чтение

- [Rajpurkar et al. (2016). SQuAD: 100,000+ Questions for Machine Comprehension of Text](https://arxiv.org/abs/1606.05250) — benchmark paper.
- [Karpukhin et al. (2020). Dense Passage Retrieval for Open-Domain QA](https://arxiv.org/abs/2004.04906) — DPR, canonical dense retriever for QA.
- [Lewis et al. (2020). Retrieval-Augmented Generation for Knowledge-Intensive NLP Tasks](https://arxiv.org/abs/2005.11401) — статья, которая дала имя RAG.
- [Gao et al. (2023). Retrieval-Augmented Generation for Large Language Models: A Survey](https://arxiv.org/abs/2312.10997) — comprehensive RAG survey.
