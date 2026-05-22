# Text Summarization

> Extractive systems говорят вам, что сказал документ. Abstractive systems говорят, что имел в виду автор. Разные задачи, разные pitfalls.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Phase 5 · 02 (BoW + TF-IDF), Phase 5 · 11 (Machine Translation)
**Время:** ~75 минут

## Проблема

Новостная статья на 2,000 слов попадает в вашу ленту. Вам нужны 120 слов, которые ее передают. Можно либо выбрать три самые важные sentences из статьи (extractive), либо переписать content своими словами (abstractive). Оба варианта называются summarization. Это совершенно разные problems.

Extractive summarization - задача ранжирования. Оцените каждое sentence, верните top-`k`. Output всегда grammatical, потому что он дословно взят из текста. Риск - пропустить content, распределенный по статье.

Abstractive summarization - задача генерации. Transformer производит новый text conditioned on input. Output fluent и compressive, но может hallucinate facts, которых не было в source. Риск - уверенная fabrication.

Этот урок строит оба подхода, вместе с failure mode, который принадлежит каждому.

## Концепция

![Extractive TextRank vs abstractive transformer](../assets/summarization.svg)

**Extractive.** Рассматривайте article как graph, где nodes - sentences, а edges - similarities. Запустите PageRank (или что-то похожее) по graph, чтобы оценить sentences по тому, насколько они связаны со всем остальным. Highest-scoring sentences становятся summary. Каноническая реализация - **TextRank** (Mihalcea and Tarau, 2004).

**Abstractive.** Fine-tune transformer encoder-decoder (BART, T5, Pegasus) на document-summary pairs. На inference модель читает document и генерирует summary token-by-token через cross-attention. Pegasus особенно использует gap-sentence pretraining objective, который делает его отличным для summarization без большого fine-tuning.

Evaluation с **ROUGE** (Recall-Oriented Understudy for Gisting Evaluation). ROUGE-1 и ROUGE-2 оценивают unigram и bigram overlap. ROUGE-L оценивает longest common subsequence. Higher is better, но 40 ROUGE-L - "good", а 50 - "exceptional." Каждая статья report-ит все три. Используйте package `rouge-score`.

## Соберите это

### Шаг 1: TextRank (extractive)

```python
import math
import re
from collections import Counter


def sentence_split(text):
    return re.split(r"(?<=[.!?])\s+", text.strip())


def similarity(s1, s2):
    w1 = Counter(s1.lower().split())
    w2 = Counter(s2.lower().split())
    intersection = sum((w1 & w2).values())
    denom = math.log(len(w1) + 1) + math.log(len(w2) + 1)
    if denom == 0:
        return 0.0
    return intersection / denom


def textrank(text, top_k=3, damping=0.85, iterations=50, epsilon=1e-4):
    sentences = sentence_split(text)
    n = len(sentences)
    if n <= top_k:
        return sentences

    sim = [[0.0] * n for _ in range(n)]
    for i in range(n):
        for j in range(n):
            if i != j:
                sim[i][j] = similarity(sentences[i], sentences[j])

    scores = [1.0] * n
    for _ in range(iterations):
        new_scores = [1 - damping] * n
        for i in range(n):
            total_out = sum(sim[i]) or 1e-9
            for j in range(n):
                if sim[i][j] > 0:
                    new_scores[j] += damping * sim[i][j] / total_out * scores[i]
        if max(abs(s - ns) for s, ns in zip(scores, new_scores)) < epsilon:
            scores = new_scores
            break
        scores = new_scores

    ranked = sorted(range(n), key=lambda k: scores[k], reverse=True)[:top_k]
    ranked.sort()
    return [sentences[i] for i in ranked]
```

Стоит назвать две вещи. Similarity function использует log-normalized word overlap, что является исходным вариантом TextRank. Cosine of TF-IDF vectors тоже работает. Damping factor 0.85 и iteration count - defaults PageRank.

### Шаг 2: abstractive с BART

```python
from transformers import pipeline

summarizer = pipeline("summarization", model="facebook/bart-large-cnn")

article = """(long news article text)"""

summary = summarizer(article, max_length=120, min_length=60, do_sample=False)
print(summary[0]["summary_text"])
```

BART-large-CNN fine-tuned на corpus CNN/DailyMail. Он из коробки производит news-style summaries. Для других domains (scientific papers, dialog, legal) используйте соответствующий checkpoint Pegasus или fine-tune на ваших target data.

### Шаг 3: ROUGE evaluation

```python
from rouge_score import rouge_scorer

scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)
scores = scorer.score(reference_summary, generated_summary)
print({k: round(v.fmeasure, 3) for k, v in scores.items()})
```

Всегда используйте stemming. Без него "running" и "run" считаются разными words, и ROUGE недосчитывает.

### Beyond ROUGE (summarization eval 2026)

ROUGE был доминирующей summarization metric двадцать лет, но в 2026 году сам по себе он недостаточен. Large-scale meta-analysis NLG papers показал:

- **BERTScore** (contextual embedding similarity) набирал позиции до 2023 года и теперь report-ится вместе с ROUGE в большинстве summarization papers.
- **BARTScore** рассматривает evaluation как generation: score summary по тому, насколько вероятным pretrained BART считает его given the source.
- **MoverScore** (Earth Mover's Distance over contextual embeddings) вышел на первое место в summarization benchmarks 2025 года, потому что лучше ROUGE ловит semantic overlap.
- **FactCC** и **QA-based faithfulness** были распространены в 2021-2023, теперь часто заменяются **G-Eval** (GPT-4 prompt chain, который scores coherence, consistency, fluency, relevance with chain-of-thought reasoning).
- **G-Eval** и похожие LLM-judge approaches совпадают с human judgment примерно в 80% случаев, когда rubrics хорошо спроектированы.

Production recommendation: report ROUGE-L для legacy comparison, BERTScore для semantic overlap, G-Eval для coherence и factuality. Calibrate на 50-100 human-labeled summaries.

### Шаг 4: проблема factuality

Abstractive summaries склонны к hallucination. Extractive summaries несут гораздо меньший hallucination risk, потому что output дословно взят из source, хотя они все равно могут вводить в заблуждение, если source sentences decontextualized, outdated или quoted out of order. Это главная причина, по которой production systems все еще предпочитают extractive methods для compliance-adjacent content.

Типы hallucination, которые стоит назвать:

- **Entity swap.** Source говорит "John Smith." Summary говорит "John Brown."
- **Number drift.** Source говорит "25,000." Summary говорит "25 million."
- **Polarity flip.** Source говорит "rejected the offer." Summary говорит "accepted the offer."
- **Fact invention.** Source не упоминает CEO. Summary говорит, что CEO одобрил.

Evaluation approaches, которые работают:

- **FactCC.** Binary classifier, обученный на entailment между source sentence и summary sentence. Predicts factual/not-factual.
- **QA-based factuality.** Задайте QA model вопросы, ответы на которые есть в source. Если summary поддерживает другие answers, flag.
- **Entity-level F1.** Сравните named entities в source и summary. Entities, присутствующие только в summary, подозрительны.

Для всего user-facing, где factuality важна (news, medical, legal, financial), extractive - более безопасный default. Abstractive требует factuality check в loop.

## Используйте это

Stack 2026 года:

| Use case | Recommended |
|---------|-------------|
| News, 3-5 sentence summary, English | `facebook/bart-large-cnn` |
| Scientific papers | `google/pegasus-pubmed` или tuned T5 |
| Multi-document, long-form | Любая LLM с 32k+ context, prompted |
| Dialog summarization | `philschmid/bart-large-cnn-samsum` |
| Extractive, low hallucination risk by construction | TextRank или `sumy`'s LSA / LexRank |

LLMs с long context часто превосходят specialized models в 2026 году, когда compute не является ограничением. Tradeoff - cost и reproducibility; specialized models дают более consistent outputs.

## Отгрузите это

Сохраните как `outputs/skill-summary-picker.md`:

```markdown
---
name: summary-picker
description: Pick extractive or abstractive, named library, factuality check.
version: 1.0.0
phase: 5
lesson: 12
tags: [nlp, summarization]
---

Given a task (document type, compliance requirement, length, compute budget), output:

1. Approach. Extractive or abstractive. Explain in one sentence why.
2. Starting model / library. Name it. `sumy.TextRankSummarizer`, `facebook/bart-large-cnn`, `google/pegasus-pubmed`, or an LLM prompt.
3. Evaluation plan. ROUGE-1, ROUGE-2, ROUGE-L (use rouge-score with stemming). Plus factuality check if abstractive.
4. One failure mode to probe. Entity swap is the most common in abstractive news summarization; flag samples where source entities do not appear in summary.

Refuse abstractive summarization for medical, legal, financial, or regulated content without a factuality gate. Flag input over the model's context window as needing chunked map-reduce summarization (not just truncation).
```

## Упражнения

1. **Easy.** Запустите TextRank на 5 news articles. Сравните top-3 sentences с reference summary. Измерьте ROUGE-L. Вы должны увидеть 30-45 ROUGE-L на articles в стиле CNN/DailyMail.
2. **Medium.** Реализуйте entity-level factuality: извлеките named entities из source и summary (spaCy), вычислите recall source entities в summary и precision summary entities относительно source. High precision и low recall означают safe but terse; low precision означает hallucinated entities.
3. **Hard.** Сравните BART-large-CNN с LLM (Claude или GPT-4) на 50 CNN/DailyMail articles. Report ROUGE-L, factuality (by entity F1) и cost per summary. Document where each wins.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|-----------------------|
| Extractive | Pick sentences | Вернуть sentences дословно из source. Никогда не hallucinate. |
| Abstractive | Rewrite | Generate new text conditioned on source. Может hallucinate. |
| ROUGE | Summary metric | N-gram / LCS overlap между system output и reference. |
| TextRank | Graph-based extractive | PageRank по graph similarity между sentences. |
| Factuality | Is it right | Поддержаны ли claims summary источником. |
| Hallucination | Made-up content | Content в summary, который source не поддерживает. |

## Дополнительное чтение

- [Mihalcea and Tarau (2004). TextRank: Bringing Order into Texts](https://aclanthology.org/W04-3252/) — canonical extractive paper.
- [Lewis et al. (2019). BART: Denoising Sequence-to-Sequence Pre-training](https://arxiv.org/abs/1910.13461) — статья BART.
- [Zhang et al. (2019). PEGASUS: Pre-training with Extracted Gap-sentences](https://arxiv.org/abs/1912.08777) — Pegasus и gap-sentence objective.
- [Lin (2004). ROUGE: A Package for Automatic Evaluation of Summaries](https://aclanthology.org/W04-1013/) — статья ROUGE.
- [Maynez et al. (2020). On Faithfulness and Factuality in Abstractive Summarization](https://arxiv.org/abs/2005.00661) — статья о factuality landscape.
