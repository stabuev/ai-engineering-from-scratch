# Оценивание LLM — RAGAS, DeepEval, G-Eval

> Exact-match и F1 пропускают семантическую эквивалентность. Ручная проверка не масштабируется. LLM-as-judge - производственный ответ, если есть достаточная калибровка, чтобы доверять числу.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 5 · 13 (Question Answering), Фаза 5 · 14 (Information Retrieval)
**Время:** ~75 минут

## Цели обучения

- Измерять фактологичность через NLI (в стиле RAGAS), релевантность ответа и кастомную метрику G-Eval.
- Встраивать оценку в CI-гейт.
- Калибровать LLM-as-judge против его смещений.

## Проблема

Ваша RAG-система отвечает: "June 29th, 2007."
Эталонная ссылка: "June 29, 2007."
Exact Match дает 0. F1 дает ~75%. Человек поставил бы 100%.

Теперь умножьте на 10 000 тестовых случаев. Еще раз умножьте на каждое изменение retriever, chunking, prompt или model. Вам нужен оценщик, который понимает смысл, дешево работает в масштабе, не лжет о регрессиях и показывает правильные режимы отказа.

В 2026 году есть три фреймворка, которые владеют этой проблемой.

- **RAGAS.** Retrieval-Augmented Generation ASsessment. Четыре RAG-метрики (faithfulness, answer-relevance, context-precision, context-recall) с backend на NLI + LLM-judge. Подкреплен исследованиями, легковесен.
- **DeepEval.** Pytest для LLM. G-Eval, task-completion, hallucination, bias metrics. Нативен для CI/CD.
- **G-Eval.** Метод (и метрика DeepEval): LLM-as-judge с chain-of-thought, пользовательскими критериями, оценкой 0-1.

Все три опираются на LLM-as-judge. Этот урок строит интуицию для метода и слоя доверия вокруг него.

## Концепция

![Четыре измерения оценивания, архитектура LLM-as-judge](../assets/llm-evaluation.svg)

**LLM-as-judge.** Замените статическую метрику LLM, которая оценивает выводы по рубрике. Для `(query, context, answer)` попросите judge LLM: "Score 0-1 on faithfulness." Верните оценку.

Почему это работает: LLM приближают человеческое суждение за крошечную долю стоимости. GPT-4o-mini примерно за ~$0.003 на оцененный случай позволяет проводить regression eval runs на 1000 примерах дешевле $5.

Почему это тихо ломается:

1. **Смещение судьи.** Судьи предпочитают более длинные ответы, ответы из собственного семейства моделей, ответы, совпадающие со стилем prompt.
2. **Ошибки разбора JSON.** Плохой JSON → NaN score → молча исключается из aggregate. Пользователи RAGAS знают эту боль. Ставьте gate через try/except + явный режим отказа.
3. **Дрейф между версиями модели.** Обновление judge меняет каждую метрику. Фиксируйте judge model + version.

**Четыре RAG-метрики.**

| Метрика | Вопрос | Backend |
|--------|----------|---------|
| Faithfulness | Каждый ли claim в ответе происходит из извлеченного context? | NLI-based entailment |
| Answer relevance | Отвечает ли answer на question? | Сгенерировать hypothetical questions из answer; сравнить с real question |
| Context precision | Какая доля retrieved chunks была релевантной? | LLM-judge |
| Context recall | Вернул ли retrieval все необходимое? | LLM-judge against gold answer |

**G-Eval.** Определите пользовательский критерий: "Did the answer cite the correct source?" Фреймворк автоматически разворачивает его в шаги оценивания chain-of-thought, затем выставляет 0-1. Хорошо для предметно-специфических измерений качества, которые RAGAS не покрывает.

**Калибровка.** Никогда не доверяйте сырой оценке judge, пока у вас нет корреляции с человеческими метками. Запустите 100 вручную размеченных примеров. Постройте график judge vs human. Посчитайте Spearman rho. Если rho < 0.7, ваша рубрика judge требует доработки.

## Соберите это

### Шаг 1: faithfulness с NLI (стиль RAGAS)

```python
from typing import Callable
from transformers import pipeline

nli = pipeline("text-classification",
               model="MoritzLaurer/DeBERTa-v3-large-mnli-fever-anli-ling-wanli",
               top_k=None)

# `llm` is any callable: prompt str -> generated str.
# Example: llm = lambda p: client.messages.create(model="claude-haiku-4-5", ...).content[0].text
LLM = Callable[[str], str]


def atomic_claims(answer: str, llm: LLM) -> list[str]:
    prompt = f"""Break this answer into simple factual claims (one per line):
{answer}
"""
    return llm(prompt).splitlines()


def faithfulness(answer: str, context: str, llm: LLM) -> float:
    claims = atomic_claims(answer, llm)
    if not claims:
        return 0.0
    supported = 0
    for claim in claims:
        result = nli({"text": context, "text_pair": claim})[0]
        entail = next((s for s in result if s["label"] == "entailment"), None)
        if entail and entail["score"] > 0.5:
            supported += 1
    return supported / len(claims)
```

Разложите ответ на атомарные claims. Проверяйте каждый claim через NLI относительно извлеченного context. Faithfulness = доля подтвержденных claims.

### Шаг 2: answer relevance

```python
import numpy as np
from sentence_transformers import SentenceTransformer

# encoder: any model implementing .encode(texts, normalize_embeddings=True) -> ndarray
# e.g., encoder = SentenceTransformer("BAAI/bge-small-en-v1.5")

def answer_relevance(question: str, answer: str, encoder, llm: LLM, n: int = 3) -> float:
    prompt = f"Write {n} questions this answer could be the answer to:\n{answer}"
    generated = [line for line in llm(prompt).splitlines() if line.strip()][:n]
    if not generated:
        return 0.0
    q_emb = np.asarray(encoder.encode([question], normalize_embeddings=True)[0])
    g_embs = np.asarray(encoder.encode(generated, normalize_embeddings=True))
    sims = [float(q_emb @ g_emb) for g_emb in g_embs]
    return sum(sims) / len(sims)
```

Если answer подразумевает другие questions, чем заданный, relevance падает.

### Шаг 3: пользовательская метрика G-Eval

```python
from deepeval.metrics import GEval
from deepeval.test_case import LLMTestCaseParams, LLMTestCase

metric = GEval(
    name="Correctness",
    criteria="The answer should be factually accurate and match the expected output.",
    evaluation_steps=[
        "Read the expected output.",
        "Read the actual output.",
        "List factual claims in the actual output.",
        "For each claim, mark supported or unsupported by the expected output.",
        "Return score = fraction supported.",
    ],
    evaluation_params=[LLMTestCaseParams.INPUT, LLMTestCaseParams.ACTUAL_OUTPUT, LLMTestCaseParams.EXPECTED_OUTPUT],
)

test = LLMTestCase(input="When was the first iPhone released?",
                   actual_output="June 29th, 2007.",
                   expected_output="June 29, 2007.")
metric.measure(test)
print(metric.score, metric.reason)
```

Шаги оценивания - это рубрика. Явные шаги стабильнее, чем неявные prompts "score 0-1".

### Шаг 4: CI gate

```python
import deepeval
from deepeval.metrics import FaithfulnessMetric, ContextualRelevancyMetric


def test_rag_system():
    cases = load_regression_cases()
    faith = FaithfulnessMetric(threshold=0.85)
    rel = ContextualRelevancyMetric(threshold=0.7)
    for case in cases:
        faith.measure(case)
        assert faith.score >= 0.85, f"faithfulness regression on {case.id}"
        rel.measure(case)
        assert rel.score >= 0.7, f"relevancy regression on {case.id}"
```

Поставляйте как pytest-файл. Запускайте на каждом PR. Блокируйте merge при регрессиях.

### Шаг 5: игрушечное eval с нуля

См. `code/main.py`. Stdlib-only приближения faithfulness (пересечение claims ответа с context) и relevance (пересечение токенов answer с tokens question). Не production. Показывает форму.

## Подводные камни

- **Нет калибровки.** Judge с корреляцией 0.3 с человеческими метками - это шум. Требуйте calibration run перед поставкой.
- **Самооценивание.** Использование одной и той же LLM для генерации и judging завышает оценки на 10-20%. Используйте для judge другое семейство моделей.
- **Позиционное смещение в попарном judging.** Судьи предпочитают первый представленный вариант. Всегда рандомизируйте порядок и запускайте оба.
- **Сырой aggregate скрывает отказы.** Mean score 0.85 часто скрывает 5% катастрофических отказов. Всегда проверяйте нижний квантиль.
- **Гниение golden dataset.** Неверсионированные eval sets, которые дрейфуют со временем, ломают продольное сравнение. Помечайте dataset при каждом изменении.
- **Стоимость LLM.** В масштабе judge calls доминируют в стоимости. Используйте самую дешевую модель, которая проходит calibration threshold. GPT-4o-mini, Claude Haiku, Mistral-small.

## Используйте это

Стек 2026 года:

| Сценарий | Фреймворк |
|---------|-----------|
| Мониторинг качества RAG | RAGAS (4 metrics) |
| Regression gates в CI/CD | DeepEval + pytest |
| Пользовательские критерии домена | G-Eval внутри DeepEval |
| Онлайн-мониторинг live traffic | RAGAS with reference-free mode |
| Выборочные проверки human-in-the-loop | LangSmith или Phoenix с annotation UI |
| Red-teaming / оценка safety | Promptfoo + DeepEval |

Типичный стек: RAGAS для мониторинга, DeepEval для CI, G-Eval для новых измерений. Запускайте все три; они полезно расходятся.

## Доведите до поставки

Сохраните как `outputs/skill-eval-architect.md`:

```markdown
---
name: eval-architect
description: Design an LLM evaluation plan with calibrated judge and CI gates.
version: 1.0.0
phase: 5
lesson: 27
tags: [nlp, evaluation, rag]
---

Given a use case (RAG / agent / generative task), output:

1. Metrics. Faithfulness / relevance / context-precision / context-recall + any custom G-Eval metrics with criteria.
2. Judge model. Named model + version, rationale for cost vs accuracy.
3. Calibration. Hand-labeled set size, target Spearman rho vs human > 0.7.
4. Dataset versioning. Tag strategy, change log, stratification.
5. CI gate. Thresholds per metric, regression-window logic, bottom-quantile alert.

Refuse to rely on a judge untested against ≥50 human-labeled examples. Refuse self-evaluation (same model generates + judges). Refuse aggregate-only reporting without bottom-10% surfacing. Flag any pipeline where judge upgrade lands without parallel baseline eval.
```

## Упражнения

1. **Легко.** Используйте RAGAS на 10 RAG-примерах с известными галлюцинациями. Проверьте, что метрика faithfulness ловит каждую.
2. **Средне.** Вручную разметьте 50 QA-ответов оценками 0-1 за correctness. Оцените через G-Eval. Измерьте Spearman rho между judge и human.
3. **Сложно.** Постройте pytest CI gate с DeepEval. Намеренно ухудшите retriever. Проверьте, что gate падает. Добавьте bottom-quantile alerting через threshold check по нижним 10%.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| LLM-as-judge | Scoring with an LLM | Prompt judge model, чтобы оценить outputs 0-1 по рубрике. |
| RAGAS | The RAG metric library | Open-source eval framework с 4 reference-free RAG metrics. |
| Faithfulness | Is the answer grounded? | Доля claims ответа, логически следующих из retrieved context. |
| Context precision | Were retrieved chunks relevant? | Доля top-K chunks, которые действительно имели значение. |
| Context recall | Did retrieval find everything? | Доля gold-answer claims, поддержанных retrieved chunks. |
| G-Eval | Custom LLM judge | Рубрика + chain-of-thought eval steps + оценка 0-1. |
| Calibration | Trust but verify | Корреляция Spearman между judge score и human score. |

## Дополнительное чтение

- [Es et al. (2023). RAGAS: Automated Evaluation of Retrieval Augmented Generation](https://arxiv.org/abs/2309.15217) — статья RAGAS.
- [Liu et al. (2023). G-Eval: NLG Evaluation using GPT-4 with Better Human Alignment](https://arxiv.org/abs/2303.16634) — статья G-Eval.
- [DeepEval docs](https://deepeval.com/docs/metrics-introduction) — открытый production-стек.
- [Zheng et al. (2023). Judging LLM-as-a-Judge with MT-Bench and Chatbot Arena](https://arxiv.org/abs/2306.05685) — смещения, калибровка, ограничения.
- [MLflow GenAI Scorer](https://mlflow.org/blog/third-party-scorers) — объединяющий фреймворк, который интегрирует RAGAS, DeepEval, Phoenix.
