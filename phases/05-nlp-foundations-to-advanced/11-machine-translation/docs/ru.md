# Машинный перевод

> Translation - это задача, которая тридцать лет оплачивала NLP research и продолжает оплачивать его сейчас.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Phase 5 · 10 (Attention Mechanism), Phase 5 · 04 (GloVe, FastText, Subword)
**Время:** ~75 минут

## Цели обучения

- Вызывать предобученную NMT-модель и оценивать ее BLEU и chrF.
- Применять трехуровневую иерархию оценки 2026 и дообучать под домен.
- Диагностировать, что ломается в продакшен-переводе (галлюцинации, constrained decoding).

## Проблема

Модель читает предложение на одном языке и производит предложение на другом. Длина меняется. Порядок слов меняется. Некоторые source words отображаются в несколько target words и наоборот. Idioms отказываются от one-to-one mapping. "I miss you" по-французски - "tu me manques", буквально "you are lacking to me". Никакое word-level alignment этого не переживает.

Machine translation - задача, которая заставила NLP изобрести encoder-decoders, attention, transformers и в итоге всю LLM paradigm. Каждый шаг вперед появлялся потому, что качество перевода было измеримым, а разрыв между человеком и машиной оставался упрямым.

Этот урок пропускает историческую лекцию и учит рабочему pipeline 2026 года: pretrained multilingual encoder-decoder (NLLB-200 или mBART), subword tokenization, beam search, BLEU и chrF evaluation, а также нескольким failure modes, которые все еще незамеченными попадают в production.

## Концепция

![Конвейер MT: токенизация → кодирование → декодирование с attention → детокенизация](../assets/mt-pipeline.svg)

Современный MT - это transformer encoder-decoder, обученный на parallel text. Encoder читает source в tokenization своего языка. Decoder генерирует target по одному subword за раз, используя output encoder через cross-attention (урок 10). Decoding использует beam search, чтобы избежать ловушки greedy-decoding. Output detokenize-ится, detruecase-ится и оценивается относительно reference.

Три operational choices управляют реальным качеством MT.

- **Tokenizer.** SentencePiece BPE, обученный на mixed-language corpus. Shared vocabulary между языками - то, что включает zero-shot pairs в NLLB.
- **Model size.** NLLB-200 distilled 600M помещается на laptop. NLLB-200 3.3B - опубликованный production default. 54.5B - research ceiling.
- **Decoding.** Beam width 4-5 для общего контента. Length penalty, чтобы избежать слишком короткого output. Constrained decoding, когда нужна consistency терминологии.

## Соберите это

### Шаг 1: pretrained MT call

```python
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

model_id = "facebook/nllb-200-distilled-600M"
tok = AutoTokenizer.from_pretrained(model_id, src_lang="eng_Latn")
model = AutoModelForSeq2SeqLM.from_pretrained(model_id)

src = "The cats are running."
inputs = tok(src, return_tensors="pt")

out = model.generate(
    **inputs,
    forced_bos_token_id=tok.convert_tokens_to_ids("fra_Latn"),
    num_beams=5,
    length_penalty=1.0,
    max_new_tokens=64,
)
print(tok.batch_decode(out, skip_special_tokens=True)[0])
```

```text
Les chats courent.
```

Здесь важны три вещи. `src_lang` сообщает tokenizer, какой script и segmentation применять. `forced_bos_token_id` сообщает decoder, какой язык генерировать. Оба являются NLLB-specific tricks; mBART и M2M-100 используют свои conventions, и они не взаимозаменяемы.

### Шаг 2: BLEU и chrF

BLEU измеряет n-gram overlap между output и reference. Четыре размера reference n-grams (1-4), geometric mean precisions, brevity penalty для слишком короткого output. Score лежит в [0, 100]. Часто используется. Раздражающе труден для интерпретации: 30 BLEU - "usable"; 40 - "good"; 50 - "exceptional"; различия меньше 1 BLEU - шум.

chrF измеряет character-level F-score. Более чувствителен к morphologically rich languages, где BLEU недосчитывает matches. Часто report-ится вместе с BLEU.

```python
import sacrebleu

hypotheses = ["Les chats courent."]
references = [["Les chats courent."]]

bleu = sacrebleu.corpus_bleu(hypotheses, references)
chrf = sacrebleu.corpus_chrf(hypotheses, references)
print(f"BLEU: {bleu.score:.1f}  chrF: {chrf.score:.1f}")
```

Всегда используйте `sacrebleu`. Он нормализует tokenization, чтобы scores были сопоставимы между статьями. Самописный BLEU computation - путь к misleading benchmarks.

### Трехуровневая иерархия evaluation (2026)

Современный MT evaluation использует три взаимодополняющих семейства metrics. Отгружайте как минимум с двумя.

- **Heuristic** (BLEU, chrF). Быстро, reference-based, интерпретируемо, нечувствительно к paraphrase. Используйте для legacy comparison и regression detection.
- **Learned** (COMET, BLEURT, BERTScore). Neural models, обученные на human judgment; сравнивают semantic similarity перевода с source и reference. COMET имеет самую сильную связь с MT research с 2023 года и является production default 2026 года там, где quality важна.
- **LLM-as-judge** (reference-free). Prompt-ите большую модель оценить translations по fluency, adequacy, tone, cultural appropriateness. GPT-4-as-judge совпадает с human agreement примерно в 80% случаев, когда rubric хорошо спроектирован. Используйте для open-ended content, где reference не существует.

Практический stack 2026: `sacrebleu` для BLEU и chrF, `unbabel-comet` для COMET и prompted LLM для финального human-facing signal. Calibrate каждую metric на 50-100 human-labeled examples, прежде чем доверять ей на production data.

Reference-free metrics (COMET-QE, BLEURT-QE, LLM-as-judge) позволяют оценивать translations без reference, что важно для long-tail language pairs, где reference translations не существуют.

### Шаг 3: что ломается в production

Рабочий pipeline выше будет переводить fluently в 80% случаев и тихо проваливаться в оставшихся 20%. Именованные failure modes:

- **Hallucination.** Модель придумывает content, которого не было в source. Часто встречается в незнакомой domain vocabulary. Симптом: output fluent, но утверждает facts, которых source не заявлял. Mitigation: constrained decoding на domain terms, human review для regulated content, monitoring output, который намного длиннее input.
- **Off-target generation.** Модель переводит на неправильный язык. NLLB неожиданно склонен к этому на rare language pairs. Mitigation: проверяйте `forced_bos_token_id` и всегда decode с language-ID model check на output.
- **Terminology drift.** "Sign up" становится "s'inscrire" в doc 1 и "créer un compte" в doc 2. Для UI text и user-facing strings consistency важнее raw quality. Mitigation: glossary-constrained decoding или post-edit dictionary.
- **Formality mismatch.** Французские "tu" vs "vous", уровни вежливости в японском. Модель выбирает форму, которая была чаще в training. Для customer-facing content это обычно неверно. Mitigation: prompt prefix с formality token, если модель это поддерживает, или fine-tune small model на formal-only corpora.
- **Length explosion on short input.** Очень короткие input sentences часто дают overlong translations, потому что length penalty резко ломается ниже ~5 source tokens. Mitigation: жесткий max-length cap, пропорциональный source length.

### Шаг 4: fine-tuning для domain

Pretrained models - универсалы. Legal, medical или game-dialog translation заметно выигрывают от fine-tuning на domain parallel data. Recipe не экзотический:

```python
from transformers import Trainer, TrainingArguments
from datasets import Dataset

pairs = [
    {"src": "The defendant pleaded guilty.", "tgt": "L'accusé a plaidé coupable."},
]

ds = Dataset.from_list(pairs)


def preprocess(ex):
    return tok(
        ex["src"],
        text_target=ex["tgt"],
        truncation=True,
        max_length=128,
        padding="max_length",
    )


ds = ds.map(preprocess, remove_columns=["src", "tgt"])

args = TrainingArguments(output_dir="out", per_device_train_batch_size=4, num_train_epochs=3, learning_rate=3e-5)
Trainer(model=model, args=args, train_dataset=ds).train()
```

Несколько тысяч high-quality parallel examples лучше нескольких сотен тысяч noisy web-scraped examples. Quality training data - главный production lever.

## Используйте это

Production stack 2026 года для MT:

| Сценарий | Рекомендуемая отправная точка |
|---------|---------------------------|
| Any-to-any, 200 языков | `facebook/nllb-200-distilled-600M` (laptop) или `nllb-200-3.3B` (production) |
| Ориентация на английский, высокое качество, 50 языков | `facebook/mbart-large-50-many-to-many-mmt` |
| Короткие прогоны, дешевый инференс, English-French/German/Spanish | Helsinki-NLP / Marian models |
| Критичная задержка на стороне браузера | ONNX-квантованный Marian (~50 MB) |
| Максимальное качество, готовы платить | GPT-4 / Claude / Gemini с промптами для перевода |

LLMs теперь превосходят specialized MT models на нескольких language pairs по состоянию на 2026 год, особенно на idiomatic content и long context. Tradeoff - per-token cost и latency. Выбирайте LLM, когда context length, stylistic consistency или domain adaptation через prompting важнее throughput.

## Отгрузите это

Сохраните как `outputs/skill-mt-evaluator.md`:

```markdown
---
name: mt-evaluator
description: Evaluate a machine translation output for shipping.
version: 1.0.0
phase: 5
lesson: 11
tags: [nlp, translation, evaluation]
---

Given a source text and a candidate translation, output:

1. Automatic score estimate. BLEU and chrF ranges you would expect. State whether a reference is available.
2. Five-point human-verifiable check list: (a) content preservation (no hallucinations), (b) correct language, (c) register / formality match, (d) terminology consistency with glossary if provided, (e) no truncation or length explosion.
3. One domain-specific issue to probe. E.g., for legal: named entities and statute citations. For medical: drug names and dosages. For UI: placeholder variables `{name}`.
4. Confidence flag. "Ship" / "Ship with review" / "Do not ship". Tie to the severity of issues found in step 2.

Refuse to ship a translation without a language-ID check on output. Refuse to evaluate without a reference unless the user explicitly opts in to reference-free scoring (COMET-QE, BLEURT-QE). Flag any content over 1000 tokens as likely needing chunked translation.
```

## Упражнения

1. **Легко.** Переведите 5-sentence English paragraph на French и обратно на English с помощью `nllb-200-distilled-600M`. Измерьте, насколько round-trip близок к original. Вы должны увидеть semantic preservation с word-choice drift.
2. **Средне.** Реализуйте language-ID check на translation outputs с помощью `fasttext lid.176` или `langdetect`. Интегрируйте в MT call, чтобы off-target generations ловились до возврата.
3. **Сложно.** Fine-tune `nllb-200-distilled-600M` на 5,000-pair domain corpus по вашему выбору. Измерьте BLEU на held-out set до и после fine-tuning. Report, какие типы sentences улучшились, а какие regressed.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| BLEU | Translation score | N-gram precision с brevity penalty. [0, 100]. |
| chrF | Character F-score | Character-level F-score. Более чувствителен для morphologically rich languages. |
| NMT | Neural MT | Transformer encoder-decoder, обученный на parallel text. Default с 2017+. |
| NLLB | No Language Left Behind | Семейство 200-language MT models от Meta. |
| Constrained decoding | Controlled output | Принудить specific tokens или n-grams появиться / не появиться в output. |
| Hallucination | Invented content | Model output, не поддержанный source. |

## Дополнительное чтение

- [Costa-jussà et al. (2022). No Language Left Behind: Scaling Human-Centered Machine Translation](https://arxiv.org/abs/2207.04672) — статья NLLB.
- [Post (2018). A Call for Clarity in Reporting BLEU Scores](https://aclanthology.org/W18-6319/) — почему `sacrebleu` - единственный корректный способ report BLEU.
- [Popović (2015). chrF: character n-gram F-score for automatic MT evaluation](https://aclanthology.org/W15-3049/) — статья chrF.
- [Hugging Face MT guide](https://huggingface.co/docs/transformers/tasks/translation) — практический walkthrough по fine-tuning.
