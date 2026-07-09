# Подсловная токенизация — BPE, WordPiece, Unigram, SentencePiece

> Словные токенизаторы давятся на невиданных словах. Символьные токенизаторы раздувают длину последовательности. Подсловные токенизаторы выбирают середину. На одном из них поставляется каждый современный LLM.

**Тип:** Обучение
**Языки:** Python
**Предварительные требования:** Фаза 5 · 01 (Обработка текста), Фаза 5 · 04 (GloVe / FastText / Subword)
**Время:** ~60 минут

## Цели обучения

- Реализовывать BPE с нуля и кодировать текст выученными слияниями.
- Использовать SentencePiece и tiktoken на практике.
- Объяснять, почему подсловная токенизация бьет пословную и посимвольную.

## Проблема

В вашем словаре 50 000 слов. Пользователь вводит "untokenizable". Ваш токенизатор возвращает `[UNK]`. Теперь у модели нет никакого сигнала о слове. Хуже того: документ 90-го процентиля в вашем корпусе содержит 40 редких слов, что означает 40 бит потерянной информации на документ.

Подсловная токенизация решает это. Частые слова остаются одиночными токенами. Редкие слова раскладываются на значимые части: `untokenizable` → `un`, `token`, `izable`. Обучающие данные покрывают все, потому что любая строка в конечном счете является последовательностью байтов.

Каждая передовая LLM в 2026 поставляется на одном из трех алгоритмов (BPE, Unigram, WordPiece), обернутых в одну из трех библиотек (tiktoken, SentencePiece, HF Tokenizers). Вы не можете поставить языковую модель, не выбрав один из них.

## Концепция

![BPE vs Unigram vs WordPiece, посимвольно](../assets/subword-tokenization.svg)

**BPE (Byte-Pair Encoding).** Начните со словаря на уровне символов. Подсчитайте каждую соседнюю пару. Объедините самую частую пару в новый токен. Повторяйте, пока не достигнете целевого размера словаря. Доминирующий алгоритм: GPT-2/3/4, Llama, Gemma, Qwen2, Mistral.

**Byte-level BPE.** Тот же алгоритм, но поверх сырых байтов (256 базовых токенов), а не Unicode-символов. Гарантирует ноль токенов `[UNK]` — любая байтовая последовательность кодируется. GPT-2 использует 50 257 токенов (256 bytes + 50,000 merges + 1 special).

**Unigram.** Начните с огромного словаря. Назначьте каждому токену unigram probability. Итеративно удаляйте токены, удаление которых меньше всего увеличивает corpus log-likelihood. Вероятностный на инференсе: можно сэмплировать токенизации (полезно для аугментации данных через subword regularization). Используется T5, mBART, ALBERT, XLNet, Gemma.

**WordPiece.** Объединяет пары, которые максимизируют likelihood обучающего корпуса, а не сырую частоту. Используется BERT, DistilBERT, ELECTRA.

**SentencePiece vs tiktoken.** SentencePiece - библиотека, которая *обучает* словари (BPE или Unigram) напрямую на сыром Unicode-тексте, кодируя пробел как `▁`. tiktoken - быстрый *encoder* OpenAI для заранее построенных словарей; он не обучает.

Практическое правило:

- **Обучение нового словаря:** SentencePiece (многоязычный, без pre-tokenization) или HF Tokenizers.
- **Быстрый инференс с GPT vocab:** tiktoken (cl100k_base, o200k_base).
- **И то и другое:** HF Tokenizers — одна библиотека, training + serving.

## Соберите это

### Шаг 1: BPE с нуля

См. `code/main.py`. Цикл:

```python
def train_bpe(corpus, num_merges):
    vocab = {tuple(word) + ("</w>",): count for word, count in corpus.items()}
    merges = []
    for _ in range(num_merges):
        pairs = Counter()
        for symbols, freq in vocab.items():
            for a, b in zip(symbols, symbols[1:]):
                pairs[(a, b)] += freq
        if not pairs:
            break
        best = pairs.most_common(1)[0][0]
        merges.append(best)
        vocab = apply_merge(vocab, best)
    return merges
```

Три факта, которые кодирует алгоритм. `</w>` помечает конец слова, поэтому "low" (суффикс) и "lower" (префикс) остаются различимыми. Взвешивание по частоте заставляет высокочастотные пары побеждать рано. Список слияний упорядочен — инференс применяет слияния в порядке обучения.

### Шаг 2: кодирование с выученными слияниями

```python
def encode_bpe(word, merges):
    symbols = list(word) + ["</w>"]
    for a, b in merges:
        i = 0
        while i < len(symbols) - 1:
            if symbols[i] == a and symbols[i + 1] == b:
                symbols = symbols[:i] + [a + b] + symbols[i + 2:]
            else:
                i += 1
    return symbols
```

Наивная сложность O(n·|merges|). Production-реализации (tiktoken, HF Tokenizers) используют merge-rank lookup с priority queues и работают почти за линейное время.

### Шаг 3: SentencePiece на практике

```python
import sentencepiece as spm

spm.SentencePieceTrainer.train(
    input="corpus.txt",
    model_prefix="my_tokenizer",
    vocab_size=8000,
    model_type="bpe",          # or "unigram"
    character_coverage=0.9995, # lower for CJK (e.g. 0.9995 for English, 0.995 for Japanese)
    normalization_rule_name="nmt_nfkc",
)

sp = spm.SentencePieceProcessor(model_file="my_tokenizer.model")
print(sp.encode("untokenizable", out_type=str))
# ['▁un', 'token', 'izable']
```

Обратите внимание: pre-tokenization не требуется, пробел кодируется как `▁`, `character_coverage` управляет тем, насколько агрессивно редкие символы сохраняются, а не отображаются в `<unk>`.

### Шаг 4: tiktoken для OpenAI-совместимых словарей

```python
import tiktoken
enc = tiktoken.get_encoding("o200k_base")
print(enc.encode("untokenizable"))        # [127340, 101028]
print(len(enc.encode("Hello, world!")))   # 4
```

Только кодирование. Быстро (Rust backend). Точное совпадение с токенизацией GPT-4/5 для подсчета байтов, оценки стоимости и планирования context-window budgeting.

## Ловушки, которые все еще попадают в поставку в 2026

- **Tokenizer drift.** Обучение на vocab A, развертывание с vocab B. Token IDs отличаются; модель выдает мусор. Проверяйте hash `tokenizer.json` в CI.
- **Неоднозначность пробелов.** BPE "hello" и " hello" дают разные токены. Всегда явно задавайте `add_special_tokens` и `add_prefix_space`.
- **Недообучение многоязычности.** Англоцентричные корпуса создают словари, которые разбивают нелатинские письменности в 5-10x больше токенов. Один и тот же prompt стоит в 5-10x дороже на японском/арабском в GPT-3.5. o200k_base частично исправил это.
- **Разбиение emoji.** Один emoji может занимать 5 токенов. Проверяйте обработку emoji при бюджетировании контекста.

## Используйте это

Стек 2026:

| Ситуация | Выбор |
|-----------|------|
| Обучение одноязычной модели с нуля | HF Tokenizers (BPE) |
| Обучение многоязычной модели | SentencePiece (Unigram, `character_coverage=0.9995`) |
| Обслуживание OpenAI-compatible API | tiktoken (`o200k_base` for GPT-4+) |
| Доменный словарь (code, math, protein) | Обучите custom BPE на доменном корпусе, объедините с base vocab |
| Edge-инференс, малая модель | Unigram (меньшие словари работают лучше) |

Размер словаря - это решение о масштабировании, а не константа. Грубая эвристика: 32k для <1B params, 50-100k для 1-10B, 200k+ для multilingual/frontier.

## Доведите до поставки

Сохраните как `outputs/skill-tokenizer-picker.md`:

```markdown
---
name: tokenizer-picker
description: Pick tokenizer algorithm, vocab size, library for a given corpus and deployment target.
version: 1.0.0
phase: 5
lesson: 19
tags: [nlp, tokenization]
---

Given a corpus (size, languages, domain) and deployment target (training from scratch / fine-tuning / API-compatible inference), output:

1. Algorithm. BPE, Unigram, or WordPiece. One-sentence reason.
2. Library. SentencePiece, HF Tokenizers, or tiktoken. Reason.
3. Vocab size. Rounded to nearest 1k. Reason tied to model size and language coverage.
4. Coverage settings. `character_coverage`, `byte_fallback`, special-token list.
5. Validation plan. Average tokens-per-word on held-out set, OOV rate, compression ratio, round-trip decode equality.

Refuse to train a character-coverage <0.995 tokenizer on corpora with rare-script content. Refuse to ship a vocab without a frozen `tokenizer.json` hash check in CI. Flag any monolingual tokenizer under 16k vocab as likely under-spec.
```

## Упражнения

1. **Легко.** Обучите BPE с 500 слияниями на маленьком корпусе из `code/main.py`. Закодируйте три отложенных слова. Сколько из них дали ровно 1 токен, а сколько >1 токена?
2. **Средне.** Сравните число токенов на 100 английских предложениях из Wikipedia между `cl100k_base`, `o200k_base` и SentencePiece BPE, который вы обучите с vocab=32k. Сообщите compression ratio для каждого.
3. **Сложно.** Обучите один и тот же корпус с BPE, Unigram и WordPiece. Измерьте downstream accuracy при использовании каждого на небольшом классификаторе тональности. Сдвигает ли выбор результат больше чем на 1 point F1?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-------------------|--------------------------------|
| BPE | Byte-Pair Encoding | Жадное слияние самых частых пар символов до достижения целевого размера словаря. |
| Byte-level BPE | Неизвестных токенов никогда нет | BPE поверх сырых 256 bytes; GPT-2 / Llama используют это. |
| Unigram | Вероятностный токенизатор | Удаляет из большого набора кандидатов по log-likelihood; используется T5, Gemma. |
| SentencePiece | Тот, что про пробелы | Библиотека, которая обучает BPE/Unigram на сыром тексте; пробел кодируется как `▁`. |
| tiktoken | Быстрый | BPE-encoder OpenAI на Rust для заранее построенных словарей. Без обучения. |
| Merge list | Магические числа | Упорядоченный список слияний `(a, b) → ab`; инференс применяет их по порядку. |
| Character coverage | Насколько редкое - слишком редкое? | Доля символов в обучающем корпусе, которую токенизатор должен покрывать; типично ~0.9995. |

## Дополнительное чтение

- [Sennrich, Haddow, Birch (2015). Neural Machine Translation of Rare Words with Subword Units](https://arxiv.org/abs/1508.07909) — статья BPE.
- [Kudo (2018). Subword Regularization with Unigram Language Model](https://arxiv.org/abs/1804.10959) — статья Unigram.
- [Kudo, Richardson (2018). SentencePiece: A simple and language independent subword tokenizer](https://arxiv.org/abs/1808.06226) — библиотека.
- [Hugging Face — Summary of the tokenizers](https://huggingface.co/docs/transformers/tokenizer_summary) — краткий справочник.
- [OpenAI tiktoken repo](https://github.com/openai/tiktoken) — cookbook + список encoding.
