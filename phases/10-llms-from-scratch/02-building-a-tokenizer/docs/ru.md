# Создание токенизатора с нуля

> Урок 01 дал вам игрушку. Этот урок дает вам инструмент.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Phase 10, Lesson 01 (Tokenizers: BPE, WordPiece, SentencePiece)
**Время:** ~90 минут

## Цели обучения

- Построить production-grade BPE tokenizer, который обрабатывает Unicode, нормализацию пробелов и специальные токены
- Реализовать byte-level fallback, чтобы токенизатор мог кодировать любой вход (включая emoji, CJK и код) без неизвестных токенов
- Добавить regex-паттерны pre-tokenization, которые делят текст по границам слов перед применением BPE merges
- Обучить кастомный токенизатор на корпусе и оценить его compression ratio относительно tiktoken на multilingual тексте

## Проблема

Ваш BPE tokenizer из Lesson 01 работает на английском тексте. Теперь дайте ему японский. Или emoji. Или Python-код со смешанными tabs и spaces.

Он ломается.

Не потому что BPE неправильный, а потому что реализация неполная. Production tokenizer обрабатывает сырые байты в любой кодировке, нормализует Unicode перед разбиением, управляет специальными токенами, которые никогда не сливаются, связывает pre-tokenization с subword splitting и делает все это достаточно быстро, чтобы не стать узким местом training pipeline, обрабатывающего 15 триллионов токенов.

У токенизатора GPT-2 50,257 токенов. У Llama 3 -- 128,256. У GPT-4 примерно 100,000. Это не игрушечные числа. Merge tables за этими словарями обучались на сотнях гигабайт текста, а окружающая механика -- normalization, pre-tokenization, special token injection, chat template formatting -- отделяет токенизатор, который справляется с "hello world", от токенизатора, который справляется со всем интернетом.

Вы построите именно эту механику.

## Концепция

### Полный pipeline

Production tokenizer -- это не один алгоритм. Это pipeline из пяти стадий, каждая из которых решает отдельную проблему.

```mermaid
graph LR
    A[Raw Text] --> B[Normalize]
    B --> C[Pre-Tokenize]
    C --> D[BPE Merge]
    D --> E[Special Tokens]
    E --> F[Token IDs]

    style A fill:#1a1a2e,stroke:#e94560,color:#fff
    style B fill:#1a1a2e,stroke:#e94560,color:#fff
    style C fill:#1a1a2e,stroke:#e94560,color:#fff
    style D fill:#1a1a2e,stroke:#e94560,color:#fff
    style E fill:#1a1a2e,stroke:#e94560,color:#fff
    style F fill:#1a1a2e,stroke:#e94560,color:#fff
```

У каждой стадии своя задача:

| Стадия | Что делает | Почему это важно |
|-------|-------------|----------------|
| Normalize | NFKC Unicode, lowercase optional, strip accents optional | Лигатура "fi" (U+FB01) становится "fi" (два символа). Без этого одно и то же слово получает разные токены. |
| Pre-Tokenize | Split text into chunks before BPE | Не дает BPE сливать через границы слов. "the cat" не должен порождать токен "e c". |
| BPE Merge | Apply learned merge rules to byte sequences | Основное сжатие. Превращает сырые байты в subword-токены. |
| Special Tokens | Inject [BOS], [EOS], [PAD], chat template markers | У этих токенов фиксированные IDs. Они никогда не участвуют в BPE merges. Модели они нужны для структуры. |
| ID Mapping | Convert token strings to integer IDs | Модель видит целые числа, а не строки. |

### Byte-Level BPE

Токенизатор из Lesson 01 работал с UTF-8 bytes. Это было правильное решение. Но мы пропустили важный вопрос: что происходит, когда эти байты не являются валидным UTF-8?

Byte-level BPE решает это, считая каждое возможное значение байта (0-255) допустимым токеном. Базовый словарь состоит ровно из 256 элементов. Любой файл -- текстовый, бинарный, поврежденный -- можно токенизировать без неизвестного токена.

GPT-2 добавил трюк: сопоставлять каждый байт печатному Unicode-символу, чтобы словарь оставался читаемым человеком. Byte 0x20 (space) становится символом "G" в их mapping. Это чистая косметика. Алгоритму все равно.

Настоящая сила в другом: byte-level BPE обрабатывает каждый язык на земле. Китайские символы -- по 3 UTF-8 bytes. Японские могут занимать 3-4 bytes. Arabic, Devanagari, emoji -- все это просто byte sequences. Алгоритм BPE находит паттерны в этих byte sequences точно так же, как находит паттерны в английских ASCII bytes.

### Pre-Tokenization

Перед тем как BPE коснется текста, его нужно разбить на chunks. Это мешает merge algorithm создавать токены, пересекающие границы слов.

GPT-2 использует regex pattern для разбиения текста:

```
'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+
```

Этот pattern делит contractions ("don't" становится "don" + "'t"), слова с необязательным ведущим пробелом, числа, пунктуацию и whitespace. Ведущий пробел остается прикрепленным к слову, поэтому "the cat" становится [" the", " cat"], а не ["the", " ", "cat"].

Llama использует SentencePiece, который полностью пропускает regex. Он рассматривает сырой byte stream как одну длинную последовательность и позволяет BPE самому выяснять границы. Это проще, но дает BPE больше свободы создавать cross-word tokens.

Выбор имеет значение. Regex GPT-2 не дает токенизатору выучить, что "the" в конце одного слова и "the" в начале следующего нужно слить. SentencePiece это допускает, что иногда дает более эффективное сжатие, но менее интерпретируемые токены.

### Специальные токены

Каждый production tokenizer резервирует token IDs для структурных маркеров:

| Token | Purpose | Used By |
|-------|---------|---------|
| `[BOS]` / `<s>` | Beginning of sequence | Llama 3, GPT |
| `[EOS]` / `</s>` | End of sequence | All models |
| `[PAD]` | Padding for batch alignment | BERT, T5 |
| `[UNK]` | Unknown token (byte-level BPE eliminates this) | BERT, WordPiece |
| `<\|im_start\|>` | Chat message boundary start | ChatGPT, Qwen |
| `<\|im_end\|>` | Chat message boundary end | ChatGPT, Qwen |
| `<\|user\|>` | User turn marker | Llama 3 |
| `<\|assistant\|>` | Assistant turn marker | Llama 3 |

Специальные токены никогда не делятся BPE. Они точно сопоставляются до запуска алгоритма слияний, заменяются на свой фиксированный ID, а окружающий текст токенизируется обычным образом.

### Chat Templates

Вот здесь большинство людей путается, и большинство реализаций ломается.

Когда вы отправляете сообщения в чат-модель, API принимает список сообщений:

```
[
  {"role": "system", "content": "You are helpful."},
  {"role": "user", "content": "Hello"},
  {"role": "assistant", "content": "Hi there!"}
]
```

Модель не видит JSON. Она видит плоскую последовательность токенов. Chat template преобразует сообщения в эту плоскую последовательность с помощью специальных токенов. Каждая модель делает это по-своему:

```
Llama 3:
<|begin_of_text|><|start_header_id|>system<|end_header_id|>

You are helpful.<|eot_id|><|start_header_id|>user<|end_header_id|>

Hello<|eot_id|><|start_header_id|>assistant<|end_header_id|>

Hi there!<|eot_id|>

ChatGPT:
<|im_start|>system
You are helpful.<|im_end|>
<|im_start|>user
Hello<|im_end|>
<|im_start|>assistant
Hi there!<|im_end|>
```

Если template неправильный, модель генерирует мусор. Она обучалась на одном точном формате. Любое отклонение -- пропущенная новая строка, перепутанный токен, лишний пробел -- помещает вход вне обучающего распределения.

### Скорость

Python слишком медленный для промышленной токенизации.

tiktoken (OpenAI) написан на Rust с Python bindings. HuggingFace tokenizers тоже на Rust. SentencePiece написан на C++. Они дают ускорение в 10-100x по сравнению с чистым Python.

Для масштаба: токенизация 15 триллионов токенов для Llama 3 pre-training со скоростью 1 million tokens per second (быстрый Python) заняла бы 174 дня. При 100 million tokens per second (Rust) это занимает 1.7 дня.

Вы пишете на Python, чтобы понять алгоритм. В production вы использовали бы compiled implementation и трогали бы только Python wrapper.

## Постройте это

### Шаг 1: Byte-Level Encoding

Основа. Преобразуйте любую строку в sequence of bytes, сопоставьте каждый byte печатному символу для отображения и выполните обратный процесс.

```python
def bytes_to_tokens(text):
    return list(text.encode("utf-8"))

def tokens_to_text(token_bytes):
    return bytes(token_bytes).decode("utf-8", errors="replace")
```

Проверьте на multilingual text, чтобы увидеть byte counts:

```python
texts = [
    ("English", "hello"),
    ("Chinese", "你好"),
    ("Emoji", "🔥"),
    ("Mixed", "hello你好🔥"),
]

for label, text in texts:
    b = bytes_to_tokens(text)
    print(f"{label}: {len(text)} chars -> {len(b)} bytes -> {b}")
```

"hello" это 5 bytes. "你好" это 6 bytes (по 3 на символ). Fire emoji это 4 bytes. Byte-level tokenizer не волнует, какой это язык. Bytes are bytes.

### Шаг 2: Pre-Tokenizer with Regex

Разбейте текст на chunks с помощью GPT-2 regex pattern. Каждый chunk токенизируется BPE независимо.

```python
import re

try:
    import regex
    GPT2_PATTERN = regex.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?\p{L}+| ?\p{N}+| ?[^\s\p{L}\p{N}]+|\s+(?!\S)|\s+"""
    )
except ImportError:
    GPT2_PATTERN = re.compile(
        r"""'(?:[sdmt]|ll|ve|re)| ?[a-zA-Z]+| ?[0-9]+| ?[^\s\w]+|\s+(?!\S)|\s+"""
    )

def pre_tokenize(text):
    return [match.group() for match in GPT2_PATTERN.finditer(text)]
```

Модуль `regex` поддерживает Unicode property escapes (`\p{L}` для букв, `\p{N}` для чисел). Модуль standard library `re` не поддерживает их, поэтому мы откатываемся к ASCII character classes. Для production multilingual tokenizers установите `regex`.

Попробуйте:

```python
print(pre_tokenize("Hello, world! Don't stop."))
# [' Hello', ',', ' world', '!', " Don", "'t", ' stop', '.']
```

Ведущий пробел остается прикрепленным к слову. Contractions делятся по apostrophe. Пунктуация становится отдельным chunk. BPE никогда не сольет токены через эти границы.

### Шаг 3: BPE on Byte Sequences

Основной алгоритм из Lesson 01, но теперь он работает с pre-tokenized chunks независимо.

```python
from collections import Counter

def get_byte_pairs(chunks):
    pairs = Counter()
    for chunk in chunks:
        byte_seq = list(chunk.encode("utf-8"))
        for i in range(len(byte_seq) - 1):
            pairs[(byte_seq[i], byte_seq[i + 1])] += 1
    return pairs

def apply_merge(byte_seq, pair, new_id):
    merged = []
    i = 0
    while i < len(byte_seq):
        if i < len(byte_seq) - 1 and byte_seq[i] == pair[0] and byte_seq[i + 1] == pair[1]:
            merged.append(new_id)
            i += 2
        else:
            merged.append(byte_seq[i])
            i += 1
    return merged
```

### Шаг 4: Special Token Handling

Специальным токенам нужны exact matching и fixed IDs. Они полностью обходят BPE.

```python
class SpecialTokenHandler:
    def __init__(self):
        self.special_tokens = {}
        self.pattern = None

    def add_token(self, token_str, token_id):
        self.special_tokens[token_str] = token_id
        escaped = [re.escape(t) for t in sorted(self.special_tokens.keys(), key=len, reverse=True)]
        self.pattern = re.compile("|".join(escaped))

    def split_with_specials(self, text):
        if not self.pattern:
            return [(text, False)]
        parts = []
        last_end = 0
        for match in self.pattern.finditer(text):
            if match.start() > last_end:
                parts.append((text[last_end:match.start()], False))
            parts.append((match.group(), True))
            last_end = match.end()
        if last_end < len(text):
            parts.append((text[last_end:], False))
        return parts
```

### Шаг 5: Full Tokenizer Class

Свяжите все вместе: normalize, split on special tokens, pre-tokenize, BPE merge, map to IDs.

```python
import unicodedata

class ProductionTokenizer:
    def __init__(self):
        self.merges = {}
        self.vocab = {i: bytes([i]) for i in range(256)}
        self.special_handler = SpecialTokenHandler()
        self.next_id = 256

    def normalize(self, text):
        return unicodedata.normalize("NFKC", text)

    def train(self, text, num_merges):
        text = self.normalize(text)
        chunks = pre_tokenize(text)
        chunk_bytes = [list(chunk.encode("utf-8")) for chunk in chunks]

        for i in range(num_merges):
            pairs = Counter()
            for seq in chunk_bytes:
                for j in range(len(seq) - 1):
                    pairs[(seq[j], seq[j + 1])] += 1
            if not pairs:
                break
            best = max(pairs, key=pairs.get)
            new_id = self.next_id
            self.next_id += 1
            self.merges[best] = new_id
            self.vocab[new_id] = self.vocab[best[0]] + self.vocab[best[1]]
            chunk_bytes = [apply_merge(seq, best, new_id) for seq in chunk_bytes]

    def add_special_token(self, token_str):
        token_id = self.next_id
        self.next_id += 1
        self.special_handler.add_token(token_str, token_id)
        self.vocab[token_id] = token_str.encode("utf-8")
        return token_id

    def encode(self, text):
        text = self.normalize(text)
        parts = self.special_handler.split_with_specials(text)
        all_ids = []
        for part_text, is_special in parts:
            if is_special:
                all_ids.append(self.special_handler.special_tokens[part_text])
            else:
                for chunk in pre_tokenize(part_text):
                    byte_seq = list(chunk.encode("utf-8"))
                    for pair, new_id in self.merges.items():
                        byte_seq = apply_merge(byte_seq, pair, new_id)
                    all_ids.extend(byte_seq)
        return all_ids

    def decode(self, ids):
        byte_parts = []
        for token_id in ids:
            if token_id in self.vocab:
                byte_parts.append(self.vocab[token_id])
        return b"".join(byte_parts).decode("utf-8", errors="replace")

    def vocab_size(self):
        return len(self.vocab)
```

### Шаг 6: Multilingual Test

Настоящий тест. Дайте ему English, Chinese, emoji и code.

```python
corpus = (
    "The quick brown fox jumps over the lazy dog. "
    "The quick brown fox runs through the forest. "
    "Machine learning models process natural language. "
    "Deep learning transforms how we build software. "
    "def train(model, data): return model.fit(data) "
    "def predict(model, x): return model(x) "
)

tok = ProductionTokenizer()
tok.train(corpus, num_merges=50)

bos = tok.add_special_token("<|begin|>")
eos = tok.add_special_token("<|end|>")

test_texts = [
    "The quick brown fox.",
    "你好世界",
    "Hello 🌍 World",
    "def foo(x): return x + 1",
    f"<|begin|>Hello<|end|>",
]

for text in test_texts:
    ids = tok.encode(text)
    decoded = tok.decode(ids)
    print(f"Input:   {text}")
    print(f"Tokens:  {len(ids)} ids")
    print(f"Decoded: {decoded}")
    print()
```

Китайские символы дают по 3 bytes каждый. Emoji дает 4 bytes. Ничто из этого не ломает токенизатор. Ничто не создает unknown tokens. В этом сила byte-level BPE.

## Используйте это

### Comparing Real Tokenizers

Загрузите реальные токенизаторы Llama 3, GPT-4 и Mistral. Посмотрите, как каждый обрабатывает один и тот же multilingual paragraph.

```python
import tiktoken

gpt4_enc = tiktoken.get_encoding("cl100k_base")

test_paragraph = "Machine learning is powerful. 机器学习很强大。 L'apprentissage automatique est puissant. 🤖💪"

tokens = gpt4_enc.encode(test_paragraph)
pieces = [gpt4_enc.decode([t]) for t in tokens]
print(f"GPT-4 ({len(tokens)} tokens): {pieces}")
```

```python
from transformers import AutoTokenizer

llama_tok = AutoTokenizer.from_pretrained("meta-llama/Meta-Llama-3-8B")
mistral_tok = AutoTokenizer.from_pretrained("mistralai/Mistral-7B-v0.1")

for name, tok in [("Llama 3", llama_tok), ("Mistral", mistral_tok)]:
    tokens = tok.encode(test_paragraph)
    pieces = tok.convert_ids_to_tokens(tokens)
    print(f"{name} ({len(tokens)} tokens): {pieces[:20]}...")
```

Вы увидите разные token counts для одного и того же текста. Llama 3 со словарем 128K агрессивнее сливает частые паттерны. GPT-4 со 100K находится посередине. Mistral с 32K производит больше токенов, но имеет меньший embedding layer.

Компромисс всегда один: больший словарь означает более короткие последовательности, но больше параметров.

## Результат

Этот урок создает prompt для построения и debugging production tokenizers. См. `outputs/prompt-tokenizer-builder.md`.

## Упражнения

1. **Easy:** Добавьте метод `get_token_bytes(id)`, который показывает raw bytes для любого token ID. Используйте его, чтобы проверить, что на самом деле представляют ваши самые частые merged tokens.
2. **Medium:** Реализуйте Llama-style pre-tokenizer, который делит по whitespace и digits, но сохраняет leading spaces. Сравните его vocabulary с подходом GPT-2 regex на одном и том же корпусе.
3. **Hard:** Добавьте метод chat template, который принимает список сообщений `{"role": ..., "content": ...}` и создает правильную token sequence для формата Llama 3 chat. Проверьте его относительно HuggingFace implementation.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Byte-level BPE | "Токенизатор, который работает с байтами" | BPE с базовым словарем из 256 byte values: обрабатывает любой input без unknown tokens |
| Pre-tokenization | "Разбиение перед BPE" | Regex- или rule-based разбиение, которое мешает BPE сливать через word boundaries |
| NFKC normalization | "Unicode cleanup" | Canonical decomposition с последующей compatibility composition: лигатура "fi" становится "fi", fullwidth "A" становится "A" |
| Chat template | "Как messages становятся tokens" | Точный формат преобразования списка role/content messages в flat token sequence: model-specific и должен совпадать с training format |
| Special tokens | "Control tokens" | Зарезервированные token IDs, обходящие BPE: [BOS], [EOS], [PAD], chat markers; сопоставляются точно перед merge |
| Fertility | "Tokens per word" | Отношение output tokens к input words: 1.3 для английского в GPT-4, 2-3 для корейского; больше значит wasted context |
| tiktoken | "OpenAI tokenizer" | Rust BPE implementation с Python bindings: в 10-100x быстрее pure Python |
| Merge table | "The vocabulary" | Упорядоченный список byte-pair merges, выученных при обучении: это и есть выученное знание токенизатора |

## Дополнительное чтение

- [OpenAI tiktoken source](https://github.com/openai/tiktoken) -- Rust BPE implementation, используемая GPT-3.5/4
- [HuggingFace tokenizers](https://github.com/huggingface/tokenizers) -- Rust tokenizer library с поддержкой BPE, WordPiece, Unigram
- [Llama 3 paper (Meta, 2024)](https://arxiv.org/abs/2407.21783) -- детали о 128K vocabulary и tokenizer training
- [SentencePiece (Kudo & Richardson, 2018)](https://arxiv.org/abs/1808.06226) -- language-agnostic tokenization
- [GPT-2 tokenizer source](https://github.com/openai/gpt-2/blob/master/src/encoder.py) -- original byte-to-Unicode mapping
