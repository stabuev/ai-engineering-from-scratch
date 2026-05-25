# ASCII Art and Visual Jailbreaks

> Jiang, Xu, Niu, Xiang, Ramasubramanian, Li, Poovendran, "ArtPrompt: ASCII Art-based Jailbreak Attacks against Aligned LLMs" (ACL 2024, arXiv:2402.11753). Маскируют safety-relevant tokens во вредоносном запросе, заменяют их ASCII-art отображениями тех же букв и отправляют замаскированный prompt. GPT-3.5, GPT-4, Gemini, Claude, Llama-2 не умеют надежно распознавать ASCII-art tokens. Атака обходит PPL (perplexity filters), Paraphrase defenses и Retokenization. Связано: benchmark ViTC измеряет распознавание non-semantic visual prompts; StructuralSleight обобщает это на Uncommon Text-Encoded Structures (trees, graphs, nested JSON) как семейство encoding attacks.

**Type:** Build
**Languages:** Python (stdlib, ArtPrompt token-masking harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 18 · 13 (MSJ)
**Time:** ~60 minutes

## Цели обучения

- Описать атаку ArtPrompt: этап word-identification, ASCII-art substitution, итоговый cloaked prompt.
- Объяснить, почему стандартные защиты (PPL, Paraphrase, Retokenization) проваливаются на ArtPrompt.
- Определить ViTC и описать, что он измеряет.
- Описать StructuralSleight как обобщение на произвольные Uncommon Text-Encoded Structures.

## Проблема

Атаки через paraphrase и roleplay (Lesson 12) и через long context (Lesson 13) работают на уровне текстового шаблона. ArtPrompt работает на уровне распознавания: модель не парсит запрещенный token. Она парсит изображение, отрисованное символами. Safety filter видит безвредную пунктуацию. Модель видит слово.

## Концепция

### ArtPrompt, two steps

Step 1. Word Identification. По вредоносному запросу атакующий использует LLM, чтобы определить safety-relevant words (например, "bomb" в "how to make a bomb"). 

Step 2. Cloaked Prompt Generation. Замените каждое найденное слово его ASCII-art отображением (блок символов 7x5 или 7x7, формирующий форму буквы). Модель получает сетку из пунктуации и пробелов, которую достаточно сильная модель может распознать как слово; safety filter видит только сетку.

Результат: GPT-4, Gemini, Claude, Llama-2, GPT-3.5 все проваливаются. Attack success rate выше 75% на их benchmark subset.

### Why the standard defenses fail

- **PPL (perplexity filter).** ASCII art имеет высокую perplexity — но то же верно для любого нового input. Пороги, которые блокируют ArtPrompt, также блокируют легитимный structured input.
- **Paraphrase.** Перефразирование prompt разрушает ASCII art. На практике paraphrase LLM часто сохраняют или восстанавливают art.
- **Retokenization.** Иное разбиение на tokens не меняет того, что "зрение" модели распознает формы букв.

Базовая проблема в том, что safety filters работают на token- или semantic-level; ArtPrompt действует на visual recognition level.

### ViTC benchmark

Распознавание non-semantic visual prompts. Измеряет способность модели читать ASCII-art, wingdings и другой non-text-semantic visual content. Эффективность ArtPrompt коррелирует с ViTC accuracy: чем лучше модель читает визуальный текст, тем лучше ArtPrompt работает против нее. Это capability-safety tradeoff.

### StructuralSleight

Обобщает ArtPrompt: Uncommon Text-Encoded Structures (UTES). Trees, graphs, nested JSON, CSV-in-JSON, diff-style code blocks. Если структура редка в training safety data, но парсится моделью, она может скрывать вредоносный контент.

Следствие для защиты: safety должна обобщаться на структурированные представления, которые модель умеет парсить. Этот набор велик и растет.

### Image-modality analog

Visual LLMs (GPT-5.2, Gemini 3 Pro, Claude Opus 4.5, Grok 4.1) расширяют поверхность атаки. ArtPrompt-style атаки с реальными изображениями сильнее, чем ASCII-art analogs, потому что image encoders создают более богатый сигнал.

### Как это вписывается в Phase 18

Lessons 12-14 описывают три ортогональных вектора атак: iterative refinement (PAIR), context length (MSJ) и encoding (ArtPrompt/StructuralSleight). Lesson 15 переходит от model-centric attacks к system-boundary attacks (indirect prompt injection). Lesson 16 описывает defensive tooling response.

## Применение

`code/main.py` строит игрушечный ArtPrompt. Вы можете замаскировать конкретные слова во вредоносном query с помощью ASCII-art glyphs, проверить, что cloaked string проходит keyword filter, и (опционально) декодировать cloaked string обратно с помощью простого recognizer.

## Результат

Этот урок создает `outputs/skill-encoding-audit.md`. Для jailbreak-defense report он перечисляет покрытые семейства encoding attacks (ASCII art, base64, leet-speak, UTF-8 homoglyph, UTES) и слой защиты, который ловит каждое из них.

## Упражнения

1. Запустите `code/main.py`. Проверьте, что cloaked string проходит простой keyword filter. Сообщите требуемое изменение на character-level.

2. Реализуйте второе кодирование: base64 для того же target word. Сравните filter-bypass rate с ArtPrompt и сложность восстановления.

3. Прочитайте Jiang et al. 2024 Section 4.3 (five-model results). Предложите причину, почему Claude's ArtPrompt-resistance выше, чем Gemini's, на том же benchmark.

4. Спроектируйте pre-generation defense, которая обнаруживает ASCII-art-shaped regions в prompt. Измерьте false-positive rate на легитимном code, tables и mathematical notation.

5. StructuralSleight перечисляет 10 encoding structures. Набросайте generalized defense, которая обрабатывает все 10, и оцените compute cost per defended prompt.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| ArtPrompt | "the ASCII-art attack" | Двухэтапный jailbreak, который маскирует safety words с помощью ASCII-art renderings |
| Cloaking | "hide the word" | Заменить запрещенный token визуальным представлением, которое модель читает, а фильтр нет |
| UTES | "uncommon structure" | Uncommon Text-Encoded Structure — tree, graph, nested JSON и т. д., используемые для скрытой передачи контента |
| ViTC | "visual-text capability" | Benchmark способности модели читать non-semantic visual encoding |
| Perplexity filter | "PPL defense" | Отклоняет prompts с высокой perplexity; проваливается, потому что легитимный structured input тоже получает высокий score |
| Retokenization | "tokenizer shift defense" | Предобрабатывает prompt другим tokenizer; проваливается, потому что распознавание визуальное |
| Homoglyph | "lookalike characters" | Unicode-символы, выглядящие идентично латинским буквам; обходят substring checks |

## Дополнительное чтение

- [Jiang et al. — ArtPrompt (ACL 2024, arXiv:2402.11753)](https://arxiv.org/abs/2402.11753) — статья об ASCII-art jailbreak
- [Li et al. — StructuralSleight (arXiv:2406.08754)](https://arxiv.org/abs/2406.08754) — обобщение UTES
- [Chao et al. — PAIR (Lesson 12, arXiv:2310.08419)](https://arxiv.org/abs/2310.08419) — дополняющая итеративная атака
- [Anil et al. — Many-shot Jailbreaking (Lesson 13)](https://www.anthropic.com/research/many-shot-jailbreaking) — дополняющая атака через длину
