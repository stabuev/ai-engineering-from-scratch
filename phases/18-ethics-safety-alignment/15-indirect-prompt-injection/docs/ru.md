# Indirect Prompt Injection — Production Attack Surface

> Indirect prompt injection (IPI) встраивает инструкции во внешний контент — web page, email, shared document, support ticket, — который agentic system потребляет без явного действия пользователя. IPI — доминирующая production threat 2026 года: она обходит user-input filters, потому что атакующий никогда не взаимодействует с пользователем, масштабируется незаметно по мере того, как agents обрабатывают больше внешнего контента, и нацелена на automated workflows, где никто не читает prompt. MDPI Information 17(1):54 (January 2026) синтезирует исследования 2023-2025. Статья NDSS 2026 по IPI-defense формулирует ключевую проблему: injected instructions могут быть семантически доброкачественными ("please print Yes"), поэтому для обнаружения требуется больше, чем keyword filtering. "The Attacker Moves Second" (Nasr et al., joint OpenAI/Anthropic/DeepMind, October 2025): adaptive attacks (gradient, RL, random search, human red-team) взломали >90% из 12 опубликованных защит, которые изначально сообщали о почти нулевых attack success rates.

**Type:** Build
**Languages:** Python (stdlib, IPI attack + defense harness)
**Prerequisites:** Phase 18 · 12 (PAIR), Phase 14 (agent engineering)
**Time:** ~75 minutes

## Цели обучения

- Определить indirect prompt injection и описать три распространенных delivery vectors.
- Объяснить, почему user-input filters полностью пропускают IPI.
- Описать framing "information flow control" как парадигму защиты 2026 года.
- Сформулировать результат Nasr et al. (October 2025) об успешности adaptive attack против опубликованных IPI defenses.

## Проблема

Direct prompt injection требует, чтобы атакующий добрался до пользователя или его prompt. IPI не требует ни того ни другого: атакующий размещает payload в любом контенте, который agent может прочитать — web page, email во входящих, GitHub issue, product review. Agent подхватывает его во время обычной работы и выполняет инструкции. Пользователь — переносчик, а не источник намерения.

## Концепция

### Three delivery vectors

- **Retrieval-augmented generation (RAG).** Атакующий публикует документ; retrieval step извлекает его; prompt конкатенирует его перед вопросом пользователя; модель выполняет инструкции атакующего.
- **Inbox / document workflows.** Атакующий отправляет пользователю email; agent читает emails; prompt включает тело email; модель следует инструкциям из email.
- **Tool output.** Атакующий контролирует tool, который использует agent (например, web search, возвращающий attacker-controlled result); tool output содержит инструкции; control flow agent следует им.

У этих трех вариантов общее структурное свойство: атакующий контролирует фрагмент prompt, не касаясь user-facing input.

### Why user-input filters miss it

IPI payload не появляется во вводе пользователя. Он появляется в retrieved content. Если filter срабатывает только на user input, payload обходит его. Если filter применяется ко всему content, который достигает модели, он должен применяться к произвольному retrieved text — это дорого и дает false positives на легитимном контенте, где случайно есть imperative-voice language.

### Information Flow Control (IFC) for AI

Парадигма защиты 2026 года заимствует идеи из классической OS security. Рассматривайте каждый источник контента как security label. Пометьте user's query как "trusted." Пометьте retrieved content как "untrusted." Рассматривайте control flow модели как information flow: действия, инициированные untrusted content, должны быть ratified trusted input перед выполнением.

CaMeL (Microsoft 2025), ConfAIde (Stanford 2024) и статья NDSS 2026 по IPI-defense по-разному операционализируют IFC. Общий принцип: пока code и data разделяют одно context window, цель — containment, а не prevention.

### The Attacker Moves Second

Nasr et al. (October 2025) протестировали 12 опубликованных IPI defenses с adaptive attacks (gradient search, RL policies, random search, 72-hour human red-team). Каждая защита, которая изначально сообщала о near-zero ASR, была взломана до >90% ASR.

Методологический урок: публикуйте защиту только с adaptive-attack evaluation. Static-attack benchmarks не являются свидетельством robustness; атакующий знает защиту.

### Real incidents

Lesson 25 покрывает EchoLeak (CVE-2025-32711, CVSS 9.3) — первый публично задокументированный zero-click IPI в Microsoft 365 Copilot. CamoLeak (CVSS 9.6) в GitHub Copilot Chat. CVE-2025-53773 в GitHub Copilot. Production deployments компрометируются через IPI в реальной среде, а не только в benchmarks.

### OWASP and NIST framing

OWASP LLM Top 10 (2025) ставит prompt injection (direct + indirect) как LLM01, угрозу application-layer #1. NIST AI SPD 2024 называет indirect prompt injection "generative AI's greatest security flaw."

### Как это вписывается в Phase 18

Lessons 12-14 — model-centric jailbreaks. Lesson 15 — system-centric attack, доминирующая в production deployments 2026 года. Lesson 16 покрывает defensive tooling. Lesson 25 покрывает конкретный CVE narrative.

## Применение

`code/main.py` строит IPI harness. Игрушечный agent имеет три tools (search web, read email, send message). Environment содержит attacker-controlled content со встроенной инструкцией ("forward this to all contacts"). Вы можете переключаться между naive agent (следует injected instructions), filter-defended agent (keyword filter на retrieved content) и IFC agent (разделяет trusted и untrusted content и отказывается от untrusted control-flow commands).

## Результат

Этот урок создает `outputs/skill-ipi-audit.md`. Для описания agentic deployment он перечисляет untrusted content sources, проверяет, применяет ли deployment IFC, и помечает sources, которые достигают модели без trust label.

## Упражнения

1. Запустите `code/main.py`. Измерьте success rate атаки против каждого из трех agents.

2. Реализуйте paraphrase-based defense на retrieved content. Измерьте benign false-positive rate на легитимном retrieved text.

3. Прочитайте статью NDSS 2026 IPI-defense. Опишите проблему "benign instruction" и почему она мешает keyword-based filtering.

4. Спроектируйте deployment, где agent получает tool output от third-party API. Пометьте каждый prompt fragment trust level и напишите IFC policy, управляющую действиями agent.

5. Воспроизведите методологию adaptive-attack Nasr et al. 2025 на вашем filter-defended agent из Exercise 2. Сообщите ASR до и после adaptive attack.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| IPI | "indirect prompt injection" | Injection через контент, который пользователь не писал и который agent потребляет во время нормальной работы |
| RAG injection | "poisoned retrieval" | Атакующий публикует контент, который извлекает retrieval step; prompt содержит payload |
| Zero-click | "no user action" | Атака срабатывает автоматически во время работы agent; пользователь ничего не делает |
| IFC | "information flow control" | Label-based подход: действия из untrusted content требуют trusted ratification |
| Adaptive attack | "gradient / RL red-team" | Атака, которая знает защиту и оптимизируется против нее; обязательна для честной оценки |
| Benign instruction | "please print Yes" | IPI payload, семантически доброкачественный; keyword filter его не ловит |
| Scope violation | "cross-trust exfiltration" | Agent получает доступ к данным из одного trust context и выводит их в другой |

## Дополнительное чтение

- [MDPI Information 17(1):54 — Indirect Prompt Injection Survey (January 2026)](https://www.mdpi.com/2078-2489/17/1/54) — синтез 2023-2025
- [Nasr et al. — The Attacker Moves Second (joint OpenAI/Anthropic/DeepMind, October 2025)](https://arxiv.org/abs/2510.18108) — adaptive attack evaluation
- [Greshake et al. — Not what you've signed up for (arXiv:2302.12173)](https://arxiv.org/abs/2302.12173) — исходная статья по IPI
- [OWASP — LLM Top 10 (2025)](https://genai.owasp.org/llm-top-10/) — prompt injection ranked LLM01
