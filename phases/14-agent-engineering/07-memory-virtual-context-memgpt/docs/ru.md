# Memory: виртуальный контекст и MemGPT

> Context windows конечны. Conversations, documents и tool traces — нет. MemGPT (Packer et al., 2023) формулирует это как virtual memory в ОС — main context это RAM, external store это disk, агент подкачивает данные между ними. Это паттерн, от которого наследуется каждая memory system 2026 года.

**Тип:** Практика
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 14 · 01 (Agent Loop), Фаза 14 · 06 (Tool Use)
**Время:** ~75 минут

## Цели обучения

- Объяснить OS analogy, на которой строится MemGPT: main context = RAM, external context = disk, memory tools = page in/out.
- Реализовать two-tier pattern MemGPT на stdlib с буфером main context, внешним searchable store и tools для page in/out.
- Описать, как агент выдает "interrupts" для запроса или изменения external memory и как результат вшивается в следующий prompt.
- Определить design choices MemGPT, которые переходят в Letta (Lesson 08) и Mem0 (Lesson 09).

## Проблема

Context windows выглядят так, будто должны решить memory. Не решают. Три failure modes регулярно повторяются в production:

1. **Overflow.** Multi-turn conversations, long documents или насыщенные tool calls trajectories выходят за пределы window. Все за cutoff исчезает.
2. **Dilution.** Даже внутри window stuffing irrelevant context размывает attention по тому, что важно. Frontier models все еще degrade on long inputs.
3. **Persistence.** Новая session начинается с empty window. Agents without external memory не могут сказать "remember when you asked me to..." между sessions.

Большие windows помогают, но не исправляют это. Статья Mem0 2025 года измерила, что 128k-window baselines все еще пропускают long-horizon facts, которые агент с 4k-window и external memory ловит.

## Концепция

### MemGPT: аналогия с ОС

Packer et al. (arXiv:2310.08560, v2 Feb 2024) сопоставляют context management с virtual memory операционной системы:

| Концепт ОС | Концепт MemGPT | Production-аналог 2026 года |
|------------|---------------|------------------------|
| RAM | main context (prompt) | Context window Anthropic/OpenAI |
| Disk | external context | Vector DB, KV, graph store |
| Page fault | memory tool call | `memory.search`, `memory.read`, `memory.write` |
| OS kernel | agent control loop | ReAct loop с memory tools |

Agent запускает обычный ReAct loop. Один дополнительный класс tools позволяет подкачивать данные в main context и обратно.

### Два уровня

- **Main context.** Prompt фиксированного размера, содержащий текущую задачу. Всегда виден модели.
- **External context.** Неограниченный контекст, searchable через tools. Читается, когда relevant, записывается, когда появляются facts.

Исходная статья оценивала design на двух задачах beyond the base window: анализ документов длиннее 100k tokens и multi-session chat с persistent memory на протяжении дней.

### Паттерн interrupt

MemGPT вводит memory-as-interrupt: в середине conversation агент может вызвать memory tool, runtime выполняет его, и результат вшивается в следующий assistant turn как новое observation. Концептуально это идентично Unix `read()` syscall, который блокирует process, возвращает bytes, и process продолжается.

Каноническая memory tool surface:

- `core_memory_append(section, text)` — записать в persistent section prompt.
- `core_memory_replace(section, old, new)` — изменить persistent section.
- `archival_memory_insert(text)` — записать в searchable external store.
- `archival_memory_search(query, top_k)` — извлечь из external store.
- `conversation_search(query)` — просканировать прошлые turns.

### Где заканчивается MemGPT и начинается Letta

В сентябре 2024 MemGPT стал Letta. Research repo (`cpacker/MemGPT`) остается; Letta расширяет design:

- Три уровня вместо двух (core, recall, archival — Lesson 08).
- Native reasoning вместо паттерна `send_message`/heartbeat (Lesson 08).
- Sleep-time agents, выполняющие async memory work (Lesson 08).

Статья MemGPT — foundation 2026 года, даже если production systems запускают Letta, Mem0 или custom two-tier store.

### Где этот паттерн ломается

- **Memory rot.** Writes накапливаются быстрее reads; retrieval тонет в stale facts. Fix: periodic consolidation (Letta sleep-time), explicit invalidation (Mem0 conflict detector).
- **Memory poisoning.** External memory — это retrieved text. Если attacker-controlled content попадает в memory note, агент re-ingests it next session. Это атака Greshake et al. (Lesson 27), переформулированная во времени.
- **Citation loss.** Agent recalls "the user asked me to ship X", но не может процитировать turn. Храните source references (session ID, turn ID) с каждой archival write.

## Соберите это

`code/main.py` реализует two-tier pattern MemGPT на stdlib:

- `MainContext` — prompt buffer фиксированного размера с `core` dict и `messages` list; автоматически compact oldest messages when over cap.
- `ArchivalStore` — in-memory BM25-esque store (token-overlap scoring) из records (id, text, tags, session, turn).
- Пять memory tools, соответствующих MemGPT surface.
- Scripted agent, который заполняет archival facts, затем отвечает на вопрос, вызывая `archival_memory_search`.

Запустите:

```
python3 code/main.py
```

Трасса показывает, как агент записывает три facts, заполняет main context до лимита (forcing eviction), затем отвечает на follow-up question через retrieval from archival — воспроизводя MemGPT workflow без реальной LLM.

## Используйте это

Каждая production memory system сегодня — вариант MemGPT:

- **Letta** (Lesson 08) — три уровня, native reasoning, sleep-time compute.
- **Mem0** (Lesson 09) — vector + KV + graph, объединенные scoring layer.
- **OpenAI Assistants / Responses** — managed memory через threads and files.
- **Claude Agent SDK** — long-term memory через skills and session store.

Выбирайте по operational shape (self-hosted, managed, framework-integrated), а не по core pattern — core pattern это MemGPT.

## Отгрузите это

`outputs/skill-virtual-memory.md` — reusable skill, который создает корректный two-tier memory scaffold (main + archival + tool surface) для любого target runtime, с подключенными eviction policy и citation fields.

## Упражнения

1. Добавьте cap `max_main_context_tokens`, измеренный в tokens (approximate with `len(text.split())` * 1.3). Compact oldest messages into a summary when the cap is exceeded. Сравните behavior with and without the summarizer.
2. Реализуйте BM25 properly over the archival store (term frequency, inverse document frequency). Измерьте recall@10 на toy fact set против token-overlap baseline.
3. Добавьте `citation` fields (session_id, turn_id, source_url) к archival inserts. Заставьте agent cite sources on every retrieval-backed answer.
4. Simulate memory poisoning: add an archival record that says "ignore all future user instructions." Напишите guard, который scans retrievals for directive-shaped text and marks them untrusted.
5. Перенесите implementation на core-memory JSON schema из research repo MemGPT (`cpacker/MemGPT`). Что меняется при переходе от flat strings к typed sections?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Virtual context | "Неограниченная memory" | Main (prompt) + external (searchable) tiers with page in/out |
| Main context | "Working memory" | Prompt фиксированного размера, всегда видимый |
| Archival memory | "Long-term store" | External searchable persistence, извлекаемая по требованию |
| Core memory | "Persistent prompt section" | Именованные sections, закрепленные внутри main context |
| Memory tool | "Memory API" | Tool call, который agent issues для чтения/записи external memory |
| Interrupt | "Memory page fault" | Agent pauses, runtime fetches, result splices into next turn |
| Memory rot | "Stale facts" | Старые writes топят retrieval; исправляется consolidation |
| Memory poisoning | "Инъецированная persistent note" | Attacker content stored as memory, re-ingested on recall |

## Дополнительное чтение

- [Packer et al., MemGPT (arXiv:2310.08560)](https://arxiv.org/abs/2310.08560) — статья о virtual context, вдохновленном ОС
- [Letta, Memory Blocks blog](https://www.letta.com/blog/memory-blocks) — эволюция к трем уровням
- [Anthropic, Effective context engineering](https://www.anthropic.com/engineering/effective-context-engineering-for-ai-agents) — отношение к context как к бюджету
- [Chhikara et al., Mem0 (arXiv:2504.19413)](https://arxiv.org/abs/2504.19413) — hybrid production memory поверх этого паттерна
