# Наследие FIPA-ACL и речевых актов

> До MCP, до A2A был FIPA-ACL. В 2000 году IEEE Foundation for Intelligent Physical Agents ратифицировала язык коммуникации агентов с двадцатью performatives, двумя content languages и набором interaction protocols — contract net, subscribe/notify, request-when. Он исчез из индустрии, потому что overhead онтологий был слишком тяжелым для web, но LLM-возрождение multi-agent систем тихо переизобретает те же идеи без формальной семантики: JSON contracts заменяют performatives, natural language заменяет ontologies. Этот урок серьезно читает FIPA-ACL, чтобы вы видели, какие protocol decisions 2026 года являются переизобретением, какие — новизной, и где текущая волна заново обнаружит проблемы, уже решенные в 2000-х.

**Тип:** Изучение
**Языки:** Python (stdlib)
**Предварительные требования:** Фаза 16 · 01 (Why Multi-Agent)
**Время:** ~60 минут

## Цели обучения

- Объяснять речевые акты и перформативы FIPA-ACL, предшествовавшие MCP и A2A.
- Читать каноническое сообщение FIPA-ACL.
- Объяснять, почему FIPA угас и что современные протоколы из него взяли.

## Проблема

Ландшафт agent protocols в 2026 году перегружен: MCP для tools, A2A для agents, ACP для enterprise audit, ANP для decentralized trust, NLIP для natural-language content, плюс CA-MCP и две дюжины research proposals. Каждая спецификация объявляет себя фундаментальной.

Честное прочтение: большинство из них заново открывают очень конкретное двадцатилетнее дерево решений. Speech-act theory Остина (1962) и Серля (1969) дала нам идею "utterances are actions". KQML (1993) превратил это в wire protocol. FIPA-ACL (ратифицирован в 2000) дал эталонную стандартизацию: двадцать performatives, content languages SL0/SL1, interaction protocols для contract-net и subscribe-notify. JADE и JACK были Java reference platforms. Усилия сошли на нет примерно к 2010 году, потому что overhead онтологий был слишком тяжелым, а web побеждал.

Когда вы смотрите на MCP `tools/call`, A2A task lifecycle или CA-MCP shared context store, вы смотрите на более мягкий, JSON-native пересказ решений FIPA. Знание наследия дает две вещи: какие новые "innovations" на самом деле являются reinventions, и какие старые failure modes новые specs обнаружат снова.

## Концепция

### Speech acts в одном абзаце

Остин заметил, что некоторые предложения не описывают мир — они меняют его. "I promise." "I request." "I declare." Он назвал их performative utterances. Серль формализовал пять категорий: assertive, directive, commissive, expressive, declarative. KQML (Finin et al., 1993) сделал это операциональным для software agents: message — это performative (действие) плюс content (то, о чем действие). FIPA-ACL закрыл пробелы KQML и стандартизовался вокруг двадцати performatives.

### Двадцать FIPA performatives (частичный список)

| Performative | Intent |
|---|---|
| `inform` | "I tell you P is true" |
| `request` | "I ask you to do X" |
| `query-if` | "Is P true?" |
| `query-ref` | "What is the value of X?" |
| `propose` | "I propose we do X" |
| `accept-proposal` | "I accept the proposal" |
| `reject-proposal` | "I reject the proposal" |
| `agree` | "I agree to do X" |
| `refuse` | "I refuse to do X" |
| `confirm` | "I confirm P is true" |
| `disconfirm` | "I deny P" |
| `not-understood` | "Your message did not parse" |
| `cfp` | "Call for proposals on X" |
| `subscribe` | "Notify me when X changes" |
| `cancel` | "Cancel the ongoing X" |
| `failure` | "I tried X and failed" |

Полный список находится в `fipa00037.pdf` (FIPA ACL Message Structure). Смысл не в том, чтобы заучить его, а в том, что каждый элемент соответствует примитиву, который LLM protocol в итоге добавляет снова.

### Каноническое сообщение FIPA-ACL

```
(inform
  :sender       agent1@platform
  :receiver     agent2@platform
  :content      "((price IBM 83))"
  :language     SL0
  :ontology     finance
  :protocol     fipa-request
  :conversation-id   conv-42
  :reply-with   msg-17
)
```

Семь полей несут protocol envelope; одно поле (`content`) несет payload. Остальные поля — ровно то, что вы заново изобретаете каждый раз, когда прикручиваете retries, threading и ontology к JSON protocol.

### Две legacy platforms

**JADE** (Java Agent DEvelopment framework, 1999–2020s) был самым используемым FIPA-compliant runtime. Агенты расширяли base class, обменивались ACL messages, работали внутри containers и координировались с помощью "behaviors". Библиотека interaction-protocol поставляла contract-net, subscribe-notify, request-when и propose-accept.

**JACK** (Agent Oriented Software, коммерческий) делал акцент на BDI (Belief-Desire-Intention) reasoning поверх FIPA messages. Более формальный, менее распространенный.

Оба пришли в упадок, когда web stack поглотил multi-agent use cases. MCP и A2A — runtime "containers" 2026 года.

### Почему FIPA исчез

- **Ontology overhead.** FIPA требовал shared ontology для parsing `content`. Согласование ontologies — процесс стандартизации длиной в годы. Web просто использовал HTTP + JSON.
- **Formal semantics nobody used.** SL (Semantic Language) давал строгие truth conditions, но большинство production systems использовали free-form content и игнорировали formalism.
- **Tooling lock-in.** JADE был Java-only; JACK был коммерческим. Polyglot teams обходили оба.
- **The internet won the stack.** REST, затем JSON-RPC, затем gRPC заменили transport ACL.

### LLM-возрождение — это FIPA-lite

Сравните FIPA `request` с MCP `tools/call`:

```
(request                                {
  :sender  agent1                         "jsonrpc": "2.0",
  :receiver tool-server                   "method":  "tools/call",
  :content "(lookup stock IBM)"           "params":  {"name":"lookup_stock",
  :ontology finance                                   "arguments":{"symbol":"IBM"}},
  :conversation-id c42                    "id": 42
)                                        }
```

Тот же envelope, другой syntax. Оба несут: who, whom, intent, payload, correlation id. Ни один не является революцией относительно другого — это разные trade-offs в одном design.

Survey 2025 года Liu et al. ("A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP", arXiv:2505.02279) явно показывает эту lineage: MCP соответствует tool-use speech acts, A2A — agent-peer speech acts, ACP — audit-trail speech acts, ANP — decentralized-identity extensions. Новые specs — потомки ACL с JSON syntax и более свободной semantics.

### Trade-off, прямо

**Что FIPA давал, а современные specs отбрасывают:**

- Formal semantics — можно доказать, что `inform` подразумевает, что sender believes content.
- Canonical catalog of performatives — не нужно заново спорить "should we have a `cancel`?".
- Десятилетия interaction-protocol patterns — contract-net, subscribe-notify, propose-accept — с известными correctness properties.

**Что современные specs дают, а FIPA не давал:**

- JSON-native payloads, совместимые с каждым современным tool.
- Natural-language content, который LLMs могут интерпретировать без hand-coded ontology.
- Web-stack transport (HTTP, SSE, WebSocket).
- Capability discovery через self-describing documents (MCP `listTools`, A2A Agent Card).

Более слабая semantics intent ради более простой implementation. Это ровно тот trade.

### Interaction protocols, которые стоит перенести

FIPA поставлял примерно 15 interaction protocols. Три стоит перенести в LLM multi-agent systems:

1. **Contract Net Protocol (CNP).** Manager отправляет `cfp` (call for proposals); bidders отвечают `propose`; manager accepts/rejects. Это канонический task-market pattern (Phase 16 · 16 Negotiation).
2. **Subscribe/Notify.** Subscriber отправляет `subscribe`; publisher отправляет `inform` всякий раз, когда topic меняется. Это каждый event-bus в 2026.
3. **Request-When.** "Do X when condition Y holds." Delayed-action с pre-conditions. Аналог 2026 года — deferred tasks в durable workflow engines (Phase 16 · 22 Production Scaling).

Каждый чисто ложится на современные message queues, HTTP + polling или SSE streaming.

### Что ломается, когда вы убираете ontology

Без shared ontology agents выводят meaning из natural-language content. Документированный failure mode 2026 года — **semantic drift**: два agents используют одно слово (`"customer"`) для слегка разных concepts, receiver's agent действует по неправильной interpretation, и никакой schema validator это не ловит. Требование ontology в FIPA отклонило бы message на parse time.

Mitigations без полного перехода в ontology:

- JSON Schema на `content` — отклоняет structural errors на wire.
- Typed artifacts (A2A) — отклоняет wrong modality.
- Explicit performative в envelope — делает intent однозначным, даже когда content на natural language.

### Specs 2026, сопоставленные с speech-act heritage

| Modern spec | FIPA analog | Что сохраняет | Что отбрасывает |
|---|---|---|---|
| MCP `tools/call` | `request` | explicit intent, correlation id | formal semantics, ontology |
| MCP `resources/read` | `query-ref` | explicit intent, correlation id | formal semantics |
| A2A Task lifecycle | contract-net + request-when | async lifecycle, state transitions | formal completeness guarantees |
| A2A streaming events | subscribe/notify | async push | typed-predicate subscription |
| CA-MCP shared context | blackboard (Hayes-Roth 1985) | multi-writer shared memory | logical consistency model |
| NLIP | natural-language content | LLM-native | schema |

Если читать таблицу сверху вниз, pattern такой: сохранить structural primitive, отбросить formalism, позволить LLMs сгладить ambiguity.

## Соберите это

`code/main.py` реализует pure-stdlib FIPA-ACL translator. Он encodes и decodes canonical ACL envelope и показывает, как любая MCP / A2A message shape сводится к тем же семи полям. Demo:

- Encodes пять MCP-style и A2A-style messages как FIPA-ACL.
- Decodes FIPA-ACL обратно в modern equivalent.
- Запускает toy Contract Net negotiation между одним manager и тремя bidders с помощью `cfp`, `propose`, `accept-proposal`, `reject-proposal`.

Запустите:

```
python3 code/main.py
```

Вывод — side-by-side trace, показывающий каждое modern message и в его JSON form 2026 года, и в FIPA-ACL form, затем round-trip contract-net bid. Одни и те же protocol primitives переживают round-trip; отличается только syntax.

## Используйте это

`outputs/skill-fipa-mapper.md` — это skill, который читает любую agent-protocol spec и производит FIPA-ACL mapping. Используйте его перед adoption нового protocol, чтобы ответить: "Это действительно новое или это `inform` с JSON syntax?"

## Доведите до production

Не возвращайте FIPA-ACL. Верните его checklist:

- Какой intent primitive (performative) у каждого message?
- Есть ли correlation id для request-response и cancellation?
- Есть ли explicit content language (JSON-RPC, plain text, structured typed artifact)?
- Interaction protocols first-class, или вы re-implementing contract-net from scratch?
- Что происходит, когда два agents disagree about content meaning (semantic drift)?

Задокументируйте эти пять вопросов для любого нового protocol до отправки в production.

## Упражнения

1. Запустите `code/main.py`. Посмотрите round-trip encoding. Определите, какой FIPA performative соответствует `tools/call`, `resources/read` и A2A task creation.
2. Расширьте contract-net demo с performative `cancel`, который позволяет manager отозвать task в середине bidding. Какой failure case решает `cancel`, который retries alone не решают?
3. Прочитайте FIPA ACL Message Structure (http://www.fipa.org/specs/fipa00037/) sections 4.1–4.3. Выберите один performative, не покрытый в этом уроке, и опишите его modern JSON-RPC analog.
4. Прочитайте Liu et al., arXiv:2505.02279. Для каждого из MCP, A2A, ACP, ANP перечислите семьи FIPA performative, которые они keep и drop.
5. Спроектируйте minimal JSON-Schema для поля `content` performative `request` в вашей системе. Что эта schema дает по сравнению с pure natural-language и сколько это стоит?

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Speech act | "An utterance that does something" | Austin/Searle: utterances as actions. Теоретический предок ACL. |
| FIPA | "That old XML thing" | IEEE Foundation for Intelligent Physical Agents. Standardized ACL in 2000. |
| ACL | "Agent Communication Language" | Envelope format FIPA: performative + content + metadata. |
| Performative | "The verb" | Класс intent message: `inform`, `request`, `propose`, `cfp`, etc. |
| KQML | "FIPA's predecessor" | Knowledge Query and Manipulation Language (1993). Проще и уже. |
| Ontology | "Shared vocabulary" | Formal definition concepts, о которых говорит content language. |
| SL0 / SL1 | "FIPA content languages" | Semantic Language levels 0 and 1 — семейство formal content language. |
| Contract Net | "Task market" | Manager отправляет cfp; bidders propose; manager accepts. Канонический interaction protocol. |
| Interaction protocol | "Pattern of messages" | Последовательность performatives с известной correctness: request-when, subscribe-notify, etc. |

## Дополнительное чтение

- [Liu et al. — A Survey of Agent Interoperability Protocols: MCP, ACP, A2A, ANP](https://arxiv.org/html/2505.02279v1) — canonical 2025 survey, связывающий modern specs с FIPA heritage
- [FIPA ACL Message Structure Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — ratified 2000 envelope format
- [FIPA Communicative Act Library Specification (fipa00037)](http://www.fipa.org/specs/fipa00037/) — полный performative catalog
- [MCP specification 2025-11-25](https://modelcontextprotocol.io/specification/2025-11-25) — modern tool-use equivalent of `request`/`query-ref`
- [A2A specification](https://a2a-protocol.org/latest/specification/) — modern agent-peer equivalent of contract-net and subscribe-notify
