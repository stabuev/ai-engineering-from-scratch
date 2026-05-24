# Case Studies and the 2026 State of the Art

> Три production-grade references для end-to-end изучения, каждая иллюстрирует отдельный срез multi-agent engineering. **Anthropic's Research system** (orchestrator-worker, 15x tokens, +90.2% over single-agent Opus 4, rainbow deployments) — канонический supervisor case. **MetaGPT / ChatDev** (SOP-encoded role specialization для software engineering; "communicative dehallucination" в ChatDev; extension MacNet до >1000 agents via DAGs, arXiv:2406.07155) — канонический role-decomposition case. **OpenClaw / Moltbook** (изначально Clawdbot by Peter Steinberger, ноябрь 2025; переименован дважды; 247k GitHub stars к марту 2026; local ReAct-loop agents; Moltbook как agent-only social network с ~2.3M agent accounts за несколько дней после launch, acquired by Meta 2026-03-10) показывает, что происходит в population scale: emergent economic activity, prompt-injection risks, state-level regulation (China restricted OpenClaw on government computers, March 2026). **Framework landscape April 2026:** LangGraph и CrewAI лидируют в production; AG2 — community AutoGen continuation; Microsoft AutoGen в maintenance mode (merged into Microsoft Agent Framework, RC Feb 2026); OpenAI Agents SDK — production Swarm successor; Google ADK (April 2025) — A2A-native entrant. Каждый major framework теперь поставляет MCP support; большинство поставляет A2A. Этот урок читает каждый case end-to-end и выделяет common patterns, чтобы вы могли выбрать правильный reference для следующей production system.

**Тип:** Изучение (capstone)
**Языки:** —
**Предварительные требования:** all of Phase 16 (Lessons 01-24)
**Время:** ~90 минут

## Проблема

Multi-agent engineering — молодая дисциплина. Production references немного, и каждый покрывает разные части пространства. Читать их по одному полезно; сравнивать их как набор полезнее. Этот урок рассматривает три канонических case studies 2026 года как end-to-end reading list, фиксирует common patterns и mapping framework landscape, чтобы вы принимали framework choices на основе знания, а не marketing.

## Концепция

### Anthropic Research system

Production supervisor-worker case. Claude Opus 4 plans and synthesizes; Claude Sonnet 4 subagents research in parallel. Published engineering post: https://www.anthropic.com/engineering/multi-agent-research-system.

Ключевые measured results:

- **+90.2%** improvement over single-agent Opus 4 on internal research evals.
- **80% of BrowseComp variance** explained by **token usage alone** — multi-agent wins largely because each subagent gets a fresh context window.
- **15x tokens per query** vs single-agent.
- **Rainbow deployment** because agents are long-running and stateful.

Зафиксированные уроки проектирования:

1. **Scale effort to query complexity.** Simple → 1 agent with 3-10 tool calls. Medium → 3 agents. Complex research → 10+ subagents.
2. **Broad first, then narrow.** Subagents делают wide searches; lead synthesizes; follow-up subagents делают targeted deeps.
3. **Rainbow deploys.** Keep old runtime versions alive until their in-flight agents finish.
4. **Verification is not optional.** Было замечено, что system hallucinate без explicit verifier roles.

Это reference case для supervisor-worker topology (Phase 16 · 05) at production scale.

### MetaGPT / ChatDev

Production SOP-role-decomposition case. Cover arXiv:2308.00352 (MetaGPT) and arXiv:2307.07924 (ChatDev).

MetaGPT кодирует software-engineering SOPs как role prompts: Product Manager, Architect, Project Manager, Engineer, QA Engineer. Framing paper: `Code = SOP(Team)`. У каждой role узкий, specialized prompt; inter-role handoffs несут structured artifacts (PRD docs, architecture docs, code).

Вклад ChatDev: **communicative dehallucination**. Агенты запрашивают specifics перед ответом — designer agent спрашивает programmer, какой язык предполагается, перед тем как sketch UI, а не guesses. Paper reports, что это measurably reduces hallucination in multi-agent pipelines.

MacNet (arXiv:2406.07155) extends ChatDev to **>1000 agents via DAGs**. Каждый DAG node — role specialization; edges encode handoff contracts. Scale возможен, потому что routing explicit and offline-computable.

Уроки проектирования:

1. **Structure matters more than size.** Tight 5-role SOP team превосходит 50-agent unstructured group.
2. **Handoff contracts in writing.** Artifacts, передаваемые между roles, follow a schema.
3. **Communicative dehallucination** — дешевый, load-bearing pattern.
4. **DAGs scale further than chat.** Когда flow knowable, encode it.

Это reference case для role specialization (Phase 16 · 08) и structured topology (Phase 16 · 15).

### OpenClaw / Moltbook ecosystem

Production population-scale case. Timeline:

- **Nov 2025:** Clawdbot (local ReAct-loop coding agent Peter Steinberger) ships.
- **Dec 2025 – Mar 2026:** renamed twice (Clawdbot → OpenClaw → continued under OpenClaw).
- **Feb 2026:** Moltbook launches as an agent-only social network on the same primitives; ~2.3M agent accounts within days.
- **Mar 2026 (2026-03-10):** Meta acquires Moltbook.
- **Mar 2026:** China restricts OpenClaw on government computers.
- **Mar 2026:** OpenClaw crosses 247k GitHub stars.

Вот как выглядит multi-agent, когда вы помещаете миллионы agents на shared substrate:

- **Emergent economic activity.** Agents buy, sell, and service each other using token-payments.
- **Prompt-injection risks at population scale.** Один malicious prompt в viral agent profile распространяется на тысячи agent-to-agent interactions за часы.
- **State-level regulatory response.** Within weeks of launch, regulation reaches the ecosystem.

Уроки проектирования из этого case частично technical, частично governance:

1. **Multi-agent at population scale is a new regime.** Individual-system best practices (verification, role clarity) still apply but are not sufficient.
2. **Prompt injection is the new XSS.** Рассматривайте agent profiles и cross-agent messages как untrusted input by default.
3. **Regulation is faster than design cycles.** Планируйте это.
4. **Open-source + viral scale compounds.** 247k stars за ~4 months unusual; design for deploy-burst-load.

См. [OpenClaw Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) и reporting CNBC / Palo Alto Networks для ecosystem detail. Для technical underpinnings repos Clawdbot / OpenClaw показывают local ReAct loop; public posts Moltbook раскрывают social-graph architecture поверх него.

### Framework landscape April 2026

| Framework | Status | Best for | Notes |
|---|---|---|---|
| **LangGraph** (LangChain) | Production leader | structured graph + checkpointing + human-in-the-loop | recommended default for production |
| **CrewAI** | Production leader | role-based crews with Sequential/Hierarchical processes | strong for role decomposition |
| **AG2** | Community maintained | GroupChat + speaker selection | AutoGen v0.2 continuation |
| **Microsoft AutoGen** | Maintenance mode (Feb 2026) | — | merged into Microsoft Agent Framework RC |
| **Microsoft Agent Framework** | RC (Feb 2026) | orchestration patterns + enterprise integration | new entrant; watch |
| **OpenAI Agents SDK** | Production | Swarm successor | tool-return handoff pattern |
| **Google ADK** | Production (April 2025) | A2A-native | Google Cloud integration |
| **Anthropic Claude Agent SDK** | Production | single-agent + Research extension | see the Research system post |

Каждый major framework теперь поставляет **MCP** support; большинство поставляет **A2A**. Protocol compatibility больше не differentiator.

### Common patterns across all three cases

1. **Orchestrator + workers** (Anthropic explicit supervisor, MetaGPT PM-as-supervisor, OpenClaw individual agents + network effects).
2. **Structured handoff contracts** (Anthropic subagent task descriptions, MetaGPT PRD/architecture docs, OpenClaw A2A artifacts).
3. **Verification as first-class role** (Anthropic's verifier, MetaGPT's QA Engineer, OpenClaw's in-network validators).
4. **Scaling is topology + substrate, not just more agents** (rainbow deploys, MacNet DAGs, population-scale substrates).
5. **Cost is material and disclosed** (15x tokens, per-role budget in MetaGPT, per-interaction pricing in Moltbook).
6. **Security posture is explicit** (Anthropic's sandboxing, MetaGPT's role restrictions, OpenClaw's prompt-injection as known attack surface).

### Choosing a reference for your next project

- **Production research / knowledge task → Anthropic Research.** Fresh-context subagents win.
- **Engineering / tool-chain workflow → MetaGPT / ChatDev.** Roles + SOPs + handoff contracts.
- **Network-effect social product → OpenClaw / Moltbook.** Substrate + emergent economy.
- **Classic enterprise automation → CrewAI or LangGraph** (production leader, stable runtime).

### The 2026 state-of-the-art summary

Где находится область в апреле 2026:

- **Frameworks are converging.** MCP + A2A support is table stakes. Handoff semantics are the remaining design choice.
- **Evaluation is hardening.** SWE-bench Pro, MARBLE, STRATUS mitigation benchmarks. Pro is the current contamination-resistant reality check.
- **Production failure rates are measurable** (Cemri 2025 MAST; 41-86.7% on real MAS). Область вышла из эпохи "looks great in demo".
- **Cost is the central engineering constraint.** Token cost per task, wall-clock per interaction, rainbow-deploy overhead. Multi-agent выигрывает по accuracy, но проигрывает по cost — and that trade is the business decision.
- **Regulation is a near-term input, not a background concern.** Jurisdictions are moving faster than individual deploy cycles.

## Используйте

`outputs/skill-case-study-mapper.md` — skill, который читает proposed multi-agent system design и maps it to the closest case study, surfacing design decisions that case study already tested.

## Запустите в production

Starter rules for production multi-agent in 2026:

- **Start from a case study, not from scratch.** Pick the closest of Anthropic Research / MetaGPT / OpenClaw and adapt.
- **Adopt MCP + A2A.** Portability across frameworks valuable; protocol support is free.
- **Measure against SWE-bench Pro or your internal Pro-equivalent.** Verified is contaminated.
- **Pay the verification tax.** Independent verifier costs ~20-30% of your token budget and buys measurable correctness.
- **Rainbow deploy long-running agents.** Expect multi-hour agent runs to be routine.
- **Read WMAC 2026 and the MAST follow-ups.** The discipline is moving fast.

## Упражнения

1. Прочитайте post Anthropic Research system end-to-end. Определите три design decisions, которые изменились бы, если заменить Opus 4 меньшей model (e.g., Haiku 4).
2. Прочитайте MetaGPT Sections 3-4 (arXiv:2308.00352). Encode one SOP from your own domain (not software) as role prompts. Сколько roles подразумевает SOP?
3. Прочитайте ChatDev (arXiv:2307.07924). Определите mechanism of "communicative dehallucination." Реализуйте его в одной из своих existing multi-agent systems.
4. Прочитайте про OpenClaw и Moltbook. Выберите один specific failure mode, возникший at population scale, который не появился бы в 5-agent system. Как бы вы engineer against it?
5. Выберите текущий multi-agent project. Which of the three case studies is the closest reference? Какие design decisions from that case study вы еще НЕ adopted? Запишите one you will adopt this quarter.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|----------------|------------------------|
| Anthropic Research | "The supervisor reference" | Claude Opus 4 + Sonnet 4 subagents; 15x tokens; +90.2% over single-agent. |
| MetaGPT | "SOP as prompts" | Role decomposition for software engineering; `Code = SOP(Team)`. |
| ChatDev | "Agents as roles" | Designer / programmer / reviewer / tester; communicative dehallucination. |
| MacNet | "Scale ChatDev via DAG" | arXiv:2406.07155; 1000+ agents via explicit DAG routing. |
| OpenClaw | "Local ReAct-loop agents" | Steinberger's project; 247k stars by March 2026. |
| Moltbook | "Agent-only social network" | 2.3M agent accounts; acquired by Meta March 2026. |
| Rainbow deploy | "Multiple versions concurrent" | Keep old runtime versions alive for in-flight long-running agents. |
| Communicative dehallucination | "Ask before answering" | Agents request specifics from peers instead of guessing. |
| WMAC 2026 | "The AAAI workshop" | April 2026 community focal point for multi-agent coordination. |

## Дополнительное чтение

- [Anthropic — How we built our multi-agent research system](https://www.anthropic.com/engineering/multi-agent-research-system) — the supervisor-worker production reference
- [MetaGPT — Meta Programming for Multi-Agent Collaborative Framework](https://arxiv.org/abs/2308.00352) — SOP-role decomposition
- [ChatDev — Communicative Agents for Software Development](https://arxiv.org/abs/2307.07924) — communicative dehallucination
- [MacNet — scaling role-based agents to 1000+](https://arxiv.org/abs/2406.07155) — DAG-based scale
- [OpenClaw on Wikipedia](https://en.wikipedia.org/wiki/OpenClaw) — ecosystem overview
- [WMAC 2026](https://multiagents.org/2026/) — AAAI 2026 Bridge Program Workshop on Multi-Agent Coordination
- [LangGraph docs](https://docs.langchain.com/oss/python/langgraph/workflows-agents) — production leader
- [CrewAI docs](https://docs.crewai.com/en/introduction) — role-based framework
