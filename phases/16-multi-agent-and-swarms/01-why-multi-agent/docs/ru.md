# Почему Multi-Agent?

> Один агент упирается в стену. Умный ход — не агент побольше, а больше агентов.

**Тип:** Изучение
**Языки:** TypeScript
**Предварительные требования:** Фаза 14 (Agent Engineering)
**Время:** ~60 минут

## Цели обучения

- Определить потолок single-agent подхода (context overflow, смешанная экспертиза, sequential bottleneck) и объяснить, когда разделение на несколько agents — правильный ход
- Сравнить orchestration patterns (pipeline, parallel fan-out, supervisor, hierarchical) и выбрать подходящий для заданной структуры задачи
- Спроектировать multi-agent system с четкими границами ролей, shared state и communication contract
- Проанализировать tradeoffs multi-agent complexity (latency, cost, debugging difficulty) по сравнению с single-agent simplicity

## Проблема

В Фазе 14 вы построили single agent. Он работает. Он умеет читать файлы, запускать команды, вызывать APIs и рассуждать о результатах. Потом вы направляете его на реальную codebase: 200 файлов, три языка, tests, зависящие от infrastructure, и требование исследовать external APIs перед написанием code.

Agent захлебывается. Не потому, что LLM глупая, а потому, что задача превышает то, с чем может справиться один agent loop. Context window заполняется содержимым файлов. Agent забывает, что читал 40 tool calls назад. Он пытается быть researcher, coder и reviewer одновременно и все три делает плохо.

Это потолок single-agent. Вы упираетесь в него каждый раз, когда задача требует:

- **Больше context, чем помещается в одно окно** - чтение 50 файлов легко пробивает 200k tokens
- **Разной экспертизы на разных stages** - research требует другого prompting, чем code generation
- **Работы, которую можно выполнять параллельно** - зачем читать три файла последовательно, если можно читать их simultaneously?

## Концепция

### Потолок Single-Agent

Single agent — это один loop, одно context window, один system prompt. Представьте:

```
┌─────────────────────────────────────────┐
│            SINGLE AGENT                 │
│                                         │
│  ┌───────────────────────────────────┐  │
│  │         Context Window            │  │
│  │                                   │  │
│  │  research notes                   │  │
│  │  + code files                     │  │
│  │  + test output                    │  │
│  │  + review feedback                │  │
│  │  + API docs                       │  │
│  │  + ...                            │  │
│  │                                   │  │
│  │  ██████████████████████ FULL ███  │  │
│  └───────────────────────────────────┘  │
│                                         │
│  One system prompt tries to cover       │
│  research + coding + review + testing   │
│                                         │
│  Result: mediocre at everything         │
└─────────────────────────────────────────┘
```

Ломаются три вещи:

1. **Context saturation** - tool results накапливаются. К 30-му turn agent уже потребил 150k tokens содержимого файлов, command outputs и предыдущих рассуждений. Критичные детали из 5-го turn теряются.

2. **Role confusion** - system prompt, который говорит "you are a researcher, coder, reviewer, and tester", порождает agent, который наполовину исследует, наполовину пишет код и так и не заканчивает review.

3. **Sequential bottleneck** - agent читает file A, затем file B, затем file C. Три serial LLM calls. Три serial tool executions. Нет parallelism.

### Multi-Agent решение

Разделите работу. Дайте каждому agent одну работу, одно context window и один system prompt, tuned for that job:

```
┌──────────────────────────────────────────────────────────┐
│                    ORCHESTRATOR                          │
│                                                          │
│  "Build a REST API for user management"                  │
│                                                          │
│         ┌──────────┬──────────┬──────────┐               │
│         │          │          │          │               │
│         ▼          ▼          ▼          ▼               │
│   ┌──────────┐ ┌──────────┐ ┌──────────┐ ┌──────────┐  │
│   │RESEARCHER│ │  CODER   │ │ REVIEWER │ │  TESTER  │  │
│   │          │ │          │ │          │ │          │  │
│   │ Reads    │ │ Writes   │ │ Checks   │ │ Runs     │  │
│   │ docs,    │ │ code     │ │ code     │ │ tests,   │  │
│   │ finds    │ │ based on │ │ quality, │ │ reports  │  │
│   │ patterns │ │ research │ │ finds    │ │ results  │  │
│   │          │ │ + spec   │ │ bugs     │ │          │  │
│   └─────┬────┘ └────┬─────┘ └────┬─────┘ └────┬─────┘  │
│         │           │            │             │         │
│         └───────────┴────────────┴─────────────┘         │
│                          │                               │
│                     Merge results                        │
└──────────────────────────────────────────────────────────┘
```

У каждого agent есть:
- Сфокусированный system prompt ("You are a code reviewer. Your only job is finding bugs.")
- Собственное context window (не загрязненное работой других agents)
- Четкий input/output contract (получает research notes, выдает code)

### Реальные системы, которые так делают

**Claude Code subagents** - когда Claude Code spawn-ит subagent через `Task`, он создает child agent со scoped task. Parent сохраняет чистый context. Child выполняет focused work и возвращает summary.

**Devin** - запускает planner agent, coder agent и browser agent. Planner разбивает работу на steps. Coder пишет code. Browser исследует documentation. У каждого отдельный context.

**Multi-agent coding teams (SWE-bench)** - лучшие системы на SWE-bench используют researcher, который читает codebase, planner, который проектирует fix, и coder, который его реализует. Single-agent systems получают более низкие scores.

**ChatGPT Deep Research** - spawn-ит несколько search agents in parallel, каждый исследует свой angle, затем results синтезируются.

### Спектр

Multi-agent — не бинарный выбор. Это спектр:

```
SIMPLE ──────────────────────────────────────────── COMPLEX

 Single        Sub-         Pipeline      Team         Swarm
 Agent         agents

 ┌───┐       ┌───┐        ┌───┐───┐    ┌───┐───┐    ┌─┐┌─┐┌─┐
 │ A │       │ A │        │ A │ B │    │ A │ B │    │ ││ ││ │
 └───┘       └─┬─┘        └───┘─┬─┘    └─┬─┘─┬─┘    └┬┘└┬┘└┬┘
               │                │        │   │       ┌┴──┴──┴┐
             ┌─┴─┐          ┌───┘───┐    │   │       │shared │
             │ a │          │ C │ D │  ┌─┴───┴─┐    │ state │
             └───┘          └───┘───┘  │  msg   │    └───────┘
                                       │  bus   │
 1 loop      Parent +      Stage by    │       │    N peers,
 1 context   child tasks   stage       └───────┘    emergent
                                       Explicit      behavior
                                       roles
```

**Single agent** - один loop, один prompt. Хорош для простых задач.

**Subagents** - parent spawn-ит children для focused subtasks. Parent поддерживает plan. Children отчитываются назад. Именно так работает Claude Code.

**Pipeline** - agents запускаются по очереди. Вывод Agent A становится входом Agent B. Хорошо для staged workflows: research -> code -> review -> test.

**Team** - agents запускаются параллельно с shared message bus. У каждого роль. Orchestrator координирует. Хорошо, когда разные skills нужны одновременно.

**Swarm** - множество одинаковых или почти одинаковых agents с shared state. Нет фиксированного orchestrator. Агенты забирают работу из queue. Хорошо для high-throughput parallel tasks.

### Четыре Multi-Agent паттерна

#### Паттерн 1: Pipeline

```
Input ──▶ Agent A ──▶ Agent B ──▶ Agent C ──▶ Output
          (research)  (code)      (review)
```

Каждый agent преобразует данные и передает их дальше. Просто рассуждать. Failure на одном stage блокирует остальные.

#### Паттерн 2: Fan-out / Fan-in

```
                ┌──▶ Agent A ──┐
                │              │
Input ──▶ Split ├──▶ Agent B ──├──▶ Merge ──▶ Output
                │              │
                └──▶ Agent C ──┘
```

Разделите работу между parallel agents, затем merge results. Хорошо для задач, раскладывающихся на independent subtasks.

#### Паттерн 3: Orchestrator-Worker

```
                    ┌──────────┐
                    │  Orch.   │
                    └──┬───┬───┘
                  task │   │ task
                 ┌─────┘   └─────┐
                 ▼               ▼
           ┌──────────┐   ┌──────────┐
           │ Worker A │   │ Worker B │
           └──────────┘   └──────────┘
```

Smart orchestrator решает, что делать, delegates to workers и synthesizes results. Orchestrator сам является agent с tools для spawning workers.

#### Паттерн 4: Peer Swarm

```
         ┌───┐ ◄──── msg ────▶ ┌───┐
         │ A │                  │ B │
         └─┬─┘                  └─┬─┘
           │                      │
      msg  │    ┌───────────┐     │ msg
           └───▶│  Shared   │◄────┘
                │  State    │
           ┌───▶│  / Queue  │◄────┐
           │    └───────────┘     │
      msg  │                      │ msg
         ┌─┴─┐                  ┌─┴─┐
         │ C │ ◄──── msg ────▶ │ D │
         └───┘                  └───┘
```

Нет центрального orchestrator. Агенты общаются peer-to-peer. Решения возникают из взаимодействия. Сложнее debugging, но масштабируется на many agents.

### Когда НЕ использовать Multi-Agent

Multi-agent добавляет complexity. Каждое message между agents — потенциальная failure point. Debugging превращается из "прочитать один conversation" в "проследить messages across five agents."

**Оставайтесь на single-agent, когда:**
- Задача помещается в одно context window (менее ~100k tokens working data)
- Вам не нужны разные system prompts для разных stages
- Sequential execution достаточно быстра
- Задача достаточно проста, и splitting добавляет больше overhead, чем value

**Цена complexity:**
- Каждая agent boundary — lossy compression step: полный context agent A summarizе-ится в message для agent B
- Coordination logic (кто что делает, когда, в каком порядке) — собственный источник bugs
- Latency растет: N agents означает минимум N serial LLM calls, больше, если им нужно общаться туда-сюда
- Cost умножается: каждый agent независимо сжигает tokens

Правило большого пальца: если задача занимает меньше 20 tool calls и помещается в 100k tokens, оставляйте single-agent.

## Соберите это

### Шаг 1: Перегруженный Single Agent

Вот single agent, который пытается делать все. У него один massive system prompt и одно context window, где лежат research, code и reviews:

```typescript
type AgentResult = {
  content: string;
  tokensUsed: number;
  toolCalls: number;
};

async function singleAgentApproach(task: string): Promise<AgentResult> {
  const systemPrompt = `You are a full-stack developer. You must:
1. Research the requirements
2. Write the code
3. Review the code for bugs
4. Write tests
Do ALL of these in a single conversation.`;

  const contextWindow: string[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const research = await fakeLLMCall(systemPrompt, `Research: ${task}`);
  contextWindow.push(research.output);
  totalTokens += research.tokens;
  totalToolCalls += research.calls;

  const code = await fakeLLMCall(
    systemPrompt,
    `Given this research:\n${contextWindow.join("\n")}\n\nNow write code for: ${task}`
  );
  contextWindow.push(code.output);
  totalTokens += code.tokens;
  totalToolCalls += code.calls;

  const review = await fakeLLMCall(
    systemPrompt,
    `Given all previous context:\n${contextWindow.join("\n")}\n\nReview the code.`
  );
  contextWindow.push(review.output);
  totalTokens += review.tokens;
  totalToolCalls += review.calls;

  return {
    content: contextWindow.join("\n---\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Проблемы этого подхода:
- Context window растет на каждом stage. К review step оно содержит research notes И code И previous reasoning.
- System prompt generic. Его нельзя tuned для каждого stage.
- Ничего не выполняется параллельно.

### Шаг 2: Specialist Agents

Теперь разделим. Каждый agent получает одну работу:

```typescript
type SpecialistAgent = {
  name: string;
  systemPrompt: string;
  run: (input: string) => Promise<AgentResult>;
};

function createSpecialist(name: string, systemPrompt: string): SpecialistAgent {
  return {
    name,
    systemPrompt,
    run: async (input: string) => {
      const result = await fakeLLMCall(systemPrompt, input);
      return {
        content: result.output,
        tokensUsed: result.tokens,
        toolCalls: result.calls,
      };
    },
  };
}

const researcher = createSpecialist(
  "researcher",
  "You are a technical researcher. Read documentation, find patterns, and summarize findings. Output only the facts needed for implementation."
);

const coder = createSpecialist(
  "coder",
  "You are a senior TypeScript developer. Given requirements and research notes, write clean, tested code. Nothing else."
);

const reviewer = createSpecialist(
  "reviewer",
  "You are a code reviewer. Find bugs, security issues, and logic errors. Be specific. Cite line numbers."
);
```

У каждого specialist focused prompt. Каждый получает clean context window только с input, который ему нужен.

### Шаг 3: Координация через Messages

Свяжем specialists явной message passing:

```typescript
type AgentMessage = {
  from: string;
  to: string;
  content: string;
  timestamp: number;
};

async function multiAgentApproach(task: string): Promise<AgentResult> {
  const messages: AgentMessage[] = [];
  let totalTokens = 0;
  let totalToolCalls = 0;

  const researchResult = await researcher.run(task);
  messages.push({
    from: "researcher",
    to: "coder",
    content: researchResult.content,
    timestamp: Date.now(),
  });
  totalTokens += researchResult.tokensUsed;
  totalToolCalls += researchResult.toolCalls;

  const coderInput = messages
    .filter((m) => m.to === "coder")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const codeResult = await coder.run(coderInput);
  messages.push({
    from: "coder",
    to: "reviewer",
    content: codeResult.content,
    timestamp: Date.now(),
  });
  totalTokens += codeResult.tokensUsed;
  totalToolCalls += codeResult.toolCalls;

  const reviewerInput = messages
    .filter((m) => m.to === "reviewer")
    .map((m) => `[From ${m.from}]: ${m.content}`)
    .join("\n");

  const reviewResult = await reviewer.run(reviewerInput);
  messages.push({
    from: "reviewer",
    to: "orchestrator",
    content: reviewResult.content,
    timestamp: Date.now(),
  });
  totalTokens += reviewResult.tokensUsed;
  totalToolCalls += reviewResult.toolCalls;

  return {
    content: messages.map((m) => `[${m.from} -> ${m.to}]: ${m.content}`).join("\n\n"),
    tokensUsed: totalTokens,
    toolCalls: totalToolCalls,
  };
}
```

Каждый agent получает только messages, адресованные ему. Никакого context pollution. 50k tokens documentation reading researcher никогда не попадают в context reviewer.

### Шаг 4: Сравнение

```typescript
async function compare() {
  const task = "Build a rate limiter middleware for an Express.js API";

  console.log("=== Single Agent ===");
  const single = await singleAgentApproach(task);
  console.log(`Tokens: ${single.tokensUsed}`);
  console.log(`Tool calls: ${single.toolCalls}`);

  console.log("\n=== Multi-Agent ===");
  const multi = await multiAgentApproach(task);
  console.log(`Tokens: ${multi.tokensUsed}`);
  console.log(`Tool calls: ${multi.toolCalls}`);
}
```

Multi-agent version использует больше total tokens (три agents, три отдельных LLM calls), но context каждого agent остается чистым. Quality каждого stage улучшается, потому что system prompt specialized.

## Используйте это

Этот урок производит reusable prompt для решения, когда переходить на multi-agent. См. `outputs/prompt-multi-agent-decision.md`.

## Упражнения

1. Добавьте четвертого specialist: agent "tester", который получает code от coder и review feedback от reviewer, затем пишет tests
2. Измените pipeline так, чтобы reviewer мог отправлять feedback обратно coder для revision loop (max 2 rounds)
3. Превратите sequential pipeline в fan-out: запустите researcher и agent "requirements analyzer" параллельно, затем merge их outputs перед передачей coder

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|----------------|----------------------|
| Swarm | "A hive mind of AI agents" | Набор peer agents с shared state и без fixed leader. Behavior возникает из local interactions. |
| Orchestrator | "The boss agent" | Agent, чьи tools включают spawning and managing other agents. Он plans и delegates, но может не делать actual work. |
| Coordinator | "The traffic cop" | Non-agent component (часто просто code, не LLM), который routes messages between agents based on rules. |
| Consensus | "The agents agree" | Protocol, где multiple agents должны reach agreement before proceeding. Используется, когда conflicting outputs need resolution. |
| Emergent behavior | "The agents figured it out themselves" | System-level patterns, возникающие из agent interactions, но не запрограммированные явно. Могут быть useful or harmful. |
| Fan-out / fan-in | "Map-reduce for agents" | Splitting task across parallel agents (fan-out), затем combining results (fan-in). |
| Message passing | "Agents talk to each other" | Communication mechanism между agents: structured data, отправляемые от одного agent к другому, заменяют shared context windows. |

## Дополнительное чтение

- [The Landscape of Emerging AI Agent Architectures](https://arxiv.org/abs/2409.02977) - survey of multi-agent patterns
- [AutoGen: Enabling Next-Gen LLM Applications](https://arxiv.org/abs/2308.08155) - Microsoft's multi-agent conversation framework
- [Claude Code subagents documentation](https://docs.anthropic.com/en/docs/claude-code) - как Claude Code delegates with Task
- [CrewAI documentation](https://docs.crewai.com/) - role-based multi-agent framework
