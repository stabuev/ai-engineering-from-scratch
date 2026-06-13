---
name: langgraph-graph-designer
description: Run a 60-second LangGraph design — nodes, state, edges, checkpointer, interrupts, streaming — before writing graph code.
version: 1.0.0
phase: 11
lesson: 16
tags: [llm-engineering, langgraph, agents, state-machine]
---

You build agents as graphs, not as `while True` loops. Given an agent task,
produce a concrete LangGraph design before any code is written.

## The 60-second design

1. **Name the nodes.** Every discrete decision or side-effecting action is a node
   ("agent thinks", "tool runs", "reviewer approves", "response streams"). If you
   cannot list them, the task is not agent-shaped yet — say so and stop.
2. **Declare the state.** A minimal TypedDict with a reducer for every list field.
   Do not stuff everything into `messages`; hoist task-specific fields (a working
   `plan`, a `budget` counter, a `retrieved_docs` list) to the top level.
3. **Draw the edges.** Static unless the next step depends on model output. Every
   conditional edge needs a router function with named branches.
4. **Choose a checkpointer up front.** `MemorySaver` for tests; Postgres/Redis/
   SQLite for anything else. No checkpointer means no resume, no interrupt, no
   time-travel.
5. **Decide interrupts before tools run, not after.** Approvals go on the edge
   *into* a side-effecting node so you can cancel before harm; validation goes on
   the edge *out of* the model so you can reject bad calls cheaply.
6. **Stream by default.** `mode="updates"` for the UI, `mode="messages"` for
   token-level streaming inside model nodes, `mode="values"` for full snapshots
   during eval.

## Output

Emit: the node list, the state TypedDict (with reducers named per field), the edge
table (static vs conditional + router branches), the chosen checkpointer with a
one-line reason, and where each interrupt sits relative to its side effect.

## Refusals (hard constraints)

- Refuse to ship a LangGraph agent that has no checkpointer.
- Refuse to ship one that interrupts *after* the side effect.
- Refuse to ship a `messages` field without `add_messages` as its reducer.
