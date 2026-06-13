---
name: agent-framework-selector
description: Pick the right agent framework (LangGraph / CrewAI / AutoGen / Agno / plain Python) from the problem shape, in one defensible sentence.
version: 1.0.0
phase: 11
lesson: 17
tags: [llm-engineering, agents, frameworks, decision-aid]
---

You can explain, in one sentence, why a given framework is right for a given agent
problem. Given a task description, run the checklist, then return a pick from the
matrix with the one-line reason.

## Pre-build checklist

1. **Draw the shape.** Is this a graph (typed state, named transitions)? A role play
   (specialists hand off work)? A chat (agents talk until done)? A single agent with
   tools?
2. **Decide who branches.** Developer-decided branching → LangGraph. Manager-agent-
   decided → CrewAI hierarchical. Chat-emergent → AutoGen. Tool-call-decided → Agno.
3. **Check the state budget.** Need resume-from-checkpoint, time-travel, or human
   interrupts mid-run? If yes, LangGraph is the default; Agno sessions cover
   conversation-scoped state.
4. **Check the cost budget.** LLM-selected routing costs extra tokens per turn. If
   the agent runs thousands of times a day, prefer explicit routing.
5. **Budget the framework overhead.** Every framework is another dependency. If the
   task is two LLM calls and a tool, write 30 lines of plain Python — no framework is
   cheaper than no framework.

## The decision matrix

| Problem shape | Preferred framework | Why |
|---------------|---------------------|-----|
| Workflow DAG with typed state, human approvals, long-running | LangGraph | First-class state, checkpointer, interrupts, time-travel. |
| Research / writing pipeline with distinct roles | CrewAI (sequential) or LangGraph subgraphs | Role-per-task is cheap in CrewAI; scale up with LangGraph when branching gets complex. |
| Proposer-critic or teacher-student dialogue | AutoGen | Two-agent chat is its native shape. |
| Single agent with tools, sessions, memory | Agno | Thinnest setup, built-in storage and memory. |
| Thousands of parallel fanouts with reducers | LangGraph + `Send` | The only one with a first-class parallel dispatch primitive. |
| Quick prototype, no framework commitment | Plain Python + provider SDK | No framework is the fastest framework. |

## Refusals

- Refuse to reach for a framework before you can draw the graph, the org chart, the
  chat, or the agent box.
- Refuse to pick one that forces you to fight its state model for the thing you
  actually need.
