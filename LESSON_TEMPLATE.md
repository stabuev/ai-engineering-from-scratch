# Lesson Template

Use this template when creating a new lesson. The quickest start is
`scripts/scaffold-lesson.sh <phase-dir> <lesson-slug> [title]` — it creates the
folder structure and prefilled `docs/en.md` + `docs/ru.md` skeletons.

After writing the lesson, register it in `lessons.json` and run
`node site/build.js` — README/ROADMAP tables and the website data are generated
from the manifest, and CI rejects lessons that exist on disk but are not
registered.

## Folder Structure

```
NN-lesson-name/
├── code/
│   ├── main.py            (primary implementation)
│   ├── main.ts            (TypeScript version, if applicable)
│   ├── main.rs            (Rust version, if applicable)
│   └── main.jl            (Julia version, if applicable)
├── docs/
│   ├── en.md              (lesson narrative, English)
│   └── ru.md              (lesson narrative, Russian — keep in sync with en.md)
├── quiz.json              (optional quiz, Russian; rendered on the website)
├── quiz_en.json           (optional quiz, English)
└── outputs/
    ├── prompt-*.md         (prompts produced by this lesson)
    └── skill-*.md          (skills produced by this lesson)
```

Quiz format: either a bare array of questions or `{"questions": [...]}`. Each
question needs `question`, `options` (2+), integer `correct` index; optional
`explanation` and `stage` (`"pre"` / `"post"`). CI validates every quiz file.

## Documentation Format (docs/en.md)

```markdown
# [Lesson Title]

> [One-line motto — the core idea that sticks]

**Type:** Build | Learn
**Languages:** Python, TypeScript, Rust, Julia (list only what code/ actually contains)
**Prerequisites:** [List prior lessons as "Phase N · MM (Name)" — the site turns these into links]
**Time:** ~[estimated time] minutes
**Requires:** Paid API (Anthropic/OpenAI) | GPU — *omit this line for CPU-only lessons.* CPU-only is the default (the from-scratch code runs on a laptop); add a Requires line only when the lesson's own code needs more — a paid API key or a GPU. No dollar estimates (they rot faster than versions). Mirror it in `lessons.json` (`"requires": "paid-api"` / `"gpu"`); CI checks the two agree.

## Learning Objectives

- [3-5 bullets: what the learner can do after this lesson]

## The Problem

[2-3 paragraphs. What can't you do without this? Why should you care?
Make it concrete — show a scenario where not knowing this hurts.]

## The Concept

[Explain with diagrams and intuition. No code yet.
Use ASCII diagrams, tables, or link to visuals in the web app.
Build mental models before implementation.]

## Build It

[Step-by-step implementation from scratch.
Start with the simplest version, then add complexity.
Every code block should be runnable on its own.
End with an "Expected output" block so the learner can check their run:]

    Expected output:
    Epoch 0 | Loss: 0.69
    ...
    [0.0, 0.0] -> 0.01 (expected 0)

### Step 1: [Name]

[Explanation]

    [code block]

### Step 2: [Name]

[Explanation]

    [code block]

[...continue...]

## Use It

[Now show how frameworks/libraries do the same thing.
Compare your from-scratch version to the library version.
This proves the concept and introduces practical tools.]

## Ship It

[What reusable artifact does this lesson produce?
Could be a prompt, a skill, an agent, an MCP server, or a tool.
Include it here and save it in the outputs/ folder.]

## Exercises

1. [Easy — reinforce the core concept]
2. [Medium — apply it to a different problem]
3. [Hard — extend or combine with prior lessons]

   <details><summary>Solution</summary>

   [Reference solution for the hard exercise — code and/or a one-line "why".
   Collapsed by default on GitHub and the site, so it does not spoil the
   attempt but closes the self-check loop for a learner without a mentor.]

   </details>

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
| [term] | [common misconception] | [actual definition] |

## Further Reading

- [Resource 1](url) — [why it's worth reading]
- [Resource 2](url) — [why it's worth reading]
```

## Code File Guidelines

- Code must run without errors
- No comments — code should be self-explanatory
- Use the language that fits best for the topic
- Include a `requirements.txt` or equivalent if there are dependencies
- Start simple, build up complexity
- Every function and class should have a clear purpose

## Output File Format

### Prompts

```markdown
---
name: prompt-name
description: What this prompt does
phase: [phase number]
lesson: [lesson number]
---

[Prompt content]
```

### Skills

```markdown
---
name: skill-name
description: What this skill teaches
version: 1.0.0
phase: [phase number]
lesson: [lesson number]
tags: [relevant, tags]
---

[Skill content]
```
