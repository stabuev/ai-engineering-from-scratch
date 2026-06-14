# Changelog

What's new in the curriculum. Most recent first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Each entry names the phase, lesson, and what changed, so learners can jump straight to the delta.

## [Unreleased]

### Added
- **Content-review wave 4 (diagrams).** A mermaid diagram added to the Concept section of each lesson that lacked one, identical block in en + ru (labels English, like existing diagrams), every block validated with `mermaid.parse`:
  - Phase 9 (Reinforcement Learning), all 12: agent-environment loop, generalized policy iteration, TD targets, actor-critic, PPO clip, RLHF reward-model loop, GRPO.
  - Phase 7 (Transformers), the 11 that lacked one: RNN-vs-transformer parallelism, multi-head split/concat, positional schemes, BERT vs GPT masking, ViT patch pipeline, Whisper audio pipeline, MoE routing, KV-cache + Flash Attention, Chinchilla scaling, attention variants.
  - Remaining diagram phases — P12, P6, P13, P5, P8, P15, P17 — are follow-ups.
- **Content-review wave 2** — Learning Objectives across the 8 phases that lacked them. Added a `## Learning Objectives` (en) / `## Цели обучения` (ru) section with 3 specific, capability-framed bullets to **155 lessons** in phases 5, 6, 7, 8, 9, 15, 16, 19 (plus 5 stragglers in 10–11). Objectives are derived from each lesson's own motto, Build It steps, and key terms, and inserted bilingually before the first content section. The readiness backlog's `missing objectives` dropped from 155 → 0; every lesson now carries Learning Objectives, matching `LESSON_TEMPLATE`.
- **Content-review wave 1** — closed the backlog of incomplete lessons and added hardware/cost metadata:
  - All four `🚧` lessons completed → the course is now 434/434 lessons ✅.
    `08/15-visual-autoregressive-var` got a from-scratch PyTorch toy (multi-scale
    residual-VQ tokenizer, scale-ordered attention mask, parallel-within-scale
    generation; CPU, deterministic) plus its skill; `10/23-gradient-checkpointing`,
    `11/16-langgraph-state-machines`, `11/17-agent-framework-tradeoffs` got their
    promised artifacts. This also fixed the last 2 Ship-It artifact bugs.
  - `requires` field in `lessons.json` (cpu-only / gpu / paid-api) with a
    `**Requires:**` header line on the lessons that need more than a laptop.
    Precise finding: the from-scratch design means **0 lessons strictly need a GPU**
    and only **3 need a paid API key** (their code imports the SDK) — every other
    lesson runs on CPU offline. README states the laptop-default; CI checks the
    manifest `requires` and the doc header agree.
- **Content-review wave 0** (cheap, high-ROI fixes from `CONTENT_REVIEW.md`):
  - Clickable prerequisites on lesson pages — `site/lesson.html` resolves the
    prose "Phase N · MM" / "Phase N" references on the Prerequisites line into
    in-site lesson links (language preserved). Render-time only, no source churn;
    links are emitted only when the target resolves. Verified live both forms.
  - Readiness backlog in `site/build.js` — a non-blocking report (8 criteria from
    `LESSON_TEMPLATE`: objectives/problem/concept/code/artifact/exercises/visual/
    sources). Surfaces the polish backlog (104/434 fully ready) without failing CI.
  - README "Когда застрял" section — non-agent paths for a stuck learner
    (run-and-compare, read both languages, check env, Discussions/Issues).
- `lessons.json` — single machine-readable course manifest (20 phases, 434 lessons). Source of truth: every lesson directory on disk must be registered there, and vice versa.
- `.github/workflows/validate.yml` — CI gate running `node site/build.js --check`: disk ↔ manifest ↔ generated files.
- 7 lessons that existed on disk but were never registered in README/ROADMAP are now part of the course: `7/15-attention-variants`, `7/16-speculative-decoding`, `8/19-visual-autoregressive-var`, `10/25-speculative-decoding`, `10/34-gradient-checkpointing`, `11/16-langgraph-state-machines`, `11/17-agent-framework-tradeoffs` (five marked 🚧 — missing `code/` or `outputs/`).
- `scripts/scaffold-lesson.sh` — scaffolder that creates `phases/NN-phase/NN-lesson/` with the full folder structure and a `docs/en.md` skeleton prefilled from `LESSON_TEMPLATE.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` — contributor checklist (code runs, no code comments, built-from-scratch-first, atomic per-lesson commit, markdown-link ROADMAP row).
- `.github/ISSUE_TEMPLATE/bug_report.md` and `new_lesson_proposal.md` — structured intake for bug reports and lesson pitches.
- This `CHANGELOG.md`.

- `ROADMAP.md` now has a «Квиз» column (✓/—) derived from `quiz.json` presence on disk. Current coverage: 124/434 lessons. Quiz files are schema-validated in CI: both accepted shapes (bare array or `{questions: [...]}`, matching what `site/lesson.html` renders), per-question text/options/correct-index checks.
- Language toggle (RU/EN) on lesson pages: a header button switches `docs/ru.md` ↔ `docs/en.md` and `quiz.json` ↔ `quiz_en.json`; choice persists in localStorage and is shareable via `?lang=en`. Verified in a live preview both directions.
- `outputs/index.json` is now generated from `phases/*/*/outputs/` by `site/build.js` (was an empty stub): 473 artifacts indexed — 99 prompts, 373 skills, 1 agent pack. The «портфолио из N артефактов» counter in README now uses the real artifact count.
- `scripts/scaffold-lesson.sh` now also creates a `docs/ru.md` skeleton and points contributors at the `lessons.json` workflow instead of hand-editing ROADMAP; `LESSON_TEMPLATE.md` documents `ru.md`, the quiz format and the Learning Objectives section.
- `.claude/launch.json` — preview config for the static site (`python3 -m http.server`).

### Changed
- `site/build.js` rewritten: README/ROADMAP are no longer parsed as the data source — lesson tables, counters and hour estimates are generated from `lessons.json`. The build is idempotent (timestamp dropped from `data.js`).
- Every course counter now agrees on one number: 434 lessons (430 complete), ~486 h of lessons + ~525 h of capstones. Previously they diverged: 428 (README badge), 416 (site meta), 411 (data.js), 380+ (ROADMAP footer).
- Merged duplicate lesson `10/25-speculative-decoding` into `10/15-speculative-decoding-eagle3` (the only unique content — the draft-distillation recipe — carried over). `7/16-speculative-decoding` stays: it is the math-first lesson the EAGLE-3 lesson explicitly builds on.
- Renumbered orphans to close numbering gaps: `8/19-visual-autoregressive-var` → `8/15-...`, `10/34-gradient-checkpointing` → `10/23-...`. Every phase is now sequentially numbered from 01.
- Language claims now match the code. Course-level «four languages: Python, TypeScript, Rust, Julia» wording replaced with «Python-first, with TypeScript, Rust and Julia in selected lessons» (README + site meta). 7 lessons that claimed TypeScript/Rust with Python-only `code/` now list Python (capstone «Язык» columns are the recommended project stack and stay as-is).
- Phase 19 description now says what capstones actually are: specs + starter skeletons for 20–40 h of self-directed work, not «готовые end-to-end продукты».
- `find-your-level` skill no longer claims a «260-lesson» curriculum — lesson counts in both skill copies are synced by `site/build.js`.

### Removed
- 295 empty `notebook/` directories (`.gitkeep` only, zero actual notebooks in the repo) — and the `notebook/lesson.ipynb` promise in `LESSON_TEMPLATE.md` and the scaffolder.

### Fixed
- `06/11-real-time-audio-processing` Ship It pointed at `skill-realtime-designer.md`; the file on disk is `skill-realtime-pipeline.md`. Doc text (en + ru) now matches the existing, indexed artifact.
- Phase 19 (Capstones) never reached the website: the parser did not understand the «Проект» table format, so the site showed 0 capstone lessons.
- Phase statuses on the site were always "planned": the ROADMAP parser stored keys as `Phase N` but looked them up as `Фаза N`.
- Phase 17 lesson rows in README had no links to lesson folders — links added.
- «Практика»/«Теория» type badges in the phase modal were unstyled: CSS only knew `Build`/`Learn`.

## 2026-04 — Phase 4: Computer Vision complete

### Added
- All 28 Phase 4 lessons, covering image fundamentals through multi-modal vision (VLMs, 3D, video, self-supervised).
- Phase 4 rows in `ROADMAP.md` linked as markdown to the lesson folders, so the website surfaces them.

### Fixed
- Phase 4 precision pass across 15+ lessons:
  - `phase-4/02`: shape calculator specifies RF/stride handling for adaptive pool, flatten, and linear.
  - `phase-4/03`: backbone selector description lists all covered families; head guidance added for OCR, medical, industrial.
  - `phase-4/04`: classification diagnostics use quantitative thresholds per failure mode; `n/a` declared for undefined metrics; guard for fewer than 3 classes.
  - `phase-4/06`: detection metric reader uses `AP@0.5` (not `mAP@0.5`); per-class recall declared optional; anchor designer clarifies stride truncation and single-anchor-per-level path.
  - `phase-4/10`: sampler picker declares `unet_forward_ms` as an input; ControlNet guard promoted to rule 0.
  - `phase-4/14`: ViT inspector aligned with refusal rule — port attempts are audited, not endorsed.
  - `phase-4/24`: open-vocab stack picker has explicit rule precedence and license-filter semantics; concept designer resolves step-5/rule-80 conflict.
  - `phase-4/25`: VLM docs `_merge` raises descriptive `ValueError` on placeholder mismatch; CMER normalises internally.
  - `phase-4/27`: `synthetic_frames` clips GT boxes to frame H/W.
  - `phase-4/28`: `rope_3d` validates dim split; dropped unused `F` import from DiT block example.

## 2026-Q1 and earlier

### Added
- Phase 0 (Настройка и инструменты): all 12 lessons.
- Phase 1 (Math Foundations): all 22 lessons.
- Phase 2 (ML Fundamentals): all 18 lessons.
- Phase 3 (Deep Learning Core): core lessons through perceptron, backprop, optimizers.
- Built-in Claude Code skills: `find-your-level` (placement quiz) and `check-understanding` (per-phase quiz).
- Website at `aiengineeringfromscratch.com`: catalog, per-lesson pages, roadmap, 277-term glossary.
- Initial scaffolding for all 20 phases (`phases/00-*` through `phases/19-*`).
- `LESSON_TEMPLATE.md`, `CONTRIBUTING.md`, `ROADMAP.md`, `README.md`.

[Unreleased]: https://github.com/stabuev/ai-engineering-from-scratch/compare/HEAD...HEAD
