# Changelog

What's new in the curriculum. Most recent first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Each entry names the phase, lesson, and what changed, so learners can jump straight to the delta.

## [Unreleased]

### Added
- **Content-review wave 5 — reproducible environment (`requirements.lock`).**
  `requirements.txt` was all `>=` with no lock, so a future learner would get
  bleeding-edge versions and silent breakage. Added a pinned, internally
  consistent `requirements.lock` (uv-resolved, dry-run verified). The from-scratch
  core the Phase 0-3 lessons were run against is pinned to its tested versions
  (numpy 1.26.4, torch 2.7.1 + matching torchvision/torchaudio, scikit-learn
  1.7.0, pandas 2.2.2, matplotlib 3.9.2, pillow 10.4.0); the rest is pinned to the
  course's target generation (transformers 4.x, datasets 3.x) for coherence. A
  naive fresh resolve pulled bleeding-edge majors (numpy 2.4, pandas 3.0,
  transformers 5.x) and a torch/torchaudio mismatch — rejected in favour of the
  course-era stack. `requirements.txt` keeps the flexible `>=` path and now points
  at the lock; README's clone-and-run uses the lock.
- **Content-review wave 5 (start) — clickable sources.** Phases 1 and 3 cited
  classic papers by name only, so Further Reading had no links. Wrapped the
  references in real, HTTP-verified canonical URLs (arxiv / DOI / PMLR / JMLR) —
  Adam, BatchNorm, GELU, Dropout, AdamW, SGDR, He/Xavier init, GCN/GAT, DDPM,
  Rumelhart 1986, Rosenblatt 1958, Cybenko 1989, and more (~40 references across
  14 lessons, en + ru). Added a short Further Reading section to the two math
  lessons that had none (linear-algebra, statistics) and converted the JAX
  lesson's bare doc URLs to links. Only links that returned HTTP 200 were used —
  one wrong path (a 3Blue1Brown page) was caught and swapped for the verified
  playlist. Readiness `missing sources` 30 → 13 (the rest are Phase 0 tooling
  tutorials, where references are optional). Fully ready: 404 → 420.
- **Content-review wave 3 (start) — self-check loop.** Added an `### Expected
  output` / `### Ожидаемый вывод` block to the end of Build It in 11 Phase 3
  (Deep Learning Core) lessons, showing the **real, captured** final output of
  each lesson's code (run twice to confirm determinism — nothing fabricated).
  Closes the beginner gap "I ran it, is this right?" for the phase. Skipped
  `11-intro-to-pytorch` (non-deterministic, no fixed seed) and `12-intro-to-jax`
  (jax not installed, no output).
- **Wave 3 (cont.) — collapsible solutions enabled.** `site/lesson.html` now
  passes `<details>`/`<summary>` through its markdown parser (it previously
  escaped raw HTML, so collapsible solutions would have shown as literal tags),
  with CSS styling — verified live (cursor, colour, left border, native toggle).
  First verified solution added to `03-backpropagation` exercise 3 (`__pow__`):
  the code was run and produces loss and gradients identical to the original
  `mse_loss`.
- **Wave 3 (chunk 3b) — Phase 3 hard-exercise solutions.** Collapsible
  `<details>` solutions for 10 more Phase 3 lessons (perceptron, multi-layer,
  activations, losses, optimizers, regularization, init, LR schedules,
  mini-framework, debugging) — 11/13 lessons now carry a verified solution
  (the two framework-intro lessons, pytorch/jax, are left out, as with their
  expected-output). Every code snippet is syntax-checked, and each numeric claim
  was verified by running: majority-of-3 is linearly separable; one-hot KL has
  the same gradient as cross-entropy (max diff 0.0); SVD orthogonal init gives
  `U·Uᵀ = I` (≈8e-16); SGDR restarts to `lr_max` each period; AdamW decoupled
  decay shrinks a 5.0 weight to ≈4.09 over 200 steps.

### Fixed
- **The website never rendered lesson figures.** `site/lesson.html`'s markdown
  parser had no image support, so the ~89 `![](…/assets/*.svg)` figures across
  phases 5–11 showed nothing on the site (they only rendered on GitHub). Added
  image rendering with relative-path resolution to a CDN (`cdn.jsdelivr.net/gh`,
  which serves the repo's assets with a correct `image/svg+xml` content-type —
  raw.githubusercontent serves `text/plain`, which browsers refuse in `<img>`).
  Verified live: the SVG figures now load on the site. The readiness `visual`
  check now counts figures too, so the backlog reflects reality (174 lessons
  genuinely have no visual, not 284).

### Added
- **Content-review wave 4 (diagrams).** Mermaid diagrams in the Concept section,
  identical block in en + ru (English labels), each validated with
  `mermaid.parse`. Scoped to lessons that have **no** existing figure:
  - Phase 12 (Multimodal AI), all 25 — the review's #1 diagram gap (these
    architecture lessons shipped with zero figures): each fusion pattern (ViT
    patches, CLIP dual-encoder, BLIP-2 Q-Former, Flamingo gated cross-attention,
    LLaVA projector, patch-n'-pack, Chameleon/Emu3 early fusion, Transfusion/
    Show-o two-loss, Janus decoupled encoders, MIO any-to-any, video TMRoPE,
    long-video paths, audio Q-former, omni Thinker-Talker, embodied VLA, OCR-free
    docs, ColPali late interaction, cross-modal RAG, computer-use loop).
  - Phase 7 · 15 (attention variants) — the one transformer lesson without a figure.
  - Phases 9 (all 12) and 7 (the other 10) were diagrammed first, then reverted:
    those lessons already ship polished SVG figures, which now render on the site,
    so a second mermaid diagram was redundant.
  - Remaining figure-less phases — P13, P14, P15, P16, P17, P18, P19 — are follow-ups.
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
