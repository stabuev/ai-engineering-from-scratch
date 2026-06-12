# Changelog

What's new in the curriculum. Most recent first.

Format loosely follows [Keep a Changelog](https://keepachangelog.com/). Each entry names the phase, lesson, and what changed, so learners can jump straight to the delta.

## [Unreleased]

### Added
- `lessons.json` — single machine-readable course manifest (20 phases, 435 lessons). Source of truth: every lesson directory on disk must be registered there, and vice versa.
- `.github/workflows/validate.yml` — CI gate running `node site/build.js --check`: disk ↔ manifest ↔ generated files.
- 7 lessons that existed on disk but were never registered in README/ROADMAP are now part of the course: `7/15-attention-variants`, `7/16-speculative-decoding`, `8/19-visual-autoregressive-var`, `10/25-speculative-decoding`, `10/34-gradient-checkpointing`, `11/16-langgraph-state-machines`, `11/17-agent-framework-tradeoffs` (five marked 🚧 — missing `code/` or `outputs/`).
- `scripts/scaffold-lesson.sh` — scaffolder that creates `phases/NN-phase/NN-lesson/` with the full folder structure and a `docs/en.md` skeleton prefilled from `LESSON_TEMPLATE.md`.
- `.github/PULL_REQUEST_TEMPLATE.md` — contributor checklist (code runs, no code comments, built-from-scratch-first, atomic per-lesson commit, markdown-link ROADMAP row).
- `.github/ISSUE_TEMPLATE/bug_report.md` and `new_lesson_proposal.md` — structured intake for bug reports and lesson pitches.
- This `CHANGELOG.md`.

### Changed
- `site/build.js` rewritten: README/ROADMAP are no longer parsed as the data source — lesson tables, counters and hour estimates are generated from `lessons.json`. The build is idempotent (timestamp dropped from `data.js`).
- Every course counter now agrees on one number: 435 lessons (430 complete), ~487 h of lessons + ~525 h of capstones. Previously they diverged: 428 (README badge), 416 (site meta), 411 (data.js), 380+ (ROADMAP footer).

### Fixed
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
