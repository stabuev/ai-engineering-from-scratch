#!/usr/bin/env bash
set -euo pipefail

if [[ $# -lt 2 ]]; then
  cat <<'USAGE' >&2
Usage: scripts/scaffold-lesson.sh <phase-dir> <lesson-slug> [title]

Examples:
  scripts/scaffold-lesson.sh 05-nlp-foundations-to-advanced 03-tokenizers
  scripts/scaffold-lesson.sh 05-nlp-foundations-to-advanced 03-tokenizers "Tokenizers from Scratch"

Creates phases/<phase-dir>/<lesson-slug>/ with code/, docs/, outputs/ and
docs/en.md + docs/ru.md skeletons prefilled from LESSON_TEMPLATE.md.
USAGE
  exit 2
fi

PHASE="$1"
LESSON="$2"
TITLE="${3:-}"

REPO_ROOT="$(git rev-parse --show-toplevel 2>/dev/null)"
if [[ -z "$REPO_ROOT" ]]; then
  echo "error: run this from inside the ai-engineering-from-scratch git repo" >&2
  exit 1
fi

PHASE_DIR="$REPO_ROOT/phases/$PHASE"
LESSON_DIR="$PHASE_DIR/$LESSON"

if [[ ! -d "$PHASE_DIR" ]]; then
  echo "error: phase dir not found: phases/$PHASE" >&2
  echo "       run: ls phases/ to see valid phases" >&2
  exit 1
fi

if [[ -e "$LESSON_DIR" ]]; then
  echo "error: lesson already exists: phases/$PHASE/$LESSON" >&2
  exit 1
fi

if [[ ! "$LESSON" =~ ^[0-9]{2}-[a-z0-9-]+$ ]]; then
  echo "error: lesson slug must match NN-kebab-case (e.g. 03-tokenizers)" >&2
  exit 1
fi

mkdir -p "$LESSON_DIR/code" "$LESSON_DIR/docs" "$LESSON_DIR/outputs"

PRETTY_TITLE="$TITLE"
if [[ -z "$PRETTY_TITLE" ]]; then
  PRETTY_TITLE="$(echo "${LESSON#[0-9][0-9]-}" | tr '-' ' ' | awk '{for (i=1; i<=NF; i++) $i=toupper(substr($i,1,1)) substr($i,2);}1')"
fi

PHASE_NUM="${PHASE%%-*}"
LESSON_NUM="${LESSON%%-*}"

cat >"$LESSON_DIR/docs/en.md" <<EOF
# $PRETTY_TITLE

> [One-line motto. The core idea that sticks.]

**Type:** Build
**Languages:** Python
**Prerequisites:** [prior lessons]
**Time:** ~75 minutes

## The Problem

[2-3 paragraphs. What can't a learner do without this? Make it concrete.]

## The Concept

[Intuition first. Diagrams, tables, mental models. No code yet.]

## Build It

### Step 1: [name]

[explanation]

\`\`\`python
# code here
\`\`\`

### Step 2: [name]

[explanation]

\`\`\`python
# code here
\`\`\`

## Use It

[How a real framework solves the same thing. Compare your version.]

## Ship It

[The reusable artifact this lesson produces. Save in outputs/.]

## Exercises

1. [Easy — reinforce core concept]
2. [Medium — apply to a different problem]
3. [Hard — extend or combine with prior lessons]

## Key Terms

| Term | What people say | What it actually means |
|------|----------------|----------------------|
|      |                |                      |

## Further Reading

- []() — []
EOF

cat >"$LESSON_DIR/code/main.py" <<'EOF'
def main():
    raise NotImplementedError("implement the lesson")


if __name__ == "__main__":
    main()
EOF

touch "$LESSON_DIR/outputs/.gitkeep"

cat >"$LESSON_DIR/docs/ru.md" <<EOF
# $PRETTY_TITLE

> [Однострочный девиз. Главная идея, которая запоминается.]

**Тип:** Build
**Языки:** Python
**Пререквизиты:** [предыдущие уроки]
**Время:** ~75 минут

## Проблема

[2-3 абзаца. Что без этого нельзя сделать? Конкретно.]

## Концепция

[Сначала интуиция. Диаграммы, таблицы, ментальные модели. Пока без кода.]

## Build It

### Шаг 1: [название]

[объяснение]

\\`\\`\\`python
\\`\\`\\`

## Use It

[Как то же самое решает реальный фреймворк. Сравните со своей версией.]

## Ship It

[Переиспользуемый артефакт этого урока. Сохраните в outputs/.]

## Упражнения

1. [Легкое — закрепить основную идею]
2. [Среднее — применить к другой задаче]
3. [Сложное — расширить или скомбинировать с прошлыми уроками]

## Ключевые термины

| Термин | Что говорят | Что это на самом деле |
|--------|-------------|----------------------|
|        |             |                      |

## Что почитать

- []() — []
EOF

echo "created phases/$PHASE/$LESSON/"
echo ""
echo "next:"
echo "  1. edit phases/$PHASE/$LESSON/docs/en.md and docs/ru.md"
echo "  2. write phases/$PHASE/$LESSON/code/main.py"
echo "  3. register the lesson in lessons.json (phase $PHASE_NUM) and run: node site/build.js"
echo "     (README/ROADMAP tables and site data regenerate from the manifest; CI rejects unregistered lessons)"
echo "  4. optional: add quiz.json + quiz_en.json (see existing lessons for the format; validated by CI)"
echo "  5. atomic commit: git add phases/$PHASE/$LESSON lessons.json README.md ROADMAP.md site/ && git commit -m \"feat(phase-$PHASE_NUM/$LESSON_NUM): $PRETTY_TITLE\""
