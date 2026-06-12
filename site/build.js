#!/usr/bin/env node
/**
 * Build script for AI Engineering from Scratch.
 *
 * Source of truth: lessons.json at the repo root. Every lesson directory on
 * disk must be registered there, and vice versa. From the manifest this
 * script:
 *   - generates site/data.js (PHASES + GLOSSARY);
 *   - rewrites the lesson tables in README.md and ROADMAP.md;
 *   - keeps every lesson/hour counter in README.md, ROADMAP.md and the
 *     site/*.html meta tags in sync.
 *
 * Run: node site/build.js          — regenerate everything in place
 *      node site/build.js --check  — verify nothing would change (CI mode);
 *                                    exits 1 and lists stale files otherwise
 *
 * Called by Vercel on every deploy (see vercel.json) and by CI on every push.
 */

const fs = require('fs');
const path = require('path');

const REPO_ROOT = path.resolve(__dirname, '..');
const MANIFEST_PATH = path.join(REPO_ROOT, 'lessons.json');
const README_PATH = path.join(REPO_ROOT, 'README.md');
const ROADMAP_PATH = path.join(REPO_ROOT, 'ROADMAP.md');
const GLOSSARY_PATH = path.join(REPO_ROOT, 'glossary', 'terms.md');
const DATA_PATH = path.join(__dirname, 'data.js');
const HTML_COUNT_PAGES = ['index.html', 'catalog.html', 'prereqs.html', 'lesson.html']
  .map(f => path.join(__dirname, f));

const GITHUB_BASE = 'https://github.com/stabuev/ai-engineering-from-scratch/tree/main/';
const STATUS_EMOJI = { 'complete': '✅', 'in-progress': '🚧', 'planned': '⬚' };

// ─── Helpers ─────────────────────────────────────────────────────────
function ruPlural(n, one, few, many) {
  const i = Math.floor(Math.abs(n));
  if (Math.abs(n) !== i) return few; // 24.5 часа
  const m100 = i % 100;
  const m10 = i % 10;
  if (m100 >= 11 && m100 <= 14) return many;
  if (m10 === 1) return one;
  if (m10 >= 2 && m10 <= 4) return few;
  return many;
}

function hoursLabel(hours) {
  const rounded = Math.round(hours * 10) / 10;
  const text = Number.isInteger(rounded) ? String(rounded) : rounded.toFixed(1);
  return `${text} ${ruPlural(rounded, 'час', 'часа', 'часов')}`;
}

function lessonRel(phase, lesson) {
  return `phases/${phase.dir}/${lesson.dir}`;
}

// Replace the first contiguous run of markdown-table lines in `segment`.
function replaceFirstTable(segment, newTableLines, label) {
  const lines = segment.split('\n');
  let start = -1;
  let end = -1;
  for (let i = 0; i < lines.length; i++) {
    if (lines[i].startsWith('|')) {
      if (start === -1) start = i;
      end = i;
    } else if (start !== -1) {
      break;
    }
  }
  if (start === -1) throw new Error(`no markdown table found in segment: ${label}`);
  return [...lines.slice(0, start), ...newTableLines, ...lines.slice(end + 1)].join('\n');
}

// ─── Manifest loading & validation ───────────────────────────────────
function loadManifest() {
  const manifest = JSON.parse(fs.readFileSync(MANIFEST_PATH, 'utf8'));
  const errors = [];

  if (!Array.isArray(manifest.phases) || manifest.phases.length !== 20) {
    errors.push(`expected 20 phases in lessons.json, got ${manifest.phases?.length}`);
  }

  const diskPhases = fs.readdirSync(path.join(REPO_ROOT, 'phases'))
    .filter(d => /^\d{2}-/.test(d)).sort();
  const manifestPhaseDirs = manifest.phases.map(p => p.dir).sort();
  for (const d of diskPhases) {
    if (!manifestPhaseDirs.includes(d)) errors.push(`phase on disk but not in lessons.json: phases/${d}`);
  }
  for (const d of manifestPhaseDirs) {
    if (!diskPhases.includes(d)) errors.push(`phase in lessons.json but not on disk: phases/${d}`);
  }

  for (const phase of manifest.phases) {
    const phaseAbs = path.join(REPO_ROOT, 'phases', phase.dir);
    if (!fs.existsSync(phaseAbs)) continue;

    const diskLessons = fs.readdirSync(phaseAbs)
      .filter(d => /^\d{2}-/.test(d) && fs.statSync(path.join(phaseAbs, d)).isDirectory())
      .sort();
    const manifestLessons = phase.lessons.map(l => l.dir);

    for (const d of diskLessons) {
      if (!manifestLessons.includes(d)) {
        errors.push(`lesson on disk but not in lessons.json: phases/${phase.dir}/${d}`);
      }
    }
    for (const lesson of phase.lessons) {
      const abs = path.join(phaseAbs, lesson.dir);
      if (!diskLessons.includes(lesson.dir)) {
        errors.push(`lesson in lessons.json but not on disk: phases/${phase.dir}/${lesson.dir}`);
        continue;
      }
      for (const doc of ['en.md', 'ru.md']) {
        if (!fs.existsSync(path.join(abs, 'docs', doc))) {
          errors.push(`missing docs/${doc}: phases/${phase.dir}/${lesson.dir}`);
        }
      }
      // Two accepted shapes, matching site/lesson.html (`data.questions || data`):
      // a bare array of questions, or an object with a `questions` array.
      for (const quizFile of ['quiz.json', 'quiz_en.json']) {
        const quizPath = path.join(abs, quizFile);
        if (!fs.existsSync(quizPath)) continue;
        const where = `phases/${phase.dir}/${lesson.dir}/${quizFile}`;
        try {
          const parsed = JSON.parse(fs.readFileSync(quizPath, 'utf8'));
          const questions = Array.isArray(parsed) ? parsed : parsed.questions;
          if (!Array.isArray(questions) || questions.length === 0) {
            errors.push(`no questions array: ${where}`);
            continue;
          }
          questions.forEach((q, i) => {
            if (typeof q.question !== 'string' || !q.question.trim()) {
              errors.push(`question ${i + 1} has no text: ${where}`);
            }
            if (!Array.isArray(q.options) || q.options.length < 2) {
              errors.push(`question ${i + 1} needs >= 2 options: ${where}`);
            } else if (!Number.isInteger(q.correct) || q.correct < 0 || q.correct >= q.options.length) {
              errors.push(`question ${i + 1} has bad correct index: ${where}`);
            }
          });
        } catch (e) {
          errors.push(`invalid JSON: ${where} (${e.message})`);
        }
      }
      if (!lesson.title || !lesson.type || !STATUS_EMOJI[lesson.status] || !(lesson.minutes > 0)) {
        errors.push(`incomplete manifest entry: phases/${phase.dir}/${lesson.dir}`);
      }
      if (lesson.title.includes('|')) {
        errors.push(`title contains "|", breaks markdown tables: phases/${phase.dir}/${lesson.dir}`);
      }
      if (lesson.dir.slice(0, 2) !== lesson.number) {
        errors.push(`number/dir mismatch: phases/${phase.dir}/${lesson.dir} has number ${lesson.number}`);
      }
    }
  }

  if (errors.length) {
    console.error('❌ lessons.json validation failed:');
    errors.forEach(e => console.error('   - ' + e));
    process.exit(1);
  }
  return manifest;
}

function computeStats(manifest) {
  let total = 0;
  let complete = 0;
  let quizzes = 0;
  let lessonMinutes = 0; // phases 0–18
  let capstoneMinutes = 0; // phase 19
  for (const phase of manifest.phases) {
    for (const lesson of phase.lessons) {
      total += 1;
      if (lesson.status === 'complete') complete += 1;
      if (hasQuiz(phase, lesson)) quizzes += 1;
      if (phase.number === 19) capstoneMinutes += lesson.minutes;
      else lessonMinutes += lesson.minutes;
    }
  }
  return {
    total,
    complete,
    quizzes,
    lessonHours: lessonMinutes / 60,
    capstoneHours: capstoneMinutes / 60,
    totalHours: (lessonMinutes + capstoneMinutes) / 60,
  };
}

// ─── Glossary ────────────────────────────────────────────────────────
function parseGlossary(content) {
  const terms = [];
  let currentTerm = null;

  for (const line of content.split('\n')) {
    const termMatch = line.match(/^###\s+(.+)/);
    if (termMatch) {
      if (currentTerm && currentTerm.says && currentTerm.means) {
        terms.push(currentTerm);
      }
      currentTerm = { term: termMatch[1].trim(), says: '', means: '' };
      continue;
    }
    if (!currentTerm) continue;

    const saysMatch = line.match(/\*\*What people say:\*\*\s*"?(.+?)"?\s*$/);
    if (saysMatch) {
      currentTerm.says = saysMatch[1].replace(/^"/, '').replace(/"$/, '').trim();
      continue;
    }
    const meansMatch = line.match(/\*\*What it actually means:\*\*\s*(.+)/);
    if (meansMatch) {
      currentTerm.means = meansMatch[1].trim();
    }
  }
  if (currentTerm && currentTerm.says && currentTerm.means) {
    terms.push(currentTerm);
  }
  return terms;
}

// ─── site/data.js ────────────────────────────────────────────────────
function renderDataJs(manifest, glossaryTerms) {
  const phases = manifest.phases.map(phase => ({
    id: phase.number,
    name: phase.name,
    status: phase.status,
    desc: phase.desc,
    lessons: phase.lessons.map(lesson => ({
      name: lesson.title,
      status: lesson.status,
      type: lesson.type,
      lang: lesson.languages,
      ...(lesson.combines && { combines: lesson.combines }),
      url: GITHUB_BASE + lessonRel(phase, lesson) + '/',
    })),
  }));

  return `// Auto-generated by build.js from lessons.json — do not edit manually.

const PHASES = ${JSON.stringify(phases, null, 2)};

const GLOSSARY = ${JSON.stringify(glossaryTerms, null, 2)};
`;
}

// ─── README.md ───────────────────────────────────────────────────────
function readmeTable(phase) {
  const isCapstone = phase.number === 19;
  const lines = isCapstone
    ? ['| # | Проект | Объединяет | Язык |', '|:---:|---------|----------|------|']
    : ['| # | Урок | Тип | Язык |', '|:---:|--------|:----:|------|'];
  for (const lesson of phase.lessons) {
    const link = `[${lesson.title}](${lessonRel(phase, lesson)}/)`;
    const middle = isCapstone ? lesson.combines : lesson.type;
    lines.push(`| ${lesson.number} | ${link} | ${middle} | ${lesson.languages} |`);
  }
  return lines;
}

function renderReadme(content, manifest, stats) {
  let out = content;

  // Badge + headline counters
  out = out.replace(/lessons-\d+-3553ff/, `lessons-${stats.total}-3553ff`);
  out = out.replace(/alt="\d+ уроков"/, `alt="${stats.total} уроков"`);
  out = out.replace(
    /\d+ уроков\. 20 фаз\. ~\d+ часов[^.\n]*\./,
    `${stats.total} уроков. 20 фаз. ~${Math.round(stats.lessonHours)} часов уроков + ~${Math.round(stats.capstoneHours)} часов capstone-проектов.`
  );
  out = out.replace(/20 фаз, \d+ уроков/, `20 фаз, ${stats.total} уроков`);
  out = out.replace(/портфолио из \d+ артефактов/, `портфолио из ${stats.total} артефактов`);

  // "С чего начать" — cumulative lesson hours from the entry phase (no capstones)
  out = out.replace(
    /^(\|[^|\n]+\| Фаза (\d+) — [^|\n]+\| )~\d+ часов( \|)$/gm,
    (match, prefix, phaseNum, suffix) => {
      const from = parseInt(phaseNum);
      const minutes = manifest.phases
        .filter(p => p.number >= from && p.number < 19)
        .reduce((s, p) => s + p.lessons.reduce((a, l) => a + l.minutes, 0), 0);
      const rounded = Math.round(minutes / 60 / 10) * 10;
      return `${prefix}~${rounded} часов${suffix}`;
    }
  );

  // Per-phase lesson counts + tables
  for (const phase of manifest.phases) {
    const count = phase.lessons.length;
    const table = readmeTable(phase);

    if (phase.number === 0) {
      out = out.replace(/(### Фаза 0: [^`\n]*`)\d+ уроков(`)/, `$1${count} уроков$2`);
      const startIdx = out.indexOf('### Фаза 0:');
      const endIdx = out.indexOf('<details id="phase-1">');
      if (startIdx === -1 || endIdx === -1) throw new Error('README: phase 0 block not found');
      const segment = out.slice(startIdx, endIdx);
      out = out.slice(0, startIdx) + replaceFirstTable(segment, table, 'README phase 0') + out.slice(endIdx);
    } else {
      const marker = `<details id="phase-${phase.number}">`;
      const startIdx = out.indexOf(marker);
      if (startIdx === -1) throw new Error(`README: ${marker} not found`);
      const endIdx = out.indexOf('</details>', startIdx);
      if (endIdx === -1) throw new Error(`README: unclosed ${marker}`);
      let segment = out.slice(startIdx, endIdx);
      const noun = phase.number === 19 ? 'проектов' : ruPlural(count, 'урок', 'урока', 'уроков');
      segment = segment.replace(/<code>\d+ (?:уроков?|урока|проектов?)<\/code>/, `<code>${count} ${noun}</code>`);
      segment = replaceFirstTable(segment, table, `README phase ${phase.number}`);
      out = out.slice(0, startIdx) + segment + out.slice(endIdx);
    }
  }

  return out;
}

// ─── ROADMAP.md ──────────────────────────────────────────────────────
function hasQuiz(phase, lesson) {
  return fs.existsSync(path.join(REPO_ROOT, lessonRel(phase, lesson), 'quiz.json'));
}

function roadmapTable(phase) {
  const isCapstone = phase.number === 19;
  const lines = isCapstone
    ? ['| # | Проект | Статус | Квиз | Оценка |', '|---|---------|--------|------|------|']
    : ['| # | Урок | Статус | Квиз | Оценка |', '|---|--------|--------|------|------|'];
  for (const lesson of phase.lessons) {
    const link = `[${lesson.title}](${lessonRel(phase, lesson)})`;
    const time = isCapstone ? `~${Math.round(lesson.minutes / 60)} ч` : `~${lesson.minutes} мин`;
    const quiz = hasQuiz(phase, lesson) ? '✓' : '—';
    lines.push(`| ${lesson.number} | ${link} | ${STATUS_EMOJI[lesson.status]} | ${quiz} | ${time} |`);
  }
  return lines;
}

function renderRoadmap(content, manifest, stats) {
  let out = content;

  out = out.replace(
    /^Общее оценочное время:.*$/m,
    `Общее оценочное время: ~${Math.round(stats.lessonHours)} часов уроков (фазы 0–18) + ~${Math.round(stats.capstoneHours)} часов capstone-проектов, в своем темпе.`
  );

  for (const phase of manifest.phases) {
    const headerRe = new RegExp(`^## Фаза ${phase.number}: (.+?) — (?:✅|🚧|⬚) \\(~[\\d.]+ час[а-я]*\\)$`, 'm');
    const headerMatch = out.match(headerRe);
    if (!headerMatch) throw new Error(`ROADMAP: header for phase ${phase.number} not found`);
    const phaseHours = phase.lessons.reduce((s, l) => s + l.minutes, 0) / 60;
    const newHeader = `## Фаза ${phase.number}: ${headerMatch[1]} — ${STATUS_EMOJI[phase.status]} (~${hoursLabel(phaseHours)})`;
    out = out.replace(headerRe, newHeader);

    const startIdx = out.indexOf(newHeader);
    let endIdx = out.indexOf('\n## ', startIdx + newHeader.length);
    if (endIdx === -1) endIdx = out.length;
    const segment = out.slice(startIdx, endIdx);
    out = out.slice(0, startIdx)
      + replaceFirstTable(segment, roadmapTable(phase), `ROADMAP phase ${phase.number}`)
      + out.slice(endIdx);
  }

  out = out.replace(
    /^\*\*Итого:.*\*\*$/m,
    `**Итого: 20 фаз, ${stats.total} уроков | ${stats.complete} завершено | ~${Math.round(stats.totalHours)} часов по оценке (включая capstone-проекты)**`
  );

  return out;
}

// ─── site/*.html lesson counters ─────────────────────────────────────
function renderHtmlCounts(content, stats) {
  // No trailing \b: JS ASCII word boundaries do not work after Cyrillic letters.
  return content.replace(/\b\d+(?=( AI engineering)? (lessons|уроков)(?![A-Za-zа-яё]))/g, String(stats.total));
}

// ─── Main ────────────────────────────────────────────────────────────
function main() {
  const checkMode = process.argv.includes('--check');

  const manifest = loadManifest();
  const stats = computeStats(manifest);
  const glossaryTerms = parseGlossary(fs.readFileSync(GLOSSARY_PATH, 'utf8'));

  const targets = [
    { path: DATA_PATH, render: () => renderDataJs(manifest, glossaryTerms) },
    { path: README_PATH, render: old => renderReadme(old, manifest, stats) },
    { path: ROADMAP_PATH, render: old => renderRoadmap(old, manifest, stats) },
    ...HTML_COUNT_PAGES.map(p => ({ path: p, render: old => renderHtmlCounts(old, stats) })),
  ];

  const stale = [];
  for (const target of targets) {
    const old = fs.existsSync(target.path) ? fs.readFileSync(target.path, 'utf8') : '';
    const next = target.render(old);
    if (next !== old) {
      if (checkMode) {
        stale.push(path.relative(REPO_ROOT, target.path));
      } else {
        fs.writeFileSync(target.path, next, 'utf8');
        console.log(`✏️  Updated ${path.relative(REPO_ROOT, target.path)}`);
      }
    }
  }

  console.log(`\n📊 Stats:`);
  console.log(`   Phases: ${manifest.phases.length}`);
  console.log(`   Lessons: ${stats.total} (${stats.complete} complete, ${stats.quizzes} with quizzes)`);
  console.log(`   Hours: ~${Math.round(stats.lessonHours)} lessons + ~${Math.round(stats.capstoneHours)} capstones`);
  console.log(`   Glossary terms: ${glossaryTerms.length}`);

  if (checkMode) {
    if (stale.length) {
      console.error(`\n❌ Stale generated content. Run \`node site/build.js\` and commit:`);
      stale.forEach(f => console.error('   - ' + f));
      process.exit(1);
    }
    console.log('\n✅ Check passed: lessons.json, disk and generated files are in sync.');
  } else {
    console.log('\n✅ Build complete.');
  }
}

main();
