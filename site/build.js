#!/usr/bin/env node
/**
 * Build script for AI Engineering from Scratch.
 *
 * Source of truth: lessons.json at the repo root. Every lesson directory on
 * disk must be registered there, and vice versa. From the manifest this
 * script:
 *   - generates site/data.js (PHASES + GLOSSARY);
 *   - prerenders every lesson (ru + en) into static SEO-friendly pages:
 *       site/<phase-slug>/<lesson-slug>.html       (ru, canonical)
 *       site/en/<phase-slug>/<lesson-slug>.html    (en)
 *     plus a phase hub page site/<phase-slug>/index.html per phase.
 *     Slugs are the repo dir names without the numeric prefix. The pages are
 *     gitignored (every dir under site/ except assets/) and rebuilt on deploy;
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
const lessonRenderer = require('./lesson-render.js');

const REPO_ROOT = path.resolve(__dirname, '..');
const MANIFEST_PATH = path.join(REPO_ROOT, 'lessons.json');
const README_PATH = path.join(REPO_ROOT, 'README.md');
const ROADMAP_PATH = path.join(REPO_ROOT, 'ROADMAP.md');
const GLOSSARY_PATH = path.join(REPO_ROOT, 'glossary', 'terms.md');
const GLOSSARY_RU_PATH = path.join(REPO_ROOT, 'glossary', 'terms.ru.md');
const DATA_PATH = path.join(__dirname, 'data.js');
const LESSON_TEMPLATE_PATH = path.join(__dirname, 'lesson-template.html');
const HTML_COUNT_PAGES = ['index.html', 'catalog.html', 'prereqs.html']
  .map(f => path.join(__dirname, f));
const SKILL_COUNT_PAGES = [
  path.join(REPO_ROOT, '.claude', 'skills', 'find-your-level', 'SKILL.md'),
  path.join(REPO_ROOT, '.agents', 'skills', 'find-your-level', 'SKILL.md'),
];

const OUTPUTS_INDEX_PATH = path.join(REPO_ROOT, 'outputs', 'index.json');

const GITHUB_BASE = 'https://github.com/stabuev/ai-engineering-from-scratch/tree/main/';
// Public site root (must match the Sitemap: line in site/robots.txt).
const SITE_BASE = 'https://datascience.xyz/courses/aicourse/';
const SITEMAP_PATH = path.join(__dirname, 'sitemap.xml');
const STATUS_EMOJI = { 'complete': '✅', 'in-progress': '🚧', 'planned': '⬚' };

// Cache-busting version for the static assets referenced by prerendered pages.
const ASSET_V = '20260709a';
// Directory names at the site root that pretty phase slugs must never shadow.
const RESERVED_SLUGS = ['en', 'assets'];

// UI strings baked into the prerendered chrome (header nav, sidebar, bottom
// nav, mermaid modal). Article-level strings live in lesson-render.js.
const UI = {
  ru: {
    navContents: 'Содержание', navCatalog: 'Каталог', navRoadmap: 'Дорожная карта', navGlossary: 'Глоссарий',
    phase: 'Фаза', prevLesson: 'Предыдущий', nextLesson: 'Следующий',
    sidebarToggle: 'Переключить боковую панель',
    mermaidModalLabel: 'Увеличенная диаграмма', diagram: 'Диаграмма', close: 'Закрыть',
    home: 'Главная', lessonsWord: 'уроков', minutes: 'мин',
    siteTitleSuffix: ' - AI Engineering from Scratch',
  },
  en: {
    navContents: 'Contents', navCatalog: 'Catalog', navRoadmap: 'Roadmap', navGlossary: 'Glossary',
    phase: 'Phase', prevLesson: 'Previous', nextLesson: 'Next',
    sidebarToggle: 'Toggle sidebar',
    mermaidModalLabel: 'Expanded diagram', diagram: 'Diagram', close: 'Close',
    home: 'Home', lessonsWord: 'lessons', minutes: 'min',
    siteTitleSuffix: ' - AI Engineering from Scratch',
  },
};

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

// Pretty URL slugs: repo dir names without the numeric ordering prefix, so
// renumbering a lesson does not change its public URL.
function stripNum(dir) {
  return dir.replace(/^\d+-/, '');
}

function phaseSlug(phase) {
  return stripNum(phase.dir);
}

function lessonSlug(phase, lesson) {
  return `${stripNum(phase.dir)}/${stripNum(lesson.dir)}`;
}

// Relative href from a prerendered page to a lesson / phase hub.
// `root` is the page's prefix to the site root ('../' for ru, '../../' for en).
function lessonHref(root, lang, phase, lesson) {
  return root + (lang === 'en' ? 'en/' : '') + lessonSlug(phase, lesson) + '.html';
}

function phaseHubHref(root, lang, phase) {
  return root + (lang === 'en' ? 'en/' : '') + phaseSlug(phase) + '/';
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
      // `requires` (default cpu-only) must match the doc header: a Requires
      // line appears only when the lesson needs more than a laptop (paid API).
      const requires = lesson.requires || 'cpu-only';
      if (!['cpu-only', 'gpu', 'paid-api'].includes(requires)) {
        errors.push(`invalid requires "${requires}": phases/${phase.dir}/${lesson.dir}`);
      }
      const enHead = fs.existsSync(path.join(abs, 'docs', 'en.md'))
        ? fs.readFileSync(path.join(abs, 'docs', 'en.md'), 'utf8') : '';
      const ruHead = fs.existsSync(path.join(abs, 'docs', 'ru.md'))
        ? fs.readFileSync(path.join(abs, 'docs', 'ru.md'), 'utf8') : '';
      const enHasReq = /^\*\*Requires:\*\*/m.test(enHead);
      const ruHasReq = /^\*\*Требуется:\*\*/m.test(ruHead);
      if (requires === 'cpu-only' && (enHasReq || ruHasReq)) {
        errors.push(`requires=cpu-only but a Requires line is present: phases/${phase.dir}/${lesson.dir}`);
      }
      if (requires !== 'cpu-only' && !(enHasReq && ruHasReq)) {
        errors.push(`requires=${requires} but the Requires line is missing in en.md and/or ru.md: phases/${phase.dir}/${lesson.dir}`);
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

  // Pretty URL slugs must stay unique and must not shadow reserved site dirs,
  // otherwise two lessons would prerender to the same file.
  const seenSlugs = new Map();
  for (const phase of manifest.phases) {
    const pSlug = phaseSlug(phase);
    if (RESERVED_SLUGS.includes(pSlug)) {
      errors.push(`phase slug "${pSlug}" collides with a reserved site directory`);
    }
    for (const lesson of phase.lessons) {
      const slug = lessonSlug(phase, lesson);
      if (seenSlugs.has(slug)) {
        errors.push(`duplicate lesson URL slug "${slug}": ${lessonRel(phase, lesson)} vs ${seenSlugs.get(slug)}`);
      }
      seenSlugs.set(slug, lessonRel(phase, lesson));
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

// ─── Lesson readiness (non-blocking backlog report) ──────────────────
// Criteria from LESSON_TEMPLATE. Never fails the build — this is a visible
// backlog so the polish tracks have a moving target and regressions show up.
//
// Two lesson templates, two readiness profiles:
//   • CONCEPT lessons (Phases 0-18 and the large Phase 19 capstones) follow the
//     classic template: Problem → Concept → Build → Exercises → Further Reading
//     with external sources, and ship a reusable artifact under outputs/.
//   • BUILD lessons (the Phase 19 mini-project capstones, type "Практика")
//     follow a hands-on template — Build It / Use It / How to read the code, a
//     runnable-and-proven code/ demo, and an architecture diagram. Their
//     "artifact" is the tested demo itself (a tests/ dir or an in-file
//     verification pass), and problem/concept/exercises/sources are folded into
//     the walkthrough rather than carried as separate required sections. They
//     are measured against BUILD_REQUIRED so the backlog reflects genuine gaps,
//     not a mismatch between two section vocabularies.
//
// `visual` is ADVISORY for both: a diagram earns its place only when a lesson
// carries structural information, so a missing visual is a triage candidate,
// not a defect, and never counts against readiness.
const CONCEPT_REQUIRED = ['objectives', 'problem', 'concept', 'code', 'artifact', 'exercises', 'sources'];
const BUILD_REQUIRED = ['objectives', 'code', 'artifact', 'application'];
const ADVISORY_CRITERIA = ['visual'];
const READINESS_CRITERIA = ['objectives', 'problem', 'concept', 'code', 'artifact', 'exercises', 'sources', 'application', 'visual'];

function isBuildLesson(phase, lesson) {
  return phase.number === 19 && lesson.type === 'Практика';
}

function lessonReadiness(manifest) {
  const missingByCriterion = Object.fromEntries(READINESS_CRITERIA.map(c => [c, 0]));
  const missingByPhase = {};
  let fullyReady = 0;

  for (const phase of manifest.phases) {
    missingByPhase[phase.number] = 0;
    for (const lesson of phase.lessons) {
      const dir = path.join(REPO_ROOT, lessonRel(phase, lesson));
      const enPath = path.join(dir, 'docs', 'en.md');
      const en = fs.existsSync(enPath) ? fs.readFileSync(enPath, 'utf8') : '';
      const build = isBuildLesson(phase, lesson);

      const codeDir = path.join(dir, 'code');
      const codeEntries = fs.existsSync(codeDir) ? fs.readdirSync(codeDir) : [];
      const hasCodeFile = codeEntries.some(f => /\.(py|ts|rs|jl)$/.test(f));
      const outDir = path.join(dir, 'outputs');
      const hasOutput = fs.existsSync(outDir)
        && fs.readdirSync(outDir).some(f => f !== '.gitkeep');
      // A build lesson's artifact is its self-contained runnable demo:
      // a code/main.py entrypoint that the Use It / Running it section walks the
      // reader through, optionally proven by a tests/ dir or a test file. This is
      // a more concrete deliverable than the static outputs/ file a concept
      // lesson ships, so it stands in for the `artifact` criterion here.
      const hasDemo = codeEntries.includes('main.py')
        || codeEntries.some(f => /^test.*\.(py|ts|rs|jl)$/.test(f))
        || codeEntries.includes('tests');

      const exMatch = en.match(/##\s+Exercises[\s\S]*?(?=\n##\s|$)/i);
      const exCount = exMatch ? (exMatch[0].match(/^\s*\d+\./gm) || []).length : 0;
      const frMatch = en.match(/##\s+Further Reading[\s\S]*?(?=\n##\s|$)/i);
      const frLinks = frMatch ? (frMatch[0].match(/\]\(https?:/g) || []).length : 0;
      const hasVisual = /```mermaid/.test(en)
        || /[┌│└├─┐┘►▼▲]|--->|═══/.test(en)
        || /!\[[^\]]*\]\([^)]*\.(svg|png|jpg)\)/.test(en); // figures render on the site too
      const hasApplication = /^##\s+(Build It|Use It|Running it|What you will build|How to read the code|Ship It)/im.test(en);

      const checks = {
        objectives: /##\s+Learning Objectives/i.test(en),
        problem: /^##\s+.*Problem/im.test(en),   // any "… Problem" heading
        concept: /^##\s+.*Concept/im.test(en),   // any "… Concept" heading
        code: lesson.type !== 'Практика' || hasCodeFile, // Learn lessons may be codeless
        artifact: build ? (hasDemo || hasOutput) : hasOutput,
        exercises: exCount >= 3,
        application: hasApplication,
        visual: hasVisual,
        sources: frLinks >= 1,
      };

      // Advisory criteria are tallied across every lesson but never block ready.
      for (const c of ADVISORY_CRITERIA) {
        if (!checks[c]) missingByCriterion[c] += 1;
      }
      const required = build ? BUILD_REQUIRED : CONCEPT_REQUIRED;
      for (const c of required) {
        if (!checks[c]) missingByCriterion[c] += 1;
      }
      const missingRequired = required.filter(c => !checks[c]).length;
      if (missingRequired === 0) fullyReady += 1;
      else missingByPhase[phase.number] += 1;
    }
  }

  return { missingByCriterion, missingByPhase, fullyReady };
}

function printReadiness(manifest, stats) {
  const r = lessonReadiness(manifest);
  console.log(`\n📋 Readiness backlog (non-blocking — see CONTENT_REVIEW.md):`);
  console.log(`   Fully ready: ${r.fullyReady}/${stats.total} (required criteria)`);
  const requiredNames = [...new Set([...CONCEPT_REQUIRED, ...BUILD_REQUIRED])];
  const order = requiredNames.sort((a, b) => r.missingByCriterion[b] - r.missingByCriterion[a]);
  for (const c of order) {
    if (r.missingByCriterion[c] > 0) {
      console.log(`   missing ${c.padEnd(11)} ${r.missingByCriterion[c]}`);
    }
  }
  const worst = Object.entries(r.missingByPhase)
    .filter(([, n]) => n > 0)
    .sort((a, b) => b[1] - a[1])
    .slice(0, 6)
    .map(([p, n]) => `P${p}:${n}`)
    .join('  ');
  if (worst) console.log(`   phases with most incomplete lessons: ${worst}`);
  for (const c of ADVISORY_CRITERIA) {
    if (r.missingByCriterion[c] > 0) {
      console.log(`   advisory: ${r.missingByCriterion[c]} lessons have no ${c} (triage candidates, not a defect)`);
    }
  }
}

// ─── Glossary ────────────────────────────────────────────────────────
// Strip a single layer of surrounding quotes (straight, curly or guillemets)
// so the display layer can add its own per-language quotes.
function stripQuotes(s) {
  return s.replace(/^["“«]/, '').replace(/["”»]$/, '').trim();
}

// English and Russian glossary sources share the same `### headword` order but
// use different section markers. The parser is parameterized by those markers
// so both files run through the same logic.
const GLOSSARY_MARKERS = {
  en: { says: /\*\*What people say:\*\*\s*(.+)/, means: /\*\*What it actually means:\*\*\s*(.+)/ },
  ru: { says: /\*\*Что говорят:\*\*\s*(.+)/, means: /\*\*Что это на самом деле:\*\*\s*(.+)/ },
};

function parseGlossary(content, markers) {
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

    const saysMatch = line.match(markers.says);
    if (saysMatch) {
      currentTerm.says = stripQuotes(saysMatch[1].trim());
      continue;
    }
    const meansMatch = line.match(markers.means);
    if (meansMatch) {
      currentTerm.means = meansMatch[1].trim();
    }
  }
  if (currentTerm && currentTerm.says && currentTerm.means) {
    terms.push(currentTerm);
  }
  return terms;
}

// Combine the ru + en glossaries into one bilingual list. Russian is the
// top-level (the site default); the English copy is nested under `en`. The two
// files must have the same term count and order (validated here) since they are
// zipped by index — the headwords differ ("Attention" vs "Attention (внимание)").
function buildBilingualGlossary() {
  const en = parseGlossary(fs.readFileSync(GLOSSARY_PATH, 'utf8'), GLOSSARY_MARKERS.en);
  const ru = parseGlossary(fs.readFileSync(GLOSSARY_RU_PATH, 'utf8'), GLOSSARY_MARKERS.ru);
  if (en.length !== ru.length) {
    console.error(`❌ glossary mismatch: terms.md has ${en.length} terms, terms.ru.md has ${ru.length}. They must match 1:1 in order.`);
    process.exit(1);
  }
  return ru.map((r, i) => ({
    term: r.term, says: r.says, means: r.means,
    en: { term: en[i].term, says: en[i].says, means: en[i].means },
  }));
}

// ─── site/data.js ────────────────────────────────────────────────────
// English lesson title = the H1 of the lesson's en.md (falls back to the
// Russian title). Used by the bilingual catalog and roadmap pages.
function lessonTitleEn(phase, lesson) {
  const enPath = path.join(REPO_ROOT, lessonRel(phase, lesson), 'docs', 'en.md');
  if (fs.existsSync(enPath)) {
    const t = lessonRenderer.extractTitle(fs.readFileSync(enPath, 'utf8'));
    if (t) return t;
  }
  return lesson.title;
}

function renderDataJs(manifest, glossaryTerms) {
  const phases = manifest.phases.map(phase => ({
    id: phase.number,
    name: phase.name,
    name_en: phase.name_en || phase.name,
    status: phase.status,
    desc: phase.desc,
    desc_en: phase.desc_en || phase.desc,
    dir: phase.dir,
    slug: phaseSlug(phase),
    lessons: phase.lessons.map(lesson => ({
      name: lesson.title,
      name_en: lessonTitleEn(phase, lesson),
      status: lesson.status,
      type: lesson.type,
      lang: lesson.languages,
      ...(lesson.combines && { combines: lesson.combines }),
      // path — repo path, also the progress.js key; slug — pretty URL path
      // (append '.html', prefix 'en/' for the English version).
      path: lessonRel(phase, lesson),
      slug: lessonSlug(phase, lesson),
      url: GITHUB_BASE + lessonRel(phase, lesson) + '/',
    })),
  }));

  return `// Auto-generated by build.js from lessons.json — do not edit manually.

const PHASES = ${JSON.stringify(phases, null, 2)};

const GLOSSARY = ${JSON.stringify(glossaryTerms, null, 2)};
`;
}

// ─── outputs/index.json ──────────────────────────────────────────────
// Index of every artifact the lessons ship. One entry per file in a lesson's
// outputs/, or per directory for multi-file packs. Bucketed by name prefix.
function renderOutputsIndex(manifest) {
  const buckets = { prompts: [], skills: [], agents: [], mcp_servers: [] };
  const bucketByPrefix = name => {
    if (name.startsWith('prompt')) return 'prompts';
    if (name.startsWith('skill')) return 'skills';
    if (name.startsWith('agent')) return 'agents';
    if (name.startsWith('mcp')) return 'mcp_servers';
    return null;
  };

  for (const phase of manifest.phases) {
    for (const lesson of phase.lessons) {
      const outDir = path.join(REPO_ROOT, lessonRel(phase, lesson), 'outputs');
      if (!fs.existsSync(outDir)) continue;
      for (const entry of fs.readdirSync(outDir)) {
        if (entry === '.gitkeep') continue;
        const name = entry.replace(/\.[^.]+$/, '');
        const bucket = bucketByPrefix(name);
        if (!bucket) {
          console.warn(`⚠️  unclassified artifact (no prompt/skill/agent/mcp prefix): ${lessonRel(phase, lesson)}/outputs/${entry}`);
          continue;
        }
        buckets[bucket].push({
          name,
          path: `${lessonRel(phase, lesson)}/outputs/${entry}`,
          phase: phase.number,
          lesson: lesson.dir,
        });
      }
    }
  }

  for (const list of Object.values(buckets)) {
    list.sort((a, b) => a.path.localeCompare(b.path));
  }

  return JSON.stringify({
    version: '1.0.0',
    note: 'Auto-generated by site/build.js from phases/*/*/outputs — do not edit manually.',
    counts: Object.fromEntries(Object.entries(buckets).map(([k, v]) => [k, v.length])),
    ...buckets,
  }, null, 2) + '\n';
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
  out = out.replace(/портфолио из \d+ артефактов/, `портфолио из ${stats.artifacts} артефактов`);

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

// ─── agent skill lesson counters ─────────────────────────────────────
function renderSkillCounts(content, stats) {
  return content
    .replace(/\d+-lesson, 20-phase/g, `${stats.total}-lesson, 20-phase`)
    .replace(/20 phases, \d+\+? lessons/g, `20 phases, ${stats.total} lessons`);
}

// ─── Prerendered lesson pages ────────────────────────────────────────
// One static page per lesson per language, written on every build (deploy):
//   site/<phase-slug>/<lesson-slug>.html      ru — canonical, x-default
//   site/en/<phase-slug>/<lesson-slug>.html   en
// plus a phase hub (site/<phase-slug>/index.html + en/) listing the lessons.
// Everything a crawler needs — article body, sidebar links, prev/next nav,
// canonical/hreflang/JSON-LD — is in the markup; site/lesson.js only hydrates.

function fillTemplate(tpl, vars) {
  return tpl.replace(/\{\{(\w+)\}\}/g, (m, key) => (key in vars ? vars[key] : m));
}

// JSON safe for embedding into <script>: no '</script>' breakouts.
function jsonForHtml(value) {
  return JSON.stringify(value).replace(/</g, '\\u003c');
}

function buildFlatLessons(manifest) {
  const flat = [];
  manifest.phases.forEach((phase, phaseIndex) => {
    phase.lessons.forEach(lesson => {
      flat.push({ phase, lesson, phaseIndex });
    });
  });
  return flat;
}

function chromeVars(lang, root) {
  const ui = UI[lang];
  return {
    HTML_LANG: lang,
    ROOT: root,
    ASSET_V,
    NAV_CONTENTS: ui.navContents,
    NAV_CATALOG: ui.navCatalog,
    NAV_ROADMAP: ui.navRoadmap,
    NAV_GLOSSARY: ui.navGlossary,
    LANG_LABEL: lang.toUpperCase(),
    SIDEBAR_TOGGLE_LABEL: ui.sidebarToggle,
    MERMAID_MODAL_LABEL: ui.mermaidModalLabel,
    MERMAID_MODAL_TITLE: ui.diagram,
    MERMAID_MODAL_CLOSE: ui.close,
  };
}

function buildSidebarHtml(manifest, phaseIndex, activeLessonDir, lang, root) {
  const esc = lessonRenderer.escapeHtml;
  const ui = UI[lang];
  const phases = manifest.phases;
  const phase = phases[phaseIndex];
  let html = '';

  const prevPhase = phaseIndex > 0 ? phases[phaseIndex - 1] : null;
  const nextPhase = phaseIndex < phases.length - 1 ? phases[phaseIndex + 1] : null;
  if (prevPhase || nextPhase) {
    html += '<div class="sidebar-phase-nav">';
    if (prevPhase) {
      html += `<a href="${phaseHubHref(root, lang, prevPhase)}">&larr; ${ui.phase} ${String(prevPhase.number).padStart(2, '0')}: ${esc(prevPhase.name)}</a>`;
    }
    if (nextPhase) {
      html += `<a href="${phaseHubHref(root, lang, nextPhase)}">${ui.phase} ${String(nextPhase.number).padStart(2, '0')}: ${esc(nextPhase.name)} &rarr;</a>`;
    }
    html += '</div>';
  }

  html += `<div class="sidebar-phase-header"><a href="${phaseHubHref(root, lang, phase)}">${ui.phase} ${String(phase.number).padStart(2, '0')} · ${esc(phase.name)}</a></div>`;
  for (const lesson of phase.lessons) {
    const isActive = lesson.dir === activeLessonDir;
    html += `<a class="sidebar-lesson-link${isActive ? ' active' : ''}" href="${lessonHref(root, lang, phase, lesson)}">`;
    html += `<span class="sidebar-lesson-dot ${lesson.status}"></span>`;
    html += esc(lesson.title);
    html += '</a>';
  }
  return html;
}

function buildBottomNavHtml(flat, index, lang, root) {
  const esc = lessonRenderer.escapeHtml;
  const ui = UI[lang];
  const prev = index > 0 ? flat[index - 1] : null;
  const next = index < flat.length - 1 ? flat[index + 1] : null;

  let nav = '<div class="lesson-nav-bottom">';
  if (prev) {
    nav += `<a class="lesson-nav-btn prev" href="${lessonHref(root, lang, prev.phase, prev.lesson)}">`;
    nav += `<span class="nav-label">&larr; ${ui.prevLesson}</span>`;
    nav += `<span class="nav-title">${esc(prev.lesson.title)}</span>`;
    nav += '</a>';
  } else {
    nav += '<div></div>';
  }
  if (next) {
    nav += `<a class="lesson-nav-btn next" href="${lessonHref(root, lang, next.phase, next.lesson)}">`;
    nav += `<span class="nav-label">${ui.nextLesson} &rarr;</span>`;
    nav += `<span class="nav-title">${esc(next.lesson.title)}</span>`;
    nav += '</a>';
  }
  nav += '</div>';
  return nav;
}

function makePrereqResolver(manifest, lang, root) {
  return function resolveHref(phaseNum, lessonNum) {
    const phase = manifest.phases.find(p => p.number === phaseNum);
    if (!phase || !phase.lessons.length) return null;
    if (lessonNum == null) return lessonHref(root, lang, phase, phase.lessons[0]);
    const lesson = phase.lessons.find(l => l.number === lessonNum);
    return lesson ? lessonHref(root, lang, phase, lesson) : null;
  };
}

function lessonJsonLd({ title, description, canonical, lang, phase, lesson }) {
  const ui = UI[lang];
  const hubUrl = `${SITE_BASE}${lang === 'en' ? 'en/' : ''}${phaseSlug(phase)}/`;
  return jsonForHtml([
    {
      '@context': 'https://schema.org',
      '@type': 'LearningResource',
      name: title,
      description,
      url: canonical,
      inLanguage: lang,
      learningResourceType: 'Lesson',
      timeRequired: `PT${lesson.minutes}M`,
      isAccessibleForFree: true,
      isPartOf: {
        '@type': 'Course',
        name: 'AI Engineering from Scratch',
        url: SITE_BASE,
      },
    },
    {
      '@context': 'https://schema.org',
      '@type': 'BreadcrumbList',
      itemListElement: [
        { '@type': 'ListItem', position: 1, name: 'AI Engineering from Scratch', item: SITE_BASE },
        { '@type': 'ListItem', position: 2, name: `${ui.phase} ${String(phase.number).padStart(2, '0')}: ${phase.name}`, item: hubUrl },
        { '@type': 'ListItem', position: 3, name: title, item: canonical },
      ],
    },
  ]);
}

function renderLessonPages(manifest) {
  const tpl = fs.readFileSync(LESSON_TEMPLATE_PATH, 'utf8');
  const flat = buildFlatLessons(manifest);
  const esc = lessonRenderer.escapeHtml;
  let written = 0;

  flat.forEach((entry, index) => {
    const { phase, lesson, phaseIndex } = entry;
    const lessonPath = lessonRel(phase, lesson);
    const slug = lessonSlug(phase, lesson);
    const ruUrl = `${SITE_BASE}${slug}.html`;
    const enUrl = `${SITE_BASE}en/${slug}.html`;

    for (const lang of ['ru', 'en']) {
      const root = lang === 'en' ? '../../' : '../';
      const md = fs.readFileSync(path.join(REPO_ROOT, lessonPath, 'docs', `${lang}.md`), 'utf8');
      const strings = lessonRenderer.STRINGS[lang];

      let quiz = null;
      const quizPath = path.join(REPO_ROOT, lessonPath, lang === 'en' ? 'quiz_en.json' : 'quiz.json');
      if (fs.existsSync(quizPath)) {
        try { quiz = JSON.parse(fs.readFileSync(quizPath, 'utf8')); } catch (e) { /* validated in loadManifest */ }
      }

      const { html: body, title } = lessonRenderer.renderArticle({ md, lessonPath, strings });
      let article = body;
      if (quiz) article = lessonRenderer.buildQuizzesHtml(article, quiz, strings);
      article = lessonRenderer.linkifyPrereqs(article, makePrereqResolver(manifest, lang, root));
      article += '<div class="ai-panels" id="aiPanels"></div>';
      article += buildBottomNavHtml(flat, index, lang, root);

      const pageTitle = (title || lesson.title) + UI[lang].siteTitleSuffix;
      const description = lessonRenderer.extractDescription(md) || lesson.title;
      const canonical = lang === 'en' ? enUrl : ruUrl;

      const page = fillTemplate(tpl, {
        ...chromeVars(lang, root),
        TITLE: esc(pageTitle),
        OG_TITLE: esc((title || lesson.title) + ' · AI Engineering from Scratch'),
        DESCRIPTION: esc(description),
        CANONICAL: canonical,
        RU_URL: ruUrl,
        EN_URL: enUrl,
        JSONLD: lessonJsonLd({ title: title || lesson.title, description, canonical, lang, phase, lesson }),
        SIDEBAR: buildSidebarHtml(manifest, phaseIndex, lesson.dir, lang, root),
        ARTICLE: article,
        LESSON_JSON: jsonForHtml({ path: lessonPath, slug, lang, root }),
      });

      const outPath = path.join(__dirname, lang === 'en' ? 'en' : '', `${slug}.html`);
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, page, 'utf8');
      written++;
    }
  });

  return written;
}

function renderPhaseHubs(manifest) {
  const tpl = fs.readFileSync(LESSON_TEMPLATE_PATH, 'utf8');
  const esc = lessonRenderer.escapeHtml;
  let written = 0;

  manifest.phases.forEach((phase, phaseIndex) => {
    const slug = phaseSlug(phase);
    const ruUrl = `${SITE_BASE}${slug}/`;
    const enUrl = `${SITE_BASE}en/${slug}/`;

    for (const lang of ['ru', 'en']) {
      const root = lang === 'en' ? '../../' : '../';
      const ui = UI[lang];
      const phaseLabel = `${ui.phase} ${String(phase.number).padStart(2, '0')}`;
      const hours = Math.round(phase.lessons.reduce((s, l) => s + l.minutes, 0) / 60 * 10) / 10;

      let article = `<h1>${phaseLabel}: ${esc(phase.name)}</h1>`;
      article += `<div class="motto">${esc(phase.desc)}</div>`;
      article += `<div class="lesson-meta-tag"><strong>${lang === 'en' ? 'Lessons' : 'Уроки'}:</strong> ${phase.lessons.length} · ~${hours} ${lang === 'en' ? 'h' : 'ч'}</div>`;
      article += '<ol>';
      for (const lesson of phase.lessons) {
        article += `<li><a href="${lessonHref(root, lang, phase, lesson)}">${esc(lesson.title)}</a> — ${esc(lesson.type)} · ~${lesson.minutes} ${ui.minutes}</li>`;
      }
      article += '</ol>';

      const pageTitle = `${phaseLabel}: ${phase.name}${ui.siteTitleSuffix}`;
      const description = phase.desc;
      const canonical = lang === 'en' ? enUrl : ruUrl;

      const jsonLd = jsonForHtml([
        {
          '@context': 'https://schema.org',
          '@type': 'BreadcrumbList',
          itemListElement: [
            { '@type': 'ListItem', position: 1, name: 'AI Engineering from Scratch', item: SITE_BASE },
            { '@type': 'ListItem', position: 2, name: `${phaseLabel}: ${phase.name}`, item: canonical },
          ],
        },
      ]);

      const page = fillTemplate(tpl, {
        ...chromeVars(lang, root),
        TITLE: esc(pageTitle),
        OG_TITLE: esc(`${phaseLabel}: ${phase.name} · AI Engineering from Scratch`),
        DESCRIPTION: esc(description),
        CANONICAL: canonical,
        RU_URL: ruUrl,
        EN_URL: enUrl,
        JSONLD: jsonLd,
        SIDEBAR: buildSidebarHtml(manifest, phaseIndex, null, lang, root),
        ARTICLE: article,
        LESSON_JSON: jsonForHtml({ path: '', slug: `${slug}/index`, lang, root }),
      });

      const outPath = path.join(__dirname, lang === 'en' ? 'en' : '', slug, 'index.html');
      fs.mkdirSync(path.dirname(outPath), { recursive: true });
      fs.writeFileSync(outPath, page, 'utf8');
      written++;
    }
  });

  return written;
}

// ─── Service pages (index / catalog / prereqs / glossary), en copies ─
// The Russian versions are committed static files at the site root; they
// self-localize the visible chrome at runtime by detecting `/en/` in the path.
// This generates the English copy under site/en/<file> by transforming the
// committed root file: point root-relative assets one level up, set lang=en,
// and swap the <head> SEO block (title/description/canonical/og) to English.
// Sibling nav links (catalog.html, glossary.html, …) are left as-is — they
// resolve to the en siblings that live in the same site/en/ directory.
const SERVICE_ROOT_ASSETS = ['style.css', 'data.js', 'progress.js', 'header.js', 'lesson.css', 'lesson.js', 'app.js'];

const SERVICE_PAGES = [
  {
    file: 'glossary.html',
    title: 'AI Glossary - AI Engineering from Scratch',
    description: 'AI glossary: what people say vs what things actually mean. Every term explained without hand-waving.',
    ogTitle: 'Glossary · AI Engineering from Scratch',
    ogDescription: 'What people say vs what things actually mean. Every AI term, defined without hand-waving.',
    twDescription: 'What people say vs what things actually mean.',
    // Exact Russian head strings to replace (must match glossary.html verbatim).
    ruTitle: 'Глоссарий ИИ - AI Engineering from Scratch',
    ruDescription: 'Глоссарий ИИ: что говорят vs что это значит на самом деле. Каждый термин объяснен без воды.',
    ruOgTitle: 'Глоссарий · AI Engineering from Scratch',
    ruOgDescription: 'Что говорят vs что это значит на самом деле. Каждый термин ИИ, определенный без воды.',
    ruTwDescription: 'Что говорят vs что это значит на самом деле.',
  },
  {
    file: 'catalog.html',
    title: 'Lesson Catalog - AI Engineering from Scratch',
    description: 'Full catalog of 502 AI Engineering lessons. Search, filter, and sort every lesson across all 20 phases.',
    ogTitle: 'Catalog · AI Engineering from Scratch',
    ogDescription: 'Search and filter 502 lessons across 20 phases. Python, TypeScript, Rust, Julia.',
    twDescription: 'Search and filter 502 lessons across 20 phases.',
    ruTitle: 'Каталог уроков - AI Engineering from Scratch',
    ruDescription: 'Полный каталог из 502 уроков по AI Engineering. Поиск, фильтрация и сортировка всех уроков во всех 20 фазах.',
    ruOgTitle: 'Каталог · AI Engineering from Scratch',
    ruOgDescription: 'Ищите и фильтруйте 502 уроков в 20 фазах. Python, TypeScript, Rust, Julia.',
    ruTwDescription: 'Ищите и фильтруйте 502 уроков в 20 фазах.',
  },
  {
    file: 'index.html',
    canonicalRu: SITE_BASE,
    canonicalEn: `${SITE_BASE}en/`,
    title: 'AI Engineering from Scratch',
    description: '502 lessons across 20 phases. Backpropagation, tokenizer, attention, and the agent loop — all by hand, from pure math, before importing any framework. Python, plus TypeScript, Rust, and Julia in dedicated lessons.',
    ogTitle: 'AI Engineering from Scratch',
    ogDescription: '502 lessons across 20 phases. Backpropagation, tokenizer, attention, and the agent loop — all by hand, from pure math, before importing any framework.',
    twDescription: '502 lessons across 20 phases. Build from pure math, by hand.',
    ruTitle: 'Разработка ИИ с нуля',
    ruDescription: '502 урока в 20 фазах. Backpropagation, токенизатор, attention и agent loop — все вручную, из чистой математики, до импорта фреймворков. Python, плюс TypeScript, Rust и Julia в отдельных уроках.',
    // og:title and twitter:title share this text with the JSON-LD Course name.
    ruOgTitle: 'Разработка ИИ с нуля',
    ruOgDescription: '502 урока в 20 фазах. Backpropagation, токенизатор, attention и agent loop — все вручную, из чистой математики, до импорта фреймворков.',
    ruTwDescription: '502 урока в 20 фазах. Стройте из чистой математики, вручную.',
    extraSwaps: [
      ['<meta property="og:locale" content="ru_RU">', '<meta property="og:locale" content="en_US">'],
      ['"description": "502 урока в 20 фазах: математика, модель, тренер, токенизатор и agent loop — все вручную, до импорта фреймворков."',
       '"description": "502 lessons across 20 phases: math, model, trainer, tokenizer, and the agent loop — all by hand, before importing any framework."'],
      ['"inLanguage": "ru"', '"inLanguage": "en"'],
      ['"teaches": "AI engineering, deep learning, трансформеры, LLM, агенты, RAG"',
       '"teaches": "AI engineering, deep learning, transformers, LLMs, agents, RAG"'],
    ],
  },
  {
    file: 'prereqs.html',
    title: 'Roadmap - AI Engineering from Scratch',
    description: 'Interactive prerequisite map for 502 AI Engineering lessons. See which phases depend on which, and plan your learning path.',
    ogTitle: 'Roadmap · AI Engineering from Scratch',
    ogDescription: 'Interactive prerequisite map. See what each phase depends on and what it unlocks next.',
    twDescription: 'Interactive prerequisite map for 20 phases.',
    ruTitle: 'Дорожная карта - AI Engineering from Scratch',
    ruDescription: 'Интерактивная карта предварительных требований для 502 уроков по AI Engineering. Посмотрите, какие фазы от каких зависят, и спланируйте свой путь обучения.',
    ruOgTitle: 'Дорожная карта · AI Engineering from Scratch',
    ruOgDescription: 'Интерактивная карта предварительных требований. Посмотрите, от чего зависит каждая фаза и что она открывает дальше.',
    ruTwDescription: 'Интерактивная карта предварительных требований для 20 фаз.',
  },
];

function replaceOnce(html, from, to, label) {
  if (!html.includes(from)) {
    console.warn(`⚠️  service-page en: expected string not found (${label}) in the source; skipping this swap`);
    return html;
  }
  return html.split(from).join(to);
}

function renderServiceEnPage(html, page) {
  let out = html;
  out = out.replace('<html lang="ru"', '<html lang="en"');
  // Root-relative asset references (href/src="name?v=…" or "name") → one level up.
  for (const asset of SERVICE_ROOT_ASSETS) {
    const re = new RegExp(`((?:href|src)=")(${asset.replace('.', '\\.')})(\\?[^"]*)?"`, 'g');
    out = out.replace(re, `$1../$2$3"`);
  }
  // Canonical + og:url point at the en URL; hreflang alternates stay untouched.
  // index.html canonicals are the site root ('' → '/', en → '/en/'), not the
  // bare filename, so pages can override the default file-based URLs.
  const ruCanonical = page.canonicalRu || `${SITE_BASE}${page.file}`;
  const enCanonical = page.canonicalEn || `${SITE_BASE}en/${page.file}`;
  out = out.replace(`<link rel="canonical" href="${ruCanonical}">`, `<link rel="canonical" href="${enCanonical}">`);
  out = out.replace(`<meta property="og:url" content="${ruCanonical}">`, `<meta property="og:url" content="${enCanonical}">`);
  // Head SEO text → English.
  out = replaceOnce(out, `<title>${page.ruTitle}</title>`, `<title>${page.title}</title>`, `${page.file} title`);
  out = replaceOnce(out, page.ruDescription, page.description, `${page.file} description`);
  out = replaceOnce(out, page.ruOgTitle, page.ogTitle, `${page.file} og:title`);
  out = replaceOnce(out, page.ruOgDescription, page.ogDescription, `${page.file} og:description`);
  out = replaceOnce(out, page.ruTwDescription, page.twDescription, `${page.file} twitter:description`);
  // Arbitrary extra head swaps (og:locale, JSON-LD name/description/inLanguage…).
  for (const [from, to] of page.extraSwaps || []) {
    out = replaceOnce(out, from, to, `${page.file} extra swap`);
  }
  return out;
}

function renderServicePages() {
  let written = 0;
  for (const page of SERVICE_PAGES) {
    const src = fs.readFileSync(path.join(__dirname, page.file), 'utf8');
    const enHtml = renderServiceEnPage(src, page);
    const outPath = path.join(__dirname, 'en', page.file);
    fs.mkdirSync(path.dirname(outPath), { recursive: true });
    fs.writeFileSync(outPath, enHtml, 'utf8');
    written++;
  }
  return written;
}

// ─── Main ────────────────────────────────────────────────────────────
// ─── sitemap.xml ─────────────────────────────────────────────────────
// robots.txt advertises this file; without it that line is a live 404.
// Deterministic (no <lastmod> timestamps) so `--check` stays stable. Every
// page lists both language versions with ru as canonical / x-default.
function renderSitemap(manifest) {
  const esc = u => u.replace(/&/g, '&amp;');
  const urls = [];

  function pushAlternates(ru, en, priority) {
    const alternates =
      `    <xhtml:link rel="alternate" hreflang="ru" href="${esc(ru)}"/>\n` +
      `    <xhtml:link rel="alternate" hreflang="en" href="${esc(en)}"/>\n` +
      `    <xhtml:link rel="alternate" hreflang="x-default" href="${esc(ru)}"/>\n`;
    for (const loc of [ru, en]) {
      urls.push(
        `  <url>\n` +
        `    <loc>${esc(loc)}</loc>\n` +
        alternates +
        `    <priority>${priority}</priority>\n` +
        `  </url>`
      );
    }
  }

  // Single-language service pages (still Russian-only). Canonical forms are the
  // real files (guaranteed to serve under the proxy); the pretty /catalog,
  // /path rewrites resolve to the same content.
  // Bilingual service pages (ru canonical + en alternate).
  pushAlternates(`${SITE_BASE}`, `${SITE_BASE}en/`, '1.0');
  pushAlternates(`${SITE_BASE}catalog.html`, `${SITE_BASE}en/catalog.html`, '0.8');
  pushAlternates(`${SITE_BASE}prereqs.html`, `${SITE_BASE}en/prereqs.html`, '0.7');
  pushAlternates(`${SITE_BASE}glossary.html`, `${SITE_BASE}en/glossary.html`, '0.5');
  for (const phase of manifest.phases) {
    pushAlternates(`${SITE_BASE}${phaseSlug(phase)}/`, `${SITE_BASE}en/${phaseSlug(phase)}/`, '0.7');
    for (const lesson of phase.lessons) {
      const slug = lessonSlug(phase, lesson);
      pushAlternates(`${SITE_BASE}${slug}.html`, `${SITE_BASE}en/${slug}.html`, '0.6');
    }
  }
  return `<?xml version="1.0" encoding="UTF-8"?>\n` +
    `<urlset xmlns="http://www.sitemaps.org/schemas/sitemap/0.9" xmlns:xhtml="http://www.w3.org/1999/xhtml">\n` +
    urls.join('\n') + `\n</urlset>\n`;
}

function main() {
  const checkMode = process.argv.includes('--check');

  const manifest = loadManifest();
  const stats = computeStats(manifest);
  const outputsIndex = renderOutputsIndex(manifest);
  stats.artifacts = Object.values(JSON.parse(outputsIndex).counts).reduce((a, b) => a + b, 0);
  const glossaryTerms = buildBilingualGlossary();

  const targets = [
    { path: DATA_PATH, render: () => renderDataJs(manifest, glossaryTerms) },
    { path: SITEMAP_PATH, render: () => renderSitemap(manifest) },
    { path: OUTPUTS_INDEX_PATH, render: () => outputsIndex },
    { path: README_PATH, render: old => renderReadme(old, manifest, stats) },
    { path: ROADMAP_PATH, render: old => renderRoadmap(old, manifest, stats) },
    ...HTML_COUNT_PAGES.map(p => ({ path: p, render: old => renderHtmlCounts(old, stats) })),
    ...SKILL_COUNT_PAGES.map(p => ({ path: p, render: old => renderSkillCounts(old, stats) })),
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

  // Prerendered pages are gitignored deploy artifacts: rebuilt on every
  // deploy, skipped in --check mode (CI verifies only tracked files).
  if (!checkMode) {
    const lessonPages = renderLessonPages(manifest);
    const hubPages = renderPhaseHubs(manifest);
    const servicePages = renderServicePages();
    console.log(`\n🗂  Prerendered ${lessonPages} lesson pages + ${hubPages} phase hubs + ${servicePages} en service pages`);
  }

  console.log(`\n📊 Stats:`);
  console.log(`   Phases: ${manifest.phases.length}`);
  console.log(`   Lessons: ${stats.total} (${stats.complete} complete, ${stats.quizzes} with quizzes)`);
  console.log(`   Hours: ~${Math.round(stats.lessonHours)} lessons + ~${Math.round(stats.capstoneHours)} capstones`);
  console.log(`   Artifacts: ${stats.artifacts}`);
  console.log(`   Glossary terms: ${glossaryTerms.length}`);

  printReadiness(manifest, stats);

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
