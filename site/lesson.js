/**
 * Client-side hydration for prerendered lesson pages.
 *
 * The article HTML (markdown, quizzes, sidebar, bottom nav) is baked in at
 * build time by site/build.js + site/lesson-render.js. This script only adds
 * interactivity: theme/lang toggles, TOC + scrollspy, code copy buttons,
 * mermaid rendering, quiz answers + progress, and the AI panels that pull
 * live data from the GitHub API.
 *
 * Expects `window.LESSON = { path, slug, lang, root }` injected by the page:
 *   path — repo lesson path, e.g. "phases/01-math-foundations/01-linear-algebra-intuition"
 *          (also the key used by progress.js — unchanged across the URL migration)
 *   slug — pretty slug, e.g. "math-foundations/linear-algebra-intuition"
 *   lang — "ru" | "en" (matches the page language, not a stored preference)
 *   root — relative prefix to the site root, e.g. "../" or "../../"
 */
(function () {
  var LESSON = window.LESSON;
  if (!LESSON) return;

  // Phase hub pages set path: '' — chrome still hydrates (theme, sidebar,
  // lang toggle), lesson-only features (progress, panels, quizzes) no-op.
  var lessonPath = LESSON.path;
  var docLang = LESSON.lang === 'en' ? 'en' : 'ru';
  var ROOT = LESSON.root || '';
  try { localStorage.setItem('aifs-lesson-lang', docLang); } catch (e) {}

  var RU = {
    copy: 'Копировать', copied: 'Скопировано!',
    copyCommand: 'Копировать команду',
    tocHeader: 'На этой странице',
    diagram: 'Диаграмма',
    diagramRendering: 'Рендеринг диаграммы...',
    diagramFailed: 'Не удалось отрендерить диаграмму.',
    correct: 'правильно',
    openOnGithub: 'Открыть на GitHub',
    openLessonOnGithub: 'Открыть урок на GitHub',
    outputsTitle: 'Что дает этот урок',
    outputsSubtitle: 'Промпты, скиллы и артефакты, которые можно использовать прямо сейчас',
    outputsLoading: 'Загрузка артефактов...',
    descLoading: 'Загрузка описания...',
    badgePrompt: 'Промпт', badgeSkill: 'Скилл', badgeArtifact: 'Артефакт',
    install: 'Установить',
    promptHint: 'Вставьте в Claude, Cursor, Codex, OpenClaw, Hermes или любого агента, который читает промпты',
    codeTitle: 'Запустите код',
    codeSubtitle: 'Исполняемые файлы из этого урока',
    codeLoading: 'Загрузка файлов с кодом...',
    pathTitle: 'Путь обучения',
    phase: 'Фаза',
    prevLessons: 'предыдущих уроков', nextLessons: 'следующих уроков',
    phaseProgress: function (done, total) { return 'Завершено ' + done + ' из ' + total + ' уроков в этой фазе'; },
    phaseNextCallout: function (num, name) { return 'Можно переходить к фазе ' + num + ': ' + name; },
    continueTitle: 'Продолжить обучение',
    phaseFinished: '✅ Эта фаза завершена!',
    allPhaseLessons: function (num) { return 'Все уроки фазы ' + num; },
    fullCatalog: 'Полный каталог курса',
    continueCallout: 'Запустите <code>/find-your-level</code> в Claude, Cursor, Codex, OpenClaw, Hermes или любом агенте с установленным SkillKit, чтобы получить персональный путь обучения'
  };
  var EN = {
    copy: 'Copy', copied: 'Copied!',
    copyCommand: 'Copy command',
    tocHeader: 'On this page',
    diagram: 'Diagram',
    diagramRendering: 'Rendering diagram...',
    diagramFailed: 'Failed to render the diagram.',
    correct: 'correct',
    openOnGithub: 'Open on GitHub',
    openLessonOnGithub: 'Open the lesson on GitHub',
    outputsTitle: 'What this lesson ships',
    outputsSubtitle: 'Prompts, skills and artifacts you can use right away',
    outputsLoading: 'Loading artifacts...',
    descLoading: 'Loading description...',
    badgePrompt: 'Prompt', badgeSkill: 'Skill', badgeArtifact: 'Artifact',
    install: 'Install',
    promptHint: 'Paste into Claude, Cursor, Codex, OpenClaw, Hermes or any agent that reads prompts',
    codeTitle: 'Run the code',
    codeSubtitle: 'Runnable files from this lesson',
    codeLoading: 'Loading code files...',
    pathTitle: 'Learning path',
    phase: 'Phase',
    prevLessons: 'previous lessons', nextLessons: 'next lessons',
    phaseProgress: function (done, total) { return 'Completed ' + done + ' of ' + total + ' lessons in this phase'; },
    phaseNextCallout: function (num, name) { return 'Ready to move on to Phase ' + num + ': ' + name; },
    continueTitle: 'Continue learning',
    phaseFinished: '✅ This phase is complete!',
    allPhaseLessons: function (num) { return 'All lessons of Phase ' + num; },
    fullCatalog: 'Full course catalog',
    continueCallout: 'Run <code>/find-your-level</code> in Claude, Cursor, Codex, OpenClaw, Hermes or any agent with SkillKit installed to get a personal learning path'
  };
  var S = docLang === 'en' ? EN : RU;

  function lessonHref(slug) {
    return ROOT + (docLang === 'en' ? 'en/' : '') + slug + '.html';
  }

  // ── Theme ──────────────────────────────────────────────────────────
  var root = document.documentElement;
  updateThemeIcon();

  function updateThemeIcon() {
    var icon = document.getElementById('themeIcon');
    if (!icon) return;
    icon.textContent = root.getAttribute('data-theme') === 'light' ? 'N' : 'D';
  }

  var themeBtn = document.getElementById('themeToggle');
  if (themeBtn) {
    themeBtn.addEventListener('click', function () {
      var next = root.getAttribute('data-theme') === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      try { localStorage.setItem('theme', next); } catch (e) {}
      updateThemeIcon();
      updateMermaidThemeAndRerender();
    });
  }

  // ── Course structure (from data.js) ────────────────────────────────
  var flatLessons = [];
  var currentPhaseIndex = -1;
  var currentLessonIndex = -1;

  if (typeof PHASES !== 'undefined') {
    for (var i = 0; i < PHASES.length; i++) {
      var p = PHASES[i];
      for (var j = 0; j < p.lessons.length; j++) {
        var l = p.lessons[j];
        flatLessons.push({
          phaseIndex: i,
          lessonIndex: j,
          phaseName: p.name,
          phaseId: p.id,
          lessonName: l.name,
          slug: l.slug,
          status: l.status,
          type: l.type,
          lang: l.lang,
          url: l.url,
          path: l.path,
          isReadable: l.status === 'complete' || !!l.url
        });
        if (l.path === lessonPath) {
          currentPhaseIndex = i;
          currentLessonIndex = flatLessons.length - 1;
        }
      }
    }
  }

  // ── Chrome ─────────────────────────────────────────────────────────
  function initSidebarToggle() {
    var toggle = document.getElementById('sidebarToggle');
    var sidebar = document.getElementById('lessonSidebar');
    if (!toggle || !sidebar) return;

    toggle.addEventListener('click', function () {
      sidebar.classList.toggle('open');
    });

    document.addEventListener('click', function (e) {
      if (window.innerWidth <= 900 && sidebar.classList.contains('open')) {
        if (!sidebar.contains(e.target) && e.target !== toggle) {
          sidebar.classList.remove('open');
        }
      }
    });
  }

  function initScrollProgress() {
    var bar = document.getElementById('scrollProgress');
    if (!bar) return;
    window.addEventListener('scroll', function () {
      var scrollTop = window.scrollY || document.documentElement.scrollTop;
      var docHeight = document.documentElement.scrollHeight - window.innerHeight;
      var pct = docHeight > 0 ? (scrollTop / docHeight) * 100 : 0;
      bar.style.width = pct + '%';
    }, { passive: true });
  }

  function initLangToggle() {
    var btn = document.getElementById('langToggle');
    if (!btn) return;
    btn.textContent = docLang.toUpperCase();
    btn.addEventListener('click', function () {
      var next = docLang === 'ru' ? 'en' : 'ru';
      try { localStorage.setItem('aifs-lesson-lang', next); } catch (e) {}
      window.location.href = ROOT + (next === 'en' ? 'en/' : '') + LESSON.slug + '.html';
    });
  }

  function initAnchorScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
      link.addEventListener('click', function (e) {
        if (link.classList && link.classList.contains('toc-link')) return;
        var target = document.querySelector(link.getAttribute('href'));
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  }

  // ── Table of contents ──────────────────────────────────────────────
  function buildTOC() {
    var sidebar = document.getElementById('tocSidebar');
    if (!sidebar) return;

    var article = document.querySelector('.lesson-article');
    if (!article) {
      sidebar.innerHTML = '';
      sidebar.setAttribute('aria-hidden', 'true');
      return;
    }

    var headings = article.querySelectorAll('h2, h3');
    if (!headings.length) {
      sidebar.innerHTML = '';
      sidebar.setAttribute('aria-hidden', 'true');
      return;
    }

    var used = Object.create(null);

    function ensureHeadingId(heading, idx) {
      if (!heading) return '';

      var base = '';
      if (heading.id) {
        base = heading.id;
      } else {
        base = String(heading.textContent || '').toLowerCase().trim().replace(/[^a-zа-яё0-9\s-]/gi, '').replace(/\s+/g, '-').replace(/-+/g, '-');
      }
      if (!base) base = 'section-' + String(idx + 1);

      var candidate = base;
      var n = 2;

      while (used[candidate] || (document.getElementById(candidate) && document.getElementById(candidate) !== heading)) {
        candidate = base + '-' + n;
        n++;
      }

      used[candidate] = true;
      heading.id = candidate;
      return candidate;
    }

    var html = '<div class="toc-header">' + S.tocHeader + '</div>';
    html += '<nav aria-label="' + S.tocHeader + '"><ul class="toc-nav">';
    for (var i = 0; i < headings.length; i++) {
      var heading = headings[i];
      var level = heading.tagName.toLowerCase();
      var id = ensureHeadingId(heading, i);
      if (!id) continue;
      html += '<li class="toc-' + level + '"><a class="toc-link" href="#' + id + '">' + escapeHtml(heading.textContent) + '</a></li>';
    }
    html += '</ul></nav>';

    sidebar.innerHTML = html;
    sidebar.setAttribute('aria-hidden', 'false');

    sidebar.querySelectorAll('.toc-link').forEach(function (a) {
      a.addEventListener('click', function (e) {
        e.preventDefault();
        var target = document.getElementById(a.getAttribute('href').slice(1));
        if (target) {
          target.scrollIntoView({ behavior: 'smooth' });
          target.setAttribute('tabindex', '-1');
          target.focus({ preventScroll: true });
          target.addEventListener('blur', function blurHandler() {
            target.removeAttribute('tabindex');
            target.removeEventListener('blur', blurHandler);
          });
          history.replaceState(null, '', '#' + target.id);
        }
      });
    });

    initScrollspy(headings, sidebar);
  }

  function initScrollspy(headings, sidebar) {
    var links = sidebar.querySelectorAll('.toc-link');
    if (!links.length) return;

    var idToLink = {};
    links.forEach(function (link) {
      idToLink[link.getAttribute('href').slice(1)] = link;
    });

    function setActive(id) {
      links.forEach(function (link) {
        link.classList.toggle('active', link.getAttribute('href').slice(1) === id);
      });
      var active = idToLink[id];
      if (active && typeof active.scrollIntoView === 'function') {
        var rect = active.getBoundingClientRect();
        var sRect = sidebar.getBoundingClientRect();
        if (rect.top < sRect.top || rect.bottom > sRect.bottom) {
          active.scrollIntoView({ block: 'nearest' });
        }
      }
    }

    function fallbackByScroll() {
      var current = null;
      for (var i = 0; i < headings.length; i++) {
        if (headings[i].getBoundingClientRect().top <= 120) {
          current = headings[i].id;
        }
      }
      if (current) setActive(current);
    }

    if ('IntersectionObserver' in window) {
      var visible = {};
      var obs = new IntersectionObserver(function (entries) {
        entries.forEach(function (entry) {
          visible[entry.target.id] = entry.isIntersecting;
        });
        var first = null;
        for (var i = 0; i < headings.length; i++) {
          if (visible[headings[i].id]) { first = headings[i].id; break; }
        }
        if (first) setActive(first);
        else fallbackByScroll();
      }, { rootMargin: '-60px 0px -70% 0px', threshold: 0 });
      Array.prototype.forEach.call(headings, function (h) {
        if (h.id) obs.observe(h);
      });
    } else {
      window.addEventListener('scroll', fallbackByScroll, { passive: true });
    }
  }

  // ── Code copy ──────────────────────────────────────────────────────
  function initCodeCopy() {
    document.querySelectorAll('.code-copy').forEach(function (btn) {
      btn.addEventListener('click', function () {
        var code = btn.getAttribute('data-code');
        navigator.clipboard.writeText(code).then(function () {
          btn.textContent = S.copied;
          setTimeout(function () { btn.textContent = S.copy; }, 1500);
        });
      });
    });
  }

  // ── Mermaid ────────────────────────────────────────────────────────
  function renderMermaidBlocks() {
    if (!document.querySelector('pre.mermaid.mermaid-source')) return;
    var fontsReady = (document.fonts && document.fonts.load)
      ? Promise.all([
          document.fonts.load('400 13px "JetBrains Mono"'),
          document.fonts.load('500 13px "JetBrains Mono"'),
          document.fonts.load('700 13px "JetBrains Mono"')
        ]).catch(function () {})
      : Promise.resolve();
    var check = setInterval(function () {
      if (!window._mermaidReady) return;
      clearInterval(check);
      fontsReady.then(function () {
        rerenderMermaidBlocks();
        bindMermaidToolbar();
        initMermaidModal();
      });
    }, 100);
    setTimeout(function () { clearInterval(check); }, 10000);
  }

  function mermaidTheme() {
    return document.documentElement.getAttribute('data-theme') === 'light' ? 'default' : 'dark';
  }

  function updateMermaidThemeAndRerender() {
    if (!window._mermaidReady) return;
    try {
      window._mermaidReady.initialize({
        startOnLoad: false,
        theme: mermaidTheme(),
        fontFamily: 'JetBrains Mono, ui-monospace, monospace',
        themeVariables: { fontSize: '13px' },
        flowchart: { useMaxWidth: true, htmlLabels: true, nodeSpacing: 40, rankSpacing: 50, padding: 12 }
      });
    } catch (_) {}
    rerenderMermaidBlocks();
    rerenderMermaidModal();
  }

  function rerenderMermaidBlocks() {
    if (!window._mermaidReady) return;
    var sources = document.querySelectorAll('pre.mermaid.mermaid-source');
    sources.forEach(function (pre) {
      var idxMatch = pre.id && pre.id.match(/mermaid-(\d+)/);
      var idx = idxMatch ? idxMatch[1] : null;
      if (!idx) return;
      var target = document.getElementById('mermaid-render-' + idx);
      if (!target) return;
      var def = pre.textContent;
      target.innerHTML = '<div class="panel-loading">' + S.diagramRendering + '</div>';

      var token = ++mermaidRenderSeq;
      target._mermaidRenderToken = token;

      window._mermaidReady.render('mermaid-' + idx + '-svg', def).then(function (result) {
        if (target._mermaidRenderToken !== token) return;
        target.innerHTML = result.svg;
      }).catch(function () {
        if (target._mermaidRenderToken !== token) return;
        target.innerHTML = '<p style="color:var(--text-muted);font-style:italic;">' + S.diagramFailed + '</p>';
      });
    });
  }

  function bindMermaidToolbar() {
    document.querySelectorAll('.mermaid-expand').forEach(function (btn) {
      if (btn._bound) return;
      btn._bound = true;
      btn.addEventListener('click', function () {
        var idx = btn.getAttribute('data-mermaid-index');
        openMermaidModal(idx, btn);
      });
    });
  }

  function initMermaidModal() {
    var overlay = document.getElementById('mermaidModalOverlay');
    var closeBtn = document.getElementById('mermaidModalClose');
    if (!overlay || !closeBtn) return;
    if (overlay._bound) return;
    overlay._bound = true;

    closeBtn.addEventListener('click', closeMermaidModal);
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) closeMermaidModal();
    });
    document.addEventListener('keydown', function (e) {
      if (!overlay.classList.contains('open')) return;
      if (e.key === 'Escape') {
        e.preventDefault();
        closeMermaidModal();
        return;
      }

      if (e.key === 'Tab') {
        var focusables = getMermaidModalFocusables();
        if (!focusables.length) return;
        var first = focusables[0];
        var last = focusables[focusables.length - 1];
        var active = document.activeElement;

        if (e.shiftKey) {
          if (active === first || active === overlay) {
            e.preventDefault();
            last.focus();
          }
        } else {
          if (active === last) {
            e.preventDefault();
            first.focus();
          }
        }
      }
    });
  }

  var lastMermaidTrigger = null;
  var openMermaidIndex = null;
  var mermaidRenderSeq = 0;

  function getMermaidModalFocusables() {
    var modal = document.querySelector('#mermaidModalOverlay .mermaid-modal');
    if (!modal) return [];
    return Array.prototype.slice.call(modal.querySelectorAll('button, [href], input, select, textarea, [tabindex]:not([tabindex="-1"])'))
      .filter(function (el) { return !el.disabled && el.offsetParent !== null; });
  }

  function openMermaidModal(idx, triggerEl) {
    var overlay = document.getElementById('mermaidModalOverlay');
    var body = document.getElementById('mermaidModalBody');
    var center = document.getElementById('mermaidModalCenter');
    var title = document.getElementById('mermaidModalTitle');
    var closeBtn = document.getElementById('mermaidModalClose');
    var pre = document.getElementById('mermaid-' + idx);
    if (!overlay || !body || !center || !pre || !closeBtn) return;

    lastMermaidTrigger = triggerEl || document.activeElement;
    openMermaidIndex = idx;

    if (title) title.textContent = S.diagram;

    overlay.classList.add('open');
    overlay.setAttribute('aria-hidden', 'false');
    document.body.style.overflow = 'hidden';

    closeBtn.focus();

    rerenderMermaidModal();
  }

  function rerenderMermaidModal() {
    var overlay = document.getElementById('mermaidModalOverlay');
    var center = document.getElementById('mermaidModalCenter');
    if (!overlay || !center) return;
    if (!overlay.classList.contains('open')) return;
    if (!openMermaidIndex) return;
    if (!window._mermaidReady) return;

    var pre = document.getElementById('mermaid-' + openMermaidIndex);
    if (!pre) return;
    var def = pre.textContent;

    center.innerHTML = '<div class="panel-loading">' + S.diagramRendering + '</div>';
    var token = ++mermaidRenderSeq;
    center._mermaidRenderToken = token;

    window._mermaidReady.render('mermaid-modal-svg', def).then(function (result) {
      if (center._mermaidRenderToken !== token) return;
      center.innerHTML = result.svg;
    }).catch(function () {
      if (center._mermaidRenderToken !== token) return;
      center.innerHTML = '<p style="color:var(--text-muted);font-style:italic;">' + S.diagramFailed + '</p>';
    });
  }

  function closeMermaidModal() {
    var overlay = document.getElementById('mermaidModalOverlay');
    var center = document.getElementById('mermaidModalCenter');
    if (!overlay) return;
    overlay.classList.remove('open');
    overlay.setAttribute('aria-hidden', 'true');
    if (center) center.innerHTML = '';
    document.body.style.overflow = '';
    openMermaidIndex = null;

    if (lastMermaidTrigger && typeof lastMermaidTrigger.focus === 'function') {
      lastMermaidTrigger.focus();
    }
  }

  // ── In-article quizzes (prerendered markup, live behavior) ────────
  window.handleQuizClick = function (el) {
    var questionDiv = el.closest('.quiz-question');
    if (questionDiv.classList.contains('answered')) return;
    applyQuizAnswer(questionDiv, parseInt(el.getAttribute('data-index')), true);
  };

  function applyQuizAnswer(questionDiv, chosen, persist) {
    if (!questionDiv || isNaN(chosen)) return;
    questionDiv.classList.add('answered');

    var correct = parseInt(questionDiv.getAttribute('data-correct'));
    var options = questionDiv.querySelectorAll('.quiz-option');

    for (var i = 0; i < options.length; i++) {
      options[i].classList.add('disabled');
      if (i === correct) {
        options[i].classList.add('correct');
        var mc = options[i].querySelector('.quiz-marker');
        if (mc) mc.textContent = '✓';
      }
    }
    if (chosen !== correct) {
      var picked = questionDiv.querySelector('.quiz-option[data-index="' + chosen + '"]');
      if (picked) {
        picked.classList.add('incorrect');
        var mp = picked.querySelector('.quiz-marker');
        if (mp) mp.textContent = '✗';
      }
    }

    var exp = questionDiv.querySelector('.quiz-explanation');
    if (exp) exp.style.display = 'block';

    if (persist && window.AIFSProgress && lessonPath) {
      window.AIFSProgress.recordAnswer(lessonPath, questionDiv.id, chosen, chosen === correct);
    }

    updateSectionScore(questionDiv.closest('.quiz-section'));
    maybeMarkLessonComplete();
  }

  function updateSectionScore(section) {
    if (!section) return;
    var allQs = section.querySelectorAll('.quiz-question');
    var answeredQs = section.querySelectorAll('.quiz-question.answered');
    if (allQs.length !== answeredQs.length) return;
    var correctCount = 0;
    for (var j = 0; j < allQs.length; j++) {
      if (!allQs[j].querySelector('.quiz-option.incorrect')) correctCount++;
    }
    var scoreEl = section.querySelector('.quiz-score');
    if (scoreEl) {
      scoreEl.textContent = correctCount + '/' + allQs.length + ' ' + S.correct;
      scoreEl.style.display = 'block';
    }
  }

  function maybeMarkLessonComplete() {
    if (!window.AIFSProgress || !lessonPath) return;
    var sections = document.querySelectorAll('.quiz-section');
    if (!sections.length) return;
    for (var s = 0; s < sections.length; s++) {
      var qs = sections[s].querySelectorAll('.quiz-question');
      var answered = sections[s].querySelectorAll('.quiz-question.answered');
      if (qs.length === 0 || qs.length !== answered.length) return;
      if (sections[s].querySelector('.quiz-option.incorrect')) return;
    }
    window.AIFSProgress.markLessonComplete(lessonPath);
  }

  function restoreQuizAnswers() {
    if (!window.AIFSProgress || !lessonPath) return;
    var lp = window.AIFSProgress.getLessonProgress(lessonPath);
    if (!lp || !lp.answers) return;
    for (var qid in lp.answers) {
      var q = document.getElementById(qid);
      if (q && !q.classList.contains('answered')) {
        applyQuizAnswer(q, lp.answers[qid].picked, false);
      }
    }
  }

  // ── AI panels (live data from GitHub) ──────────────────────────────
  function renderAIPanels() {
    var container = document.getElementById('aiPanels');
    if (!container || !lessonPath) return;

    renderOutputsPanel(container);
    renderCodePanel(container);
    if (docLang !== 'en') renderQuizPanel(container); // hardcoded quiz bank is RU-only
    renderLearningPathPanel(container);
    renderContinuePanel(container);
  }

  var ghApiCache = {};

  function ghApiFetch(apiPath, cb) {
    if (ghApiCache[apiPath]) {
      cb(null, ghApiCache[apiPath]);
      return;
    }
    fetch('https://api.github.com/repos/stabuev/ai-engineering-from-scratch/contents/' + apiPath, {
      headers: { 'Accept': 'application/vnd.github.v3+json' }
    }).then(function (res) {
      if (!res.ok) throw new Error(res.status);
      return res.json();
    }).then(function (data) {
      ghApiCache[apiPath] = data;
      cb(null, data);
    }).catch(function (err) {
      cb(err, null);
    });
  }

  function classifyOutput(name) {
    var lower = name.toLowerCase();
    if (lower.indexOf('prompt') >= 0) return 'prompt';
    if (lower.indexOf('skill') >= 0) return 'skill';
    return 'other';
  }

  function langIcon(filename) {
    var ext = filename.split('.').pop().toLowerCase();
    var map = { py: 'PY', python: 'PY', ts: 'TS', js: 'JS', rust: 'RS', rs: 'RS', jl: 'JL', julia: 'JL', sh: 'SH', bash: 'SH' };
    return map[ext] || '··';
  }

  function langCommand(filename) {
    var ext = filename.split('.').pop().toLowerCase();
    if (ext === 'py') return 'python';
    if (ext === 'ts') return 'npx tsx';
    if (ext === 'js') return 'node';
    if (ext === 'rs') return 'cargo run --';
    if (ext === 'jl') return 'julia';
    if (ext === 'sh') return 'bash';
    return 'run';
  }

  function formatSize(bytes) {
    if (bytes < 1024) return bytes + ' B';
    return (bytes / 1024).toFixed(1) + ' KB';
  }

  function extractFrontmatterDesc(content) {
    var match = content.match(/^---[\s\S]*?description:\s*(.+?)[\r\n]/m);
    if (match) return match[1].replace(/^["']|["']$/g, '').trim();
    var firstLine = content.split('\n').find(function (l) { return l.trim() && !l.match(/^[#-]/) && !l.match(/^---/); });
    return firstLine ? firstLine.trim().substring(0, 120) : '';
  }

  function renderOutputsPanel(container) {
    var panel = document.createElement('div');
    panel.className = 'ai-panel';
    panel.innerHTML = '<div class="ai-panel-header"><div class="ai-panel-icon">O</div><div class="ai-panel-title">' + S.outputsTitle + '</div></div><div class="ai-panel-subtitle">' + S.outputsSubtitle + '</div><div id="outputsContent" class="panel-loading">' + S.outputsLoading + '</div>';
    container.appendChild(panel);

    ghApiFetch(lessonPath + '/outputs', function (err, data) {
      var el = document.getElementById('outputsContent');
      if (err || !data || !Array.isArray(data) || data.length === 0) {
        el.innerHTML = '<div class="panel-fallback"><a href="https://github.com/stabuev/ai-engineering-from-scratch/tree/main/' + lessonPath + '" target="_blank" rel="noopener">' + S.openLessonOnGithub + '</a></div>';
        return;
      }

      var html = '<div class="output-cards">';
      data.forEach(function (file, idx) {
        var type = classifyOutput(file.name);
        var badgeClass = type;
        var badgeText = type === 'prompt' ? S.badgePrompt : type === 'skill' ? S.badgeSkill : S.badgeArtifact;
        var installId = 'install-hint-' + idx;
        var installHint = '';
        if (type === 'prompt') {
          installHint = S.promptHint;
        } else if (type === 'skill') {
          var skillName = file.name.replace(/\.md$/, '').replace(/^skill-/, '');
          installHint = 'npx skillkit install ' + skillName;
        }

        html += '<div class="output-card">';
        html += '<div class="output-badge ' + badgeClass + '">' + badgeText + '</div>';
        html += '<div class="output-card-name">' + escapeHtml(file.name) + '</div>';
        html += '<div class="output-desc" id="desc-' + idx + '">' + S.descLoading + '</div>';
        html += '<div class="output-actions">';
        html += '<a class="output-btn" href="' + file.html_url + '" target="_blank" rel="noopener">' + S.openOnGithub + '</a>';
        if (installHint) {
          html += '<button class="output-btn" onclick="var h=document.getElementById(\'' + installId + '\');h.classList.toggle(\'visible\')">' + S.install + '</button>';
        }
        html += '</div>';
        if (installHint) {
          html += '<div class="output-install-hint" id="' + installId + '">' + escapeHtml(installHint) + '</div>';
        }
        html += '</div>';

        fetch(file.download_url).then(function (r) { return r.text(); }).then(function (content) {
          var descEl = document.getElementById('desc-' + idx);
          if (descEl) {
            var desc = extractFrontmatterDesc(content);
            descEl.textContent = desc || file.name;
          }
        }).catch(function () {
          var descEl = document.getElementById('desc-' + idx);
          if (descEl) descEl.textContent = file.name;
        });
      });
      html += '</div>';
      el.innerHTML = html;
    });
  }

  function renderCodePanel(container) {
    var panel = document.createElement('div');
    panel.className = 'ai-panel';
    panel.innerHTML = '<div class="ai-panel-header"><div class="ai-panel-icon">C</div><div class="ai-panel-title">' + S.codeTitle + '</div></div><div class="ai-panel-subtitle">' + S.codeSubtitle + '</div><div id="codeContent" class="panel-loading">' + S.codeLoading + '</div>';
    container.appendChild(panel);

    ghApiFetch(lessonPath + '/code', function (err, data) {
      var el = document.getElementById('codeContent');
      if (err || !data || !Array.isArray(data) || data.length === 0) {
        el.innerHTML = '<div class="panel-fallback"><a href="https://github.com/stabuev/ai-engineering-from-scratch/tree/main/' + lessonPath + '" target="_blank" rel="noopener">' + S.openLessonOnGithub + '</a></div>';
        return;
      }

      var html = '<div class="code-cards">';
      data.forEach(function (file) {
        var icon = langIcon(file.name);
        var cmd = langCommand(file.name);
        var runCmd = cmd + ' ' + lessonPath + '/code/' + file.name;
        html += '<div class="code-card">';
        html += '<div class="code-card-header"><span class="code-lang-icon">' + icon + '</span><span class="code-card-name">' + escapeHtml(file.name) + '</span></div>';
        html += '<div class="code-card-size">' + formatSize(file.size) + '</div>';
        html += '<div class="code-card-run">' + escapeHtml(runCmd) + '</div>';
        html += '<div class="code-card-actions">';
        html += '<a class="code-card-btn" href="' + file.html_url + '" target="_blank" rel="noopener">' + S.openOnGithub + '</a>';
        html += '<button class="code-card-btn code-card-copy" data-command="' + escapeAttr(runCmd) + '">' + S.copyCommand + '</button>';
        html += '</div>';
        html += '</div>';
      });
      html += '</div>';
      el.innerHTML = html;
      el.querySelectorAll('.code-card-copy').forEach(function (btn) {
        btn.addEventListener('click', function () {
          var command = btn.getAttribute('data-command');
          navigator.clipboard.writeText(command).then(function () {
            btn.textContent = S.copied;
            setTimeout(function () { btn.textContent = S.copyCommand; }, 1500);
          });
        });
      });
    });
  }

  function getQuizQuestions() {
    var pathLower = (lessonPath || '').toLowerCase();

    if (pathLower.indexOf('math') >= 0 || pathLower.indexOf('linear-algebra') >= 0 || pathLower.indexOf('calculus') >= 0) {
      return [
        { q: 'Что измеряет скалярное произведение двух векторов?', opts: ['Их сумму', 'Насколько они сонаправлены', 'Их векторное произведение', 'Расстояние между ними'], answer: 1, explain: 'Скалярное произведение измеряет сходство или направленную согласованность двух векторов. Если оно равно нулю, векторы ортогональны.' },
        { q: 'Куда указывает градиент функции?', opts: ['В сторону минимума', 'В сторону наискорейшего возрастания', 'К ближайшей седловой точке', 'В случайном направлении'], answer: 1, explain: 'Градиент всегда указывает направление наискорейшего роста. Градиентный спуск движется в противоположную сторону, чтобы искать минимумы.' },
        { q: 'Матрица формы (3, 5), умноженная на матрицу формы (5, 2), какую форму дает?', opts: ['(3, 2)', '(5, 5)', '(3, 5)', '(2, 3)'], answer: 0, explain: 'Умножение матриц: (m, n) x (n, p) = (m, p). Поэтому (3, 5) x (5, 2) = (3, 2).' }
      ];
    }

    if (pathLower.indexOf('ml') >= 0 || pathLower.indexOf('regression') >= 0 || pathLower.indexOf('classification') >= 0 || pathLower.indexOf('supervised') >= 0) {
      return [
        { q: 'Для чего нужна функция потерь в машинном обучении?', opts: ['Генерировать данные', 'Измерять, насколько ошибочны предсказания', 'Выбирать признаки', 'Разбивать данные'], answer: 1, explain: 'Функция потерь количественно оценивает различие между предсказанными и фактическими значениями. Оптимизатор минимизирует это значение во время обучения.' },
        { q: 'Зачем мы делим данные на обучающую и тестовую выборки?', opts: ['Чтобы экономить память', 'Чтобы оценить обобщение на невидимых данных', 'Чтобы ускорить обучение', 'Чтобы уменьшить размер датасета'], answer: 1, explain: 'Тестовая выборка имитирует невидимые данные и показывает, запомнила ли модель обучающие данные или выучила обобщаемые закономерности.' },
        { q: 'Что означает переобучение (overfitting)?', opts: ['Модель слишком простая', 'Модель запоминает обучающие данные, но плохо работает на новых данных', 'Модель обучается слишком медленно', 'Потери слишком низкие'], answer: 1, explain: 'Переобучение возникает, когда модель хорошо работает на обучающих данных, но плохо на новых: она выучила шум вместо сигнала.' }
      ];
    }

    if (pathLower.indexOf('deep-learning') >= 0 || pathLower.indexOf('neural') >= 0 || pathLower.indexOf('backprop') >= 0 || pathLower.indexOf('activation') >= 0) {
      return [
        { q: 'Что вычисляет обратное распространение ошибки (backpropagation)?', opts: ['Прямые предсказания', 'Градиент функции потерь по каждому весу', 'Скорость обучения', 'Новые обучающие данные'], answer: 1, explain: 'Backpropagation использует правило цепочки, чтобы вычислить, насколько каждый вес внес вклад в ошибку, а затем веса корректируются.' },
        { q: 'Зачем нейронным сетям нужны нелинейные функции активации?', opts: ['Чтобы ускорить обучение', 'Без них стек слоев эквивалентен одному линейному слою', 'Чтобы снизить использование памяти', 'Чтобы нормализовать выходы'], answer: 1, explain: 'Без нелинейностей любая композиция линейных слоев сводится к одному линейному преобразованию. Активации вроде ReLU позволяют сети изучать сложные закономерности.' },
        { q: 'Что контролирует скорость обучения (learning rate)?', opts: ['Сколько эпох обучаться', 'Размер каждого шага обновления весов', 'Количество слоев', 'Размер батча'], answer: 1, explain: 'Скорость обучения масштабирует градиентное обновление. Слишком большая вызывает расходимость, слишком маленькая делает обучение медленным или застревающим.' }
      ];
    }

    if (pathLower.indexOf('llm') >= 0 || pathLower.indexOf('transformer') >= 0 || pathLower.indexOf('attention') >= 0 || pathLower.indexOf('tokeniz') >= 0 || pathLower.indexOf('fine-tun') >= 0) {
      return [
        { q: 'Что self-attention позволяет делать трансформеру?', opts: ['Обрабатывать токены по порядку', 'Взвешивать важность каждого токена относительно каждого другого токена', 'Уменьшать размер словаря', 'Сжимать модель'], answer: 1, explain: 'Self-attention вычисляет попарные оценки релевантности по всем позициям, позволяя модели связывать далекие токены без рекуррентности.' },
        { q: 'Почему LLM используют токенизаторы вместо сырых символов?', opts: ['Символы слишком большие', 'Токены сжимают частые паттерны в отдельные единицы, уменьшая длину последовательности', 'Токенизаторы быстрее обучать', 'Символы нельзя эмбеддить'], answer: 1, explain: 'Подсловные токенизаторы вроде BPE балансируют размер словаря и длину последовательности: частые слова становятся отдельными токенами, а редкие обрабатываются частями.' },
        { q: 'В чем ключевое различие между pre-training и fine-tuning?', opts: ['Pre-training использует размеченные данные', 'Pre-training изучает общий язык; fine-tuning адаптирует модель к конкретной задаче', 'Fine-tuning использует больше данных', 'Разницы нет'], answer: 1, explain: 'Pre-training изучает языковые закономерности на огромном неразмеченном корпусе. Fine-tuning берет эту основу и специализирует ее на меньшем наборе данных под задачу.' }
      ];
    }

    if (pathLower.indexOf('rag') >= 0 || pathLower.indexOf('retrieval') >= 0 || pathLower.indexOf('embed') >= 0 || pathLower.indexOf('vector') >= 0) {
      return [
        { q: 'В чем основная идея RAG?', opts: ['Обучить более крупную модель', 'Перед генерацией ответа извлечь релевантный контекст', 'Использовать больше GPU', 'Уменьшить размер модели'], answer: 1, explain: 'RAG (Retrieval-Augmented Generation) привязывает ответы LLM к найденным документам, снижая галлюцинации и позволяя обновлять знания без дообучения.' },
        { q: 'Что производят модели эмбеддингов?', opts: ['Краткие пересказы текста', 'Плотные векторные представления текста', 'Количество токенов', 'Исправления грамматики'], answer: 1, explain: 'Модели эмбеддингов отображают текст в векторы фиксированной размерности, где семантическая близость соответствует геометрической близости.' },
        { q: 'Зачем использовать косинусное сходство для сравнения эмбеддингов?', opts: ['Это единственная метрика', 'Оно измеряет угловое сходство независимо от длины вектора', 'Оно быстрее скалярного произведения', 'Оно работает только с целыми числами'], answer: 1, explain: 'Косинусное сходство нормирует длину и фокусируется на направлении. Два текста об одной теме будут иметь высокое косинусное сходство независимо от длины.' }
      ];
    }

    if (pathLower.indexOf('agent') >= 0 || pathLower.indexOf('tool') >= 0 || pathLower.indexOf('mcp') >= 0) {
      return [
        { q: 'Чем AI-агент отличается от простого чатбота?', opts: ['Агенты быстрее', 'Агенты могут автономно выполнять действия и использовать инструменты', 'Агенты используют более крупные модели', 'Разницы нет'], answer: 1, explain: 'У агентов есть цикл: наблюдать, решать, действовать. Они могут вызывать инструменты, читать файлы, искать в вебе и связывать несколько шагов для выполнения сложных задач.' },
        { q: 'Что такое MCP (Model Context Protocol)?', opts: ['Формат обучения модели', 'Стандартизированный протокол для подключения AI-моделей к инструментам и источникам данных', 'Алгоритм сжатия', 'Фреймворк для тестирования'], answer: 1, explain: 'MCP задает универсальный интерфейс между AI-ассистентами и внешними инструментами/данными, заменяя разовые интеграции стандартным протоколом.' },
        { q: 'Почему tool-use важен для LLM?', opts: ['Он снижает стоимость', 'Он позволяет LLM получать информацию в реальном времени и выполнять действия за пределами генерации текста', 'Он делает ответы короче', 'Он полностью устраняет галлюцинации'], answer: 1, explain: 'Инструменты расширяют LLM за пределы даты отсечения обучающих данных: позволяют делать актуальные запросы, вычисления, выполнять код и взаимодействовать с внешними системами.' }
      ];
    }

    if (pathLower.indexOf('eval') >= 0 || pathLower.indexOf('safety') >= 0 || pathLower.indexOf('alignment') >= 0 || pathLower.indexOf('deploy') >= 0) {
      return [
        { q: 'Почему evals критически важны для production AI-систем?', opts: ['Чтобы сэкономить деньги', 'Чтобы объективно измерять качество до и после изменений', 'Чтобы сделать модель больше', 'Evals необязательны'], answer: 1, explain: 'Evals - это тесты AI Engineering. Без них нельзя понять, улучшило изменение качество или привело к регрессии. Они должны запускаться на каждом изменении кода.' },
        { q: 'Что означает "alignment" в AI safety?', opts: ['Выравнивание текста на экране', 'Обеспечение того, чтобы AI-системы действовали согласно человеческим намерениям и ценностям', 'Ускорение моделей', 'Использование одних и тех же обучающих данных'], answer: 1, explain: 'Alignment помогает гарантировать, что по мере роста возможностей AI-системы остаются полезными, честными и безопасными, действуя в соответствии с целями человека.' },
        { q: 'Что такое guardrail в контексте развернутых LLM?', opts: ['Физический барьер', 'Проверка, которая предотвращает вредные, нерелевантные или нарушающие политику ответы', 'Резервная модель', 'Слой кеширования'], answer: 1, explain: 'Guardrails фильтруют входы и выходы во время выполнения, перехватывая токсичность, prompt injection, утечки PII и другие риски до того, как они попадут к пользователю.' }
      ];
    }

    return [
      { q: 'Что означает "from scratch" в этом курсе?', opts: ['Не использовать компьютер', 'Строить каждую концепцию через собственную реализацию, а не только читать теорию', 'Начинать с языка ассемблера', 'Использовать только ручку и бумагу'], answer: 1, explain: 'Курс следует методологии Build-Use-Ship: сначала понять, построив самому, затем применить, затем оформить результат как реальный артефакт.' },
      { q: 'Почему этот курс объединяет математику, ML и инженерную практику?', opts: ['Чтобы сделать его длиннее', 'Потому что реальная AI-инженерия требует всех трех компонентов для построения production-систем', 'Математика нужна просто для развлечения', 'Инженерия необязательна'], answer: 1, explain: 'Production AI-системам нужны математические основы (для понимания), знание ML (для моделирования) и инженерные навыки (для развертывания и надежности).' },
      { q: 'Что такое шаг "Ship" во фреймворке Build-Use-Ship?', opts: ['Отправка физических товаров', 'Создание переиспользуемого артефакта: промпта, skill или инструмента на основе изученного', 'Публикация статьи', 'Удаление вашего кода'], answer: 2, explain: 'Шаг Ship превращает обучение во что-то осязаемое: промпт, skill-файл, MCP tool или CLI-утилиту, которую другие люди или вы в будущем смогут сразу использовать.' }
    ];
  }

  function renderQuizPanel(container) {
    var questions = getQuizQuestions();
    var panel = document.createElement('div');
    panel.className = 'ai-panel';

    var phaseSlugForSkill = '';
    if (currentLessonIndex >= 0 && typeof PHASES !== 'undefined') {
      var curPhase = PHASES[flatLessons[currentLessonIndex].phaseIndex];
      phaseSlugForSkill = curPhase.dir || '';
    }

    var html = '<div class="ai-panel-header"><div class="ai-panel-icon">Q</div><div class="ai-panel-title">Проверьте понимание</div></div>';
    html += '<div class="ai-panel-subtitle">Уложилось ли в голове?</div>';
    html += '<div class="quiz-container" id="quizContainer">';

    var letters = ['A', 'B', 'C', 'D'];
    questions.forEach(function (q, qi) {
      html += '<div class="quiz-question" id="quiz-q-' + qi + '">';
      html += '<div class="quiz-question-num">Вопрос ' + (qi + 1) + ' из ' + questions.length + '</div>';
      html += '<div class="quiz-question-text">' + escapeHtml(q.q) + '</div>';
      html += '<div class="quiz-options">';
      q.opts.forEach(function (opt, oi) {
        html += '<button class="quiz-option" data-qi="' + qi + '" data-oi="' + oi + '" onclick="handleQuizAnswer(' + qi + ',' + oi + ')">';
        html += '<span class="opt-letter">' + letters[oi] + '</span>';
        html += '<span>' + escapeHtml(opt) + '</span>';
        html += '</button>';
      });
      html += '</div>';
      html += '<div class="quiz-explanation" id="quiz-explain-' + qi + '">' + escapeHtml(q.explain) + '</div>';
      html += '</div>';
    });

    html += '<div class="quiz-score" id="quizScore"><div class="quiz-score-number" id="quizScoreNum">0/' + questions.length + '</div><div class="quiz-score-label">Ответьте на все вопросы, чтобы увидеть результат</div></div>';
    html += '<div class="quiz-deeper">Нужен более глубокий тест? Запустите <code>/check-understanding ' + escapeHtml(phaseSlugForSkill) + '</code> в Claude, Cursor, Codex, OpenClaw, Hermes или любом агенте с установленным SkillKit</div>';
    html += '</div>';

    panel.innerHTML = html;
    container.appendChild(panel);

    window._quizData = questions;
    window._quizAnswered = 0;
    window._quizCorrect = 0;
  }

  window.handleQuizAnswer = function (qi, oi) {
    var questions = window._quizData;
    if (!questions) return;

    var qEl = document.getElementById('quiz-q-' + qi);
    if (!qEl) return;
    var opts = qEl.querySelectorAll('.quiz-option');
    if (opts[0].classList.contains('disabled')) return;

    var correct = questions[qi].answer;
    opts.forEach(function (opt, i) {
      opt.classList.add('disabled');
      if (i === oi) {
        opt.classList.add('selected');
        opt.classList.add(i === correct ? 'correct' : 'wrong');
      }
      if (i === correct) {
        opt.classList.add('is-answer');
      }
    });

    var explain = document.getElementById('quiz-explain-' + qi);
    if (explain) explain.classList.add('visible');

    window._quizAnswered++;
    if (oi === correct) window._quizCorrect++;

    if (window._quizAnswered === questions.length) {
      var scoreEl = document.getElementById('quizScore');
      var scoreNum = document.getElementById('quizScoreNum');
      if (scoreEl && scoreNum) {
        scoreNum.textContent = window._quizCorrect + '/' + questions.length;
        scoreEl.querySelector('.quiz-score-label').textContent =
          window._quizCorrect === questions.length ? 'Идеальный результат!' :
          window._quizCorrect >= 2 ? 'Отличная работа!' : 'Продолжайте разбираться!';
        scoreEl.classList.add('visible');
      }
    }
  };

  function renderLearningPathPanel(container) {
    if (currentLessonIndex < 0 || typeof PHASES === 'undefined') return;

    var panel = document.createElement('div');
    panel.className = 'ai-panel';

    var current = flatLessons[currentLessonIndex];
    var phaseIdx = current.phaseIndex;
    var phase = PHASES[phaseIdx];
    var lessonsInPhase = [];
    var currentIdxInPhase = -1;

    for (var i = 0; i < flatLessons.length; i++) {
      if (flatLessons[i].phaseIndex === phaseIdx) {
        if (i === currentLessonIndex) currentIdxInPhase = lessonsInPhase.length;
        lessonsInPhase.push({ flat: flatLessons[i], flatIdx: i });
      }
    }

    var html = '<div class="ai-panel-header"><div class="ai-panel-icon">P</div><div class="ai-panel-title">' + S.pathTitle + '</div></div>';
    html += '<div class="ai-panel-subtitle">' + S.phase + ' ' + String(phase.id).padStart(2, '0') + ': ' + escapeHtml(phase.name) + '</div>';

    // Окно: 2 урока до текущего, текущий и 2 после; максимум 5 видимых.
    var windowSize = 5;
    var startIdx = Math.max(0, currentIdxInPhase - 2);
    var endIdx = Math.min(lessonsInPhase.length, startIdx + windowSize);
    startIdx = Math.max(0, endIdx - windowSize);
    var visible = lessonsInPhase.slice(startIdx, endIdx);
    var hiddenBefore = startIdx;
    var hiddenAfter = lessonsInPhase.length - endIdx;

    html += '<div class="learning-timeline">';
    if (hiddenBefore > 0) {
      html += '<span class="timeline-ellipsis" title="' + hiddenBefore + ' ' + S.prevLessons + '">&laquo; ' + hiddenBefore + '</span>';
      html += '<div class="timeline-line done"></div>';
    }
    visible.forEach(function (item, vIdx) {
      var idx = startIdx + vIdx;
      var isCurrent = idx === currentIdxInPhase;
      var isPrev = idx < currentIdxInPhase;
      var cls = isCurrent ? 'current' : isPrev ? 'prev' : '';

      if (vIdx > 0) {
        var lineCls = idx <= currentIdxInPhase ? (idx === currentIdxInPhase ? 'active' : 'done') : '';
        html += '<div class="timeline-line ' + lineCls + '"></div>';
      }

      if (item.flat.isReadable && item.flat.slug) {
        html += '<a class="timeline-item ' + cls + '" href="' + lessonHref(item.flat.slug) + '" title="' + escapeAttr(item.flat.lessonName) + '">';
      } else {
        html += '<span class="timeline-item ' + cls + ' disabled" title="' + escapeAttr(item.flat.lessonName) + '">';
      }
      html += '<div class="timeline-dot"></div>';
      html += '<div class="timeline-label">' + escapeHtml(item.flat.lessonName) + '</div>';
      html += (item.flat.isReadable && item.flat.slug) ? '</a>' : '</span>';
    });
    if (hiddenAfter > 0) {
      html += '<div class="timeline-line"></div>';
      html += '<span class="timeline-ellipsis" title="' + hiddenAfter + ' ' + S.nextLessons + '">' + hiddenAfter + ' &raquo;</span>';
    }
    html += '</div>';

    var completedCount = currentIdxInPhase + 1;
    var totalCount = lessonsInPhase.length;
    var pct = Math.round((completedCount / totalCount) * 100);

    html += '<div class="phase-progress-bar"><div class="phase-progress-fill" style="width:' + pct + '%"></div></div>';
    html += '<div class="phase-progress-text">' + S.phaseProgress(completedCount, totalCount) + '</div>';

    if (completedCount >= totalCount && phaseIdx < PHASES.length - 1) {
      var nextPhase = PHASES[phaseIdx + 1];
      html += '<div class="phase-complete-callout visible">' + S.phaseNextCallout(String(nextPhase.id).padStart(2, '0'), escapeHtml(nextPhase.name)) + '</div>';
    }

    panel.innerHTML = html;
    container.appendChild(panel);
  }

  function renderContinuePanel(container) {
    if (currentLessonIndex < 0) return;

    var panel = document.createElement('div');
    panel.className = 'ai-panel';

    var next = null;
    for (var ci = currentLessonIndex + 1; ci < flatLessons.length; ci++) {
      if (flatLessons[ci].isReadable) { next = flatLessons[ci]; break; }
    }
    var current = flatLessons[currentLessonIndex];
    var phase = PHASES[current.phaseIndex];

    var html = '<div class="ai-panel-header"><div class="ai-panel-icon">N</div><div class="ai-panel-title">' + S.continueTitle + '</div></div>';
    html += '<div class="continue-panel">';

    if (!next || !next.slug) {
      html += '<div class="phase-finished">' + S.phaseFinished + '</div>';
    }

    html += '<div class="continue-links">';

    var firstLesson = phase.lessons[0];
    if (firstLesson && firstLesson.slug) {
      html += '<a class="continue-link" href="' + lessonHref(firstLesson.slug) + '">' + S.allPhaseLessons(String(phase.id).padStart(2, '0')) + '</a>';
    }
    html += '<a class="continue-link" href="' + ROOT + 'catalog.html">' + S.fullCatalog + '</a>';
    html += '</div>';

    html += '<div class="continue-callout">' + S.continueCallout + '</div>';
    html += '</div>';

    panel.innerHTML = html;
    container.appendChild(panel);
  }

  function escapeHtml(str) {
    return String(str)
      .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
      .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
  }

  function escapeAttr(str) {
    return escapeHtml(str);
  }

  // ── Boot ───────────────────────────────────────────────────────────
  // Runs last so every module-level `var` above is initialized: the panels
  // render synchronously on hydration (no fetch-then-render step anymore).
  if (window.AIFSProgress && lessonPath) {
    window.AIFSProgress.recordVisit(lessonPath);
  }

  initSidebarToggle();
  initScrollProgress();
  initLangToggle();
  initCodeCopy();
  renderMermaidBlocks();
  renderAIPanels();
  buildTOC();
  initAnchorScroll();
  restoreQuizAnswers();
})();
