(function () {
  var PAGE_LANG = location.pathname.indexOf('/en/') !== -1 ? 'en' : 'ru';
  document.documentElement.lang = PAGE_LANG;

  // Landing-page UI strings. Static prose carries data-i18n / data-i18n-html /
  // data-nav attributes; the dynamic phase grid and modal read from T below.
  var I18N = {
    ru: {
      logoSuffix: 'AI / С НУЛЯ',
      figLabel: 'FIG_000 · учебная программа v1.0 · 2026',
      mastheadTitle: 'Разработка ИИ<br>с нуля',
      tagline: '502 уроков. 20 фаз (разделов). Каждый алгоритм создается на основе базовых математических принципов, прежде чем импортируется какой-либо фреймворк.',
      attribution: 'Разработано Рохитом Гумаре и другими участниками. Запускайте на своем компьютере.',
      prefaceEyebrow: 'Как это устроено',
      prefaceBody: '<p>Большинство материалов по искусственному интеллекту представляют собой разрозненные фрагменты. Здесь — статья, там — пост о тонкой настройке, где-то — яркая демонстрация работы агента. Фрагменты редко складываются в общую картину. Вы выпускаете чат-бота, но не можете объяснить его кривую потерь. Вы подключаете функцию к агенту, но не можете сказать, как работает механизм внимания в модели, которая ее вызывает.</p>' +
        '<p>Эта учебная программа — основа основ. 20 этапов, 502 уроков. Основной язык — Python; отдельные уроки добавляют TypeScript, Rust и Julia. С одной стороны — линейная алгебра, с другой — автономные роевые системы. Каждый алгоритм сначала строится на основе базовых математических принципов. Обратное распространение. Токенизатор. Внимание. Цикл агента. К тому времени, когда появляется PyTorch, вы уже знаете, как он работает.</p>' +
        '<p>Каждый урок проходит по одному и тому же сценарию: прочитайте задачу, посчитайте, напишите код, запустите тест, сохраните результат. Никаких пятиминутных видео, копирования и вставки, никаких подсказок. Бесплатно, с открытым исходным кодом и для запуска на вашем ноутбуке.</p>',
      statProgress: 'Текущий прогресс',
      statLessonsReady: 'Готовых уроков', statPhases: 'Разделов', statLanguages: 'Языков', statGlossary: 'Терминов в глоссарии',
      tocTitle: 'Учебная программа · 20 разделов · 502 уроков',
      tocSubtitle: 'Нажмите на этап, чтобы открыть его уроки. Каждый из них будет готов, когда будут написаны все математические задачи, код и тесты.',
      legendDone: 'Готово', legendProgress: 'В процессе', legendPlanned: 'Запланировано',
      colophonEyebrow: 'Колофон',
      colophonBody: 'Весь учебный план доступен на GitHub. Клонируйте его, создавайте форки, учитесь в удобном для вас темпе. Никаких платных подписок, никакой регистрации. К каждому уроку прилагается исполняемый код на Python, TypeScript, Rust или Julia — в зависимости от того, какой язык лучше всего подходит для раскрытия темы. Исходный репозиторий по первой ссылке, репозиторий с переводом по второй.',
      modalFooterNote: 'Прогресс сохраняется только в браузере',
      modalReset: 'Сбросить прогресс',
      report: 'Отчет',
      nav: { 'contents': 'Содержание', 'catalog': 'Каталог', 'roadmap': 'Дорожная карта', 'glossary': 'Глоссарий', 'home': 'Главная' },
      phaseWord: 'ФАЗА', read: 'Читать', review: 'Повторить', completed: 'завершено',
      youCompleted: 'Вы прошли этот урок', markDone: 'Отметить пройденным', markNotDone: 'Снять отметку',
      combines: 'Объединяет: ', resetConfirm: 'Очистить весь локальный прогресс (ответы на тесты и пройденные уроки)? Это необратимо.'
    },
    en: {
      logoSuffix: 'AI / FROM SCRATCH',
      figLabel: 'FIG_000 · curriculum v1.0 · 2026',
      mastheadTitle: 'AI Engineering<br>from Scratch',
      tagline: '502 lessons. 20 phases. Every algorithm is built from first mathematical principles before any framework is imported.',
      attribution: 'Created by Rohit Gumare and other contributors. Run it on your own machine.',
      prefaceEyebrow: 'How this works',
      prefaceBody: '<p>Most AI material is scattered fragments. An article here, a fine-tuning post there, a flashy agent demo somewhere else. The pieces rarely add up to the whole. You ship a chatbot but can\'t explain its loss curve. You wire a function into an agent but can\'t say how attention works in the model that calls it.</p>' +
        '<p>This curriculum is the foundation of the foundations. 20 phases, 502 lessons. The primary language is Python; individual lessons add TypeScript, Rust, and Julia. Linear algebra on one end, autonomous swarm systems on the other. Every algorithm is first built from first mathematical principles. Backpropagation. The tokenizer. Attention. The agent loop. By the time PyTorch shows up, you already know how it works.</p>' +
        '<p>Every lesson follows the same script: read the problem, do the math, write the code, run the test, save the result. No five-minute videos, no copy-paste, no hand-waving. Free, open source, and built to run on your laptop.</p>',
      statProgress: 'Current progress',
      statLessonsReady: 'Lessons ready', statPhases: 'Phases', statLanguages: 'Languages', statGlossary: 'Glossary terms',
      tocTitle: 'Curriculum · 20 phases · 502 lessons',
      tocSubtitle: 'Click a phase to open its lessons. Each is ready once all its math problems, code, and tests are written.',
      legendDone: 'Done', legendProgress: 'In progress', legendPlanned: 'Planned',
      colophonEyebrow: 'Colophon',
      colophonBody: 'The entire curriculum is on GitHub. Clone it, fork it, learn at your own pace. No paid subscriptions, no sign-up. Every lesson ships runnable code in Python, TypeScript, Rust, or Julia — whichever best reveals the topic. The source repository is the first link, the translated repository the second.',
      modalFooterNote: 'Progress is saved only in your browser',
      modalReset: 'Reset progress',
      report: 'Report',
      nav: { 'contents': 'Contents', 'catalog': 'Catalog', 'roadmap': 'Roadmap', 'glossary': 'Glossary', 'home': 'Home' },
      phaseWord: 'PHASE', read: 'Read', review: 'Review', completed: 'completed',
      youCompleted: 'You completed this lesson', markDone: 'Mark complete', markNotDone: 'Mark as not done',
      combines: 'Combines: ', resetConfirm: 'Clear all your local progress (quiz answers and completed lessons)? This cannot be undone.'
    }
  };
  var T = I18N[PAGE_LANG];

  function nm(o) { return (PAGE_LANG === 'en' && o.name_en) ? o.name_en : o.name; }
  function dsc(o) { return (PAGE_LANG === 'en' && o.desc_en) ? o.desc_en : o.desc; }

  function localizeStatic() {
    document.querySelectorAll('[data-i18n]').forEach(function (el) {
      var k = el.getAttribute('data-i18n');
      if (T[k] != null) el.textContent = T[k];
    });
    document.querySelectorAll('[data-i18n-html]').forEach(function (el) {
      var k = el.getAttribute('data-i18n-html');
      if (T[k] != null) el.innerHTML = T[k];
    });
    document.querySelectorAll('[data-nav]').forEach(function (el) {
      var k = el.getAttribute('data-nav');
      if (T.nav && T.nav[k] != null) el.textContent = T.nav[k];
    });
    var langBtn = document.getElementById('langToggle');
    if (langBtn) {
      langBtn.textContent = PAGE_LANG.toUpperCase();
      langBtn.addEventListener('click', function () {
        window.location.href = PAGE_LANG === 'en' ? '../index.html' : 'en/index.html';
      });
    }
  }

  var root = document.documentElement;
  var stored = localStorage.getItem('theme');
  if (stored) {
    root.setAttribute('data-theme', stored);
  } else if (window.matchMedia && window.matchMedia('(prefers-color-scheme: dark)').matches) {
    root.setAttribute('data-theme', 'dark');
  } else {
    root.setAttribute('data-theme', 'light');
  }
  updateThemeIcon();

  document.addEventListener('DOMContentLoaded', function () {
    localizeStatic();
    initThemeToggle();
    populateStats();
    renderPhases();
    initStaggerIndex();
    initModal();
    initCopyButton();
    initSmoothScroll();
    initFadeObserver();
    initScrollExplode();
  });

  function updateThemeIcon() {
    var icon = document.getElementById('themeIcon');
    if (!icon) return;
    var theme = root.getAttribute('data-theme');
    icon.textContent = theme === 'light' ? 'N' : 'D';
  }

  function initThemeToggle() {
    var btn = document.getElementById('themeToggle');
    if (!btn) return;
    btn.addEventListener('click', function () {
      var current = root.getAttribute('data-theme');
      var next = current === 'light' ? 'dark' : 'light';
      root.setAttribute('data-theme', next);
      localStorage.setItem('theme', next);
      updateThemeIcon();
    });
    updateThemeIcon();
  }

  function computeStats() {
    var totalLessons = 0;
    var completeLessons = 0;
    var hasProgress = !!window.AIFSProgress;
    for (var i = 0; i < PHASES.length; i++) {
      var lessons = PHASES[i].lessons;
      totalLessons += lessons.length;
      for (var j = 0; j < lessons.length; j++) {
        var staticDone = lessons[j].status === 'complete';
        var userDone = false;
        if (hasProgress && lessons[j].url) {
          var lp = window.AIFSProgress.extractPath(lessons[j].url);
          if (lp) userDone = window.AIFSProgress.isLessonComplete(lp);
        }
        if (staticDone || userDone) completeLessons++;
      }
    }
    var completePhases = 0;
    for (var p = 0; p < PHASES.length; p++) {
      if (PHASES[p].status === 'complete') completePhases++;
    }
    return {
      lessons: totalLessons,
      phases: PHASES.length,
      complete: completeLessons,
      completePhases: completePhases
    };
  }

  function setBar(selector, pct) {
    var el = document.querySelector(selector);
    if (!el) return;
    var clamped = Math.max(0, Math.min(100, pct));
    el.setAttribute('data-target-pct', clamped.toFixed(1));
    if (el.classList.contains('in-view') || !window.IntersectionObserver) {
      el.style.setProperty('--bar-pct', clamped.toFixed(1) + '%');
    } else {
      el.style.setProperty('--bar-pct', '0%');
    }
  }

  function populateStats() {
    var stats = computeStats();
    var pct = stats.lessons > 0 ? (stats.complete / stats.lessons) * 100 : 0;
    var phasePct = stats.phases > 0 ? (stats.completePhases / stats.phases) * 100 : 0;
    var glossaryCount = (typeof GLOSSARY !== 'undefined') ? GLOSSARY.length : 0;

    setText('[data-stat="complete-frac"]', stats.complete + ' / ' + stats.lessons);
    setText('[data-stat="phases-frac"]', stats.completePhases + ' / ' + stats.phases);
    setText('[data-stat="glossary-count"]', String(glossaryCount));
    setBar('[data-bar="complete"]', pct);
    setBar('[data-bar="phases"]', phasePct);
    setBar('[data-bar="languages"]', 100);
    setBar('[data-bar="glossary"]', glossaryCount > 0 ? 100 : 0);
  }

  function setText(selector, value) {
    var el = document.querySelector(selector);
    if (el) el.textContent = value;
  }

  function renderPhases() {
    var grid = document.getElementById('phasesGrid');
    if (!grid) return;
    var hasProgress = !!window.AIFSProgress;
    var html = '';
    for (var i = 0; i < PHASES.length; i++) {
      var p = PHASES[i];
      var total = p.lessons.length;
      var done = 0;
      for (var j = 0; j < p.lessons.length; j++) {
        var staticDone = p.lessons[j].status === 'complete';
        var userDone = false;
        if (hasProgress && p.lessons[j].url) {
          var lp = window.AIFSProgress.extractPath(p.lessons[j].url);
          if (lp) userDone = window.AIFSProgress.isLessonComplete(lp);
        }
        if (staticDone || userDone) done++;
      }
      var statusClass = p.status.replace(/ /g, '-');
      var roman = toRoman(p.id);
      var num = String(p.id).padStart(2, '0');
      html += '<div class="toc-row" data-phase="' + i + '">';
      html += '<span class="toc-num">' + roman + '.</span>';
      html += '<div><span class="toc-status ' + statusClass + '"></span><span class="toc-name">' + escapeHtml(nm(p)) + '</span></div>';
      html += '<span class="toc-meta">' + done + ' / ' + total + '</span>';
      html += '<span class="toc-meta">' + num + '</span>';
      html += '</div>';
    }
    grid.innerHTML = html;
  }

  function toRoman(num) {
    var lookup = [
      ['M', 1000], ['CM', 900], ['D', 500], ['CD', 400],
      ['C', 100], ['XC', 90], ['L', 50], ['XL', 40],
      ['X', 10], ['IX', 9], ['V', 5], ['IV', 4], ['I', 1]
    ];
    var n = parseInt(num, 10);
    if (isNaN(n) || n <= 0) return String(num);
    var out = '';
    for (var k = 0; k < lookup.length; k++) {
      while (n >= lookup[k][1]) {
        out += lookup[k][0];
        n -= lookup[k][1];
      }
    }
    return out;
  }

  function initModal() {
    var overlay = document.getElementById('modalOverlay');
    var closeBtn = document.getElementById('modalClose');
    if (!overlay || !closeBtn) return;

    document.addEventListener('click', function (e) {
      var row = e.target.closest('.toc-row, .phase-card');
      if (row) {
        var idx = parseInt(row.getAttribute('data-phase'), 10);
        if (!isNaN(idx)) openModal(idx);
      }
    });

    closeBtn.addEventListener('click', closeModal);
    overlay.addEventListener('click', function (e) {
      if (e.target === overlay) closeModal();
    });
    document.addEventListener('keydown', function (e) {
      if (e.key === 'Escape') closeModal();
    });

    var resetBtn = document.getElementById('modalReset');
    if (resetBtn) {
      resetBtn.addEventListener('click', function () {
        if (!window.AIFSProgress) return;
        var ok = window.confirm(T.resetConfirm);
        if (!ok) return;
        window.AIFSProgress.reset();
      });
    }
  }

  var currentPhaseIdx = -1;

  function openModal(idx) {
    var p = PHASES[idx];
    if (!p) return;
    currentPhaseIdx = idx;

    document.getElementById('modalPhaseNum').textContent = T.phaseWord + ' ' + String(p.id).padStart(2, '0');
    document.getElementById('modalTitle').textContent = nm(p);
    document.getElementById('modalDesc').textContent = dsc(p);

    renderModalLessons(p);

    document.getElementById('modalOverlay').classList.add('open');
    document.body.style.overflow = 'hidden';
  }

  function renderModalLessons(p) {
    var container = document.getElementById('modalLessons');
    if (!container) return;

    var hasProgress = !!window.AIFSProgress;
    var userDone = 0;
    var html = '';

    for (var i = 0; i < p.lessons.length; i++) {
      var l = p.lessons[i];
      var lessonPath = l.path || ''; // progress.js key
      var lessonUrl = l.slug ? l.slug + '.html' : ''; // prerendered page (ru)
      var userComplete = hasProgress && lessonPath && window.AIFSProgress.isLessonComplete(lessonPath);
      if (userComplete) userDone++;

      var statusClass = l.status.replace(/ /g, '-');
      if (userComplete) statusClass = 'complete';

      // In /en/index.html, lesson slugs resolve to the en lesson pages in the
      // same /en/ directory, so the relative href needs no language prefix.
      html += '<div class="modal-lesson' + (userComplete ? ' user-done' : '') + '">';
      html += '<span class="modal-lesson-status ' + statusClass + '"' + (userComplete ? ' title="' + escapeAttr(T.youCompleted) + '"' : '') + '></span>';
      if (lessonUrl) {
        html += '<a href="' + lessonUrl + '">' + escapeHtml(nm(l)) + '</a>';
      } else {
        html += '<a>' + escapeHtml(nm(l)) + '</a>';
      }
      html += '<span class="modal-lesson-type" data-type="' + escapeHtml(l.type) + '"' + (l.combines ? ' title="' + escapeAttr(T.combines + l.combines) + '"' : '') + '>' + escapeHtml(l.type) + '</span>';
      html += '<span class="modal-lesson-lang">' + escapeHtml(l.lang) + '</span>';

      var actionHtml = '';
      if ((l.status === 'complete' || userComplete) && lessonUrl) {
        actionHtml = '<a href="' + lessonUrl + '" class="modal-lesson-read">' + (userComplete ? T.review : T.read) + '</a>';
      }
      var toggleHtml = '';
      if (hasProgress && lessonPath) {
        toggleHtml = '<button type="button" class="modal-lesson-toggle' + (userComplete ? ' done' : '') + '" data-path="' + lessonPath + '" title="' + escapeAttr(userComplete ? T.markNotDone : T.markDone) + '" aria-label="' + escapeAttr(userComplete ? T.markNotDone : T.markDone) + '">' + (userComplete ? '✓' : '+') + '</button>';
      }
      html += (actionHtml || '<span class="modal-lesson-read-placeholder" aria-hidden="true"></span>') + toggleHtml;
      html += '</div>';
    }

    container.innerHTML = html;

    var toggles = container.querySelectorAll('.modal-lesson-toggle');
    for (var t = 0; t < toggles.length; t++) {
      toggles[t].addEventListener('click', function (e) {
        e.preventDefault();
        e.stopPropagation();
        var path = this.getAttribute('data-path');
        if (!path || !window.AIFSProgress) return;
        if (window.AIFSProgress.isLessonComplete(path)) {
          window.AIFSProgress.unmarkLessonComplete(path);
        } else {
          window.AIFSProgress.markLessonComplete(path);
        }
      });
    }

    var progEl = document.getElementById('modalProgress');
    var barEl = document.getElementById('modalProgressBar');
    var barFill = document.getElementById('modalProgressBarFill');
    if (hasProgress && p.lessons.length > 0) {
      var pct = Math.round((userDone / p.lessons.length) * 100);
      if (progEl) {
        progEl.style.display = '';
        progEl.innerHTML = '<span class="modal-progress-count">' + userDone + ' / ' + p.lessons.length + '</span> <span class="modal-progress-label">' + escapeHtml(T.completed) + '</span> <span class="modal-progress-pct">' + pct + '%</span>';
      }
      if (barEl && barFill) {
        barEl.style.display = '';
        barFill.style.width = pct + '%';
      }
    } else {
      if (progEl) progEl.style.display = 'none';
      if (barEl) barEl.style.display = 'none';
    }
  }

  if (window.AIFSProgress) {
    window.AIFSProgress.onChange(function () {
      if (currentPhaseIdx >= 0 && PHASES[currentPhaseIdx]) {
        renderModalLessons(PHASES[currentPhaseIdx]);
      }
      populateStats();
      renderPhases();
    });
  }

  function closeModal() {
    document.getElementById('modalOverlay').classList.remove('open');
    document.body.style.overflow = '';
  }

  function initCopyButton() {
    var btn = document.getElementById('copyBtn');
    var code = document.getElementById('cloneCmd');
    if (!btn || !code) return;
    var originalLabel = btn.textContent;
    var revertTimer = null;
    btn.addEventListener('click', function () {
      navigator.clipboard.writeText(code.textContent).then(function () {
        btn.textContent = '✓';
        if (revertTimer) clearTimeout(revertTimer);
        revertTimer = setTimeout(function () { btn.textContent = originalLabel; }, 1500);
      });
    });
  }

  function initSmoothScroll() {
    document.querySelectorAll('a[href^="#"]').forEach(function (link) {
      link.addEventListener('click', function (e) {
        var target = document.querySelector(link.getAttribute('href'));
        if (target) {
          e.preventDefault();
          target.scrollIntoView({ behavior: 'smooth' });
        }
      });
    });
  }

  function initFadeObserver() {
    var prefersReduced = window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches;

    if (!window.IntersectionObserver || prefersReduced) {
      document.querySelectorAll('.reveal, .fade-in, .stat-row-bar').forEach(function (el) {
        el.classList.add('in-view', 'visible');
        var target = el.getAttribute('data-target-pct');
        if (target !== null) el.style.setProperty('--bar-pct', target + '%');
      });
      return;
    }

    document.body.classList.add('js-anim');

    var els = document.querySelectorAll('.reveal, .fade-in, .stat-row-bar, .ascii-rule, .toc-row');
    if (!els.length) return;
    var observer = new IntersectionObserver(function (entries) {
      for (var i = 0; i < entries.length; i++) {
        if (entries[i].isIntersecting) {
          var el = entries[i].target;
          el.classList.add('in-view', 'visible');
          var target = el.getAttribute('data-target-pct');
          if (target !== null) {
            el.style.setProperty('--bar-pct', target + '%');
          }
          observer.unobserve(el);
        }
      }
    }, { threshold: 0.12, rootMargin: '0px 0px -40px 0px' });
    for (var i = 0; i < els.length; i++) {
      observer.observe(els[i]);
    }
  }

  function initStaggerIndex() {
    var rows = document.querySelectorAll('.toc-list .toc-row');
    for (var i = 0; i < rows.length; i++) {
      rows[i].style.setProperty('--stagger-delay', (i * 30) + 'ms');
    }
  }

  function initScrollExplode() {
    var containers = document.querySelectorAll('[data-svg-explode]');
    if (!containers.length) return;
    if (window.matchMedia && window.matchMedia('(prefers-reduced-motion: reduce)').matches) {
      for (var c = 0; c < containers.length; c++) applyExplode(containers[c], 1);
      return;
    }

    var ticking = false;
    function update() {
      ticking = false;
      var vh = window.innerHeight || document.documentElement.clientHeight;
      for (var i = 0; i < containers.length; i++) {
        var rect = containers[i].getBoundingClientRect();
        var startEdge = vh;
        var endEdge = vh * 0.35;
        var raw = (startEdge - rect.top) / (startEdge - endEdge);
        var progress = Math.max(0, Math.min(1, raw));
        progress = 1 - Math.pow(1 - progress, 3);
        applyExplode(containers[i], progress);
      }
    }
    function onScroll() {
      if (ticking) return;
      ticking = true;
      window.requestAnimationFrame(update);
    }
    window.addEventListener('scroll', onScroll, { passive: true });
    window.addEventListener('resize', onScroll);
    update();
  }

  function applyExplode(container, progress) {
    // Each layer / label animates over its own window in [stagger_start, stagger_start + window].
    // Sequential reveal: layer N waits for layer N-1 to mostly settle before starting.
    var STAGGER_DENOM = 720; // higher → wider gaps between layer entrances
    var WINDOW = 0.55;       // each layer's local animation duration as fraction of global progress

    function localProgress(staggerAttr) {
      var stagger = parseFloat(staggerAttr) || 0;
      var start = stagger / STAGGER_DENOM;
      var local = (progress - start) / WINDOW;
      if (local < 0) local = 0;
      if (local > 1) local = 1;
      // ease-out cubic on the local segment
      return 1 - Math.pow(1 - local, 3);
    }

    var layers = container.querySelectorAll('.explode-layer');
    for (var i = 0; i < layers.length; i++) {
      var final = parseFloat(layers[i].getAttribute('data-final')) || 0;
      var lp = localProgress(layers[i].getAttribute('data-stagger'));
      var dy = -final * lp;
      layers[i].setAttribute('transform', 'translate(0, ' + dy.toFixed(2) + ')');
      layers[i].setAttribute('opacity', lp.toFixed(3));
    }
    var labels = container.querySelectorAll('.explode-label');
    for (var j = 0; j < labels.length; j++) {
      var final2 = parseFloat(labels[j].getAttribute('data-final')) || 0;
      var lp2 = localProgress(labels[j].getAttribute('data-stagger'));
      var dy2 = -final2 * lp2;
      labels[j].setAttribute('transform', 'translate(0, ' + dy2.toFixed(2) + ')');
      labels[j].setAttribute('opacity', lp2.toFixed(3));
    }
  }

  function escapeHtml(str) {
    var div = document.createElement('div');
    div.textContent = str == null ? '' : str;
    return div.innerHTML;
  }

  function escapeAttr(str) {
    return String(str == null ? '' : str)
      .replace(/&/g, '&amp;').replace(/"/g, '&quot;').replace(/'/g, '&#39;')
      .replace(/</g, '&lt;').replace(/>/g, '&gt;');
  }
})();
