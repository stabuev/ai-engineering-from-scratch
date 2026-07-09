/**
 * Shared lesson markdown renderer.
 *
 * Ported from the former inline renderer in site/lesson.html so that
 * site/build.js can prerender every lesson into static HTML at deploy time
 * (full SEO-friendly markup, no client-side fetch). Pure string-in/string-out:
 * no DOM, no network. The client script (site/lesson.js) only hydrates the
 * prerendered markup — it never parses markdown.
 *
 * Exposed API:
 *   renderArticle({ md, lessonPath, strings }) -> { html, title }
 *   buildQuizzesHtml(articleHtml, quizData, strings) -> articleHtml with quizzes
 *   linkifyPrereqs(articleHtml, resolveHref) -> articleHtml with prereq links
 *   escapeHtml / escapeAttr / slugify / extractTitle / extractDescription
 *   STRINGS — per-language UI strings used inside the rendered article
 */
'use strict';

const STRINGS = {
  ru: {
    objectivesTitle: '&#127919; Цели обучения',
    labChallengeTitle: '&#128171; Практическое задание',
    copyCode: 'Копировать',
    mermaidExpand: 'Развернуть',
    quizPreTitle: 'Проверка перед уроком',
    quizPostTitle: 'Тест после урока',
    quizAllTitle: 'Тест',
  },
  en: {
    objectivesTitle: '&#127919; Learning Objectives',
    labChallengeTitle: '&#128171; Lab Challenge',
    copyCode: 'Copy',
    mermaidExpand: 'Expand',
    quizPreTitle: 'Check before the lesson',
    quizPostTitle: 'Quiz after the lesson',
    quizAllTitle: 'Quiz',
  },
};

function escapeHtml(str) {
  return String(str)
    .replace(/&/g, '&amp;').replace(/</g, '&lt;').replace(/>/g, '&gt;')
    .replace(/"/g, '&quot;').replace(/'/g, '&#39;');
}

function escapeAttr(str) {
  return escapeHtml(str);
}

// Keeps Cyrillic so Russian headings get stable ids (used by the TOC and the
// quiz insertion anchors). Latin-only headings keep their previous slugs.
function slugify(text) {
  return String(text).toLowerCase().replace(/[^a-zа-яё0-9]+/g, '-').replace(/(^-|-$)/g, '');
}

function extractTitle(md) {
  const m = md.match(/^# (.+)/m);
  return m ? m[1].trim() : '';
}

function stripInlineMd(text) {
  return text
    .replace(/!\[([^\]]*)\]\([^)]+\)/g, '$1')
    .replace(/\[([^\]]+)\]\([^)]+\)/g, '$1')
    .replace(/\*\*\*|\*\*|\*|`/g, '')
    .replace(/\s+/g, ' ')
    .trim();
}

// Meta description: the first paragraph after the Problem heading (the most
// informative lead), else the motto blockquote, else the first plain paragraph.
function extractDescription(md, maxLen) {
  maxLen = maxLen || 160;
  const lines = md.split('\n');
  let candidate = '';

  const problemIdx = lines.findIndex(l => /^##\s+.*(Проблема|Problem)/i.test(l));
  if (problemIdx >= 0) {
    for (let i = problemIdx + 1; i < lines.length; i++) {
      const t = lines[i].trim();
      if (!t) continue;
      if (/^(#|```|\||>|[-*]\s|\d+\.\s)/.test(t)) break;
      candidate = t;
      break;
    }
  }
  if (!candidate) {
    const motto = lines.find(l => /^>\s+\S/.test(l));
    if (motto) candidate = motto.replace(/^>\s+/, '');
  }
  if (!candidate) {
    candidate = lines.find(l => l.trim() && !/^(#|```|\||>|\*\*)/.test(l.trim())) || '';
  }

  let text = stripInlineMd(candidate);
  if (text.length > maxLen) {
    text = text.slice(0, maxLen - 1).replace(/\s+\S*$/, '') + '…';
  }
  return text;
}

function highlightSyntax(code, lang) {
  let keywords, commentPattern;

  if (lang === 'python' || lang === 'py') {
    keywords = /\b(def|class|return|if|elif|else|for|while|in|import|from|as|with|try|except|finally|raise|yield|lambda|not|and|or|is|None|True|False|print|self|pass|break|continue|assert|global)\b/g;
    commentPattern = /(#[^\n]*)/g;
  } else if (lang === 'julia') {
    keywords = /\b(function|end|if|elseif|else|for|while|in|using|import|return|struct|mutable|abstract|type|module|export|let|const|begin|do|try|catch|finally|throw|true|false|nothing|println)\b/g;
    commentPattern = /(#[^\n]*)/g;
  } else if (lang === 'javascript' || lang === 'js' || lang === 'typescript' || lang === 'ts') {
    keywords = /\b(function|const|let|var|return|if|else|for|while|class|new|this|import|export|from|async|await|try|catch|throw|typeof|instanceof|true|false|null|undefined|console)\b/g;
    commentPattern = /(\/\/[^\n]*|\/\*[\s\S]*?\*\/)/g;
  } else if (lang === 'rust') {
    keywords = /\b(fn|let|mut|struct|enum|impl|trait|pub|use|mod|self|super|crate|return|if|else|for|while|loop|match|async|await|true|false|None|Some|Ok|Err|println)\b/g;
    commentPattern = /(\/\/[^\n]*)/g;
  } else {
    keywords = /\b(function|def|class|return|if|else|for|while|import|from|const|let|var|true|false|null|None|print|println)\b/g;
    commentPattern = /(#[^\n]*|\/\/[^\n]*)/g;
  }

  const tokens = [];
  function stash(match) {
    const id = '\x00TOK' + tokens.length + '\x00';
    tokens.push(match);
    return id;
  }

  code = code.replace(commentPattern, m => stash('<span class="syn-comment">' + m + '</span>'));
  code = code.replace(/(&quot;(?:[^&]|&(?!quot;))*?&quot;|&#39;(?:[^&]|&(?!#39;))*?&#39;|&quot;&quot;&quot;[\s\S]*?&quot;&quot;&quot;)/g, m => stash('<span class="syn-string">' + m + '</span>'));
  code = code.replace(/"([^"]*?)"/g, m => stash('<span class="syn-string">' + m + '</span>'));

  code = code.replace(keywords, '<span class="syn-keyword">$1</span>');
  code = code.replace(/\b(\d+\.?\d*)\b/g, '<span class="syn-number">$1</span>');

  for (let ti = 0; ti < tokens.length; ti++) {
    code = code.replace('\x00TOK' + ti + '\x00', tokens[ti]);
  }
  return code;
}

function renderCodeBlock(code, lang, strings) {
  const highlighted = highlightSyntax(escapeHtml(code), lang);
  const langLabel = lang ? '<span class="code-lang">' + escapeHtml(lang) + '</span>' : '';
  return '<pre>' + langLabel + '<button class="code-copy" data-code="' + escapeAttr(code) + '">' + strings.copyCode + '</button><code>' + highlighted + '</code></pre>';
}

// Lesson figures referenced relatively in docs/*.md are served from the
// jsDelivr CDN mirror of the repo (same behavior as the old client renderer).
function makeAssetResolver(lessonPath) {
  return function resolveAssetUrl(src) {
    if (!lessonPath) return src;
    const parts = (lessonPath + '/docs/' + src).split('/');
    const stack = [];
    for (let i = 0; i < parts.length; i++) {
      if (parts[i] === '..') stack.pop();
      else if (parts[i] !== '.' && parts[i] !== '') stack.push(parts[i]);
    }
    return 'https://cdn.jsdelivr.net/gh/stabuev/ai-engineering-from-scratch@main/' + stack.join('/');
  };
}

function makeInlineFormat(resolveAssetUrl) {
  return function inlineFormat(text) {
    text = escapeHtml(text);
    text = text.replace(/\*\*\*(.+?)\*\*\*/g, '<strong><em>$1</em></strong>');
    text = text.replace(/\*\*(.+?)\*\*/g, '<strong>$1</strong>');
    text = text.replace(/\*(.+?)\*/g, '<em>$1</em>');
    text = text.replace(/`([^`]+)`/g, '<code>$1</code>');
    text = text.replace(/!\[([^\]]*)\]\(([^)]+)\)/g, function (m, alt, src) {
      const url = /^https?:\/\//i.test(src) ? src : resolveAssetUrl(src);
      if (!url) return '';
      return '<img src="' + url + '" alt="' + alt + '" loading="lazy" class="lesson-figure">';
    });
    text = text.replace(/\[([^\]]+)\]\(([^)]+)\)/g, function (m, label, href) {
      if (/^https?:\/\/|^mailto:/i.test(href)) {
        return '<a href="' + href + '" target="_blank" rel="noopener">' + label + '</a>';
      }
      return label;
    });
    return text;
  };
}

function parseMd(md, opts) {
  const strings = opts.strings;
  const inlineFormat = makeInlineFormat(makeAssetResolver(opts.lessonPath));

  const lines = md.split('\n');
  let out = '';
  let inCodeBlock = false;
  let codeLang = '';
  let codeLines = [];
  let inTable = false;
  let tableRows = [];
  let inList = false;
  let listType = '';
  let listItems = [];
  let firstBlockquoteDone = false;
  let firstParaAfterMottoDone = false;
  let inBlockquote = false;
  let blockquoteLines = [];
  let mermaidBlocks = 0;
  let inLabChallenge = false;

  function flushLabChallenge() {
    if (!inLabChallenge) return '';
    inLabChallenge = false;
    return '</div>';
  }

  function flushList() {
    if (!inList) return '';
    inList = false;
    const tag = listType === 'ol' ? 'ol' : 'ul';
    let h = '<' + tag + '>';
    for (let li = 0; li < listItems.length; li++) {
      h += '<li>' + inlineFormat(listItems[li]) + '</li>';
    }
    h += '</' + tag + '>';
    listItems = [];
    return h;
  }

  function flushTable() {
    if (!inTable) return '';
    inTable = false;
    if (tableRows.length < 2) return '';
    const headers = tableRows[0];
    let isKeyTerms = false;
    for (let th = 0; th < headers.length; th++) {
      if (headers[th].toLowerCase().indexOf('what people say') >= 0 || headers[th].toLowerCase().indexOf('people say') >= 0) {
        isKeyTerms = true;
      }
    }
    let h = '<div class="table-wrap' + (isKeyTerms ? ' key-terms-table' : '') + '"><table><thead><tr>';
    for (let ti = 0; ti < headers.length; ti++) {
      h += '<th>' + inlineFormat(headers[ti].trim()) + '</th>';
    }
    h += '</tr></thead><tbody>';
    for (let ri = 2; ri < tableRows.length; ri++) {
      const cells = tableRows[ri];
      h += '<tr>';
      for (let ci = 0; ci < cells.length; ci++) {
        let cls = '';
        if (isKeyTerms) {
          const headerLower = (headers[ci] || '').toLowerCase();
          if (headerLower.indexOf('say') >= 0) cls = ' class="col-says"';
        }
        h += '<td' + cls + '>' + inlineFormat((cells[ci] || '').trim()) + '</td>';
      }
      h += '</tr>';
    }
    h += '</tbody></table></div>';
    tableRows = [];
    return h;
  }

  function flushBlockquote() {
    if (!inBlockquote) return '';
    inBlockquote = false;
    const text = blockquoteLines.join(' ');
    blockquoteLines = [];
    if (!firstBlockquoteDone) {
      firstBlockquoteDone = true;
      return '<div class="motto">' + inlineFormat(text) + '</div>';
    }
    return '<blockquote><p>' + inlineFormat(text) + '</p></blockquote>';
  }

  for (var i = 0; i < lines.length; i++) {
    const line = lines[i];

    if (inCodeBlock) {
      if (line.match(/^```\s*$/)) {
        inCodeBlock = false;
        const raw = codeLines.join('\n');
        if (codeLang === 'mermaid') {
          mermaidBlocks++;
          out += '<div class="mermaid-container">';
          out += '<div class="mermaid-block" data-mermaid-index="' + mermaidBlocks + '">';
          out += '<div class="mermaid-toolbar">';
          out += '<button type="button" class="mermaid-btn mermaid-expand" data-mermaid-index="' + mermaidBlocks + '">' + strings.mermaidExpand + '</button>';
          out += '</div>';
          out += '<pre class="mermaid mermaid-source" id="mermaid-' + mermaidBlocks + '">' + escapeHtml(raw) + '</pre>';
          out += '<div class="mermaid-render" id="mermaid-render-' + mermaidBlocks + '"></div>';
          out += '</div>';
          out += '</div>';
        } else {
          out += renderCodeBlock(raw, codeLang, strings);
        }
        codeLines = [];
        codeLang = '';
      } else {
        codeLines.push(line);
      }
      continue;
    }

    const codeStart = line.match(/^```(\w*)/);
    if (codeStart) {
      out += flushList();
      out += flushTable();
      out += flushBlockquote();
      inCodeBlock = true;
      codeLang = codeStart[1] || '';
      codeLines = [];
      continue;
    }

    // Collapsible solution blocks: pass <details>/<summary>/</details>
    // lines through as raw HTML (the parser otherwise escapes them).
    const detailLine = line.match(/^\s*(<details>|<\/details>|<summary>[\s\S]*?<\/summary>)\s*$/);
    if (detailLine) {
      out += flushList();
      out += flushTable();
      out += flushBlockquote();
      const rawTag = line.trim();
      if (rawTag.indexOf('<summary>') === 0) {
        const inner = rawTag.replace(/^<summary>/, '').replace(/<\/summary>$/, '');
        out += '<summary>' + inlineFormat(inner) + '</summary>';
      } else {
        out += rawTag;
      }
      continue;
    }

    if (line.match(/^\|.+\|/)) {
      out += flushList();
      out += flushBlockquote();
      if (!inTable) inTable = true;
      const cells = line.split('|').slice(1, -1);
      tableRows.push(cells.map(c => c.trim()));
      continue;
    } else if (inTable) {
      out += flushTable();
    }

    if (line.match(/^>\s/)) {
      out += flushList();
      out += flushTable();
      if (!inBlockquote) inBlockquote = true;
      blockquoteLines.push(line.replace(/^>\s?/, ''));
      continue;
    } else if (inBlockquote) {
      out += flushBlockquote();
    }

    const h1 = line.match(/^# (.+)/);
    if (h1) {
      out += flushList();
      out += flushLabChallenge();
      out += '<h1 id="' + slugify(h1[1]) + '">' + inlineFormat(h1[1]) + '</h1>';
      continue;
    }

    const h2 = line.match(/^## (.+)/);
    if (h2) {
      out += flushList();
      const slug2 = slugify(h2[1]);
      let sectionClass = '';
      const txt = h2[1].toLowerCase();
      if (txt.indexOf('build it') >= 0 || txt.indexOf('build ') === 0 || txt.indexOf('соберите') >= 0 || txt.indexOf('постройте') >= 0) sectionClass = ' section-build';
      else if (txt.indexOf('use it') >= 0 || txt.indexOf('use ') === 0 || txt.indexOf('используйте') >= 0 || txt.indexOf('примените') >= 0) sectionClass = ' section-use';
      else if (txt.indexOf('ship it') >= 0 || txt.indexOf('ship ') === 0 || txt.indexOf('доведите до результата') >= 0 || txt.indexOf('оформите результат') >= 0) sectionClass = ' section-ship';
      if (txt.indexOf('цели обучения') >= 0 || txt.indexOf('learning objectives') >= 0) {
        const objItems = [];
        var lo;
        for (lo = i + 1; lo < lines.length; lo++) {
          const loLine = lines[lo];
          if (loLine.match(/^[-*]\s+(.+)/)) {
            objItems.push(loLine.replace(/^[-*]\s+/, ''));
          } else if (loLine.trim() === '') {
            continue;
          } else {
            break;
          }
        }
        out += '<div class="learning-objectives"><div class="learning-objectives-title">' + strings.objectivesTitle + '</div><ul>';
        for (let oi = 0; oi < objItems.length; oi++) {
          out += '<li>' + inlineFormat(objItems[oi]) + '</li>';
        }
        out += '</ul></div>';
        i = lo - 1;
        continue;
      }
      if (txt.indexOf('lab challenge') >= 0 || txt.indexOf('практическое задание') >= 0 || txt.indexOf('лабораторное задание') >= 0) {
        out += flushLabChallenge();
        inLabChallenge = true;
        out += '<div class="lab-challenge">';
        out += '<h2 id="' + slug2 + '">' + strings.labChallengeTitle + '</h2>';
        continue;
      }
      out += flushLabChallenge();
      out += '<h2 id="' + slug2 + '" class="' + sectionClass + '">' + inlineFormat(h2[1]) + '</h2>';
      continue;
    }

    const h3 = line.match(/^### (.+)/);
    if (h3) {
      out += flushList();
      out += '<h3 id="' + slugify(h3[1]) + '">' + inlineFormat(h3[1]) + '</h3>';
      continue;
    }

    if (line.match(/^---+$/)) {
      out += flushList();
      out += '<hr>';
      continue;
    }

    const ulMatch = line.match(/^[-*]\s+(.+)/);
    if (ulMatch) {
      out += flushTable();
      out += flushBlockquote();
      if (!inList || listType !== 'ul') {
        out += flushList();
        inList = true;
        listType = 'ul';
      }
      listItems.push(ulMatch[1]);
      continue;
    }

    const olMatch = line.match(/^\d+\.\s+(.+)/);
    if (olMatch) {
      out += flushTable();
      out += flushBlockquote();
      if (!inList || listType !== 'ol') {
        out += flushList();
        inList = true;
        listType = 'ol';
      }
      listItems.push(olMatch[1]);
      continue;
    }

    if (inList) {
      out += flushList();
    }

    if (line.trim() === '') {
      continue;
    }

    // Meta lines write the colon either inside or after the bold:
    // "**Тип:** Изучение" and "**Type**: Learn" are both meta tags.
    const metaMatch = line.match(/^\*\*([^*]+?):\*\*\s*(.+)/) || line.match(/^\*\*([^*]+?)\*\*:\s*(.+)/);

    let paraClass = '';
    if (firstBlockquoteDone && !firstParaAfterMottoDone) {
      if (!metaMatch) {
        paraClass = ' class="drop-cap"';
        firstParaAfterMottoDone = true;
      }
    }

    if (metaMatch && !firstParaAfterMottoDone) {
      out += '<div class="lesson-meta-tag"><strong>' + escapeHtml(metaMatch[1]) + ':</strong> ' + inlineFormat(metaMatch[2]) + '</div>';
      continue;
    }

    out += '<p' + paraClass + '>' + inlineFormat(line) + '</p>';
  }

  out += flushList();
  out += flushTable();
  out += flushBlockquote();
  out += flushLabChallenge();

  return out;
}

function renderArticle(opts) {
  const strings = opts.strings || STRINGS.ru;
  return {
    html: parseMd(opts.md, { lessonPath: opts.lessonPath, strings }),
    title: extractTitle(opts.md),
  };
}

// ─── Quizzes ─────────────────────────────────────────────────────────
function buildQuizHtml(questions, id, title) {
  if (!questions.length) return '';
  let html = '<div class="quiz-section" id="quiz-' + id + '">';
  html += '<div class="quiz-title">&#9989; ' + escapeHtml(title) + '</div>';
  for (let qi = 0; qi < questions.length; qi++) {
    const q = questions[qi];
    const qid = id + '-q' + qi;
    html += '<div class="quiz-question" data-correct="' + q.correct + '" id="' + qid + '">';
    html += '<div class="quiz-question-text">' + (qi + 1) + '. ' + escapeHtml(q.question) + '</div>';
    html += '<div class="quiz-options">';
    for (let oi = 0; oi < q.options.length; oi++) {
      html += '<div class="quiz-option" data-index="' + oi + '" onclick="handleQuizClick(this)">';
      html += '<span class="quiz-marker"></span>';
      html += '<span>' + escapeHtml(q.options[oi]) + '</span>';
      html += '</div>';
    }
    html += '</div>';
    if (q.explanation) {
      html += '<div class="quiz-explanation" id="' + qid + '-exp">' + escapeHtml(q.explanation) + '</div>';
    }
    html += '</div>';
  }
  html += '<div class="quiz-score" id="quiz-' + id + '-score"></div>';
  html += '</div>';
  return html;
}

function insertBeforeAnchor(html, anchorRe, insertHtml) {
  const m = html.match(anchorRe);
  if (!m) return null;
  return html.slice(0, m.index) + insertHtml + html.slice(m.index);
}

// String-based port of the old client-side renderQuiz(): pre-quiz goes before
// the Concept section, post-quiz before Key Terms (RU headings now carry
// Cyrillic slugs, so both languages anchor reliably).
function buildQuizzesHtml(articleHtml, quizData, strings) {
  const questions = quizData && (quizData.questions || quizData);
  if (!questions || !questions.length) return articleHtml;

  const conceptRe = /<h2 [^>]*id="(?:the-concept|concept|концепция)[^"]*"/;
  const buildRe = /<h2 [^>]*id="(?:build-it|build|соберите|постройте)[^"]*"/;
  const keyTermsRe = /<h2 [^>]*id="(?:key-terms|ключевые-термины)[^"]*"/;

  const preQuizHtml = buildQuizHtml(questions.filter(q => q.stage === 'pre'), 'pre', strings.quizPreTitle);
  const postQuizHtml = buildQuizHtml(questions.filter(q => q.stage === 'post'), 'post', strings.quizPostTitle);

  if (!preQuizHtml && !postQuizHtml) {
    const allHtml = buildQuizHtml(questions, 'all', strings.quizAllTitle);
    if (!allHtml) return articleHtml;
    return insertBeforeAnchor(articleHtml, keyTermsRe, allHtml) || (articleHtml + allHtml);
  }

  if (preQuizHtml) {
    articleHtml = insertBeforeAnchor(articleHtml, conceptRe, preQuizHtml)
      || insertBeforeAnchor(articleHtml, buildRe, preQuizHtml)
      || articleHtml;
  }
  if (postQuizHtml) {
    articleHtml = insertBeforeAnchor(articleHtml, keyTermsRe, postQuizHtml) || (articleHtml + postQuizHtml);
  }
  return articleHtml;
}

// ─── Prerequisites links ─────────────────────────────────────────────
// Turn the prose "Phase N · MM" / "Phase N" references on the Prerequisites
// meta line into in-site lesson links. Scoped to that one block so we never
// linkify inside code or body prose. resolveHref(phaseNum, lessonNum|null)
// returns an href or null (no dead links).
function linkifyPrereqs(articleHtml, resolveHref) {
  const blockRe = /<(div class="lesson-meta-tag"|p)>(?:(?!<\/(?:div|p)>)[\s\S])*?(?:Prerequisites|Предварительные требования)[\s\S]*?<\/(?:div|p)>/;
  const m = articleHtml.match(blockRe);
  if (!m) return articleHtml;

  const linked = m[0]
    .replace(/Phase\s+(\d+)\s+·\s+(\d+)/g, function (full, p, l) {
      const href = resolveHref(parseInt(p, 10), l.length === 1 ? '0' + l : l);
      if (!href) return full;
      return '<a href="' + href + '">' + full + '</a>';
    })
    .replace(/Phase\s+(\d+)(?!\s*[·\d])/g, function (full, p) {
      const href = resolveHref(parseInt(p, 10), null);
      if (!href) return full;
      return '<a href="' + href + '">' + full + '</a>';
    });

  return articleHtml.slice(0, m.index) + linked + articleHtml.slice(m.index + m[0].length);
}

module.exports = {
  STRINGS,
  escapeHtml,
  escapeAttr,
  slugify,
  extractTitle,
  extractDescription,
  renderArticle,
  buildQuizHtml,
  buildQuizzesHtml,
  linkifyPrereqs,
};
