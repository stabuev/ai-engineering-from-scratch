# Capstone 17 — Personal AI Tutor (Adaptive, Multimodal, with Memory)

> Khanmigo (Khan Academy), Duolingo Max, Google LearnLM / Gemini for Education, Quizlet Q-Chat и Synthesis Tutor в 2026 году запустили adaptive multimodal tutoring в масштабе. Общая форма: Socratic policy (никогда просто не выдавать answer), learner model, которая обновляется после каждого interaction (в стиле Bayesian knowledge tracing), voice + text + photo-math input, curriculum graph retrieval, spaced-repetition scheduling и жесткие safety filters для age-appropriate content. Capstone — запустить subject-specific tutor (K-12 algebra или intro Python), провести двухнедельное efficacy study с 10 learners и пройти content-safety audit.

**Тип:** Capstone
**Языки:** Python (backend, learner model), TypeScript (web app), SQL (curriculum graph via Postgres + Neo4j)
**Пререквизиты:** Phase 5 (NLP), Phase 6 (speech), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 14 (agents), Phase 17 (infrastructure), Phase 18 (safety)
**Отрабатываемые фазы:** P5 · P6 · P11 · P12 · P14 · P17 · P18
**Время:** 30 часов

## Цели обучения

- Построить адаптивного мультимодального AI-репетитора, отслеживающего состояние ученика.
- Персонализировать объяснения и генерировать практику с обратной связью.
- Добавить модель освоения, адаптирующую сложность в течение сессии.

## Задача

Adaptive tutoring раньше был нишей ed-tech research. К 2026 году это consumer product. Khanmigo развернут в большинстве US school districts. Duolingo Max достиг десятков миллионов MAUs. Google LearnLM / Gemini for Education поддерживает tutoring в Google Classroom. Quizlet Q-Chat находится рядом с flashcards. Synthesis Tutor стал вирусным как tutor-for-curious-kids. Общие элементы: multimodal input (type, speak, photograph equations), Socratic pedagogy (сначала спросить, потом объяснить), learner model, которая обновляется после каждого interaction, и strict age-appropriate safety.

Ты построишь такую систему для конкретной cohort. Планка измерения — настоящее efficacy study: pre-test and post-test scores за две недели с 10 learners. Voice loop должен ощущаться естественно (capstone 03 sub-stack). Memory должна уважать privacy. Safety filter должен пройти COPPA-aware red-team для K-12.

## Концепция

Четыре компонента. **Tutor policy** — Socratic loop: когда learner просит answer, policy задает наводящий question; когда learner отвечает правильно, она переходит к следующему concept; когда learner застрял, предлагает scaffolded hint. **Learner model** — Bayesian knowledge tracing (или простой вариант), который обновляет mastery probability по каждому curriculum node после каждого interaction. **Curriculum graph** — Neo4j concepts с prerequisite edges; policy проходит по graph, чтобы выбрать следующий concept. **Memory** — episodic + semantic store (agentmemory-style), хранящий past interactions, mistakes и preferences.

UX — multimodal. Text input для typed answers. Voice input через LiveKit + Whisper (переиспользуй capstone 03). Photo input для math problems через dots.ocr или PaliGemma 2. Voice output через Cartesia Sonic-2. Safety использует Llama Guard 4 плюс age-appropriate filter (блокирует adult content, violence, self-harm) и COPPA-aware memory retention policy.

Efficacy study — deliverable. 10 learners, pre-test and post-test, две недели. Отчитай learning gain delta и confidence interval. Сравни с non-adaptive baseline (тот же content, доставленный линейно без tutor policy).

## Архитектура

```
learner device
  |
  +-- text         -> web app
  +-- voice        -> LiveKit Agents (ASR + TTS)
  +-- photo math   -> dots.ocr / PaliGemma 2
       |
       v
  tutor policy (LangGraph)
       - Socratic decision head
       - next-concept chooser (curriculum graph walk)
       - hint scaffolder
       - mastery update
       |
       v
  learner model (BKT / item-response theory)
       - per-concept mastery probability
       - spaced-repetition scheduler (SM-2 or FSRS)
       |
       v
  memory (agentmemory-style)
       - episodic: every interaction
       - semantic: learned mistakes, preferences
       - retention policy: COPPA / GDPR aware
       |
       v
  curriculum graph (Neo4j)
       - prerequisite edges
       - OER content attached
       |
       v
  safety:
    Llama Guard 4 + age-appropriate filter
    memory access guarded by learner ID scope
```

## Стек

- Subject choice: K-12 algebra или intro Python (выбери одно для глубины)
- Tutor policy: LangGraph поверх Claude Sonnet 4.7 (with prompt caching)
- Learner model: Bayesian knowledge tracing (classic) или FSRS for spacing
- Curriculum graph: Neo4j с concepts + prerequisite edges + OER content
- Memory: agentmemory-style persistent vector + episodic + semantic store
- Voice: LiveKit Agents 1.0 + Cartesia Sonic-2 (reuse capstone 03 sub-stack)
- Photo math: dots.ocr или PaliGemma 2 for equation recognition
- Safety: Llama Guard 4 + custom age-appropriate filter
- Eval: Bloom-level question generation, pre/post test harness, tooling для efficacy study

## Сборка

1. **Curriculum graph.** Построй Neo4j из 50-150 concept nodes (например, K-12 algebra от "number line" до "quadratic formula") с prerequisite edges. Прикрепи OER content к каждому node (Open Textbook, OpenStax).

2. **Learner model.** Инициализируй Bayesian knowledge tracing с priors: guess, slip, learn-rate. Обновляй per-concept mastery после каждого interaction. Persist per learner.

3. **Tutor policy.** LangGraph с nodes: `read_signal` (was the learner's answer correct / partial / stuck?), `select_concept` (walk curriculum graph picking the highest-priority concept), `scaffold` (Socratic prompt), `update_mastery`.

4. **Memory.** Каждый interaction пишется в episodic store. Mistakes и preferences promote to semantic memory. COPPA-aware retention policy: auto-delete after 1 year, parent-accessible.

5. **Voice path.** LiveKit Agents worker подключен к tutor policy. ASR через Whisper-v3-turbo. TTS через Cartesia Sonic-2. Поддерживается barge-in (reuse capstone 03 mechanics).

6. **Photo-math path.** Upload or capture image; запусти dots.ocr или PaliGemma 2, чтобы распознать equation; передай tutor как structured input.

7. **Safety.** Каждый model output проходит Llama Guard 4 + age-appropriate filter (blocks self-harm, adult content, violence). Memory access scoped by learner ID; parental access surface для deletion.

8. **Efficacy study.** 10 learners, pre-test (standardized 30-question baseline), две недели tutor interaction (3 sessions/week), post-test. Сравни с non-adaptive baseline cohort из 10 learners на том же content.

9. **Weekly progress reports.** Для каждого learner автоматически генерируй PDF summary of topics explored, mastery trajectories, and recommended next steps.

## Использование

```
learner: "I don't understand why 3x + 6 = 12 means x = 2"
[signal]   stuck
[concept]  'isolating variables' (prerequisite: addition-subtraction-equality)
[scaffold] "what number would you subtract from both sides to start?"
learner: "6"
[signal]   correct
[mastery]  addition-subtraction-equality: 0.62 -> 0.77
[concept]  continue 'isolating variables'
[scaffold] "great. now what is 3x / 3 equal to?"
```

## Что сдать

`outputs/skill-ai-tutor.md` — deliverable. Subject-specific adaptive tutor с multimodal input, learner model, memory, safety и measured efficacy.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Learning gain delta | Pre/post-test delta в двухнедельном study с 10 learners |
| 20 | Socratic fidelity | Rubric score на transcript samples |
| 20 | Multimodal UX | Связность voice + photo + text end to end |
| 20 | Safety + privacy posture | Llama Guard 4 pass rate + COPPA-aware retention |
| 15 | Curriculum breadth and graph quality | Concept coverage + prerequisite graph consistency |
| **100** | | |

## Упражнения

1. Проведи efficacy study с adaptive learner model и без нее (random concept order). Отчитай delta. Ожидается, что adaptive победит, но интересен размер эффекта.

2. Добавь multimodal probe: один и тот же concept question доставляется как text, voice и photo. Измерь, сходятся ли learners быстрее через preferred modality.

3. Построй parent dashboard: topics practiced, mastery trajectories, upcoming concepts, safety events (любые guardrail hits). COPPA-aligned.

4. Добавь language-switch mode: tutor принимает Spanish input и обучает на Spanish. Измерь X-Guard coverage.

5. Проверь memory privacy под нагрузкой: убедись, что learner A не может увидеть data learner B даже через voice-clip re-ingest attack. Залогируй attempted access и отправь alert.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| Socratic policy | "Ask, do not dump" | Tutor задает наводящий question вместо выдачи answer |
| Bayesian knowledge tracing | "BKT" | Classic learner-model equations для mastery probability per concept |
| FSRS | "Free Spaced Repetition Scheduler" | 2024 spaced-repetition scheduler, лучше SM-2 |
| Curriculum graph | "Concept DAG" | Neo4j из concepts с prerequisite edges |
| Episodic memory | "Per-interaction log" | Каждый interaction сохраняется для later retrieval |
| Semantic memory | "Learned pattern store" | Сжатые mistakes и preferences, promoted from episodic |
| COPPA | "Kids privacy law" | US law, ограничивающий data collection from children under 13 |

## Дополнительное чтение

- [Khanmigo (Khan Academy)](https://www.khanmigo.ai) — reference consumer K-12 tutor
- [Duolingo Max](https://blog.duolingo.com/duolingo-max/) — reference language-learning tutor
- [Google LearnLM / Gemini for Education](https://blog.google/technology/google-deepmind/learnlm) — hosted reference model
- [Quizlet Q-Chat](https://quizlet.com) — alternate reference
- [Synthesis Tutor](https://www.synthesis.com) — startup reference
- [FSRS algorithm](https://github.com/open-spaced-repetition/fsrs4anki) — spaced-repetition scheduler
- [Bayesian Knowledge Tracing](https://en.wikipedia.org/wiki/Bayesian_knowledge_tracing) — классика learner model
- [LiveKit Agents](https://github.com/livekit/agents) — voice stack
