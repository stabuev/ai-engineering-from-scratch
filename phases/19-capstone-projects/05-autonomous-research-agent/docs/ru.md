# Capstone 05 — Автономный исследовательский агент класса AI-Scientist

> Sakana's AI-Scientist-v2 опубликовал полноценные статьи. Agent Laboratory запускал эксперименты. Allen AI поделилась трассами. Форма 2026 года — это plan-execute-verify tree search по экспериментам, ограниченный бюджет, sandboxed code execution, LaTeX-писатель с vision-feedback и автоматизированный ансамбль рецензентов в стиле NeurIPS. Capstone — построить такой агент, запустить его end to end в пределах $30 на статью и выдержать red team на sandbox-escape, описанный Sakana.

**Тип:** Capstone
**Языки:** Python (agent + sandbox), LaTeX (output)
**Предварительные требования:** Phase 2 (ML), Phase 3 (deep learning), Phase 7 (transformers), Phase 10 (LLMs from scratch), Phase 14 (agents), Phase 15 (autonomous), Phase 16 (multi-agent), Phase 18 (safety)
**Задействованные фазы:** P0 · P2 · P3 · P7 · P10 · P14 · P15 · P16 · P18
**Время:** 40 hours

## Проблема

Autonomous research agents перешагнули важный порог в 2026 году. Sakana AI's AI-Scientist-v2 был опубликован в Nature с generated papers, прошедшими workshop peer review. ShinkaEvolve (ICLR 2026) расширил эту линию до evolving hypotheses. AMD's Agent Laboratory выпустил воспроизводимые трассы. Эти агенты не магия — это цикл plan-execute-verify, работающий по дереву candidate experiments, с cost caps, seed-bound sandboxes и automated review. Мастерство находится в цикле, бюджете и истории безопасности.

Вы изучаете этот цикл, реализуя его для seed idea в узкой области, например для attention-sparsity ablations на 100M-parameter transformer. Ценность не в том, чтобы с первого запуска открыть что-то новое. Ценность в инфраструктуре: tree-search, experiment sandbox, writer-reviewer loop, red-team report. Команда Sakana задокументировала sandbox-escape failures; ваш агент должен пройти тот же red team.

## Концепция

Агент — это best-first tree search. Узлы — это спецификации экспериментов: (hypothesis, config, code, expected outcome). Шаг expand предлагает дочерние узлы с небольшими правками: заменить optimizer, сдвинуть batch size, провести ablation компонента. Каждый дочерний узел запускается в свежем sandbox с жестким resource cap. Результаты возвращаются в scoring function, ранжирующую узлы по (novelty × quality × remaining budget). Дерево растет, пока не исчерпан бюджет, затем лучшая ветка оформляется в статью.

Писатель мультимодальный. Он генерирует LaTeX draft, компилирует его, рендерит figures и передает отрендеренный PDF обратно в Claude Opus 4.7's vision mode для критики layout, читаемости figures и соответствия claim-evidence. Ансамбль из пяти LLM judges выдает оценки в стиле NeurIPS (novelty, rigor, clarity, reproducibility, impact); если среднее ниже порога, статья возвращается писателю с critique.

Безопасность здесь несущая часть системы. Каждый эксперимент выполняется в E2B or Daytona sandbox без network egress, с ограниченным wall-clock и закрепленными resource limits. Шаг code-generation у агента проходит через policy layer, блокирующий syscalls, которые выводят за пределы sandbox. Red-team report воспроизводит attack surface, задокументированную Sakana: fork bombs, filesystem escapes, LLM-written network calls.

## Архитектура

```
seed idea + domain
      |
      v
  literature search (Semantic Scholar + OpenAlex + FAISS cache)
      |
      v
  LangGraph plan-execute-verify tree
      |
      v
  +--- expand node ----+      per-node sandbox
  |                    |      (E2B / Daytona)
  v                    v      resource caps
  child_1           child_k   no network egress
  |                    |      deterministic seeds
  v                    v
  run experiment       run experiment
  |                    |
  v                    v
  score nodes by (novelty, quality, budget)
      |
      v
  best branch -> LaTeX writer
      |
      v
  compile + vision critique (Opus 4.7 vision)
      |
      v
  reviewer ensemble (5 LLM judges, NeurIPS rubric)
      |
      v
  paper.pdf + review.md + trace.json
```

## Стек

- Оркестрация: LangGraph with checkpointing and human-approval gates
- Tree search: custom best-first over experiment nodes (AB-MCTS-style from Sakana v2)
- Sandbox: E2B per experiment, Docker-in-Docker fallback; resource caps via cgroups
- Литература: Semantic Scholar Graph API + OpenAlex + local FAISS cache of abstracts
- Писатель: LaTeX template + Claude Opus 4.7 (vision mode) for figure critique and layout
- Рецензент: ensemble of 5 judges (Opus 4.7, GPT-5.4, Gemini 3 Pro, DeepSeek R1, Qwen3-Max) with weighted aggregation
- Фреймворк экспериментов: PyTorch 2.5 для физических экспериментов, W&B для logging
- Наблюдаемость: Langfuse for agent traces, $30 hard budget per paper

## Постройте это

1. **Seed and domain scoping.** Возьмите seed idea, например "investigate sparsity patterns in attention maps of sub-1B transformers". Определите search space: models, datasets, compute budget.

2. **Literature pass.** Запросите в Semantic Scholar + OpenAlex 50 наиболее цитируемых релевантных статей; закешируйте abstracts локально; сгенерируйте 1-page domain digest.

3. **Tree scaffolding.** Инициализируйте root с seed hypothesis. Реализуйте `expand(node) -> children` с small-edit proposals (одно изменение config на дочерний узел). Реализуйте `score(node)` как взвешенный член novelty × quality × budget.

4. **Sandbox wrapping.** Каждый эксперимент запускается через `docker run --network=none --memory=8g --cpus=2 --pids-limit=256 --read-only` или эквивалентную E2B policy. Seeds записываются в sandbox; outputs монтируются обратно наружу read-only.

5. **Plan-execute-verify loop.** `plan` предлагает дочерние узлы. `execute` запускает sandbox, собирает logs and metrics. `verify` выполняет unit checks на metrics: снизился ли loss, изолировала ли ablation эффект. Неудачные узлы получают failure reason, сохраненный в tree.

6. **Writer.** После исчерпания бюджета выберите лучшую ветку. Отрендерите figures с matplotlib. Сгенерируйте LaTeX draft через Claude Opus 4.7, передав branch trace в контекст. Скомпилируйте. Передайте compiled PDF обратно в Opus 4.7 vision для critique. Итерируйте.

7. **Reviewer ensemble.** Пять judges оценивают draft по (novelty, rigor, clarity, reproducibility, impact) с rubrics в стиле NeurIPS. Если mean < 4.0/5, верните писателю с critique. Hard stop после 3 rewrites.

8. **Red team.** Соберите или интегрируйте набор adversarial tasks, атакующих sandbox: fork bombs, network exfiltration attempts, filesystem escapes, LLM-written shell metacharacters. Подтвердите, что все заблокировано. Опишите findings.

9. **Reproducibility.** Каждая статья поставляется с tree-search trace JSON, seeds, W&B run links, sandbox configs и README, воспроизводящим ее end to end.

## Используйте это

```
$ ai-scientist run --seed "attention sparsity in sub-1B transformers" --budget 30
[lit]    50 papers, digest in 12s
[tree]   expanded 8 nodes, budget 12/30
[exec]   node #3 sparsity=top-8, loss=2.83 (best so far)
[exec]   node #6 sparsity=top-4, loss=3.12 (worse)
[exec]   ...
[tree]   chose branch rooted at node #3 (novelty 0.62, quality 0.81)
[write]  LaTeX draft v1 complete
[vision] critique: figure 2 legend too small, claim-evidence ok
[write]  draft v2 after 3 edits
[review] mean 4.2/5 (novelty 3.9, rigor 4.3, clarity 4.1, repro 4.5, impact 4.2)
[done]   paper.pdf + review.md + trace.json     $28.40 spent
```

## Отгрузите это

`outputs/skill-ai-scientist.md` — deliverable. При наличии seed idea + domain + $30 budget он запускает полный pipeline и выдает reviewable paper плюс reproducibility bundle.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Качество статьи | Blind rubric review against published workshop papers |
| 20 | Экспериментальная строгость | Baselines, seeds, ablations; каждое claim подкреплено ячейкой в results table |
| 20 | Дисциплина стоимости и compute | $30/paper ceiling enforced, Langfuse-traced |
| 20 | Безопасность | Sandbox red team проходит; network policy and kill-switch verified |
| 15 | Воспроизводимость | One-command rerun with identical seeds reproduces the paper |
| **100** | | |

## Упражнения

1. Запустите pipeline на трех разных seed ideas в одной области. Сравните, какие части tree-search пересекаются. Найдите duplicated wasted compute.

2. Добавьте human-in-the-loop gate перед выполнением эксперимента для узлов, оцененных дороже $5. Измерьте, насколько падает total cost.

3. Замените reviewer ensemble на одного judge. Измерьте false-accept rate на held-out наборе known-bad papers.

4. Введите red team test на network-exfiltration: агент пишет код, который пытается выполнить `curl` на внешний адрес. Подтвердите, что policy `--network=none` блокирует это. Залогируйте попытку.

5. Сравните ваш tree-search с flat random baseline (тот же бюджет, без expansion strategy). Сообщите прирост novelty × quality.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| Tree search | "AB-MCTS-style expansion" | Best-first exploration over experiment nodes with a novelty×quality×budget score |
| Sandbox | "Experiment isolation" | Container with no network, bounded CPU/memory, pinned seeds, read-only inputs |
| Vision critique | "Render-then-read" | Compile the paper to PDF, feed the PDF back to a VLM for layout and claim-evidence critique |
| Reviewer ensemble | "Automated peer review" | Multiple LLM judges scoring the paper with a NeurIPS rubric; weighted aggregate gates the pipeline |
| Novelty score | "Is this new?" | Heuristic that penalizes proximity to the 50-paper literature cache |
| Cost ceiling | "$ budget" | Hard cap on total spend per paper; Langfuse counters + pre-run estimates |
| Red team | "Sandbox-escape audit" | Adversarial tasks that would escape the sandbox if the policy is wrong |

## Дополнительное чтение

- [Sakana AI-Scientist-v2 repository](https://github.com/SakanaAI/AI-Scientist-v2) — эталонный production research agent
- [Sakana AI-Scientist-v1 paper (arXiv:2408.06292)](https://arxiv.org/abs/2408.06292) — исходная methodology
- [ShinkaEvolve (Sakana ICLR 2026)](https://sakana.ai) — evolutionary extension
- [Agent Laboratory (AMD)](https://github.com/SamuelSchmidgall/AgentLaboratory) — multi-role research-lab framework
- [LangGraph documentation](https://langchain-ai.github.io/langgraph/) — эталонный orchestration layer
- [Semantic Scholar Graph API](https://api.semanticscholar.org/) — literature search
- [E2B sandboxes](https://e2b.dev) — эталонная experiment isolation
- [NeurIPS reviewer guidelines](https://neurips.cc/Conferences/2026/Reviewer-Guidelines) — rubric, которую кодирует reviewer ensemble
