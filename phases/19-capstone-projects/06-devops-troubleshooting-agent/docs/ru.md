# Capstone 06 — DevOps Troubleshooting Agent for Kubernetes

> AWS's DevOps Agent вышел в GA, Resolve AI опубликовала свои K8s playbooks, NeuBird показала semantic monitoring, а Metoro связала AI SRE с per-service SLOs. Production-форма устоялась: alert webhook срабатывает, агент читает telemetry, обходит graph of K8s objects, ранжирует root-cause hypotheses и публикует Slack brief с approval buttons. По умолчанию read-only. Каждая remediation проходит через human gate. Этот capstone — такой агент, оцененный на 20 synthetic incidents и сравненный с AWS's Agent на трех общих cases.

**Тип:** Capstone
**Языки:** Python (agent), TypeScript (Slack integration)
**Предварительные требования:** Phase 11 (LLM engineering), Phase 13 (tools and MCP), Phase 14 (agents), Phase 15 (autonomous), Phase 17 (infrastructure), Phase 18 (safety)
**Задействованные фазы:** P11 · P13 · P14 · P15 · P17 · P18
**Время:** 30 hours

## Проблема

SRE-нарратив 2025-2026 стал таким: "AI agents triage incidents, humans approve remediations." AWS DevOps Agent, Resolve AI, NeuBird, Metoro, PagerDuty AIOps — все поставляют эту форму в production. Агент читает Prometheus metrics, Loki logs, Tempo traces, kube-state-metrics и knowledge graph of K8s objects. Он выдает ranked root-cause hypothesis с telemetry citations менее чем за пять минут. Он никогда не выполняет destructive commands без явного human approval через Slack.

Большая часть сложной работы — это scoping and safety, а не reasoning. Агенту нужна RBAC surface, read-only by default, hardened MCP tool server и audit logs каждой команды: considered vs executed. Он должен понимать, когда задача вне его глубины, и передавать ее на escalation. И он должен работать достаточно дешево, чтобы OOM-kill cascades не создавали $5k agent bill.

## Концепция

Агент работает поверх knowledge graph. Узлы — K8s objects (Pods, Deployments, Services, Nodes, HPAs, PVCs) плюс telemetry sources (Prometheus series, Loki streams, Tempo traces). Ребра кодируют ownership (Pod -> ReplicaSet -> Deployment), scheduling (Pod -> Node) и observation (Pod -> Prometheus series). Graph поддерживается свежим через kube-state-metrics sync и повторно сэмплируется при каждом alert.

Когда срабатывает alert, агент ищет root cause, начиная с affected object. Он обходит edges, достает релевантные telemetry slices за последние 15 minutes и черновиком формирует hypothesis. Hypothesis ранжируется по evidence: сколько telemetry citations ее поддерживают, насколько они свежие и специфичные. Top-3 hypotheses уходят в Slack с graph-path visualizations и approval buttons для remediation actions.

Remediation проходит через gate. Разрешенные default actions — read-only. Destructive actions (scaling down, rolling back, deleting Pods) требуют Slack approval; ArgoCD rollback hooks требуют auth token, которого у агента никогда нет. Audit log записывает каждую command, которую агент *considered*, а не только executed, чтобы review process ловил near-misses.

## Архитектура

```
PagerDuty / Alertmanager webhook
           |
           v
     FastAPI receiver
           |
           v
   LangGraph root-cause agent
           |
           +---- read-only MCP tools ----+
           |                             |
           v                             v
   K8s knowledge graph              telemetry slices
     (Neo4j / kuzu)              Prometheus, Loki, Tempo
   ownership + scheduling          last 15m, scoped
           |
           v
   hypothesis ranking (evidence weight)
           |
           v
   Slack brief + approval buttons
           |
           v (approved)
   ArgoCD rollback hook / PagerDuty escalate
           |
           v
   audit log: considered vs executed, every command
```

## Стек

- Источники наблюдаемости: Prometheus, Loki, Tempo, kube-state-metrics
- Knowledge graph: Neo4j (managed) or kuzu (embedded) of K8s objects + telemetry edges
- Агент: LangGraph with per-tool allow-list, read-only by default
- Tool transport: FastMCP over StreamableHTTP; separate server for destructive tools behind approval gate
- Модели: Claude Sonnet 4.7 for root-cause reasoning, Gemini 2.5 Flash for log summarization
- Remediation: ArgoCD rollback webhook, PagerDuty escalate, Slack approval card
- Audit: append-only structured log (considered, executed, approved, outcome)
- Deployment: K8s deployment with its own narrow RBAC role; separate namespace

## Постройте это

1. **Graph ingestion.** Синхронизируйте kube-state-metrics в Neo4j/kuzu каждые 30s. Узлы: Pod, Deployment, Node, Service, PVC, HPA. Ребра: OWNED_BY, SCHEDULED_ON, EXPOSES, MOUNTS, SCALES. Telemetry overlay edges: OBSERVED_BY (Pod наблюдается Prometheus series).

2. **Alert receiver.** FastAPI endpoint, принимающий PagerDuty or Alertmanager webhooks. Извлеките affected object(s) и SLO breach.

3. **Read-only tool surface.** Оберните kubectl, Prometheus query, Loki logql, Tempo traceql через FastMCP. У каждого tool узкий RBAC verb ("get", "list", "describe"). Никаких "delete", "exec", "scale" в default server.

4. **Root-cause agent.** LangGraph с тремя узлами: `sample` забирает telemetry slice за last-15-minutes, `walk` запрашивает у graph neighboring objects, `hypothesize` формирует ranked root-cause candidates с telemetry citations.

5. **Evidence scoring.** У каждой hypothesis score = recency * specificity * graph-path length inverse * citation count. Верните top-3.

6. **Slack brief.** Опубликуйте attachment с hypothesis, graph-path visualization (subgraph image, rendered server-side) и approval buttons максимум для одного remediation action.

7. **Remediation gate.** Destructive tools (scale down, roll back, delete) живут на втором MCP server за approval token. Агент может вызывать их только после того, как Slack card approved by a human.

8. **Audit log.** Append-only JSONL: для каждой candidate command логируйте, была ли она considered, была ли executed, кто approved it. Ежедневно отправляйте в S3.

9. **Synthetic incident suite.** Соберите 20 scenarios: OOMKill cascade, DNS flap, HPA thrash, PVC fill, noisy neighbor, faulty sidecar, bad ConfigMap rollout, certificate rotation, image-pull backoff и т.д. Оценивайте агента по root-cause accuracy и time-to-hypothesis.

## Используйте это

```
webhook: alert.pagerduty.com -> checkout-api SLO breach, error rate 14%
[graph]   affected: Deployment checkout-api (3 Pods, Node ip-10-2-3-4)
[walk]    neighbors: ReplicaSet checkout-api-abc, Service checkout-api,
           recent rollout 14m ago
[sample]  prometheus error_rate 14%, up-trend; loki 500s on /api/v2/pay
[hypo]    #1 bad rollout: latest image checkout-api:v2.41 fails /healthz
          citations: deploy.yaml (rev 42), prometheus errorRate, loki 500 stack
[slack]   [ROLL BACK to v2.40]  [ESCALATE]  [IGNORE]
          (approval required; agent does not roll back unilaterally)
```

## Отгрузите это

`outputs/skill-devops-agent.md` — deliverable. При наличии K8s cluster и alert source агент выдает ranked root-cause hypotheses и Slack-gated remediation flow.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | RCA accuracy on scenario suite | ≥80% correct root cause across 20 synthetic incidents |
| 20 | Безопасность | Destructive-action guard never fires without Slack approval in the audit log |
| 20 | Time-to-hypothesis | p50 under 5 minutes from alert to Slack brief |
| 20 | Объяснимость | Каждая hypothesis содержит graph paths and telemetry citations |
| 15 | Полнота интеграции | PagerDuty, Slack, ArgoCD, Prometheus работают end-to-end |
| **100** | | |

## Упражнения

1. Запустите своего агента на тех же трех incidents, которые демонстрирует AWS's DevOps Agent. Опубликуйте side-by-side. Сообщите, где агент расходится.

2. Добавьте "near-miss" audit, который помечает любую command, которую агент *considered* и которая была бы destructive без approval. Измерьте near-miss rate за одну неделю.

3. Замените hypothesis model с Claude Sonnet 4.7 на self-hosted Llama 3.3 70B. Измерьте delta RCA accuracy и dollar per incident.

4. Постройте causal filter: отличайте correlated telemetry spikes от true root cause. Обучите small classifier на labels из 20-scenario.

5. Добавьте rollback dry-run: ArgoCD rollback против staging cluster с тем же manifest. Проверьте rollback plan в live cluster перед Slack approval button.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле значит |
|------|-----------------|------------------------|
| K8s knowledge graph | "Cluster graph" | Nodes = K8s objects + telemetry series; edges = ownership, scheduling, observation |
| Read-only-by-default | "Scoped RBAC" | Agent's service account has only get/list/describe verbs; destructive verbs live in a separate server behind approval |
| Audit log | "Considered vs executed" | Append-only record of every candidate command, whether it ran, who approved |
| Hypothesis ranking | "Evidence score" | Recency × specificity × graph-path length inverse × citation count |
| Slack approval card | "HITL gate" | Interactive Slack message with remediation buttons; agent cannot proceed until a human clicks |
| Telemetry citation | "Evidence pointer" | A Prometheus query, Loki selector, or Tempo trace URL that supports a claim |
| MTTR | "Time to resolution" | Wall-clock from alert fire to SLO recovery |

## Дополнительное чтение

- [AWS DevOps Agent GA](https://aws.amazon.com/blogs/aws/aws-devops-agent-helps-you-accelerate-incident-response-and-improve-system-reliability-preview/) — canonical 2026 reference
- [Resolve AI K8s troubleshooting](https://resolve.ai/blog/kubernetes-troubleshooting-in-resolve-ai) — competitor reference
- [NeuBird semantic monitoring](https://www.neubird.ai) — semantic-graph approach
- [Metoro AI SRE](https://metoro.io) — SLO-first production framing
- [kube-state-metrics](https://github.com/kubernetes/kube-state-metrics) — cluster-state source
- [LangGraph](https://langchain-ai.github.io/langgraph/) — эталонный agent orchestrator
- [FastMCP](https://github.com/jlowin/fastmcp) — Python MCP server framework
- [ArgoCD rollback](https://argo-cd.readthedocs.io/en/stable/user-guide/commands/argocd_app_rollback/) — gated remediation target
