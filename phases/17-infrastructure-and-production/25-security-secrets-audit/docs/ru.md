# Security — Secrets, API Key Rotation, Audit Logs, Guardrails

> Устраните secret sprawl через централизованные vaults (HashiCorp Vault, AWS Secrets Manager, Azure Key Vault). Никогда не храните credentials в config files, env files в VCS, spreadsheets. Используйте IAM roles вместо static keys; OIDC для CI/CD. AI-gateway pattern — решение 2026: apps → gateway → model provider, где gateway подтягивает credentials из vault во время runtime. Rotate in vault, и все apps подхватывают изменения за минуты — без redeploys, без Slack-сообщений "у кого новый key". Rotation policy ≤90 days; scan with TruffleHog / GitGuardian / Gitleaks on every commit. Zero-trust: MFA, SSO, RBAC/ABAC, short-lived tokens, device posture. PII scrubbing использует entity recognition, чтобы маскировать PHI/PII перед forwarding; consistent tokenization (Mesh approach) сопоставляет sensitive values со stable placeholders, чтобы LLM сохраняла code/relationship semantics. Network egress: LLM services в dedicated VPC/VNet subnet, whitelist только `api.openai.com`, `api.anthropic.com` etc; блокируйте весь другой outbound. Incident driver 2026: Vercel supply-chain attack через compromised CI/CD credentials, exfiltrated env vars across thousands of customer deployments.

**Тип:** Learn
**Языки:** Python (stdlib, toy PII-scrubber + audit-log writer)
**Предварительные требования:** Phase 17 · 19 (AI Gateways), Phase 17 · 13 (Observability)
**Время:** ~60 minutes

## Цели обучения

- Перечислить четыре антипаттерна secret-management (config files in VCS, hardcoded env, spreadsheets, static keys) и назвать их replacements.
- Объяснить AI-gateway-pulls-from-vault pattern как production standard 2026.
- Реализовать PII scrubber с consistent tokenization (same value → same placeholder), чтобы semantics сохранялись.
- Назвать Vercel supply-chain incident 2026 и чему он научил про CI/CD credential hygiene.

## Проблема

Стажер коммитит `.env` с API keys. Он быстро удаляет файл. Но keys уже в git history — GitGuardian scan ловит это, а ваш rotation process: "написать команде в Slack, обновить 40 config files, redeploy всех services." Через 8 часов половина services live, половина ждет deploy windows.

Отдельно, user prompts включают "My SSN is 123-45-6789." Prompt уходит в OpenAI. У вас есть BAA, но internal policy требует маскировать PII перед forwarding. Вы этого не сделали.

Отдельно, LLM pod в вашем EKS cluster может ходить на любой internet host. Кто-то exfils data через DNS lookup на attacker-controlled domain. Ничего это не заблокировало.

Security для LLM services должна закрывать все три вектора. Vault-backed credentials. PII scrubbing. Network egress filtering. Audit logs.

## Концепция

### Centralized vault + IAM-role pull

**Vault**: HashiCorp Vault, AWS Secrets Manager, Azure Key Vault, GCP Secret Manager. Один source of truth.

**IAM role**: app/gateway аутентифицируется через свою IAM identity, а не static key. Vault возвращает secret на срок жизни token.

**The AI-gateway pattern**: gateway подтягивает `OPENAI_API_KEY` из vault во время request time. Rotate in vault; следующий request получает новый key. No redeploys.

### Rotation policy ≤ 90 days

Все API keys, vault root tokens, CI/CD credentials. Automated rotation where possible. Manual rotation logged and tracked.

### Secret scanning

- **TruffleHog** — regex + entropy on commits.
- **GitGuardian** — commercial, high accuracy.
- **Gitleaks** — OSS, runs in CI.

Запускайте на every commit. Блокируйте PR, если найден new secret.

### Zero-trust posture

- MFA required on all accounts.
- SSO via SAML/OIDC.
- RBAC (role-based) или ABAC (attribute-based) для fine grained access.
- Short-lived tokens (hours, not days).
- Device posture — только corp devices with disk encryption.

### PII / PHI scrubbing

Перед тем как prompt покинет вашу infra:

1. Entity recognition (spaCy NER, Presidio, commercial).
2. Mask matched entities: `"My SSN is 123-45-6789"` → `"My SSN is [SSN_TOKEN_A3F]"`.
3. Consistent tokenization (Mesh approach): same value maps to the same placeholder so the LLM preserves relationships.
4. Optional reverse mapping for LLM response.

Static regex filters ловят базовые patterns; NER ловит больше. Используйте оба.

### Input + output guardrails

Input: block known jailbreaks, forbidden topics; rate-limit per-user.

Output: regex scrub for leaked secrets (API key patterns, email patterns in refusal contexts), classifier for policy violations.

### Network egress whitelist

LLM services в dedicated subnet:
- Whitelist: `api.openai.com`, `api.anthropic.com`, vector DB endpoints, vault endpoints.
- Everything else: drop.
- DNS via allowlist-only resolver (avoid DNS-tunneling exfil).

### Audit log

Immutable log каждого LLM call с:
- Timestamp.
- User / tenant.
- Prompt hash (not raw prompt for privacy).
- Model + version.
- Token counts.
- Cost.
- Response hash.
- Any guardrail trips.

Retain per regulatory requirement (SOC 2 1 year, HIPAA 6 years).

### The 2026 Vercel incident

Supply-chain attack: compromised CI/CD credentials exfiltrated env vars across thousands of customer deployments. Урок: CI/CD credentials are prod-equivalent. Store in vault. Scope narrowly. Rotate aggressively.

### Числа, которые стоит запомнить

- Rotation policy: ≤ 90 days.
- Scan on every commit: TruffleHog / GitGuardian / Gitleaks.
- Vercel 2026: CI/CD creds compromised → thousands of customer env vars leaked.
- Audit log retention: SOC 2 = 1 year, HIPAA = 6 years.

## Используйте это

`code/main.py` реализует toy PII scrubber с consistent tokenization и append-only audit log.

## Отгрузите это

Этот урок создает `outputs/skill-llm-security-plan.md`. По regulatory scope и current state планирует vault migration, scrubber, egress, audit log.

## Упражнения

1. Запустите `code/main.py`. Отправьте два prompts, ссылающихся на один SSN. Подтвердите, что оба получают same placeholder.
2. Спроектируйте network egress policy для vLLM-on-EKS deployment, вызывающего OpenAI + Anthropic + Weaviate.
3. Вы обнаружили key в git history (2 years old). Какой правильный response — rotate the key, scrub history или both? Обоснуйте.
4. Ваш audit log растет на 10 GB/day. Спроектируйте retention tiers (hot 30d, warm 12mo, cold 6yr).
5. Аргументируйте, стоит ли reverse-tokenization (substituting real values back into LLM response) своей сложности по сравнению с visible placeholders.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| Vault | "secrets store" | Centralized credential management service |
| IAM role | "identity-based auth" | Role, принимаемая app; возвращает short-lived creds |
| OIDC for CI/CD | "cloud-issued tokens" | Нет static keys в CI — identity через OIDC |
| TruffleHog / GitGuardian / Gitleaks | "secret scanners" | Secret detection во время commit |
| RBAC / ABAC | "access control" | Role-based vs attribute-based |
| PII scrubbing | "data masking" | Удаление или токенизация sensitive entities |
| Consistent tokenization | "stable placeholders" | Same value → same token each time |
| Mesh approach | "Mesh tokenization" | Semantic-preserving tokenization pattern |
| Egress whitelist | "outbound allowlist" | Доступны только permitted domains |
| Audit log | "immutable history" | Append-only record for compliance |

## Дополнительное чтение

- [Doppler — Advanced LLM Security](https://www.doppler.com/blog/advanced-llm-security)
- [Portkey — Manage LLM API keys with secret references](https://portkey.ai/blog/secret-references-ai-api-key-management/)
- [Datadog — LLM Guardrails Best Practices](https://www.datadoghq.com/blog/llm-guardrails-best-practices/)
- [JumpServer — Secrets Management Best Practices 2026](https://www.jumpserver.com/blog/secret-management-best-practices-2026)
- [Microsoft Presidio](https://github.com/microsoft/presidio) — PII detection and anonymization.
- [HashiCorp Vault docs](https://developer.hashicorp.com/vault/docs)
