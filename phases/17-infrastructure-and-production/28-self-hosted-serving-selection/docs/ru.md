# Self-Hosted Serving Selection — llama.cpp, Ollama, TGI, vLLM, SGLang

> Четыре engines доминируют в self-hosted inference в 2026. Выбирайте по hardware, scale и ecosystem. **llama.cpp** fastest on CPU — widest model support, full control over quantization and threading. **Ollama** — dev-laptop one-command install, ~15-30% slower than llama.cpp (Go + CGo + HTTP serialization), 3x throughput gap under prod-like load. **TGI entered maintenance mode December 11, 2025** — only bug fixes, ~10% slower raw throughput than vLLM, но исторически top observability and HF-ecosystem integration. Этот maintenance status делает его рискованной long-term bet — SGLang или vLLM безопаснее как defaults для новых проектов. **vLLM** — general-purpose production default — v0.15.1 (February 2026) добавляет PyTorch 2.10, RTX Blackwell SM120, H200 optimization. **SGLang** — specialist для agentic multi-turn / prefix-heavy workloads — 400,000+ GPUs in production (xAI, LinkedIn, Cursor, Oracle, GCP, Azure, AWS). Hardware constraints: CPU-only → llama.cpp only. AMD / non-NVIDIA → vLLM only (TRT-LLM is NVIDIA-locked). 2026 pipeline pattern: dev = Ollama, staging = llama.cpp, prod = vLLM or SGLang. Same GGUF/HF weights throughout.

**Тип:** Learn
**Языки:** Python (stdlib, engine-decision tree walker)
**Предварительные требования:** All Phase 17 lessons covering engines (04, 06, 07, 09, 18)
**Время:** ~45 minutes

## Цели обучения

- Выбрать engine по hardware (CPU / AMD / NVIDIA Hopper / Blackwell), scale (1 user / 100 / 10,000) и workload (general chat / agent / long-context).
- Назвать TGI maintenance-mode status 2026 (December 11, 2025) и почему он смещает новые проекты к vLLM или SGLang.
- Описать dev/staging/prod pipeline с теми же GGUF или HF weights throughout.
- Объяснить, почему "CPU only" принуждает к llama.cpp, а "AMD" исключает TRT-LLM.

## Проблема

Ваша команда начинает новый self-hosted LLM project. Один engineer говорит Ollama, другой — vLLM, третий — "doesn't TGI just work out of the box?" Все трое правы в разных contexts. Ни один не прав для всех.

В 2026 decision tree важен: hardware first, scale second, workload third. И одно конкретное событие 2025 — переход TGI в maintenance mode December 11 — меняет default для новых проектов.

## Концепция

### Пять engines

| Engine | Best for | Notes |
|--------|----------|-------|
| **llama.cpp** | CPU / edge / minimal deps / widest model support | Fastest on CPU, full control |
| **Ollama** | Dev laptops, single user, one-command install | 15-30% slower than llama.cpp; 3x prod throughput gap |
| **TGI** | HF ecosystem, regulated industries | **Maintenance mode Dec 11, 2025** |
| **vLLM** | General-purpose production, 100+ users | Broad production default; v0.15.1 Feb 2026 |
| **SGLang** | Agentic multi-turn, prefix-heavy workloads | 400,000+ GPUs in production |

### Hardware-first decision

**CPU only** → llama.cpp. Ollama тоже работает, но медленнее. Ни один другой engine не конкурентен на CPU.

**AMD GPU** → vLLM (AMD ROCm support). SGLang тоже работает. TRT-LLM is NVIDIA-locked, поэтому исключается.

**NVIDIA Hopper (H100 / H200)** → vLLM or SGLang or TRT-LLM. Все три top-tier.

**NVIDIA Blackwell (B200 / GB200)** → TRT-LLM is the throughput leader (Phase 17 · 07). vLLM and SGLang follow close.

**Apple Silicon (M-series)** → llama.cpp (Metal). Ollama wraps this.

### Scale-second decision

**1 user / local dev** → Ollama. One command, first-token in seconds.

**10-100 users / small team** → vLLM single-GPU.

**100-10k users / production** → vLLM production-stack (Phase 17 · 18) или SGLang.

**10k+ users / enterprise** → vLLM production-stack + disaggregated (Phase 17 · 17) + LMCache (Phase 17 · 18).

### Workload-third decision

**General chat / Q&A** → vLLM wins on broad default.

**Agentic multi-turn (tools, planning, memory)** → SGLang's RadixAttention (Phase 17 · 06) dominates.

**RAG with heavy prefix reuse** → SGLang.

**Code generation** → vLLM fine; SGLang slightly better on cache.

**Long context (128K+)** → vLLM + chunked prefill; SGLang + tiered KV.

### The TGI maintenance trap

Hugging Face TGI entered maintenance mode December 11, 2025 — only bug fixes going forward. Исторически: top-tier observability, best-in-class HF-ecosystem integration (model cards, safety tools), slightly behind vLLM on raw throughput.

Для новых проектов в 2026: default away from TGI. Existing TGI deployments can continue but should migrate eventually. SGLang and vLLM are the safer defaults.

### The pipeline pattern

Dev (Ollama) → staging (llama.cpp) → prod (vLLM). Same GGUF or HF weights throughout. Engineers быстро итерируют на laptops; staging mirrors production quantization; prod is the serving target.

### Ollama caveat

Ollama отлично подходит для dev. Он не очень подходит для shared production: Go HTTP serialization adds overhead, concurrency management проще, чем у vLLM, OpenTelemetry support lags. Use Ollama where it shines — one user, one command — and switch to vLLM for shared.

### Self-hosted vs managed is a separate decision

Phase 17 · 01 (managed hyperscalers), · 02 (inference platforms) cover managed. Этот урок предполагает, что вы уже решили self-host. Reasons to self-host: data residency, custom fine-tune, total cost ownership at scale, domain model not available on hosted.

### Числа, которые стоит запомнить

- TGI maintenance mode: December 11, 2025.
- vLLM v0.15.1: February 2026; PyTorch 2.10; Blackwell SM120 support.
- SGLang production footprint: 400,000+ GPUs.
- Ollama throughput gap vs llama.cpp: 15-30% slower; 3x under prod load.

## Используйте это

`code/main.py` — decision-tree walker: по hardware + scale + workload выбирает engine и объясняет почему.

## Отгрузите это

Этот урок создает `outputs/skill-engine-picker.md`. По constraints выбирает engine и пишет migration plan.

## Упражнения

1. Запустите `code/main.py` с вашим hardware / scale / workload. Совпадает ли output с вашей intuition?
2. Ваша infra — 12 H100s и 8 MI300X AMD. Какой engine? Почему TRT-LLM off the table?
3. Команда хочет использовать TGI в 2026, потому что "it's what we know." Аргументируйте migration case.
4. Ollama dev to vLLM prod: что меняется в quantization, configuration и observability?
5. RAG product с P99 prefix length 8K и high reuse across tenants. Выберите engine и stack it with Phase 17 · 11 + 18.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| llama.cpp | "the CPU one" | Widest model support, fastest on CPU |
| Ollama | "the laptop one" | One-command install, dev-grade throughput |
| TGI | "HF's serving" | Maintenance mode since Dec 2025 |
| vLLM | "the default" | Broad production baseline 2026 |
| SGLang | "the agentic one" | Prefix-heavy, RadixAttention |
| TRT-LLM | "NVIDIA-locked" | Blackwell throughput leader, NVIDIA only |
| GGUF | "llama.cpp format" | Bundled K-quant variants |
| Production-stack | "vLLM K8s" | Phase 17 · 18 reference deployment |
| Pipeline pattern | "dev→stage→prod" | Ollama → llama.cpp → vLLM on same weights |

## Дополнительное чтение

- [AI Made Tools — vLLM vs Ollama vs llama.cpp vs TGI 2026](https://www.aimadetools.com/blog/vllm-vs-ollama-vs-llamacpp-vs-tgi/)
- [Morph — llama.cpp vs Ollama 2026](https://www.morphllm.com/comparisons/llama-cpp-vs-ollama)
- [n1n.ai — Comprehensive LLM Inference Engine Comparison](https://explore.n1n.ai/blog/llm-inference-engine-comparison-vllm-tgi-tensorrt-sglang-2026-03-13)
- [PremAI — 10 Best vLLM Alternatives 2026](https://blog.premai.io/10-best-vllm-alternatives-for-llm-inference-in-production-2026/)
- [TGI maintenance announcement](https://github.com/huggingface/text-generation-inference) — release notes.
- [vLLM v0.15.1 release notes](https://github.com/vllm-project/vllm/releases)
