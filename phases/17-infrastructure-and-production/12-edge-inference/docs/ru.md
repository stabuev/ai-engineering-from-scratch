# Edge Inference — Apple Neural Engine, Qualcomm Hexagon, WebGPU/WebLLM, Jetson

> Главное edge-ограничение — memory bandwidth, а не compute. Mobile DRAM дает 50-90 GB/s; datacenter HBM3 превышает 2-3 TB/s — разрыв 30-50x. Decode ограничен памятью, поэтому разрыв решающий. В 2026 ландшафт делится на четыре направления. Apple M4/A18 Neural Engine достигает 38 TOPS с unified memory (без CPU↔NPU copy). Qualcomm Snapdragon X Elite / 8 Gen 4 Hexagon дает 45 TOPS. WebGPU + WebLLM запускает Llama 3.1 8B (Q4) at ~41 tok/s on M3 Max (примерно 70-80% native); 17.6k GitHub stars, OpenAI-compatible API, ~70-75% mobile coverage. NVIDIA Jetson Orin Nano Super (8GB) помещает Llama 3.2 3B / Phi-3; AGX Orin запускает gpt-oss-20b через vLLM at ~40 tok/s; Jetson T4000 (JetPack 7.1) — 2x AGX Orin. TensorRT Edge-LLM поддерживает EAGLE-3, NVFP4, chunked prefill — показано на CES 2026 Bosch, ThunderSoft, MediaTek.

**Тип:** Learn
**Языки:** Python (stdlib, toy bandwidth-bound decode simulator)
**Предварительные требования:** Phase 17 · 04 (vLLM Serving Internals), Phase 17 · 09 (Production Quantization)
**Время:** ~60 minutes

## Цели обучения

- Объяснить, почему mobile LLM inference ограничен memory bandwidth, а compute вторичен.
- Перечислить четыре edge targets (Apple ANE, Qualcomm Hexagon, WebGPU/WebLLM, NVIDIA Jetson) и сопоставить каждый с use case.
- Назвать 2026 WebGPU coverage gap (Firefox Android catching up) и Safari iOS 26 landing.
- Выбрать quantization format для каждого target (Core ML INT4 + FP16 for ANE, QNN INT8/INT4 for Hexagon, WebGPU Q4 for browser, NVFP4 for Jetson Thor).

## Проблема

Клиент хочет on-device chatbot: voice-first, private-by-default, works offline. На MacBook Pro M3 Max Llama 3.1 8B Q4 работает at ~55 tok/s — нормально. На iPhone 16 Pro та же model работает at 3 tok/s — плохо. На mid-range Android со Snapdragon 8 Gen 3 — 7 tok/s. В browser через WebGPU on Chrome Android v121+ — 4-8 tok/s depending on the device.

Разброс throughput — не porting issue. Это bandwidth gap, умноженный на quantization format и на то, доступен ли NPU из user-space. Edge inference в 2026 — это четыре разные проблемы с четырьмя разными решениями.

## Концепция

### Bandwidth — настоящий потолок

Decode читает полный set of weights для каждого token. Одна 7B model in Q4 — 3.5 GB. Чтение 3.5 GB at 50 GB/s занимает 70 ms — теоретический потолок ~14 tok/s. At 90 GB/s (high-end mobile DRAM) потолок сдвигается к ~25 tok/s. Ни один compute не поможет ниже этого числа.

Datacenter HBM3 at 3 TB/s читает те же 3.5 GB за 1.2 ms — потолок 830 tok/s. Та же model, те же weights. Другая memory subsystem.

### Apple Neural Engine (M4 / A18)

- Up to 38 TOPS. Unified memory (CPU and ANE share the same pool) — no copy overhead.
- Access via Core ML + `.mlmodel` compiled models, or via Metal Performance Shaders (MPS) through PyTorch.
- Llama.cpp Metal backend uses MPS, not ANE directly; native ANE requires Core ML conversion.
- Best practical path for iOS apps in 2026: Core ML with INT4 weights + FP16 activations.

### Qualcomm Hexagon (Snapdragon X Elite / 8 Gen 4)

- Up to 45 TOPS. Integrated with CPU and GPU in the SoC but separate memory domain.
- QNN (Qualcomm Neural Network) SDK and AI Hub provide conversion from PyTorch/ONNX.
- Chat templates, Llama 3.2, Phi-3 all ship as first-class artifacts on AI Hub.

### Intel / AMD NPUs (Lunar Lake, Ryzen AI 300)

- 40-50 TOPS. Software lags behind Apple/Qualcomm; OpenVINO is improving but niche.
- Best for Windows ARM copilot apps; native on AMD/Intel desktops for local-first.

### WebGPU + WebLLM

- Запускает models в browser через WebGPU compute shaders; без установки.
- Llama 3.1 8B Q4 at ~41 tok/s on M3 Max — roughly 70-80% of native через тот же backend.
- 17.6k GitHub stars у WebLLM; OpenAI-compatible JS API; Apache 2.0.
- 2026 coverage: Chrome Android v121+, Safari iOS 26 GA, Firefox Android still catching up. Overall ~70-75% mobile coverage.

### NVIDIA Jetson family

- Orin Nano Super (8GB): fits Llama 3.2 3B, Phi-3 at good tok/s.
- AGX Orin: runs gpt-oss-20b via vLLM at ~40 tok/s.
- Thor / T4000 (JetPack 7.1): 2x AGX Orin performance, EAGLE-3 and NVFP4 supported.
- TensorRT Edge-LLM (2026) supports EAGLE-3 speculative decoding, NVFP4 weights, chunked prefill — datacenter optimizations ported to edge.

### Выбор quantization для target

| Target | Format | Notes |
|--------|--------|-------|
| Apple ANE | INT4 weights + FP16 activations | Core ML conversion path |
| Qualcomm Hexagon | QNN INT8 / INT4 | AI Hub converters |
| WebGPU / WebLLM | Q4 MLC (q4f16_1) | Use `mlc_llm convert_weight` + compiled `.wasm`; GGUF is not supported |
| Jetson Orin Nano | Q4 GGUF or TRT-LLM INT4 | Memory-bound |
| Jetson AGX / Thor | NVFP4 + FP8 KV | Edge-LLM path |

### Ловушка long-context на edge

128K context у Llama 3.1 — datacenter feature. На телефоне с 8 GB RAM: 4 GB model + 2 GB KV cache для 32K tokens + OS overhead = OOM. Edge deployments держат context at 4K-8K, если не принимают aggressive KV quantization (Q4 KV).

### Voice — killer app

Voice agents чувствительны к latency (first token < 500 ms). Local inference полностью устраняет network latency. В сочетании со speech-to-text (Whisper Turbo variants run on edge) edge inference становится production-quality voice loop.

### Числа, которые нужно помнить

- Apple M4 / A18 ANE: 38 TOPS.
- Qualcomm Hexagon SD X Elite: 45 TOPS.
- WebLLM M3 Max: ~41 tok/s on Llama 3.1 8B Q4.
- AGX Orin: ~40 tok/s on gpt-oss-20b via vLLM.
- Datacenter-edge bandwidth gap: 30-50x.
- WebGPU mobile coverage: ~70-75% (Firefox Android lagging).

## Используйте это

`code/main.py` вычисляет theoretical decode throughput ceilings из bandwidth-bound math по edge targets. Сравнивает с observed benchmarks и показывает, где bottleneck — bandwidth, а не compute.

## Отправьте в прод

Этот урок создает `outputs/skill-edge-target-picker.md`. По platform (iOS/Android/browser/Jetson), model и latency/memory budget выбирает quantization format и conversion pipeline.

## Упражнения

1. Запустите `code/main.py`. Для 7B model in Q4 на Snapdragon 8 Gen 3 (~77 GB/s bandwidth) вычислите decode ceiling. Сравните с observed 6-8 tok/s — runtime эффективен?
2. WebGPU on Android требует Chrome v121+. Спроектируйте fallback для older browsers — server-side via the same OpenAI-compatible API.
3. Вашему iOS app нужен 4K-context streaming. Какая model/format combination позволяет остаться under 4 GB active memory на iPhone 16?
4. Jetson AGX Orin runs gpt-oss-20b at 40 tok/s. Jetson Nano fits only a 3B. Если product targets both, как унифицировать inference stack?
5. Аргументируйте, production-ready ли "WebLLM is production-ready in 2026." Укажите coverage, performance и Firefox Android gap.

## Ключевые термины

| Термин | Как обычно говорят | Что это на самом деле означает |
|------|----------------|------------------------|
| ANE | "Apple neural engine" | On-device NPU in M-series and A-series; unified memory |
| Hexagon | "Qualcomm NPU" | Snapdragon NPU; QNN SDK for access |
| WebGPU | "browser GPU" | W3C-standardized browser GPU API; Chrome/Safari 2026 |
| WebLLM | "browser LLM runtime" | MLC-LLM project; Apache 2.0; OpenAI-compatible JS |
| Jetson | "NVIDIA edge" | Orin Nano / AGX / Thor / T4000 family |
| TRT Edge-LLM | "edge TensorRT" | 2026 edge port of TensorRT-LLM; EAGLE-3 + NVFP4 |
| Unified memory | "shared pool" | CPU and NPU see same RAM; no copy overhead |
| Bandwidth-bound | "memory limited" | Decode gated by bytes/sec reading weights |
| Core ML | "Apple conversion" | Apple framework for ANE-native models |
| QNN | "Qualcomm stack" | Qualcomm Neural Network SDK |

## Дополнительное чтение

- [On-Device LLMs State of the Union 2026](https://v-chandra.github.io/on-device-llms/) — landscape and benchmarks.
- [NVIDIA Jetson Edge AI](https://developer.nvidia.com/blog/getting-started-with-edge-ai-on-nvidia-jetson-llms-vlms-and-foundation-models-for-robotics/) — Orin / AGX / Thor.
- [NVIDIA TensorRT Edge-LLM](https://developer.nvidia.com/blog/accelerating-llm-and-vlm-inference-for-automotive-and-robotics-with-nvidia-tensorrt-edge-llm/) — 2026 edge port announcement.
- [WebLLM (arXiv:2412.15803)](https://arxiv.org/html/2412.15803v2) — design and benchmarks.
- [Apple Core ML](https://developer.apple.com/documentation/coreml) — ANE-native conversion.
- [Qualcomm AI Hub](https://aihub.qualcomm.com/) — pre-converted models for Hexagon.
