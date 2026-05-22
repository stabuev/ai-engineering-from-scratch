# Аудио-языковые модели — Qwen2.5-Omni, Audio Flamingo, GPT-4o Audio

> Аудио-языковые модели 2026 года рассуждают о речи + звуках окружения + музыке. Qwen2.5-Omni-7B сравнялась с GPT-4o Audio на MMAU-Pro. Audio Flamingo Next обходит Gemini 2.5 Pro на LongAudioBench. Разрыв между open и closed почти закрыт — кроме multi-audio задач, где все близки к случайности.

**Тип:** Изучение
**Языки:** Python
**Предварительные требования:** Фаза 6 · 04 (ASR), Фаза 12 · 03 (Vision-Language Models), Фаза 7 · 10 (Audio Transformers)
**Время:** ~45 минут

## Проблема

У вас есть 5 секунд аудио: собака лает, кто-то кричит "stop!", затем тишина. Полезные вопросы лежат на нескольких осях:

- **Transcription.** "What was said?" — территория ASR.
- **Semantic reasoning.** "Is the person in danger?" — требует совместного понимания bark + yell + silence.
- **Music reasoning.** "What instruments play the melody?"
- **Long-audio retrieval.** "Where in this 90-minute lecture did the instructor explain gradient descent?"

Одна модель, отвечающая на все это одним prompt, — **audio-language model** (LALM / ALM). Это не pure ASR: LALMs порождают свободные natural-language answers, а не только transcripts.

## Концепция

![Audio-language model: audio encoder + projector + LLM decoder](../assets/alm-architecture.svg)

### Шаблон из трех компонентов

Каждая LALM 2026 года имеет один скелет:

1. **Audio encoder.** Whisper encoder · BEATs · CLAP · WavLM · или custom encoder.
2. **Projector.** Linear или MLP, соединяющий audio-encoder features с token embedding space LLM.
3. **LLM.** Decoder на базе Llama / Qwen / Gemma. Принимает interleaved text + audio tokens; генерирует text.

Обучение:

- **Stage 1.** Заморозить encoder + LLM; обучать только projector на ASR / captioning data.
- **Stage 2.** Full / LoRA fine-tune на instruction-following audio tasks (QA, reasoning, music understanding).
- **Stage 3 (optional).** Voice-in / voice-out добавляет speech decoder. Qwen2.5-Omni и AF3-Chat делают это.

### Карта моделей 2026 года

| Model | Backbone | Audio encoder | Output modality | Access |
|-------|----------|---------------|-----------------|--------|
| Qwen2.5-Omni-7B | Qwen2.5-7B | Custom + Whisper | text + speech | Apache-2.0 |
| Qwen3-Omni | Qwen3 | Custom | text + speech | Apache-2.0 |
| Audio Flamingo 3 | Qwen2 | AF-CLAP | text | NVIDIA non-commercial |
| Audio Flamingo Next | Qwen2 | AF-CLAP v2 | text | NVIDIA non-commercial |
| SALMONN | Vicuna | Whisper + BEATs | text | Apache-2.0 |
| LTU / LTU-AS | Llama | CAV-MAE | text | Apache-2.0 |
| GAMA | Llama | AST + Q-Former | text | Apache-2.0 |
| Gemini 2.5 Flash/Pro (closed) | Gemini | proprietary | text + speech | API |
| GPT-4o Audio (closed) | GPT-4o | proprietary | text + speech | API |

### Проверка реальностью benchmark (2026)

**MMAU-Pro.** 1800 QA pairs по speech / sound / music / mixed. Включает multi-audio subset.

| Model | Overall | Speech | Sound | Music | Multi-audio |
|-------|---------|--------|-------|-------|-------------|
| Gemini 2.5 Pro | ~60% | 73.4% | 51.9% | 64.9% | ~22% |
| Gemini 2.5 Flash | ~57% | 73.4% | 50.5% | 64.9% | 21.2% |
| GPT-4o Audio | 52.5% | — | — | — | 26.5% |
| Qwen2.5-Omni-7B | 52.2% | 57.4% | 47.6% | 61.5% | ~20% |
| Audio Flamingo 3 | ~54% | — | — | — | — |
| Audio Flamingo Next | SOTA on LongAudioBench | — | — | — | — |

**Колонка multi-audio плоха у всех.** Случайное угадывание в multiple choice на 4 варианта = 25%; большинство моделей около этого. LALMs все еще плохо сравнивают два клипа.

### Где LALMs полезны в 2026 году

- **Compliance audit call-center recordings.** "Did the agent mention the required disclosure?"
- **Accessibility.** Описывать sound events для deaf users, а не только транскрибировать.
- **Content moderation.** Detect violent language + threatening tone + background context.
- **Podcast / meeting chaptering.** Semantic summary, а не только speaker turns.
- **Music catalog analysis.** "Find all tracks with a B-section key change."

### Где они пока НЕ полезны

- Fine-grained music theory (ниже chord-level).
- Speaker-attributed reasoning по длинным conversations (ухудшается после 10 минут).
- Multi-audio comparison (22-26% едва выше random).
- Real-time streaming reasoning (большинство — offline batch inference).

## Соберите это

### Шаг 1: запросите Qwen2.5-Omni

```python
from transformers import AutoModelForCausalLM, AutoProcessor

processor = AutoProcessor.from_pretrained("Qwen/Qwen2.5-Omni-7B")
model = AutoModelForCausalLM.from_pretrained("Qwen/Qwen2.5-Omni-7B", torch_dtype="auto")

audio, sr = load_wav("clip.wav", sr=16000)
messages = [{
    "role": "user",
    "content": [
        {"type": "audio", "audio": audio},
        {"type": "text", "text": "What sounds do you hear, and what's happening?"},
    ],
}]
inputs = processor.apply_chat_template(messages, tokenize=True, return_tensors="pt")
output = model.generate(**inputs, max_new_tokens=200)
print(processor.decode(output[0], skip_special_tokens=True))
```

### Шаг 2: pattern projector

```python
import torch.nn as nn

class AudioProjector(nn.Module):
    def __init__(self, audio_dim=1280, llm_dim=4096):
        super().__init__()
        self.down = nn.Linear(audio_dim, llm_dim)
        self.act = nn.GELU()
        self.up = nn.Linear(llm_dim, llm_dim)

    def forward(self, audio_features):
        return self.up(self.act(self.down(audio_features)))
```

Вот и все. Projector обычно состоит из 1-3 linear layers. Обучение на ASR pairs (audio → transcript) — Stage-1 pretext task.

### Шаг 3: benchmarking MMAU / LongAudioBench

```python
from datasets import load_dataset
mmau = load_dataset("MMAU/MMAU-Pro")

correct = 0
for item in mmau["test"]:
    answer = call_model(item["audio"], item["question"], item["choices"])
    if answer == item["correct_choice"]:
        correct += 1
print(f"Accuracy: {correct / len(mmau['test']):.3f}")
```

Сообщайте per-category (speech / sound / music / multi-audio) отдельно. Aggregate numbers скрывают, где модель ломается.

## Используйте это

| Задача | Выбор 2026 |
|------|-----------|
| Free-form audio QA (open) | Qwen2.5-Omni-7B |
| Best open on long audio | Audio Flamingo Next |
| Best closed | Gemini 2.5 Pro |
| Voice-in / voice-out agent | Qwen2.5-Omni или GPT-4o Audio |
| Music reasoning | Audio Flamingo 3 или 2 (music-specialized AF-CLAP) |
| Call-center audit | Gemini 2.5 Pro via API, with RAG over your policy docs |

## Ловушки

- **Over-trust on multi-audio.** Если задача требует "which clip has X," performance уровня случайности реален.
- **Long-audio degradation.** После 10 минут у большинства моделей ломается speaker attribution. Сначала diarize (Lesson 6), затем summarize.
- **Hallucinations on silence.** Та же Whisper-style проблема у LALMs, использующих Whisper encoder. VAD-gate.
- **Benchmark cherry-picking.** Vendor blog posts показывают best-case categories. Запустите MMAU-Pro multi-audio subset сами.

## Доведите до результата

Сохраните как `outputs/skill-alm-picker.md`. Выберите LALM + benchmark subset + output-modality (text vs speech) для заданной audio-understanding task.

## Упражнения

1. **Легко.** Запустите `code/main.py`, чтобы увидеть toy projector pattern + fake LALM routing (audio-embedding, text-tokens) → output tokens.
2. **Средне.** Оцените Qwen2.5-Omni-7B на 100 speech items MMAU-Pro. Сравните с reported number из статьи.
3. **Сложно.** Соберите минимальный audio-captioning baseline: BEATs encoder + 2-layer projector + frozen Llama-3.2-1B. Дообучайте только projector на AudioCaps. Сравните с SALMONN на Clotho-AQA.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|-----------------|-----------------------|
| LALM | Audio ChatGPT | Audio encoder + projector + LLM decoder. |
| Projector | Adapter | Маленький MLP, отображающий audio features в LLM embedding space. |
| MMAU | Benchmark | 10k audio-QA pairs по speech, sound, music. |
| MMAU-Pro | Более сложный MMAU | 1800 multi-audio / reasoning-heavy questions. |
| LongAudioBench | Long-form eval | Multi-minute clips с semantic queries. |
| Voice-in / voice-out | Speech-native | Модель принимает речь и выдает речь без text detour. |

## Дополнительное чтение

- [Chu et al. (2024). Qwen2-Audio](https://arxiv.org/abs/2407.10759) — reference architecture.
- [Alibaba (2025). Qwen2.5-Omni](https://huggingface.co/Qwen/Qwen2.5-Omni-7B) — speech-in-speech-out.
- [NVIDIA (2025). Audio Flamingo 3](https://arxiv.org/abs/2507.08128) — open long-audio leader.
- [NVIDIA (2026). Audio Flamingo Next](https://arxiv.org/abs/2604.10905) — LongAudioBench SOTA.
- [Tang et al. (2023). SALMONN](https://arxiv.org/abs/2310.13289) — dual-encoder pioneer.
- [MMAU-Pro leaderboard](https://mmaubenchmark.github.io/) — live rankings 2026 года.
