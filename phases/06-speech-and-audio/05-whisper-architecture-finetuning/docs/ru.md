# Whisper — архитектура и дообучение

> Whisper — transformer encoder-decoder с 30-секундным окном, обученный на 680k часах многоязычных слабо размеченных пар аудио-текст. Одна архитектура, много задач, устойчивость на 99 языках. Эталонный ASR 2026 года.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 04 (ASR), Фаза 5 · 10 (Attention), Фаза 7 · 05 (Full Transformer)
**Время:** ~75 минут

## Проблема

Whisper, выпущенный OpenAI в сентябре 2022 года, был первой ASR-моделью, ставшей товарной: вставьте аудио, получите текст, 99 языков, устойчивость к шуму, запуск на ноутбуке. К 2024 году OpenAI выпустила варианты Large-v3 и Turbo; к 2026 году Whisper стал дефолтным бейзлайном для всего: от транскрибации подкастов до голосовых ассистентов и субтитров YouTube.

Но Whisper нельзя вечно считать черным ящиком. Domain shift ломает его — технический жаргон, акценты, имена собственные, короткие клипы, тишина. Вам нужно знать:

1. Что он реально представляет собой внутри.
2. Как корректно подавать chunked, streaming или long-form аудио.
3. Когда дообучать и как.

## Концепция

![Whisper encoder-decoder, tasks, chunked inference, fine-tune](../assets/whisper.svg)

**Архитектура.** Стандартный transformer encoder-decoder.

- Вход: 30-секундная log-mel спектрограмма, 80 mels, шаг 10 ms → 3000 фреймов. Более короткие клипы дополняются нулями, более длинные режутся на chunks.
- Encoder: conv-downsample (stride 2) + `N` transformer blocks. Для Large-v3: 32 layers, 1280-dim, 20 heads.
- Decoder: `N` transformer blocks с causal self-attn + cross-attn к encoder output. Тот же размер, что у encoder.
- Выход: BPE-токены над словарем 51,865 токенов.

Large-v3 имеет 1.55B параметров. Turbo использует 4-layer decoder (вместо 32), уменьшая задержку в 8× с потерей WER <1%.

**Формат prompt.** Whisper — multitask-модель, управляемая специальными токенами в decoder prompt:

```
<|startoftranscript|><|en|><|transcribe|><|notimestamps|> Hello world.<|endoftext|>
```

- `<|en|>` — language tag; задает поведение translation-vs-transcription.
- `<|transcribe|>` или `<|translate|>` — перевод в английский из входа на любом языке или дословная транскрипция.
- `<|notimestamps|>` — пропустить word-level timestamps (быстрее).

Prompt позволяет одной модели выполнять много задач. Замените `<|en|>` на `<|fr|>`, и она будет транскрибировать французский.

**30-секундное окно.** Все привязано к 30 секундам. Более длинные клипы нужно разбивать на chunks; более короткие — дополнять padding. Окна нативно не поддерживают streaming — поэтому существуют WhisperX, Whisper-Streaming и faster-whisper.

**Log-mel нормализация.** `(log_mel - mean) / std`, где статистики взяты из обучающего корпуса Whisper. Вы *обязаны* использовать preprocessing Whisper (`whisper.audio.log_mel_spectrogram`), а не `librosa.feature.melspectrogram`.

### Варианты в 2026 году

| Variant | Params | Latency (A100) | WER (LibriSpeech-clean) |
|---------|--------|----------------|------------------------|
| Tiny | 39M | 1× realtime | 5.4% |
| Base | 74M | 1× | 4.1% |
| Small | 244M | 1× | 3.0% |
| Medium | 769M | 1× | 2.7% |
| Large-v3 | 1.55B | 2× | 1.8% |
| Large-v3-turbo | 809M | 8× | 1.58% |
| Whisper-Streaming (2024) | 1.55B | streaming | 2.0% |

### Дообучение

Канонический workflow 2026 года:

1. Соберите 10–100 часов аудио целевого домена с выровненными транскриптами.
2. Запустите `transformers.Seq2SeqTrainer` с callback `generate_with_loss`.
3. Parameter-efficient: LoRA на attention layers `q_proj`, `k_proj`, `v_proj` снижает память GPU в 4× с ценой <0.3 WER.
4. Заморозьте encoder, если у вас <10 часов. Дообучайте только decoder.
5. Используйте собственный tokenizer и prompt format Whisper; никогда не меняйте tokenizer.

Результаты сообщества: дообучение Medium на 20 часах медицинской диктовки снижает WER с 12% до 4.5% на медицинской лексике. Дообучение Turbo на 4 часах исландского снижает WER с 18% до 6%.

## Соберите это

### Шаг 1: запустите Whisper из коробки

```python
import whisper
model = whisper.load_model("large-v3-turbo")
result = model.transcribe(
    "clip.wav",
    language="en",
    task="transcribe",
    temperature=0.0,
    condition_on_previous_text=False,  # prevents runaway repetition
)
print(result["text"])
for seg in result["segments"]:
    print(f"[{seg['start']:.2f}–{seg['end']:.2f}] {seg['text']}")
```

Ключевые значения по умолчанию, которые стоит переопределять всегда: `temperature=0.0` (sampling по умолчанию использует fallback chain 0.0 → 0.2 → 0.4 …), `condition_on_previous_text=False` (предотвращает каскадные галлюцинации) и `no_speech_threshold=0.6` (детекция тишины).

### Шаг 2: chunked long-form

```python
# whisperx is the 2026 reference for long-form with word-level timestamps
import whisperx
model = whisperx.load_model("large-v3-turbo", device="cuda", compute_type="float16")
segments = model.transcribe("1hour.mp3", batch_size=16, chunk_size=30)
```

WhisperX добавляет (1) Silero VAD gating, (2) word-level alignment через wav2vec 2.0, (3) diarization через `pyannote.audio`. Это рабочая лошадка 2026 года для продакшен-транскрибации.

### Шаг 3: дообучите с LoRA

```python
from transformers import WhisperForConditionalGeneration, WhisperProcessor
from peft import LoraConfig, get_peft_model

model = WhisperForConditionalGeneration.from_pretrained("openai/whisper-large-v3-turbo")
lora = LoraConfig(
    r=16, lora_alpha=32, target_modules=["q_proj", "v_proj"],
    lora_dropout=0.1, bias="none", task_type="SEQ_2_SEQ_LM",
)
model = get_peft_model(model, lora)
# model.print_trainable_parameters()  -> ~3M trainable / 809M total
```

Дальше стандартный цикл Trainer. Checkpoint каждые 1000 шагов. Оценивайте WER на held-out.

### Шаг 4: посмотрите, что учит каждый слой

```python
# Grab cross-attention weights during decode to see what the decoder attends to.
with torch.inference_mode():
    out = model.generate(
        input_features=features,
        return_dict_in_generate=True,
        output_attentions=True,
    )
# out.cross_attentions: layer × head × step × src_len
```

Визуализируйте heatmap — вы увидите диагональное выравнивание, пока шаги decoder сканируют encoder frames. Эта диагональ — представление Whisper о word timestamps.

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| General English, offline | Large-v3-turbo через `whisperx` |
| Mobile / edge | Whisper-Tiny quantized (int8) или Moonshine |
| Multilingual long-form | Large-v3 через `whisperx` + diarization |
| Low-resource language | Дообучить Medium или Turbo с LoRA |
| Streaming (2 s latency) | Whisper-Streaming или Parakeet-TDT |
| Word-level timestamps | WhisperX (forced alignment через wav2vec 2.0) |

`faster-whisper` (backend CTranslate2) — самый быстрый CPU+GPU runtime для инференса в 2026 году: 4× быстрее vanilla при идентичном output.

## Ловушки, которые все еще попадают в продакшен в 2026 году

- **Галлюцинации текста на тишине.** Whisper обучался на captions и включает "Thanks for watching!", "Subscribe!", lyrics. Всегда используйте VAD-gate перед вызовом.
- **Каскад `condition_on_previous_text`.** Одна галлюцинация загрязняет последующие окна. Ставьте `False`, если вам не нужна fluency между chunks.
- **Padding коротких клипов.** 2-секундный клип, дополненный до 30 секунд, может галлюцинировать в хвостовой тишине. Используйте `pad=False` или VAD-gate.
- **Неверные статистики mel.** Использование mels из librosa вместо Whisper дает почти случайный output. Используйте `whisper.audio.log_mel_spectrogram`.

## Доведите до результата

Сохраните как `outputs/skill-whisper-tuner.md`. Спроектируйте дообучение Whisper или inference pipeline для заданного домена.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он токенизирует Whisper-style prompt, считает decoded shape budgets и печатает chunk schedule для 10-минутного клипа.
2. **Средне.** Установите `faster-whisper`, транскрибируйте 10-минутный подкаст, сравните WER с человеческим transcript. Попробуйте `language="auto"` против принудительного `language="en"`.
3. **Сложно.** Используя HF `datasets`, выберите язык, с которым Whisper справляется плохо (например, Urdu), дообучите Medium с LoRA на 2 эпохи на 2 часах и сообщите WER delta.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| 30-sec window | Лимит Whisper | Жесткий входной предел; режьте длинное аудио на chunks. |
| SOT | Start-of-transcript | `<|startoftranscript|>` запускает decoder prompt. |
| Timestamps token | Временное выравнивание | Каждое смещение 0.02 s — специальный токен в словаре 51k. |
| Turbo | Быстрый вариант | 4 decoder layers, 8× быстрее, <1% WER regression. |
| WhisperX | Long-form wrapper | VAD + Whisper + wav2vec alignment + diarization. |
| LoRA fine-tune | Эффективное дообучение | Добавить low-rank adapters в attention; обучить ~0.3% параметров. |
| Hallucination | Тихий отказ | Whisper порождает беглый английский из шума/тишины. |

## Дополнительное чтение

- [Radford et al. (2022). Whisper paper](https://arxiv.org/abs/2212.04356) — исходная архитектура и рецепт обучения.
- [OpenAI (2024). Whisper Large-v3-turbo release](https://github.com/openai/whisper/discussions/2363) — 4-layer decoder, ускорение 8×.
- [Bain et al. (2023). WhisperX](https://arxiv.org/abs/2303.00747) — long-form, word-aligned, diarized.
- [Systran — faster-whisper repo](https://github.com/SYSTRAN/faster-whisper) — на CTranslate2, 4× быстрее.
- [HuggingFace — Whisper fine-tune tutorial](https://huggingface.co/blog/fine-tune-whisper) — канонический walkthrough LoRA / full-FT.
