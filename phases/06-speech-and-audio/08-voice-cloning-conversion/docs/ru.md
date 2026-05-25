# Клонирование и преобразование голоса

> Клонирование голоса читает ваш текст чужим голосом. Преобразование голоса переписывает ваш голос в чужой, сохраняя сказанное. Обе задачи держатся на одном примитиве: отделить идентичность говорящего от содержания.

**Тип:** Сборка
**Языки:** Python
**Предварительные требования:** Фаза 6 · 06 (Speaker Recognition), Фаза 6 · 07 (TTS)
**Время:** ~75 минут

## Проблема

В 2026 году 5-секундного аудиоклипа достаточно, чтобы получить качественный клон любого голоса на потребительском GPU. ElevenLabs, F5-TTS, OpenVoice v2, VoiceBox поставляют zero-shot или few-shot cloning. Технология одновременно благо (accessibility TTS, dubbing, assistive voices) и оружие (scam calls, political deepfakes, IP theft).

Две близкие задачи:

- **Voice cloning (TTS-side):** text + 5-секундный reference voice → аудио этим голосом.
- **Voice conversion (speech-side):** source audio (человек A говорит X) + reference voice человека B → аудио, где B говорит X.

Обе факторизуют waveform на (content, speaker, prosody) и рекомбинируют content из одного источника со speaker из другого.

Ключевое ограничение в 2026 году: **watermarking и consent gates юридически обязательны в EU (AI Act, применим с августа 2026) и California (AB 2905, действует с 2025)**. Ваш pipeline должен добавлять неслышимый watermark и отказывать в non-consensual clones.

## Концепция

![Voice cloning vs conversion: factorize, swap speaker, recombine](../assets/voice-cloning.svg)

**Zero-shot cloning.** Передайте 5-секундный клип модели, обученной на тысячах speakers. Speaker encoder отображает клип в speaker embedding; TTS decoder conditioned on that embedding plus text.

Используют: F5-TTS (2024), YourTTS (2022), XTTS v2 (2024), OpenVoice v2 (2024).

**Few-shot fine-tuning.** Запишите 5-30 минут целевого голоса. LoRA-fine-tune базовую модель за час. Качество прыгает от «приемлемо» до «неотличимо». Coqui и ElevenLabs поддерживают этот паттерн; community использует его с F5-TTS.

**Voice conversion (VC).** Два семейства:

- **Recognition-synthesis.** Запустить ASR-like model, чтобы извлечь content representation (например, soft phoneme posteriors, PPGs), затем ресинтезировать с target speaker embedding. Устойчиво к языку и акценту. KNN-VC (2023), Diff-HierVC (2023).
- **Disentanglement.** Обучить autoencoder, который разделяет content, speaker и prosody в latent space на bottleneck. При инференсе заменить speaker embedding. Качество ниже, но быстрее. AutoVC (2019), VITS-VC variants.

**Neural codec-based cloning (2024+).** VALL-E, VALL-E 2, NaturalSpeech 3, VoiceBox — рассматривают аудио как дискретные токены SoundStream / EnCodec и обучают большой autoregressive или flow-matching model над codec tokens. Качество сравнимо с ElevenLabs на коротких prompts.

### Этика — не надстройка

**Watermarking.** PerTh (Perth) и SilentCipher (2024) встраивают ~16-32 bit ID незаметно для слуха. Выживает после re-encoding, streaming и типичных edits. Production-ready open source.

**Consent gates.** Каждый cloned output должен иметь verifiable consent record. Например: «Я, Rohit, 2026-04-22 разрешаю использовать этот голос для цели X». Храните в tamper-evident log.

**Detection.** AASIST, RawNet2 и Wav2Vec2-AASIST поставляются как detectors. ASVspoof 2025 challenge опубликовал EER 0.8–2.3% для state-of-the-art detectors против outputs ElevenLabs, VALL-E 2 и Bark.

### Числа (2026)

| Model | Zero-shot? | SECS (target sim) | WER (intel.) | Params |
|-------|-----------|--------------------|--------------|--------|
| F5-TTS | Yes | 0.72 | 2.1% | 335M |
| XTTS v2 | Yes | 0.65 | 3.5% | 470M |
| OpenVoice v2 | Yes | 0.70 | 2.8% | 220M |
| VALL-E 2 | Yes | 0.77 | 2.4% | 370M |
| VoiceBox | Yes | 0.78 | 2.1% | 330M |

SECS > 0.70 обычно неотличим от цели для большинства слушателей.

## Соберите это

### Шаг 1: разложите через recognition-synthesis (code-only demo в main.py)

```python
def clone_pipeline(ref_audio, text, target_embedder, tts_model):
    speaker_emb = target_embedder.encode(ref_audio)
    mel = tts_model(text, speaker=speaker_emb)
    return vocoder(mel)
```

Концептуально просто; основная масса реализации — в `tts_model` и speaker encoder.

### Шаг 2: zero-shot clone с F5-TTS

```python
from f5_tts.api import F5TTS
tts = F5TTS()
wav = tts.infer(
    ref_file="rohit_5s.wav",
    ref_text="The quick brown fox jumps over the lazy dog.",
    gen_text="Please add milk and bread to my list.",
)
```

Reference transcript должен точно соответствовать audio; mismatch ломает alignment.

### Шаг 3: voice conversion с KNN-VC

```python
import torch
from knnvc import KNNVC  # 2023 model, https://github.com/bshall/knn-vc
vc = KNNVC.load("wavlm-base-plus")
out_wav = vc.convert(source="my_voice.wav", target_pool=["alice_1.wav", "alice_2.wav"])
```

KNN-VC запускает WavLM для per-frame embeddings source и target pool, затем заменяет каждый source frame на nearest neighbor из pool. Non-parametric, работает с минутой target speech.

### Шаг 4: встройте watermark

```python
from silentcipher import SilentCipher
sc = SilentCipher(model="2024-06-01")
payload = b"consent_id:abc123;ts:1745353200"
watermarked = sc.embed(wav, sr=24000, message=payload)
detected = sc.detect(watermarked, sr=24000)   # returns payload bytes
```

~32 bits payload, детектируется после MP3 re-encode и легкого шума.

### Шаг 5: consent gate

```python
def cloned_inference(text, ref_audio, consent_record):
    assert verify_signature(consent_record), "Signed consent required"
    assert consent_record["speaker_id"] == hash_speaker(ref_audio)
    wav = tts.infer(ref_file=ref_audio, gen_text=text)
    wav = watermark(wav, payload=consent_record["id"])
    return wav
```

## Используйте это

Стек 2026 года:

| Ситуация | Выбор |
|-----------|------|
| 5-секундный zero-shot clone, open-source | F5-TTS или OpenVoice v2 |
| Commercial production cloning | ElevenLabs Instant Voice Clone v2.5 |
| Voice conversion (rewriting) | KNN-VC или Diff-HierVC |
| Many-speaker fine-tune | StyleTTS 2 + speaker adapter |
| Cross-lingual cloning | XTTS v2 или VALL-E X |
| Deepfake detection | Wav2Vec2-AASIST |

## Ловушки

- **Misaligned reference transcript.** F5-TTS и похожие модели требуют точного соответствия reference text и reference audio, включая punctuation.
- **Reverberant reference.** Echo убивает clone. Записывайте сухо, близко к микрофону.
- **Emotional mismatch.** Reference "cheerful" дает cheerful clones всего. Согласуйте emotion reference с target use.
- **Language leakage.** Клонирование English speaker и запрос French часто сохраняет accent; используйте cross-lingual models (XTTS, VALL-E X).
- **Нет watermark.** Юридически нельзя ship в EU с Aug 2026.

## Доведите до результата

Сохраните как `outputs/skill-voice-cloner.md`. Спроектируйте cloning или conversion pipeline с consent gate + watermark + целевым качеством.

## Упражнения

1. **Легко.** Запустите `code/main.py`. Он демонстрирует speaker-embedding swap, считая cosine между двумя "speakers" до и после swap.
2. **Средне.** Используйте OpenVoice v2, чтобы клонировать собственный голос. Измерьте SECS между reference и clone. Измерьте CER через Whisper.
3. **Сложно.** Примените SilentCipher watermark к 20 clones, прогоните их через 128 kbps MP3 encode+decode, детектируйте payload. Сообщите bit-accuracy.

## Ключевые термины

| Термин | Как говорят | Что это на самом деле значит |
|------|----------------|----------------------|
| Zero-shot clone | 5 секунд достаточно | Pretrained model + speaker embedding; без обучения. |
| PPG | Phonetic posteriorgram | Per-frame ASR posteriors как language-agnostic content rep. |
| KNN-VC | Nearest-neighbor conversion | Заменить каждый source frame ближайшим target-pool frame. |
| Neural codec TTS | VALL-E style | AR model над EnCodec/SoundStream tokens. |
| Watermark | Неслышимая подпись | Биты, встроенные в audio и переживающие re-encode. |
| SECS | Cloning fidelity | Cosine между target и clone speaker embeddings. |
| AASIST | Deepfake detector | Anti-spoof model; детектирует синтезированную речь. |

## Дополнительное чтение

- [Chen et al. (2024). F5-TTS](https://arxiv.org/abs/2410.06885) — open-source SOTA для zero-shot cloning.
- [Baevski et al. / Microsoft (2023). VALL-E](https://arxiv.org/abs/2301.02111) and [VALL-E 2 (2024)](https://arxiv.org/abs/2406.05370) — neural-codec TTS.
- [Qian et al. (2019). AutoVC](https://arxiv.org/abs/1905.05879) — voice conversion на disentanglement.
- [Baas, Waubert de Puiseau, Kamper (2023). KNN-VC](https://arxiv.org/abs/2305.18975) — retrieval-based VC.
- [SilentCipher (2024) — Audio Watermarking](https://github.com/sony/silentcipher) — production-ready 32-bit audio watermark.
- [ASVspoof 2025 results](https://www.asvspoof.org/) — гонка detector vs synthesizer, updated 2026.
