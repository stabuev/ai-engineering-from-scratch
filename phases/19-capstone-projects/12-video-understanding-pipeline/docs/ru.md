# Capstone 12 — Pipeline понимания видео (сцены, QA, поиск)

> Twelve Labs вывела Marengo + Pegasus в продукт. VideoDB выпустила CRUD-for-video API. AI2 Molmo 2 опубликовала open VLM checkpoints. Gemini long-context нативно обрабатывает часы видео. TimeLens-100K задал temporal grounding в масштабе. Pipeline 2026 года устоялся: scene segmentation, per-scene caption + embedding, transcript alignment, multi-vector index и query, который отвечает с timestamps (start, end) и frame previews. Капстоун: ingest 100 часов, попадание в публичные benchmarks и измерение hallucination на counting и action questions.

**Тип:** Capstone
**Языки:** Python (pipeline), TypeScript (UI)
**Пререквизиты:** Phase 4 (CV), Phase 6 (speech), Phase 7 (transformers), Phase 11 (LLM engineering), Phase 12 (multimodal), Phase 17 (infrastructure)
**Отрабатываемые фазы:** P4 · P6 · P7 · P11 · P12 · P17
**Время:** 30 часов

## Цели обучения

- Построить пайплайн понимания видео от сегментации сцен до ответов на вопросы.
- Сочетать video-VLM с длинноконтекстным поиском по кадрам.
- Отвечать на временные вопросы, опираясь на конкретные сцены.

## Проблема

Long-form video QA — самая требовательная к bandwidth multimodal problem в масштабе 2026 года. Gemini 2.5 Pro может нативно читать двухчасовое video, но ingest 100 часов видео в queryable corpus все равно требует scene-level index. Продакшен-форма объединяет scene segmentation (TransNetV2 или PySceneDetect), per-scene captioning с VLM (Gemini 2.5, Qwen3-VL-Max или Molmo 2), transcript alignment (Whisper-v3-turbo с word timestamps) и multi-vector index, который хранит caption, frame embedding и transcript рядом. Query pipeline отвечает timestamps (start, end) плюс frame previews.

Бенчмарки публичны (ActivityNet-QA, NeXT-GQA) плюс ваш собственный custom set из 100 queries. Hallucination на counting и action-type questions — известный сложный класс отказов; капстоун явно его измеряет.

## Концепция

При ingest параллельно работают три pipelines. **Scene segmentation** режет video на scenes. **VLM captioning** создает caption для каждой scene и frame embedding из keyframe. **ASR alignment** выдает word-level timestamps. Три streams объединяются по (scene_id, time range). Каждая scene получает три vector types в multi-vector index (Qdrant): caption embedding, keyframe embedding, transcript embedding.

Во время query natural-language question запускается по всем трем vectors; results объединяются через RRF; temporal-grounding adapter (TimeLens-style) уточняет window (start, end) внутри top scene. VLM synthesizer (Gemini 2.5 Pro или Qwen3-VL-Max) получает query + top scenes + cropped frames и отвечает с cited timestamps и frame preview.

Измерение hallucination важно. Counting ("how many people enter the room?") и action-type ("does the chef pour before stirring?") questions печально ненадежны. Отчитайтесь по accuracy отдельно от descriptive questions.

## Архитектура

```
video file / URL
      |
      v
PySceneDetect / TransNetV2  (scene segmentation)
      |
      +--- per-scene keyframe --- VLM caption + frame embedding
      |                            (Gemini 2.5 Pro / Qwen3-VL-Max / Molmo 2)
      |
      +--- audio channel --- Whisper-v3-turbo ASR + word timestamps
      |
      v
multi-vector Qdrant: {caption_emb, keyframe_emb, transcript_emb}
      |
query:
  dense queries against all three -> RRF merge -> top-k scenes
      |
      v
TimeLens / VideoITG temporal grounding (refine start/end within scene)
      |
      v
VLM synth: query + top scenes + frame previews
      |
      v
answer + (start, end) timestamps + frame thumbs + citations
```

## Стек

- Scene segmentation: TransNetV2 (state-of-the-art 2024-26) или PySceneDetect
- ASR: Whisper-v3-turbo через faster-whisper с word timestamps
- VLM captioner + answerer: Gemini 2.5 Pro или Qwen3-VL-Max или Molmo 2
- Temporal grounding: adapter, обученный на TimeLens-100K, или VideoITG
- Index: Qdrant с multi-vector support (caption / frame / transcript)
- UI: Next.js 15 с HTML5 video player и scene thumbnails
- Eval: ActivityNet-QA, NeXT-GQA, custom hand-labeled set из 100 questions
- Hallucination benchmark: counting и action-type subsets с hand labels

## Соберите это

1. **Ingest walker.** Принимайте YouTube URLs или local MP4s. При необходимости downscale до 720p. Persist `{video_id, file_path}`.

2. **Scene segmentation.** Запустите TransNetV2 или PySceneDetect, чтобы получить `[{scene_id, start_ms, end_ms, keyframe_path}]`. Цель на 100 часов: ~6k-8k scenes.

3. **ASR pass.** Запустите Whisper-v3-turbo на audio; экспортируйте word-level timestamps; разделите на per-scene transcript slices.

4. **VLM captioning.** Для каждой scene вызовите Gemini 2.5 Pro (или Qwen3-VL-Max) с keyframe и коротким caption template. Создайте caption + frame embedding.

5. **Multi-vector index.** Qdrant collection с тремя named vectors. Payload: `{video_id, scene_id, start_ms, end_ms, keyframe_url}`.

6. **Query.** Natural-language question запускает три dense queries; объедините через reciprocal rank fusion; top-k=5 scenes.

7. **Temporal grounding.** Запустите TimeLens-style adapter на top scene, чтобы уточнить window (start, end) внутри scene.

8. **VLM synth.** Вызовите Gemini 2.5 Pro с query + top-3 scene clips (как images или short clips) + transcripts. Требуйте citations `(video_id, start_ms, end_ms)`.

9. **Eval.** Запустите ActivityNet-QA и NeXT-GQA. Постройте custom set из 100 queries. Отчитайтесь по overall accuracy + breakdown по class (counting, action, descriptive).

## Используйте это

```
$ video-qa ask --url=https://youtube.com/watch?v=X "how many cars pass the intersection in the first minute?"
[scene]    23 scenes detected
[asr]      transcript complete, 4m12s
[index]    69 vectors written (23 scenes x 3)
[query]    top scene: scene 3 [01:32-01:54], confidence 0.84
[ground]   refined window: [00:12-00:58]
[synth]    gemini 2.5 pro, 1.4s
answer:    5 cars pass the intersection between 00:12 and 00:58.
citations: [scene 3: 00:12-00:58]
          [frame preview at 00:14, 00:27, 00:44, 00:51, 00:57]
```

## Сдайте это

`outputs/skill-video-qa.md` — deliverable. Для заданного YouTube URL или uploaded video pipeline индексирует scenes и отвечает на questions с timestamped citations.

| Вес | Критерий | Как измеряется |
|:-:|---|---|
| 25 | Temporal grounding IoU | Intersection-over-union на held-out grounding set |
| 20 | QA accuracy | NeXT-GQA и custom 100-query |
| 20 | Ingest throughput | Hours of video per dollar spent |
| 20 | UI and citation UX | Timestamp links, thumbnail strip, jump-to-frame |
| 15 | Hallucination rate | Accuracy для counting и action-type отдельно |
| **100** | | |

## Упражнения

1. Замените Gemini 2.5 Pro на Qwen3-VL-Max в captioning pass. Отчитайтесь о разнице в caption quality на human-rated sample из 50 scenes.

2. Сократите per-scene frame embedding до одного pooled vector вместо multi-vector. Измерьте retrieval regression.

3. Постройте режим "counting strict": synthesizer извлекает каждый counted instance с timestamp, а user кликает для проверки. Измерьте, снижает ли user-verification hallucination.

4. Проведите benchmark ingest cost: hours-of-video-per-dollar для трех VLM choices. Выберите оптимальный вариант.

5. Добавьте speaker-diarized transcript: запустите pyannote speaker diarization на audio и embed per-speaker transcripts. Продемонстрируйте queries вида "what did Alice say about X?".

## Ключевые термины

| Термин | Как говорят | Что это на самом деле означает |
|------|-----------------|------------------------|
| Scene segmentation | "Shot detection" | Нарезка video на scenes по shot boundaries |
| Multi-vector index | "Caption + frame + transcript" | Qdrant collection с named vectors для каждого representation |
| Temporal grounding | "When exactly did it happen" | Уточнение window (start, end) для query answer |
| Frame embedding | "Visual representation" | Vector embedding keyframe; используется для scene-visual similarity |
| RRF fusion | "Reciprocal rank fusion" | Merge strategy для нескольких ranked lists; классический hybrid-retrieval прием |
| Counting hallucination | "Miscount" | Известный failure mode VLMs на questions "how many X" |
| ActivityNet-QA | "Video-QA benchmark" | Benchmark accuracy для long-form video QA |

## Дополнительное чтение

- [AI2 Molmo 2](https://allenai.org/blog/molmo2) — open VLM checkpoints
- [TimeLens (CVPR 2026)](https://github.com/TencentARC/TimeLens) — temporal grounding в масштабе
- [Gemini Video long-context](https://deepmind.google/technologies/gemini) — hosted reference
- [VideoDB](https://videodb.io) — CRUD-for-video API reference
- [Twelve Labs Marengo + Pegasus](https://www.twelvelabs.io) — commercial reference
- [TransNetV2](https://github.com/soCzech/TransNetV2) — model для scene segmentation
- [PySceneDetect](https://github.com/Breakthrough/PySceneDetect) — классическая open alternative
- [ActivityNet-QA](https://arxiv.org/abs/1906.02467) — reference eval benchmark
