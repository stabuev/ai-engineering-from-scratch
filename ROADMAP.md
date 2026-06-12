# Дорожная карта

Трекер статусов для каждой фазы и каждого урока. Статусные символы в этом файле питают
сайт (`site/build.js` парсит их в `site/data.js`); не меняйте
их форму.

Общее оценочное время: ~487 часов уроков (фазы 0–18) + ~525 часов capstone-проектов, в своем темпе.

**Легенда:** ✅ Завершено &nbsp;·&nbsp; 🚧 В работе &nbsp;·&nbsp; ⬚ Запланировано

## Фаза 0: Настройка и инструменты — ✅ (~13.5 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Среда разработки](phases/00-setup-and-tooling/01-dev-environment) | ✅ | ~75 мин |
| 02 | [Git и совместная работа](phases/00-setup-and-tooling/02-git-and-collaboration) | ✅ | ~45 мин |
| 03 | [Настройка GPU и облако](phases/00-setup-and-tooling/03-gpu-setup-and-cloud) | ✅ | ~75 мин |
| 04 | [API и ключи](phases/00-setup-and-tooling/04-apis-and-keys) | ✅ | ~75 мин |
| 05 | [Jupyter Notebook](phases/00-setup-and-tooling/05-jupyter-notebooks) | ✅ | ~75 мин |
| 06 | [Окружения Python](phases/00-setup-and-tooling/06-python-environments) | ✅ | ~75 мин |
| 07 | [Docker для AI](phases/00-setup-and-tooling/07-docker-for-ai) | ✅ | ~75 мин |
| 08 | [Настройка редактора](phases/00-setup-and-tooling/08-editor-setup) | ✅ | ~75 мин |
| 09 | [Управление данными](phases/00-setup-and-tooling/09-data-management) | ✅ | ~75 мин |
| 10 | [Терминал и shell](phases/00-setup-and-tooling/10-terminal-and-shell) | ✅ | ~45 мин |
| 11 | [Linux для AI](phases/00-setup-and-tooling/11-linux-for-ai) | ✅ | ~45 мин |
| 12 | [Отладка и профилирование](phases/00-setup-and-tooling/12-debugging-and-profiling) | ✅ | ~75 мин |

## Фаза 1: Математические основы — ✅ (~22.5 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Интуиция линейной алгебры](phases/01-math-foundations/01-linear-algebra-intuition) | ✅ | ~45 мин |
| 02 | [Векторы, матрицы и операции](phases/01-math-foundations/02-vectors-matrices-operations) | ✅ | ~75 мин |
| 03 | [Матричные преобразования и собственные значения](phases/01-math-foundations/03-matrix-transformations) | ✅ | ~75 мин |
| 04 | [Математический анализ для ML: производные и градиенты](phases/01-math-foundations/04-calculus-for-ml) | ✅ | ~45 мин |
| 05 | [Правило цепочки и автоматическое дифференцирование](phases/01-math-foundations/05-chain-rule-and-autodiff) | ✅ | ~75 мин |
| 06 | [Вероятность и распределения](phases/01-math-foundations/06-probability-and-distributions) | ✅ | ~45 мин |
| 07 | [Теорема Байеса и статистическое мышление](phases/01-math-foundations/07-bayes-theorem) | ✅ | ~75 мин |
| 08 | [Оптимизация: семейство градиентного спуска](phases/01-math-foundations/08-optimization) | ✅ | ~75 мин |
| 09 | [Теория информации: энтропия, KL-дивергенция](phases/01-math-foundations/09-information-theory) | ✅ | ~45 мин |
| 10 | [Снижение размерности: PCA, t-SNE, UMAP](phases/01-math-foundations/10-dimensionality-reduction) | ✅ | ~75 мин |
| 11 | [Сингулярное разложение](phases/01-math-foundations/11-singular-value-decomposition) | ✅ | ~75 мин |
| 12 | [Тензорные операции](phases/01-math-foundations/12-tensor-operations) | ✅ | ~75 мин |
| 13 | [Численная устойчивость](phases/01-math-foundations/13-numerical-stability) | ✅ | ~45 мин |
| 14 | [Нормы и расстояния](phases/01-math-foundations/14-norms-and-distances) | ✅ | ~45 мин |
| 15 | [Статистика для ML](phases/01-math-foundations/15-statistics-for-ml) | ✅ | ~45 мин |
| 16 | [Методы выборки](phases/01-math-foundations/16-sampling-methods) | ✅ | ~75 мин |
| 17 | [Линейные системы](phases/01-math-foundations/17-linear-systems) | ✅ | ~75 мин |
| 18 | [Выпуклая оптимизация](phases/01-math-foundations/18-convex-optimization) | ✅ | ~75 мин |
| 19 | [Комплексные числа для AI](phases/01-math-foundations/19-complex-numbers) | ✅ | ~45 мин |
| 20 | [Преобразование Фурье](phases/01-math-foundations/20-fourier-transform) | ✅ | ~75 мин |
| 21 | [Теория графов для ML](phases/01-math-foundations/21-graph-theory) | ✅ | ~45 мин |
| 22 | [Стохастические процессы](phases/01-math-foundations/22-stochastic-processes) | ✅ | ~45 мин |

## Фаза 2: Основы ML — ✅ (~21 час)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Что такое машинное обучение](phases/02-ml-fundamentals/01-what-is-machine-learning) | ✅ | ~45 мин |
| 02 | [Линейная регрессия с нуля](phases/02-ml-fundamentals/02-linear-regression) | ✅ | ~75 мин |
| 03 | [Логистическая регрессия и классификация](phases/02-ml-fundamentals/03-logistic-regression) | ✅ | ~75 мин |
| 04 | [Деревья решений и случайные леса](phases/02-ml-fundamentals/04-decision-trees) | ✅ | ~75 мин |
| 05 | [Метод опорных векторов](phases/02-ml-fundamentals/05-support-vector-machines) | ✅ | ~75 мин |
| 06 | [KNN и метрики расстояния](phases/02-ml-fundamentals/06-knn-and-distances) | ✅ | ~75 мин |
| 07 | [Обучение без учителя: K-Means, DBSCAN](phases/02-ml-fundamentals/07-unsupervised-learning) | ✅ | ~75 мин |
| 08 | [Проектирование и отбор признаков](phases/02-ml-fundamentals/08-feature-engineering) | ✅ | ~75 мин |
| 09 | [Оценка моделей: метрики, кросс-валидация](phases/02-ml-fundamentals/09-model-evaluation) | ✅ | ~75 мин |
| 10 | [Смещение, дисперсия и кривая обучения](phases/02-ml-fundamentals/10-bias-variance) | ✅ | ~45 мин |
| 11 | [Ансамблевые методы: boosting, bagging, stacking](phases/02-ml-fundamentals/11-ensemble-methods) | ✅ | ~75 мин |
| 12 | [Настройка гиперпараметров](phases/02-ml-fundamentals/12-hyperparameter-tuning) | ✅ | ~75 мин |
| 13 | [ML-пайплайны и отслеживание экспериментов](phases/02-ml-fundamentals/13-ml-pipelines) | ✅ | ~75 мин |
| 14 | [Наивный Байес](phases/02-ml-fundamentals/14-naive-bayes) | ✅ | ~75 мин |
| 15 | [Основы временных рядов](phases/02-ml-fundamentals/15-time-series) | ✅ | ~45 мин |
| 16 | [Обнаружение аномалий](phases/02-ml-fundamentals/16-anomaly-detection) | ✅ | ~75 мин |
| 17 | [Работа с несбалансированными данными](phases/02-ml-fundamentals/17-imbalanced-data) | ✅ | ~75 мин |
| 18 | [Отбор признаков](phases/02-ml-fundamentals/18-feature-selection) | ✅ | ~75 мин |

## Фаза 3: Ядро deep learning — ✅ (~14.5 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Перцептрон: с чего все началось](phases/03-deep-learning-core/01-the-perceptron) | ✅ | ~45 мин |
| 02 | [Многослойные сети и прямой проход](phases/03-deep-learning-core/02-multi-layer-networks) | ✅ | ~75 мин |
| 03 | [Backpropagation с нуля](phases/03-deep-learning-core/03-backpropagation) | ✅ | ~75 мин |
| 04 | [Функции активации: ReLU, Sigmoid, GELU и зачем они нужны](phases/03-deep-learning-core/04-activation-functions) | ✅ | ~45 мин |
| 05 | [Функции потерь: MSE, Cross-Entropy, Contrastive](phases/03-deep-learning-core/05-loss-functions) | ✅ | ~45 мин |
| 06 | [Оптимизаторы: SGD, Momentum, Adam, AdamW](phases/03-deep-learning-core/06-optimizers) | ✅ | ~75 мин |
| 07 | [Регуляризация: Dropout, Weight Decay, BatchNorm](phases/03-deep-learning-core/07-regularization) | ✅ | ~75 мин |
| 08 | [Инициализация весов и стабильность обучения](phases/03-deep-learning-core/08-weight-initialization) | ✅ | ~45 мин |
| 09 | [Расписания learning rate и warmup](phases/03-deep-learning-core/09-learning-rate-schedules) | ✅ | ~45 мин |
| 10 | [Соберите собственный мини-фреймворк](phases/03-deep-learning-core/10-mini-framework) | ✅ | ~120 мин |
| 11 | [Введение в PyTorch](phases/03-deep-learning-core/11-intro-to-pytorch) | ✅ | ~75 мин |
| 12 | [Введение в JAX](phases/03-deep-learning-core/12-intro-to-jax) | ✅ | ~75 мин |
| 13 | [Отладка нейросетей](phases/03-deep-learning-core/13-debugging-neural-networks) | ✅ | ~75 мин |

## Фаза 4: Computer Vision — ✅ (~31.3 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Основы изображений: пиксели, каналы, цветовые пространства](phases/04-computer-vision/01-image-fundamentals) | ✅ | ~45 мин |
| 02 | [Свертки с нуля](phases/04-computer-vision/02-convolutions-from-scratch) | ✅ | ~75 мин |
| 03 | [CNN: от LeNet до ResNet](phases/04-computer-vision/03-cnns-lenet-to-resnet) | ✅ | ~75 мин |
| 04 | [Классификация изображений](phases/04-computer-vision/04-image-classification) | ✅ | ~75 мин |
| 05 | [Transfer learning и fine-tuning](phases/04-computer-vision/05-transfer-learning) | ✅ | ~75 мин |
| 06 | [Обнаружение объектов — YOLO с нуля](phases/04-computer-vision/06-object-detection-yolo) | ✅ | ~75 мин |
| 07 | [Семантическая сегментация — U-Net](phases/04-computer-vision/07-semantic-segmentation-unet) | ✅ | ~75 мин |
| 08 | [Instance segmentation — Mask R-CNN](phases/04-computer-vision/08-instance-segmentation-mask-rcnn) | ✅ | ~75 мин |
| 09 | [Генерация изображений — GAN](phases/04-computer-vision/09-image-generation-gans) | ✅ | ~75 мин |
| 10 | [Генерация изображений — diffusion models](phases/04-computer-vision/10-image-generation-diffusion) | ✅ | ~75 мин |
| 11 | [Stable Diffusion — архитектура и fine-tuning](phases/04-computer-vision/11-stable-diffusion) | ✅ | ~75 мин |
| 12 | [Понимание видео — временное моделирование](phases/04-computer-vision/12-video-understanding) | ✅ | ~45 мин |
| 13 | [3D vision: облака точек, NeRF](phases/04-computer-vision/13-3d-vision-nerf) | ✅ | ~45 мин |
| 14 | [Vision Transformers (ViT)](phases/04-computer-vision/14-vision-transformers) | ✅ | ~45 мин |
| 15 | [Зрение в реальном времени: edge-деплой](phases/04-computer-vision/15-real-time-edge) | ✅ | ~75 мин |
| 16 | [Соберите полный vision-пайплайн](phases/04-computer-vision/16-vision-pipeline-capstone) | ✅ | ~120 мин |
| 17 | [Self-supervised vision — SimCLR, DINO, MAE](phases/04-computer-vision/17-self-supervised-vision) | ✅ | ~75 мин |
| 18 | [Open-vocabulary vision — CLIP](phases/04-computer-vision/18-open-vocab-clip) | ✅ | ~45 мин |
| 19 | [OCR и понимание документов](phases/04-computer-vision/19-ocr-document-understanding) | ✅ | ~45 мин |
| 20 | [Поиск изображений и metric learning](phases/04-computer-vision/20-image-retrieval-metric) | ✅ | ~45 мин |
| 21 | [Обнаружение ключевых точек и оценка позы](phases/04-computer-vision/21-keypoint-pose) | ✅ | ~45 мин |
| 22 | [3D Gaussian Splatting с нуля](phases/04-computer-vision/22-3d-gaussian-splatting) | ✅ | ~90 мин |
| 23 | [Diffusion Transformers и Rectified Flow](phases/04-computer-vision/23-diffusion-transformers-rectified-flow) | ✅ | ~75 мин |
| 24 | [SAM 3 и open-vocabulary сегментация](phases/04-computer-vision/24-sam3-open-vocab-segmentation) | ✅ | ~60 мин |
| 25 | [Vision-language models (ViT-MLP-LLM)](phases/04-computer-vision/25-vision-language-models) | ✅ | ~75 мин |
| 26 | [Монокулярная глубина и оценка геометрии](phases/04-computer-vision/26-monocular-depth) | ✅ | ~60 мин |
| 27 | [Multi-object tracking и видеопамять](phases/04-computer-vision/27-multi-object-tracking) | ✅ | ~60 мин |
| 28 | [Модели мира и video diffusion](phases/04-computer-vision/28-world-models-video-diffusion) | ✅ | ~75 мин |

## Фаза 5: NLP — от основ к продвинутому уровню — ✅ (~30.5 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Обработка текста: токенизация, stemming, лемматизация](phases/05-nlp-foundations-to-advanced/01-text-processing) | ✅ | ~45 мин |
| 02 | [Bag of Words, TF-IDF и представление текста](phases/05-nlp-foundations-to-advanced/02-bag-of-words-tfidf) | ✅ | ~75 мин |
| 03 | [Word embeddings: Word2Vec с нуля](phases/05-nlp-foundations-to-advanced/03-word-embeddings-word2vec) | ✅ | ~75 мин |
| 04 | [GloVe, FastText и subword embeddings](phases/05-nlp-foundations-to-advanced/04-glove-fasttext-subword) | ✅ | ~45 мин |
| 05 | [Анализ тональности](phases/05-nlp-foundations-to-advanced/05-sentiment-analysis) | ✅ | ~75 мин |
| 06 | [Распознавание именованных сущностей (NER)](phases/05-nlp-foundations-to-advanced/06-named-entity-recognition) | ✅ | ~75 мин |
| 07 | [POS-tagging и синтаксический парсинг](phases/05-nlp-foundations-to-advanced/07-pos-tagging-parsing) | ✅ | ~45 мин |
| 08 | [Классификация текста — CNN и RNN для текста](phases/05-nlp-foundations-to-advanced/08-cnns-rnns-for-text) | ✅ | ~75 мин |
| 09 | [Sequence-to-sequence модели](phases/05-nlp-foundations-to-advanced/09-sequence-to-sequence) | ✅ | ~75 мин |
| 10 | [Механизм attention — прорыв](phases/05-nlp-foundations-to-advanced/10-attention-mechanism) | ✅ | ~45 мин |
| 11 | [Машинный перевод](phases/05-nlp-foundations-to-advanced/11-machine-translation) | ✅ | ~75 мин |
| 12 | [Суммаризация текста](phases/05-nlp-foundations-to-advanced/12-text-summarization) | ✅ | ~75 мин |
| 13 | [Системы ответов на вопросы](phases/05-nlp-foundations-to-advanced/13-question-answering) | ✅ | ~75 мин |
| 14 | [Information retrieval и поиск](phases/05-nlp-foundations-to-advanced/14-information-retrieval-search) | ✅ | ~75 мин |
| 15 | [Topic modeling: LDA, BERTopic](phases/05-nlp-foundations-to-advanced/15-topic-modeling) | ✅ | ~45 мин |
| 16 | [Генерация текста](phases/05-nlp-foundations-to-advanced/16-text-generation-pre-transformer) | ✅ | ~45 мин |
| 17 | [Чат-боты: от правил к нейросетям](phases/05-nlp-foundations-to-advanced/17-chatbots-rule-to-neural) | ✅ | ~75 мин |
| 18 | [Многоязычный NLP](phases/05-nlp-foundations-to-advanced/18-multilingual-nlp) | ✅ | ~45 мин |
| 19 | [Subword tokenization: BPE, WordPiece, Unigram, SentencePiece](phases/05-nlp-foundations-to-advanced/19-subword-tokenization) | ✅ | ~60 мин |
| 20 | [Структурированные выходы и constrained decoding](phases/05-nlp-foundations-to-advanced/20-structured-outputs-constrained-decoding) | ✅ | ~60 мин |
| 21 | [NLI и textual entailment](phases/05-nlp-foundations-to-advanced/21-nli-textual-entailment) | ✅ | ~60 мин |
| 22 | [Глубокий разбор embedding models](phases/05-nlp-foundations-to-advanced/22-embedding-models-deep-dive) | ✅ | ~60 мин |
| 23 | [Стратегии chunking для RAG](phases/05-nlp-foundations-to-advanced/23-chunking-strategies-rag) | ✅ | ~60 мин |
| 24 | [Разрешение кореференции](phases/05-nlp-foundations-to-advanced/24-coreference-resolution) | ✅ | ~60 мин |
| 25 | [Entity linking и устранение неоднозначности](phases/05-nlp-foundations-to-advanced/25-entity-linking) | ✅ | ~60 мин |
| 26 | [Извлечение отношений и построение knowledge graph](phases/05-nlp-foundations-to-advanced/26-relation-extraction-kg) | ✅ | ~60 мин |
| 27 | [Оценка LLM: RAGAS, DeepEval, G-Eval](phases/05-nlp-foundations-to-advanced/27-llm-evaluation-frameworks) | ✅ | ~75 мин |
| 28 | [Оценка длинного контекста: NIAH, RULER, LongBench, MRCR](phases/05-nlp-foundations-to-advanced/28-long-context-evaluation) | ✅ | ~60 мин |
| 29 | [Отслеживание состояния диалога](phases/05-nlp-foundations-to-advanced/29-dialogue-state-tracking) | ✅ | ~75 мин |

## Фаза 6: Speech & Audio — ✅ (~18.5 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Основы аудио: волновые формы, sampling, FFT](phases/06-speech-and-audio/01-audio-fundamentals) | ✅ | ~45 мин |
| 02 | [Спектрограммы, Mel scale и audio features](phases/06-speech-and-audio/02-spectrograms-mel-features) | ✅ | ~45 мин |
| 03 | [Классификация аудио](phases/06-speech-and-audio/03-audio-classification) | ✅ | ~75 мин |
| 04 | [Распознавание речи (ASR)](phases/06-speech-and-audio/04-speech-recognition-asr) | ✅ | ~45 мин |
| 05 | [Whisper: архитектура и fine-tuning](phases/06-speech-and-audio/05-whisper-architecture-finetuning) | ✅ | ~75 мин |
| 06 | [Распознавание и верификация говорящего](phases/06-speech-and-audio/06-speaker-recognition-verification) | ✅ | ~45 мин |
| 07 | [Text-to-Speech (TTS)](phases/06-speech-and-audio/07-text-to-speech) | ✅ | ~75 мин |
| 08 | [Клонирование и преобразование голоса](phases/06-speech-and-audio/08-voice-cloning-conversion) | ✅ | ~75 мин |
| 09 | [Генерация музыки](phases/06-speech-and-audio/09-music-generation) | ✅ | ~75 мин |
| 10 | [Audio-language models](phases/06-speech-and-audio/10-audio-language-models) | ✅ | ~45 мин |
| 11 | [Обработка аудио в реальном времени](phases/06-speech-and-audio/11-real-time-audio-processing) | ✅ | ~75 мин |
| 12 | [Соберите пайплайн голосового ассистента](phases/06-speech-and-audio/12-voice-assistant-pipeline) | ✅ | ~120 мин |
| 13 | [Нейронные аудиокодеки — EnCodec, SNAC, Mimi, DAC](phases/06-speech-and-audio/13-neural-audio-codecs) | ✅ | ~60 мин |
| 14 | [Voice activity detection и turn-taking](phases/06-speech-and-audio/14-voice-activity-detection-turn-taking) | ✅ | ~45 мин |
| 15 | [Потоковый speech-to-speech — Moshi, Hibiki](phases/06-speech-and-audio/15-streaming-speech-to-speech-moshi-hibiki) | ✅ | ~75 мин |
| 16 | [Voice anti-spoofing и аудиоводяные знаки](phases/06-speech-and-audio/16-anti-spoofing-audio-watermarking) | ✅ | ~75 мин |
| 17 | [Оценка аудио — WER, MOS, MMAU, лидерборды](phases/06-speech-and-audio/17-audio-evaluation-metrics) | ✅ | ~60 мин |

## Фаза 7: Глубокий разбор Transformers — ✅ (~16.3 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Зачем нужны Transformers: проблемы RNN](phases/07-transformers-deep-dive/01-why-transformers) | ✅ | ~45 мин |
| 02 | [Self-attention с нуля](phases/07-transformers-deep-dive/02-self-attention-from-scratch) | ✅ | ~75 мин |
| 03 | [Multi-head attention](phases/07-transformers-deep-dive/03-multi-head-attention) | ✅ | ~75 мин |
| 04 | [Позиционное кодирование: sinusoidal, RoPE, ALiBi](phases/07-transformers-deep-dive/04-positional-encoding) | ✅ | ~45 мин |
| 05 | [Полный Transformer: encoder + decoder](phases/07-transformers-deep-dive/05-full-transformer) | ✅ | ~75 мин |
| 06 | [BERT — masked language modeling](phases/07-transformers-deep-dive/06-bert-masked-language-modeling) | ✅ | ~45 мин |
| 07 | [GPT — causal language modeling](phases/07-transformers-deep-dive/07-gpt-causal-language-modeling) | ✅ | ~75 мин |
| 08 | [T5, BART — encoder-decoder модели](phases/07-transformers-deep-dive/08-t5-bart-encoder-decoder) | ✅ | ~45 мин |
| 09 | [Vision Transformers (ViT)](phases/07-transformers-deep-dive/09-vision-transformers) | ✅ | ~45 мин |
| 10 | [Audio Transformers — архитектура Whisper](phases/07-transformers-deep-dive/10-audio-transformers-whisper) | ✅ | ~45 мин |
| 11 | [Mixture of Experts (MoE)](phases/07-transformers-deep-dive/11-mixture-of-experts) | ✅ | ~45 мин |
| 12 | [KV cache, Flash Attention и оптимизация inference](phases/07-transformers-deep-dive/12-kv-cache-flash-attention) | ✅ | ~75 мин |
| 13 | [Законы масштабирования](phases/07-transformers-deep-dive/13-scaling-laws) | ✅ | ~45 мин |
| 14 | [Соберите Transformer с нуля](phases/07-transformers-deep-dive/14-build-a-transformer-capstone) | ✅ | ~120 мин |
| 15 | [Варианты Attention — Sliding Window, Sparse, Differential](phases/07-transformers-deep-dive/15-attention-variants) | ✅ | ~60 мин |
| 16 | [Speculative Decoding — черновик, проверка, повтор](phases/07-transformers-deep-dive/16-speculative-decoding) | ✅ | ~60 мин |

## Фаза 8: Generative AI — ✅ (~15.5 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Генеративные модели: таксономия и история](phases/08-generative-ai/01-generative-models-taxonomy-history) | ✅ | ~45 мин |
| 02 | [Автоэнкодеры и VAE](phases/08-generative-ai/02-autoencoders-vae) | ✅ | ~75 мин |
| 03 | [GAN: генератор против дискриминатора](phases/08-generative-ai/03-gans-generator-discriminator) | ✅ | ~75 мин |
| 04 | [Conditional GAN и Pix2Pix](phases/08-generative-ai/04-conditional-gans-pix2pix) | ✅ | ~75 мин |
| 05 | [StyleGAN](phases/08-generative-ai/05-stylegan) | ✅ | ~45 мин |
| 06 | [Diffusion models — DDPM с нуля](phases/08-generative-ai/06-diffusion-ddpm-from-scratch) | ✅ | ~75 мин |
| 07 | [Latent diffusion и Stable Diffusion](phases/08-generative-ai/07-latent-diffusion-stable-diffusion) | ✅ | ~75 мин |
| 08 | [ControlNet, LoRA и conditioning](phases/08-generative-ai/08-controlnet-lora-conditioning) | ✅ | ~75 мин |
| 09 | [Inpainting, outpainting и редактирование](phases/08-generative-ai/09-inpainting-outpainting-editing) | ✅ | ~75 мин |
| 10 | [Генерация видео](phases/08-generative-ai/10-video-generation) | ✅ | ~45 мин |
| 11 | [Генерация аудио](phases/08-generative-ai/11-audio-generation) | ✅ | ~45 мин |
| 12 | [Генерация 3D](phases/08-generative-ai/12-3d-generation) | ✅ | ~45 мин |
| 13 | [Flow matching и Rectified Flows](phases/08-generative-ai/13-flow-matching-rectified-flows) | ✅ | ~45 мин |
| 14 | [Оценка: FID, CLIP Score](phases/08-generative-ai/14-evaluation-fid-clip-score) | ✅ | ~45 мин |
| 19 | [Visual Autoregressive Modeling (VAR): Next-Scale Prediction](phases/08-generative-ai/19-visual-autoregressive-var) | 🚧 | ~90 мин |

## Фаза 9: Reinforcement Learning — ✅ (~13 часов)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [MDP, состояния, действия и награды](phases/09-reinforcement-learning/01-mdps-states-actions-rewards) | ✅ | ~45 мин |
| 02 | [Динамическое программирование](phases/09-reinforcement-learning/02-dynamic-programming) | ✅ | ~75 мин |
| 03 | [Методы Monte Carlo](phases/09-reinforcement-learning/03-monte-carlo-methods) | ✅ | ~75 мин |
| 04 | [Q-learning, SARSA](phases/09-reinforcement-learning/04-q-learning-sarsa) | ✅ | ~75 мин |
| 05 | [Deep Q-Networks (DQN)](phases/09-reinforcement-learning/05-dqn) | ✅ | ~75 мин |
| 06 | [Policy gradients — REINFORCE](phases/09-reinforcement-learning/06-policy-gradients-reinforce) | ✅ | ~75 мин |
| 07 | [Actor-critic — A2C, A3C](phases/09-reinforcement-learning/07-actor-critic-a2c-a3c) | ✅ | ~75 мин |
| 08 | [PPO](phases/09-reinforcement-learning/08-ppo) | ✅ | ~75 мин |
| 09 | [Reward modeling и RLHF](phases/09-reinforcement-learning/09-reward-modeling-rlhf) | ✅ | ~45 мин |
| 10 | [Multi-agent RL](phases/09-reinforcement-learning/10-multi-agent-rl) | ✅ | ~45 мин |
| 11 | [Sim-to-real transfer](phases/09-reinforcement-learning/11-sim-to-real-transfer) | ✅ | ~45 мин |
| 12 | [RL для игр](phases/09-reinforcement-learning/12-rl-for-games) | ✅ | ~75 мин |

## Фаза 10: LLM с нуля — ✅ (~28.4 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Токенизаторы: BPE, WordPiece, SentencePiece](phases/10-llms-from-scratch/01-tokenizers) | ✅ | ~45 мин |
| 02 | [Создание токенизатора с нуля](phases/10-llms-from-scratch/02-building-a-tokenizer) | ✅ | ~75 мин |
| 03 | [Пайплайны данных для pre-training](phases/10-llms-from-scratch/03-data-pipelines) | ✅ | ~75 мин |
| 04 | [Pre-training мини-GPT (124M)](phases/10-llms-from-scratch/04-pre-training-mini-gpt) | ✅ | ~120 мин |
| 05 | [Распределенное обучение, FSDP, DeepSpeed](phases/10-llms-from-scratch/05-scaling-distributed) | ✅ | ~75 мин |
| 06 | [Instruction tuning — SFT](phases/10-llms-from-scratch/06-instruction-tuning-sft) | ✅ | ~75 мин |
| 07 | [RLHF — reward model + PPO](phases/10-llms-from-scratch/07-rlhf) | ✅ | ~75 мин |
| 08 | [DPO — Direct Preference Optimization](phases/10-llms-from-scratch/08-dpo) | ✅ | ~75 мин |
| 09 | [Constitutional AI и self-improvement](phases/10-llms-from-scratch/09-constitutional-ai-self-improvement) | ✅ | ~45 мин |
| 10 | [Оценка — бенчмарки, evals](phases/10-llms-from-scratch/10-evaluation) | ✅ | ~75 мин |
| 11 | [Квантизация: INT8, GPTQ, AWQ, GGUF](phases/10-llms-from-scratch/11-quantization) | ✅ | ~75 мин |
| 12 | [Оптимизация inference](phases/10-llms-from-scratch/12-inference-optimization) | ✅ | ~75 мин |
| 13 | [Создание полного LLM-пайплайна](phases/10-llms-from-scratch/13-building-complete-llm-pipeline) | ✅ | ~120 мин |
| 14 | [Открытые модели: разбор архитектур](phases/10-llms-from-scratch/14-open-models-architecture-walkthroughs) | ✅ | ~45 мин |
| 15 | [Speculative decoding и EAGLE-3](phases/10-llms-from-scratch/15-speculative-decoding-eagle3) | ✅ | ~75 мин |
| 16 | [Differential Attention (V2)](phases/10-llms-from-scratch/16-differential-attention-v2) | ✅ | ~60 мин |
| 17 | [Native Sparse Attention (DeepSeek NSA)](phases/10-llms-from-scratch/17-native-sparse-attention) | ✅ | ~60 мин |
| 18 | [Multi-token prediction (MTP)](phases/10-llms-from-scratch/18-multi-token-prediction) | ✅ | ~60 мин |
| 19 | [Параллелизм DualPipe](phases/10-llms-from-scratch/19-dualpipe-parallelism) | ✅ | ~60 мин |
| 20 | [Разбор архитектуры DeepSeek-V3](phases/10-llms-from-scratch/20-deepseek-v3-walkthrough) | ✅ | ~75 мин |
| 21 | [Jamba — гибридный SSM-Transformer](phases/10-llms-from-scratch/21-jamba-hybrid-ssm-transformer) | ✅ | ~60 мин |
| 22 | [Async и Hogwild! inference](phases/10-llms-from-scratch/22-async-hogwild-inference) | ✅ | ~60 мин |
| 25 | [Speculative Decoding and EAGLE](phases/10-llms-from-scratch/25-speculative-decoding) | 🚧 | ~75 мин |
| 34 | [Gradient Checkpointing и Activation Recomputation](phases/10-llms-from-scratch/34-gradient-checkpointing) | 🚧 | ~70 мин |

## Фаза 11: LLM Engineering — ✅ (~18.8 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Prompt engineering: техники и паттерны](phases/11-llm-engineering/01-prompt-engineering) | ✅ | ~45 мин |
| 02 | [Few-shot, CoT, Tree-of-Thought](phases/11-llm-engineering/02-few-shot-cot) | ✅ | ~45 мин |
| 03 | [Структурированные выходы](phases/11-llm-engineering/03-structured-outputs) | ✅ | ~75 мин |
| 04 | [Embeddings и векторные представления](phases/11-llm-engineering/04-embeddings) | ✅ | ~75 мин |
| 05 | [Context engineering](phases/11-llm-engineering/05-context-engineering) | ✅ | ~75 мин |
| 06 | [RAG: Retrieval-Augmented Generation](phases/11-llm-engineering/06-rag) | ✅ | ~75 мин |
| 07 | [Продвинутый RAG: chunking, reranking](phases/11-llm-engineering/07-advanced-rag) | ✅ | ~75 мин |
| 08 | [Fine-tuning с LoRA и QLoRA](phases/11-llm-engineering/08-fine-tuning-lora) | ✅ | ~75 мин |
| 09 | [Function calling и использование инструментов](phases/11-llm-engineering/09-function-calling) | ✅ | ~75 мин |
| 10 | [Оценка и тестирование](phases/11-llm-engineering/10-evaluation) | ✅ | ~45 мин |
| 11 | [Кеширование, rate limiting и стоимость](phases/11-llm-engineering/11-caching-cost) | ✅ | ~45 мин |
| 12 | [Guardrails и безопасность](phases/11-llm-engineering/12-guardrails) | ✅ | ~45 мин |
| 13 | [Создание production LLM-приложения](phases/11-llm-engineering/13-production-app) | ✅ | ~120 мин |
| 14 | [Model Context Protocol (MCP)](phases/11-llm-engineering/14-model-context-protocol) | ✅ | ~75 мин |
| 15 | [Prompt caching и context caching](phases/11-llm-engineering/15-prompt-caching) | ✅ | ~60 мин |
| 16 | [LangGraph — конечные автоматы для агентов](phases/11-llm-engineering/16-langgraph-state-machines) | 🚧 | ~75 мин |
| 17 | [Компромиссы агентных фреймворков — LangGraph, CrewAI, AutoGen и Agno](phases/11-llm-engineering/17-agent-framework-tradeoffs) | 🚧 | ~45 мин |

## Фаза 12: Multimodal AI — ✅ (~67 часов)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Vision Transformers и примитив patch-token](phases/12-multimodal-ai/01-vision-transformer-patch-tokens) | ✅ | ~120 мин |
| 02 | [CLIP и contrastive vision-language pretraining](phases/12-multimodal-ai/02-clip-contrastive-pretraining) | ✅ | ~180 мин |
| 03 | [BLIP-2 Q-Former как мост между модальностями](phases/12-multimodal-ai/03-blip2-qformer-bridge) | ✅ | ~180 мин |
| 04 | [Flamingo и gated cross-attention](phases/12-multimodal-ai/04-flamingo-gated-cross-attention) | ✅ | ~120 мин |
| 05 | [LLaVA и visual instruction tuning](phases/12-multimodal-ai/05-llava-visual-instruction-tuning) | ✅ | ~180 мин |
| 06 | [Any-resolution vision — Patch-n'-Pack и NaFlex](phases/12-multimodal-ai/06-any-resolution-patch-n-pack) | ✅ | ~120 мин |
| 07 | [Рецепты open-weight VLM: что действительно важно](phases/12-multimodal-ai/07-open-weight-vlm-recipes) | ✅ | ~180 мин |
| 08 | [LLaVA-OneVision: single, multi, video](phases/12-multimodal-ai/08-llava-onevision-single-multi-video) | ✅ | ~180 мин |
| 09 | [Семейство Qwen-VL и видео с dynamic FPS](phases/12-multimodal-ai/09-qwen-vl-family-dynamic-fps) | ✅ | ~120 мин |
| 10 | [InternVL3 native multimodal pretraining](phases/12-multimodal-ai/10-internvl3-native-multimodal) | ✅ | ~120 мин |
| 11 | [Chameleon early-fusion token-only](phases/12-multimodal-ai/11-chameleon-early-fusion-tokens) | ✅ | ~180 мин |
| 12 | [Emu3 next-token prediction для генерации](phases/12-multimodal-ai/12-emu3-next-token-for-generation) | ✅ | ~120 мин |
| 13 | [Transfusion autoregressive + diffusion](phases/12-multimodal-ai/13-transfusion-autoregressive-diffusion) | ✅ | ~180 мин |
| 14 | [Show-o unified discrete diffusion](phases/12-multimodal-ai/14-show-o-discrete-diffusion-unified) | ✅ | ~120 мин |
| 15 | [Janus-Pro decoupled encoders](phases/12-multimodal-ai/15-janus-pro-decoupled-encoders) | ✅ | ~120 мин |
| 16 | [MIO any-to-any streaming](phases/12-multimodal-ai/16-mio-any-to-any-streaming) | ✅ | ~120 мин |
| 17 | [Video-language temporal grounding](phases/12-multimodal-ai/17-video-language-temporal-grounding) | ✅ | ~180 мин |
| 18 | [Длинное видео в million-token context](phases/12-multimodal-ai/18-long-video-million-token) | ✅ | ~180 мин |
| 19 | [Audio-language models: от Whisper до AF3](phases/12-multimodal-ai/19-audio-language-whisper-to-af3) | ✅ | ~180 мин |
| 20 | [Omni models: thinker-talker streaming](phases/12-multimodal-ai/20-omni-models-thinker-talker) | ✅ | ~180 мин |
| 21 | [Embodied VLA: RT-2, OpenVLA, π0, GR00T](phases/12-multimodal-ai/21-embodied-vlas-openvla-pi0-groot) | ✅ | ~180 мин |
| 22 | [Понимание документов и диаграмм](phases/12-multimodal-ai/22-document-diagram-understanding) | ✅ | ~180 мин |
| 23 | [ColPali vision-native document RAG](phases/12-multimodal-ai/23-colpali-vision-native-rag) | ✅ | ~180 мин |
| 24 | [Multimodal RAG и cross-modal retrieval](phases/12-multimodal-ai/24-multimodal-rag-cross-modal) | ✅ | ~180 мин |
| 25 | [Мультимодальные агенты и computer-use (capstone)](phases/12-multimodal-ai/25-multimodal-agents-computer-use) | ✅ | ~240 мин |

## Фаза 13: Инструменты и протоколы — ✅ (~24.8 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Интерфейс инструмента](phases/13-tools-and-protocols/01-the-tool-interface) | ✅ | ~45 мин |
| 02 | [Глубокий разбор function calling](phases/13-tools-and-protocols/02-function-calling-deep-dive) | ✅ | ~75 мин |
| 03 | [Параллельные и потоковые вызовы инструментов](phases/13-tools-and-protocols/03-parallel-and-streaming-tool-calls) | ✅ | ~75 мин |
| 04 | [Структурированный выход](phases/13-tools-and-protocols/04-structured-output) | ✅ | ~75 мин |
| 05 | [Проектирование схем инструментов](phases/13-tools-and-protocols/05-tool-schema-design) | ✅ | ~45 мин |
| 06 | [Основы MCP](phases/13-tools-and-protocols/06-mcp-fundamentals) | ✅ | ~45 мин |
| 07 | [Создание MCP-сервера](phases/13-tools-and-protocols/07-building-an-mcp-server) | ✅ | ~75 мин |
| 08 | [Создание MCP-клиента](phases/13-tools-and-protocols/08-building-an-mcp-client) | ✅ | ~75 мин |
| 09 | [Транспорты MCP](phases/13-tools-and-protocols/09-mcp-transports) | ✅ | ~45 мин |
| 10 | [MCP resources и prompts](phases/13-tools-and-protocols/10-mcp-resources-and-prompts) | ✅ | ~45 мин |
| 11 | [MCP sampling](phases/13-tools-and-protocols/11-mcp-sampling) | ✅ | ~75 мин |
| 12 | [MCP roots и elicitation](phases/13-tools-and-protocols/12-mcp-roots-and-elicitation) | ✅ | ~45 мин |
| 13 | [MCP async tasks](phases/13-tools-and-protocols/13-mcp-async-tasks) | ✅ | ~75 мин |
| 14 | [MCP apps](phases/13-tools-and-protocols/14-mcp-apps) | ✅ | ~75 мин |
| 15 | [Безопасность MCP I — tool poisoning](phases/13-tools-and-protocols/15-mcp-security-tool-poisoning) | ✅ | ~45 мин |
| 16 | [Безопасность MCP II — OAuth 2.1](phases/13-tools-and-protocols/16-mcp-security-oauth-2-1) | ✅ | ~75 мин |
| 17 | [MCP gateways и registries](phases/13-tools-and-protocols/17-mcp-gateways-and-registries) | ✅ | ~45 мин |
| 18 | [MCP auth в production — DCR + JWKS на iii](phases/13-tools-and-protocols/18-mcp-auth-production) | ✅ | ~90 мин |
| 19 | [Протокол A2A](phases/13-tools-and-protocols/19-a2a-protocol) | ✅ | ~75 мин |
| 20 | [OpenTelemetry GenAI](phases/13-tools-and-protocols/20-opentelemetry-genai) | ✅ | ~75 мин |
| 21 | [Слой маршрутизации LLM](phases/13-tools-and-protocols/21-llm-routing-layer) | ✅ | ~45 мин |
| 22 | [Skills и agent SDK](phases/13-tools-and-protocols/22-skills-and-agent-sdks) | ✅ | ~45 мин |
| 23 | [Capstone — экосистема инструментов](phases/13-tools-and-protocols/23-capstone-tool-ecosystem) | ✅ | ~120 мин |

## Фаза 14: Agent Engineering — ✅ (~42.9 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Цикл агента](phases/14-agent-engineering/01-the-agent-loop) | ✅ | ~60 мин |
| 02 | [ReWOO и plan-and-execute](phases/14-agent-engineering/02-rewoo-plan-and-execute) | ✅ | ~60 мин |
| 03 | [Reflexion и verbal reinforcement learning](phases/14-agent-engineering/03-reflexion-verbal-rl) | ✅ | ~60 мин |
| 04 | [Tree of Thoughts и LATS](phases/14-agent-engineering/04-tree-of-thoughts-lats) | ✅ | ~75 мин |
| 05 | [Self-Refine и CRITIC](phases/14-agent-engineering/05-self-refine-and-critic) | ✅ | ~60 мин |
| 06 | [Использование инструментов и function calling](phases/14-agent-engineering/06-tool-use-and-function-calling) | ✅ | ~60 мин |
| 07 | [Память — virtual context и MemGPT](phases/14-agent-engineering/07-memory-virtual-context-memgpt) | ✅ | ~75 мин |
| 08 | [Блоки памяти и sleep-time compute](phases/14-agent-engineering/08-memory-blocks-sleep-time-compute) | ✅ | ~75 мин |
| 09 | [Гибридная память — Mem0 vector + graph + KV](phases/14-agent-engineering/09-hybrid-memory-mem0) | ✅ | ~75 мин |
| 10 | [Библиотеки skills и lifelong learning — Voyager](phases/14-agent-engineering/10-skill-libraries-voyager) | ✅ | ~75 мин |
| 11 | [Планирование с HTN и evolutionary search](phases/14-agent-engineering/11-planning-htn-and-evolutionary) | ✅ | ~75 мин |
| 12 | [Паттерны workflow Anthropic](phases/14-agent-engineering/12-anthropic-workflow-patterns) | ✅ | ~60 мин |
| 13 | [LangGraph — stateful graphs и durable execution](phases/14-agent-engineering/13-langgraph-stateful-graphs) | ✅ | ~75 мин |
| 14 | [AutoGen v0.4 — actor model](phases/14-agent-engineering/14-autogen-actor-model) | ✅ | ~75 мин |
| 15 | [CrewAI — role-based crews и flows](phases/14-agent-engineering/15-crewai-role-based-crews) | ✅ | ~60 мин |
| 16 | [OpenAI Agents SDK — handoffs, guardrails, tracing](phases/14-agent-engineering/16-openai-agents-sdk) | ✅ | ~75 мин |
| 17 | [Claude Agent SDK — subagents и session store](phases/14-agent-engineering/17-claude-agent-sdk) | ✅ | ~75 мин |
| 18 | [Agno и Mastra — production runtimes](phases/14-agent-engineering/18-agno-and-mastra-runtimes) | ✅ | ~45 мин |
| 19 | [Benchmarks — SWE-bench, GAIA, AgentBench](phases/14-agent-engineering/19-benchmarks-swebench-gaia) | ✅ | ~60 мин |
| 20 | [Бенчмарки — WebArena и OSWorld](phases/14-agent-engineering/20-benchmarks-webarena-osworld) | ✅ | ~60 мин |
| 21 | [Computer use — Claude, OpenAI CUA, Gemini](phases/14-agent-engineering/21-computer-use-agents) | ✅ | ~60 мин |
| 22 | [Голосовые агенты — Pipecat и LiveKit](phases/14-agent-engineering/22-voice-agents-pipecat-livekit) | ✅ | ~60 мин |
| 23 | [Семантические соглашения OpenTelemetry GenAI](phases/14-agent-engineering/23-otel-genai-conventions) | ✅ | ~60 мин |
| 24 | [Agent observability — Langfuse, Phoenix, Opik](phases/14-agent-engineering/24-agent-observability-platforms) | ✅ | ~45 мин |
| 25 | [Multi-agent debate и collaboration](phases/14-agent-engineering/25-multi-agent-debate) | ✅ | ~60 мин |
| 26 | [Failure modes — почему агенты ломаются](phases/14-agent-engineering/26-failure-modes-agentic) | ✅ | ~60 мин |
| 27 | [Prompt injection и защита PVE](phases/14-agent-engineering/27-prompt-injection-defense) | ✅ | ~75 мин |
| 28 | [Паттерны оркестрации — supervisor, swarm, hierarchical](phases/14-agent-engineering/28-orchestration-patterns) | ✅ | ~60 мин |
| 29 | [Production runtimes — queue, event, cron](phases/14-agent-engineering/29-production-runtimes) | ✅ | ~60 мин |
| 30 | [Eval-driven разработка агентов](phases/14-agent-engineering/30-eval-driven-agent-development) | ✅ | ~60 мин |
| 31 | [Agent Workbench: почему способные модели все еще ошибаются](phases/14-agent-engineering/31-agent-workbench-why-models-fail) | ✅ | ~45 мин |
| 32 | [Минимальный Agent Workbench](phases/14-agent-engineering/32-minimal-agent-workbench) | ✅ | ~45 мин |
| 33 | [Инструкции агента как исполняемые ограничения](phases/14-agent-engineering/33-instructions-as-executable-constraints) | ✅ | ~50 мин |
| 34 | [Память репозитория и durable state](phases/14-agent-engineering/34-repo-memory-and-state) | ✅ | ~60 мин |
| 35 | [Скрипты инициализации для агентов](phases/14-agent-engineering/35-initialization-scripts) | ✅ | ~45 мин |
| 36 | [Scope contracts и границы задачи](phases/14-agent-engineering/36-scope-contracts) | ✅ | ~50 мин |
| 37 | [Runtime feedback loops](phases/14-agent-engineering/37-runtime-feedback-loops) | ✅ | ~50 мин |
| 38 | [Verification gates](phases/14-agent-engineering/38-verification-gates) | ✅ | ~55 мин |
| 39 | [Reviewer agent: отделить builder от marker](phases/14-agent-engineering/39-reviewer-agent) | ✅ | ~55 мин |
| 40 | [Multi-session handoff](phases/14-agent-engineering/40-multi-session-handoff) | ✅ | ~50 мин |
| 41 | [Workbench на реальном репозитории](phases/14-agent-engineering/41-workbench-for-real-repos) | ✅ | ~60 мин |
| 42 | [Capstone: поставьте переиспользуемый пакет Agent Workbench](phases/14-agent-engineering/42-agent-workbench-capstone) | ✅ | ~75 мин |

## Фаза 15: Автономные системы — ✅ (~20 часов)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [От чат-ботов к долгосрочным агентам (METR)](phases/15-autonomous-systems/01-long-horizon-agents) | ✅ | ~45 мин |
| 02 | [STaR, V-STaR, Quiet-STaR: самообучающееся рассуждение](phases/15-autonomous-systems/02-star-family-reasoning) | ✅ | ~60 мин |
| 03 | [AlphaEvolve: эволюционные coding agents](phases/15-autonomous-systems/03-alphaevolve-evolutionary-coding) | ✅ | ~60 мин |
| 04 | [Darwin Gödel Machine: самомодифицирующиеся агенты](phases/15-autonomous-systems/04-darwin-godel-machine) | ✅ | ~60 мин |
| 05 | [AI Scientist v2: исследования уровня workshop](phases/15-autonomous-systems/05-ai-scientist-v2) | ✅ | ~60 мин |
| 06 | [Автоматизированные alignment-исследования (Anthropic AAR)](phases/15-autonomous-systems/06-automated-alignment-research) | ✅ | ~60 мин |
| 07 | [Рекурсивное self-improvement: capability vs alignment](phases/15-autonomous-systems/07-recursive-self-improvement) | ✅ | ~60 мин |
| 08 | [Дизайны ограниченного self-improvement](phases/15-autonomous-systems/08-bounded-self-improvement) | ✅ | ~60 мин |
| 09 | [Ландшафт автономных coding agents (SWE-bench, CodeAct)](phases/15-autonomous-systems/09-coding-agent-landscape) | ✅ | ~45 мин |
| 10 | [Режимы разрешений Claude Code и auto mode](phases/15-autonomous-systems/10-claude-code-permission-modes) | ✅ | ~45 мин |
| 11 | [Браузерные агенты и indirect prompt injection](phases/15-autonomous-systems/11-browser-agents) | ✅ | ~45 мин |
| 12 | [Durable execution для долгих запусков агентов](phases/15-autonomous-systems/12-durable-execution) | ✅ | ~60 мин |
| 13 | [Бюджеты действий, лимиты итераций, cost governors](phases/15-autonomous-systems/13-cost-governors) | ✅ | ~60 мин |
| 14 | [Kill switches, circuit breakers, canary tokens](phases/15-autonomous-systems/14-kill-switches-canaries) | ✅ | ~60 мин |
| 15 | [HITL: propose-then-commit](phases/15-autonomous-systems/15-propose-then-commit) | ✅ | ~60 мин |
| 16 | [Checkpoints и rollback](phases/15-autonomous-systems/16-checkpoints-rollback) | ✅ | ~60 мин |
| 17 | [Constitutional AI и переопределения правил](phases/15-autonomous-systems/17-constitutional-ai) | ✅ | ~60 мин |
| 18 | [Llama Guard и классификация input/output](phases/15-autonomous-systems/18-llama-guard) | ✅ | ~45 мин |
| 19 | [Anthropic Responsible Scaling Policy v3.0](phases/15-autonomous-systems/19-anthropic-rsp) | ✅ | ~45 мин |
| 20 | [OpenAI Preparedness Framework и DeepMind FSF](phases/15-autonomous-systems/20-openai-preparedness-deepmind-fsf) | ✅ | ~45 мин |
| 21 | [Временные горизонты METR и внешняя оценка](phases/15-autonomous-systems/21-metr-external-evaluation) | ✅ | ~60 мин |
| 22 | [CAIS, CAISI и риски общественного масштаба](phases/15-autonomous-systems/22-cais-caisi-societal-risk) | ✅ | ~45 мин |

## Фаза 16: Multi-Agent и Swarms — ✅ (~29.3 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Зачем нужен multi-agent подход](phases/16-multi-agent-and-swarms/01-why-multi-agent) | ✅ | ~45 мин |
| 02 | [Наследие FIPA-ACL и speech acts](phases/16-multi-agent-and-swarms/02-fipa-acl-heritage) | ✅ | ~60 мин |
| 03 | [Коммуникационные протоколы](phases/16-multi-agent-and-swarms/03-communication-protocols) | ✅ | ~45 мин |
| 04 | [Примитивная multi-agent модель](phases/16-multi-agent-and-swarms/04-primitive-model) | ✅ | ~60 мин |
| 05 | [Паттерн supervisor / orchestrator-worker](phases/16-multi-agent-and-swarms/05-supervisor-orchestrator-pattern) | ✅ | ~75 мин |
| 06 | [Иерархическая архитектура и decomposition drift](phases/16-multi-agent-and-swarms/06-hierarchical-architecture) | ✅ | ~60 мин |
| 07 | [Society of Mind и multi-agent debate](phases/16-multi-agent-and-swarms/07-society-of-mind-debate) | ✅ | ~75 мин |
| 08 | [Специализация ролей — planner / critic / executor / verifier](phases/16-multi-agent-and-swarms/08-role-specialization) | ✅ | ~75 мин |
| 09 | [Параллельный swarm и сетевые архитектуры](phases/16-multi-agent-and-swarms/09-parallel-swarm-networks) | ✅ | ~60 мин |
| 10 | [Групповой чат и выбор говорящего](phases/16-multi-agent-and-swarms/10-group-chat-speaker-selection) | ✅ | ~60 мин |
| 11 | [Handoffs и routines (stateless orchestration)](phases/16-multi-agent-and-swarms/11-handoffs-and-routines) | ✅ | ~60 мин |
| 12 | [A2A — протокол agent-to-agent](phases/16-multi-agent-and-swarms/12-a2a-protocol) | ✅ | ~75 мин |
| 13 | [Shared memory и blackboard patterns](phases/16-multi-agent-and-swarms/13-shared-memory-blackboard) | ✅ | ~75 мин |
| 14 | [Консенсус и Byzantine fault tolerance](phases/16-multi-agent-and-swarms/14-consensus-and-bft) | ✅ | ~75 мин |
| 15 | [Голосование, self-consistency и debate topology](phases/16-multi-agent-and-swarms/15-voting-debate-topology) | ✅ | ~75 мин |
| 16 | [Переговоры и bargaining](phases/16-multi-agent-and-swarms/16-negotiation-bargaining) | ✅ | ~75 мин |
| 17 | [Generative agents и emergent simulation](phases/16-multi-agent-and-swarms/17-generative-agents-simulation) | ✅ | ~75 мин |
| 18 | [Theory of mind и emergent coordination](phases/16-multi-agent-and-swarms/18-theory-of-mind-coordination) | ✅ | ~75 мин |
| 19 | [Swarm optimization (PSO, ACO)](phases/16-multi-agent-and-swarms/19-swarm-optimization-pso-aco) | ✅ | ~75 мин |
| 20 | [MARL — MADDPG, QMIX, MAPPO](phases/16-multi-agent-and-swarms/20-marl-maddpg-qmix-mappo) | ✅ | ~90 мин |
| 21 | [Agent economies, token incentives, reputation](phases/16-multi-agent-and-swarms/21-agent-economies) | ✅ | ~75 мин |
| 22 | [Production scaling — очереди, checkpoints, durability](phases/16-multi-agent-and-swarms/22-production-scaling-queues-checkpoints) | ✅ | ~75 мин |
| 23 | [Failure modes — MAST, groupthink, monoculture](phases/16-multi-agent-and-swarms/23-failure-modes-mast-groupthink) | ✅ | ~75 мин |
| 24 | [Оценка и coordination benchmarks](phases/16-multi-agent-and-swarms/24-evaluation-coordination-benchmarks) | ✅ | ~75 мин |
| 25 | [Case studies и state of the art 2026 года](phases/16-multi-agent-and-swarms/25-case-studies-2026-sota) | ✅ | ~90 мин |

## Фаза 17: Infrastructure и Production — ✅ (~29 часов)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Управляемые LLM-платформы — Bedrock, Azure OpenAI, Vertex AI](phases/17-infrastructure-and-production/01-managed-llm-platforms) | ✅ | ~60 мин |
| 02 | [Экономика inference-платформ — Fireworks, Together, Baseten, Modal](phases/17-infrastructure-and-production/02-inference-platform-economics) | ✅ | ~60 мин |
| 03 | [GPU autoscaling в Kubernetes — Karpenter, KAI Scheduler](phases/17-infrastructure-and-production/03-gpu-autoscaling-kubernetes) | ✅ | ~75 мин |
| 04 | [Внутреннее устройство vLLM serving — PagedAttention, continuous batching, chunked prefill](phases/17-infrastructure-and-production/04-vllm-serving-internals) | ✅ | ~75 мин |
| 05 | [EAGLE-3 speculative decoding в production](phases/17-infrastructure-and-production/05-eagle3-speculative-decoding) | ✅ | ~60 мин |
| 06 | [SGLang и RadixAttention для prefix-heavy нагрузок](phases/17-infrastructure-and-production/06-sglang-radixattention) | ✅ | ~60 мин |
| 07 | [TensorRT-LLM на Blackwell с FP8 и NVFP4](phases/17-infrastructure-and-production/07-tensorrt-llm-blackwell) | ✅ | ~75 мин |
| 08 | [Метрики inference — TTFT, TPOT, ITL, goodput, P99](phases/17-infrastructure-and-production/08-inference-metrics-goodput) | ✅ | ~60 мин |
| 09 | [Production quantization — AWQ, GPTQ, GGUF, FP8, NVFP4](phases/17-infrastructure-and-production/09-production-quantization) | ✅ | ~75 мин |
| 10 | [Смягчение cold start для serverless LLM](phases/17-infrastructure-and-production/10-cold-start-mitigation) | ✅ | ~60 мин |
| 11 | [Multi-region LLM serving и локальность KV cache](phases/17-infrastructure-and-production/11-multi-region-kv-locality) | ✅ | ~60 мин |
| 12 | [Edge inference — ANE, Hexagon, WebGPU, Jetson](phases/17-infrastructure-and-production/12-edge-inference) | ✅ | ~60 мин |
| 13 | [Выбор стека observability для LLM](phases/17-infrastructure-and-production/13-llm-observability) | ✅ | ~60 мин |
| 14 | [Prompt caching и экономика semantic caching](phases/17-infrastructure-and-production/14-prompt-semantic-caching) | ✅ | ~60 мин |
| 15 | [Batch APIs — скидка 50% как отраслевой стандарт](phases/17-infrastructure-and-production/15-batch-apis) | ✅ | ~45 мин |
| 16 | [Model routing как примитив снижения стоимости](phases/17-infrastructure-and-production/16-model-routing) | ✅ | ~60 мин |
| 17 | [Раздельные prefill/decode — NVIDIA Dynamo и llm-d](phases/17-infrastructure-and-production/17-disaggregated-prefill-decode) | ✅ | ~75 мин |
| 18 | [Production-стек vLLM с LMCache KV offloading](phases/17-infrastructure-and-production/18-vllm-production-stack-lmcache) | ✅ | ~60 мин |
| 19 | [AI gateways — LiteLLM, Portkey, Kong, Bifrost](phases/17-infrastructure-and-production/19-ai-gateways) | ✅ | ~60 мин |
| 20 | [Shadow, canary и progressive deployment](phases/17-infrastructure-and-production/20-shadow-canary-progressive) | ✅ | ~60 мин |
| 21 | [A/B testing LLM-функций — GrowthBook и Statsig](phases/17-infrastructure-and-production/21-ab-testing-llm-features) | ✅ | ~60 мин |
| 22 | [Load testing LLM API — k6, LLMPerf, GenAI-Perf](phases/17-infrastructure-and-production/22-load-testing-llm-apis) | ✅ | ~75 мин |
| 23 | [SRE для AI — multi-agent incident response](phases/17-infrastructure-and-production/23-sre-for-ai) | ✅ | ~60 мин |
| 24 | [Chaos engineering для LLM production](phases/17-infrastructure-and-production/24-chaos-engineering-llm) | ✅ | ~60 мин |
| 25 | [Безопасность — secrets, PII scrubbing, audit logs](phases/17-infrastructure-and-production/25-security-secrets-audit) | ✅ | ~60 мин |
| 26 | [Compliance — SOC 2, HIPAA, GDPR, EU AI Act, ISO 42001](phases/17-infrastructure-and-production/26-compliance-frameworks) | ✅ | ~60 мин |
| 27 | [FinOps для LLM — unit economics и multi-tenant attribution](phases/17-infrastructure-and-production/27-finops-llms) | ✅ | ~60 мин |
| 28 | [Выбор self-hosted serving — llama.cpp, Ollama, TGI, vLLM, SGLang](phases/17-infrastructure-and-production/28-self-hosted-serving-selection) | ✅ | ~45 мин |

## Фаза 18: Этика, безопасность и alignment — ✅ (~30.8 часа)

| # | Урок | Статус | Оценка |
|---|--------|--------|------|
| 01 | [Следование инструкциям как alignment signal](phases/18-ethics-safety-alignment/01-instruction-following-alignment-signal) | ✅ | ~45 мин |
| 02 | [Reward hacking и закон Гудхарта](phases/18-ethics-safety-alignment/02-reward-hacking-goodhart) | ✅ | ~60 мин |
| 03 | [Семейство Direct Preference Optimization](phases/18-ethics-safety-alignment/03-direct-preference-optimization-family) | ✅ | ~60 мин |
| 04 | [Sycophancy как усиление RLHF](phases/18-ethics-safety-alignment/04-sycophancy-rlhf-amplification) | ✅ | ~45 мин |
| 05 | [Constitutional AI и RLAIF](phases/18-ethics-safety-alignment/05-constitutional-ai-rlaif) | ✅ | ~60 мин |
| 06 | [Mesa-optimization и deceptive alignment](phases/18-ethics-safety-alignment/06-mesa-optimization-deceptive-alignment) | ✅ | ~75 мин |
| 07 | [Sleeper agents — устойчивый обман](phases/18-ethics-safety-alignment/07-sleeper-agents-persistent-deception) | ✅ | ~60 мин |
| 08 | [In-context scheming во frontier models](phases/18-ethics-safety-alignment/08-in-context-scheming-frontier-models) | ✅ | ~60 мин |
| 09 | [Alignment faking](phases/18-ethics-safety-alignment/09-alignment-faking) | ✅ | ~60 мин |
| 10 | [AI control — безопасность несмотря на subversion](phases/18-ethics-safety-alignment/10-ai-control-subversion) | ✅ | ~75 мин |
| 11 | [Scalable oversight и weak-to-strong](phases/18-ethics-safety-alignment/11-scalable-oversight-weak-to-strong) | ✅ | ~60 мин |
| 12 | [Red-teaming: PAIR и автоматизированные атаки](phases/18-ethics-safety-alignment/12-red-teaming-pair-automated-attacks) | ✅ | ~75 мин |
| 13 | [Many-shot jailbreaking](phases/18-ethics-safety-alignment/13-many-shot-jailbreaking) | ✅ | ~45 мин |
| 14 | [ASCII art и визуальные jailbreaks](phases/18-ethics-safety-alignment/14-ascii-art-visual-jailbreaks) | ✅ | ~60 мин |
| 15 | [Indirect prompt injection](phases/18-ethics-safety-alignment/15-indirect-prompt-injection) | ✅ | ~75 мин |
| 16 | [Инструменты red-team: Garak, Llama Guard, PyRIT](phases/18-ethics-safety-alignment/16-red-team-tooling-garak-llamaguard-pyrit) | ✅ | ~75 мин |
| 17 | [WMDP и оценка dual-use capabilities](phases/18-ethics-safety-alignment/17-wmdp-dual-use-evaluation) | ✅ | ~60 мин |
| 18 | [Frontier safety frameworks — RSP, PF, FSF](phases/18-ethics-safety-alignment/18-frontier-safety-frameworks-rsp-pf-fsf) | ✅ | ~75 мин |
| 19 | [Исследования model welfare](phases/18-ethics-safety-alignment/19-model-welfare-research) | ✅ | ~45 мин |
| 20 | [Bias и representational harm](phases/18-ethics-safety-alignment/20-bias-representational-harm) | ✅ | ~60 мин |
| 21 | [Критерии fairness: group, individual, counterfactual](phases/18-ethics-safety-alignment/21-fairness-criteria-group-individual-counterfactual) | ✅ | ~60 мин |
| 22 | [Differential privacy для LLM](phases/18-ethics-safety-alignment/22-differential-privacy-for-llms) | ✅ | ~60 мин |
| 23 | [Watermarking: SynthID, Stable Signature, C2PA](phases/18-ethics-safety-alignment/23-watermarking-synthid-stable-signature-c2pa) | ✅ | ~75 мин |
| 24 | [Регуляторные frameworks: EU, US, UK, Korea](phases/18-ethics-safety-alignment/24-regulatory-frameworks-eu-us-uk-korea) | ✅ | ~75 мин |
| 25 | [EchoLeak и CVE для AI](phases/18-ethics-safety-alignment/25-echoleak-cves-for-ai) | ✅ | ~45 мин |
| 26 | [Model, system и dataset cards](phases/18-ethics-safety-alignment/26-model-system-dataset-cards) | ✅ | ~60 мин |
| 27 | [Data provenance и управление training data](phases/18-ethics-safety-alignment/27-data-provenance-training-governance) | ✅ | ~60 мин |
| 28 | [Экосистема alignment research: MATS, Redwood, Apollo, METR](phases/18-ethics-safety-alignment/28-alignment-research-ecosystem) | ✅ | ~45 мин |
| 29 | [Системы модерации: OpenAI, Perspective, Llama Guard](phases/18-ethics-safety-alignment/29-moderation-systems-openai-perspective-llamaguard) | ✅ | ~60 мин |
| 30 | [Dual-use risk: cyber, bio, chem, nuclear](phases/18-ethics-safety-alignment/30-dual-use-risk-cyber-bio-chem-nuclear) | ✅ | ~75 мин |

## Фаза 19: Capstone-проекты — ✅ (~525 часов)

| # | Проект | Статус | Оценка |
|---|---------|--------|------|
| 01 | [Terminal-native coding agent](phases/19-capstone-projects/01-terminal-native-coding-agent) | ✅ | ~35 ч |
| 02 | [RAG поверх codebase (cross-repo semantic search)](phases/19-capstone-projects/02-rag-over-codebase) | ✅ | ~30 ч |
| 03 | [Голосовой ассистент в реальном времени (ASR → LLM → TTS)](phases/19-capstone-projects/03-realtime-voice-assistant) | ✅ | ~30 ч |
| 04 | [Multimodal document QA (vision-first)](phases/19-capstone-projects/04-multimodal-document-qa) | ✅ | ~30 ч |
| 05 | [Автономный исследовательский агент (класс AI Scientist)](phases/19-capstone-projects/05-autonomous-research-agent) | ✅ | ~40 ч |
| 06 | [DevOps-агент для troubleshooting Kubernetes](phases/19-capstone-projects/06-devops-troubleshooting-agent) | ✅ | ~30 ч |
| 07 | [End-to-end пайплайн fine-tuning](phases/19-capstone-projects/07-end-to-end-fine-tuning-pipeline) | ✅ | ~35 ч |
| 08 | [Production RAG chatbot для регулируемой вертикали](phases/19-capstone-projects/08-production-rag-chatbot) | ✅ | ~30 ч |
| 09 | [Агент миграции кода (repo-level upgrade)](phases/19-capstone-projects/09-code-migration-agent) | ✅ | ~30 ч |
| 10 | [Multi-agent команда software engineering](phases/19-capstone-projects/10-multi-agent-software-team) | ✅ | ~40 ч |
| 11 | [LLM observability и eval dashboard](phases/19-capstone-projects/11-llm-observability-dashboard) | ✅ | ~25 ч |
| 12 | [Пайплайн понимания видео (scene → QA)](phases/19-capstone-projects/12-video-understanding-pipeline) | ✅ | ~30 ч |
| 13 | [MCP-сервер с registry и governance](phases/19-capstone-projects/13-mcp-server-with-registry) | ✅ | ~25 ч |
| 14 | [Inference server для speculative decoding](phases/19-capstone-projects/14-speculative-decoding-server) | ✅ | ~30 ч |
| 15 | [Constitutional safety harness + red-team range](phases/19-capstone-projects/15-constitutional-safety-harness) | ✅ | ~25 ч |
| 16 | [Автономный агент GitHub issue-to-PR](phases/19-capstone-projects/16-github-issue-to-pr-agent) | ✅ | ~30 ч |
| 17 | [Персональный AI tutor (adaptive, multimodal)](phases/19-capstone-projects/17-personal-ai-tutor) | ✅ | ~30 ч |

---

**Итого: 20 фаз, 435 уроков | 430 завершено | ~1012 часов по оценке (включая capstone-проекты)**

Хотите помочь? Выберите любой урок со статусом ⬚ и отправьте PR. См. [CONTRIBUTING.md](CONTRIBUTING.md).
