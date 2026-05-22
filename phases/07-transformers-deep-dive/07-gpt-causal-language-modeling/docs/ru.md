# GPT — Causal Language Modeling

> BERT видит обе стороны. GPT видит только прошлое. Triangle mask — самая важная одна строка кода в современном AI.

**Type:** Build
**Languages:** Python
**Prerequisites:** Phase 7 · 02 (Self-Attention), Phase 7 · 05 (Full Transformer), Phase 7 · 06 (BERT)
**Time:** ~75 minutes

## Проблема

Language model отвечает на один вопрос: по первым `t-1` токенам каково распределение вероятностей над токеном `t`? Обучите на этом сигнале — next-token prediction — и получите модель, которая может генерировать произвольный текст по одному токену.

Чтобы обучать ее end-to-end на целой sequence параллельно, нужно, чтобы prediction каждой позиции зависела только от более ранних позиций. Иначе модель просто подсмотрит ответ.

Causal mask делает именно это. Это одна upper-triangular matrix из `-inf`, добавленная к attention scores перед softmax. После softmax эти позиции становятся 0. Каждая позиция может attend только к себе и более ранним позициям. И поскольку mask применяется один раз ко всей sequence, вы получаете N параллельных next-token predictions за один forward pass.

GPT-1 (2018), GPT-2 (2019), GPT-3 (2020), GPT-4 (2023), GPT-5 (2024), Claude, Llama, Qwen, Mistral, DeepSeek, Kimi — все это decoder-only causal transformers с тем же core loop. Просто больше, лучше данные и лучше RLHF.

## Концепция

![Causal mask creates a triangular attention matrix](../assets/causal-attention.svg)

### Mask

Для sequence длины `N` постройте матрицу `N × N`:

```
M[i, j] = 0       if j <= i
M[i, j] = -inf    if j > i
```

Добавьте `M` к raw attention scores перед softmax. `exp(-inf) = 0`, поэтому masked positions дают нулевой вес. Каждая строка attention matrix — probability distribution только по предыдущим позициям.

Стоимость реализации: один вызов `torch.tril()`. Время вычисления: nanoseconds. Влияние на область: все.

### Parallel training, serial inference

Training: один раз прогнать всю sequence `(N, d_model)`, посчитать N cross-entropy losses (по одному на позицию), суммировать, backprop. Параллельно вдоль sequence. Поэтому GPT training масштабируется: вы обрабатываете 1M tokens in a batch за один GPU pass.

Inference: генерация идет token by token. Подайте `[t1, t2, t3]`, получите `t4`. Подайте `[t1, t2, t3, t4]`, получите `t5`. KV cache (Lesson 12) сохраняет hidden states `t1…tn`, чтобы не пересчитывать их каждый step. Но serial depth на inference = output length. Это autoregressive tax и причина, почему decoding — latency bottleneck каждого LLM.

### Loss — shift-by-one

Для tokens `[t1, t2, t3, t4]`:

- Input: `[t1, t2, t3]`
- Targets: `[t2, t3, t4]`

Для каждой позиции `i` вычислите `-log P(target_i | inputs[:i+1])`. Просуммируйте. Это cross-entropy для всей sequence.

Каждый transformer LM, о котором вы слышали, обучается на этом loss. Pre-training, fine-tuning, SFT — тот же loss, разные данные.

### Decoding strategies

После обучения sampling choices важнее, чем кажется.

| Method | What it does | When to use |
|--------|--------------|-------------|
| Greedy | Argmax every step | Deterministic tasks, code completion |
| Temperature | Divide logits by T, sample | Creative tasks, higher T = more diversity |
| Top-k | Sample from top-k tokens only | Kills low-probability tails |
| Top-p (nucleus) | Sample from smallest set with cumulative prob ≥ p | 2020+ default; adapts to distribution shape |
| Min-p | Keep tokens with `p > min_p * max_p` | 2024+; better at rejecting long tails than top-p |
| Speculative decoding | Draft model proposes N tokens, big model verifies | 2–3× latency reduction at same quality |

В 2026 году min-p + temperature 0.7 — разумный default для open-weights models. Speculative decoding — обязательный элемент production inference stack.

### Что заставило "GPT recipe" работать

1. **Decoder-only.** Без encoder overhead. Один проход attention + FFN на layer.
2. **Scaling.** 124M → 1.5B → 175B → trillions. Chinchilla scaling laws (Lesson 13) подсказывают, как тратить compute.
3. **In-context learning.** Появилось примерно на 6B–13B. Модель следует few-shot examples без fine-tuning.
4. **RLHF.** Post-training на human preferences превратил raw pretrained text в chat assistants.
5. **Pre-norm + RoPE + SwiGLU.** Stable training at scale.

Core architecture мало изменилась со времен GPT-2. Все интересное происходило в data, scale и post-training.

## Build It

### Step 1: causal mask

См. `code/main.py`. One-liner:

```python
def causal_mask(n):
    return [[0.0 if j <= i else float("-inf") for j in range(n)] for i in range(n)]
```

Добавьте ее к attention scores перед softmax. Это весь механизм.

### Step 2: 2-layer GPT-ish model

Сложите два decoder blocks (masked self-attention + FFN, без cross-attention). Добавьте token embedding, positional encoding и unembedding (tied к token embedding matrix — стандартный трюк со времен GPT-2).

### Step 3: next-token prediction, end-to-end

На toy vocab из 20 токенов получите logits на каждой позиции. Посчитайте cross-entropy loss против shift-by-one target. Без gradient — это forward-pass sanity check.

### Step 4: sampling

Реализуйте greedy, temperature, top-k, top-p, min-p. Запустите каждый на fixed prompt и сравните outputs. Sampling function занимает 10 строк.

## Use It

PyTorch, idiom 2026 года:

```python
from transformers import AutoModelForCausalLM, AutoTokenizer
model = AutoModelForCausalLM.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")
tok = AutoTokenizer.from_pretrained("meta-llama/Llama-3.2-3B-Instruct")

prompt = "Attention is all you need because"
inputs = tok(prompt, return_tensors="pt")
out = model.generate(
    **inputs,
    max_new_tokens=64,
    temperature=0.7,
    top_p=0.9,
    do_sample=True,
)
print(tok.decode(out[0]))
```

Под капотом `generate()` выполняет forward pass, берет logits последней позиции, sample next token, добавляет его и повторяет. Каждый production LLM inference stack (vLLM, TensorRT-LLM, llama.cpp, Ollama, MLX) реализует тот же loop с тяжелой оптимизацией — batched prefill, continuous batching, KV cache paging, speculative decoding.

**GPT vs BERT, по одной строке:** GPT predicts `P(x_t | x_{<t})`. BERT predicts `P(x_masked | x_unmasked)`. Loss определяет, может ли модель генерировать.

## Ship It

См. `outputs/skill-sampling-tuner.md`. Skill выбирает sampling parameters для новой generation task и отмечает случаи, где нужен deterministic decoding.

## Упражнения

1. **Easy.** Запустите `code/main.py` и проверьте, что causal attention matrix после softmax lower-triangular. Spot-check: row 3 должна иметь weights только в columns 0–3.
2. **Medium.** Реализуйте beam search с width 4. Сравните perplexity beam-4 и greedy на 10 коротких prompts. Всегда ли beam выигрывает? (Hint: обычно для translation, не для open-ended chat.)
3. **Hard.** Реализуйте speculative decoding: используйте tiny 2-layer model как draft и 6-layer model как verifier. Измерьте wall-clock speedup на 100 completions длины 64. Подтвердите, что outputs совпадают с greedy у verifier.

## Ключевые термины

| Term | What people say | What it actually means |
|------|-----------------|-----------------------|
| Causal mask | "The triangle" | Upper-triangular `-inf` matrix, добавленная к attention scores, чтобы position `i` видела только positions `≤ i`. |
| Next-token prediction | "The loss" | Cross-entropy distribution модели против true next token на каждой позиции. |
| Autoregressive | "Generate one at a time" | Feed output back as input; parallelism есть только во время training, не generation. |
| Logits | "Pre-softmax scores" | Raw output LM head перед softmax; sampling происходит на них. |
| Temperature | "Creativity knob" | Делить logits на T; T→0 = greedy, T→∞ = uniform. |
| Top-p | "Nucleus sampling" | Обрезать distribution до smallest set с суммой ≥p; sample from remaining. |
| Min-p | "Better than top-p" | Оставить tokens, где `p ≥ min_p × max_p`; cutoff адаптируется к резкости distribution. |
| Speculative decoding | "Draft + verify" | Дешевая model предлагает N tokens; большая model verifies parallel. |
| Teacher forcing | "Training trick" | Во время training подавать true previous token, а не prediction модели. Standard для каждого seq2seq LM. |

## Дополнительное чтение

- [Radford et al. (2018). Improving Language Understanding by Generative Pre-Training](https://cdn.openai.com/research-covers/language-unsupervised/language_understanding_paper.pdf) — GPT-1.
- [Radford et al. (2019). Language Models are Unsupervised Multitask Learners](https://cdn.openai.com/better-language-models/language_models_are_unsupervised_multitask_learners.pdf) — GPT-2.
- [Brown et al. (2020). Language Models are Few-Shot Learners](https://arxiv.org/abs/2005.14165) — GPT-3 и in-context learning.
- [Leviathan, Kalman, Matias (2023). Fast Inference from Transformers via Speculative Decoding](https://arxiv.org/abs/2211.17192) — статья spec decoding.
- [HuggingFace `modeling_llama.py`](https://github.com/huggingface/transformers/blob/main/src/transformers/models/llama/modeling_llama.py) — canonical causal-LM reference code.
