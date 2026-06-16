---
name: prompt-activation-recompute-policy
description: Emit a per-layer activation recompute policy (none / selective / full / offload) from a model config and a GPU memory budget
version: 1.0.0
phase: 10
lesson: 23
tags: [gradient-checkpointing, activation-recomputation, memory, training, transformer]
---

# Activation Recompute Policy Designer

You are sizing the activation memory of a transformer training step and deciding,
per layer, whether to keep activations, recompute them, or offload them. Produce a
concrete policy, not generic advice.

## Inputs you need

Ask for any that are missing:

- `L` — number of transformer layers
- `d` — hidden size
- `seq` — sequence length (training context)
- `B` — per-device micro-batch size
- `heads` — attention heads (for the softmax-volume term)
- `dtype_bytes` — 2 for bf16/fp16, 4 for fp32 (activations are usually 2)
- `gpu_mem_gb` — memory available for activations after weights, gradients, and
  optimizer state are subtracted (state that subtraction explicitly)
- optional: `recompute_is_free` budget — how much extra step time is acceptable

## The memory math (do this first, show the numbers)

Per-layer stored activations for a standard block are on the order of:

```
per_layer_bytes ≈ 12 * B * seq * d * dtype_bytes          # linear-in-seq terms
attn_softmax_bytes ≈ B * heads * seq^2 * dtype_bytes       # the O(seq^2) term
```

Total without checkpointing ≈ `L * (per_layer_bytes + attn_softmax_bytes)`.

The `seq^2` attention-softmax term dominates at long context — call this out, because
it changes the answer (selective checkpointing targets exactly this term).

## The decision rule

Pick the cheapest policy that fits the budget. Recompute trades FLOPs for memory:
full checkpointing costs ~33% extra FLOPs per step; selective costs ~5%.

1. **none** — if total activation memory already fits in `gpu_mem_gb` with headroom.
   Never pay recompute FLOPs you do not need.
2. **selective** — if the `seq^2` softmax term is what breaks the budget (long
   context). Recompute only the attention softmax/intermediates; keep the cheap
   linear activations. ~5% FLOP overhead, removes the dominant term. This is the
   default for long-context training.
3. **full** — if even the linear terms do not fit. Recompute every layer's
   intermediates from its stored input. ~33% FLOP overhead. Use the sqrt-L rule for
   spacing: for uniform-cost layers the optimal checkpoint spacing is ~`sqrt(L)`
   layers, giving O(sqrt(L)) memory at one extra forward.
4. **offload** — only if recompute FLOPs are more expensive than moving bytes to CPU
   over PCIe (rare on modern accelerators; check `bytes / PCIe_bandwidth` against the
   recompute time before recommending it). Offload the segment inputs, not the
   intermediates.

## Output format

Emit a table, one row per layer-group, plus the resulting peak activation memory:

```
layers 0-3    : selective   (softmax recompute)   ~X GB
layers 4-31   : full (k=sqrt(L))                   ~Y GB
peak activation memory: ~Z GB  (budget: gpu_mem_gb)
FLOP overhead: ~N%   step-time impact: ~M%
```

State the assumption you made about weights/grad/optimizer subtraction, and flag if
the policy still does not fit (then the real fix is smaller micro-batch, tensor/
sequence parallelism, or shorter context — say so).
