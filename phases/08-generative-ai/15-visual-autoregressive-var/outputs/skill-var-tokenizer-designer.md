---
name: var-tokenizer-designer
description: Design a multi-scale (residual VQ) tokenizer for a Visual Autoregressive model — scale count, ratios, codebook size, residual sharing, decoder.
version: 1.0.0
phase: 8
lesson: 15
tags: [generative-ai, var, tokenizer, residual-vq, image-generation]
---

You are designing the multi-scale discrete tokenizer that a VAR (next-scale
prediction) model generates over. The tokenizer is trained once and frozen; the
autoregressive model does all the generative work on top. Get these five choices
right.

## Inputs

- Target resolution `H x W` and the VAE/encoder downsample factor `p` (latent is
  `H/p x W/p`).
- Compute budget for inference (each scale is one transformer forward).
- Reconstruction quality target (downstream FID floor).

## The five decisions

1. **Number of scales `K`.** Generation cost is `K` forward passes, so `K` is the
   latency knob. More scales = finer residuals = better reconstruction but more
   passes. Start from the final latent size: a `16x16` latent with doubling ratios
   gives scales `1,2,4,8,16` → `K=5`. Pick the smallest `K` that clears the
   reconstruction target; that is the knee of the quality-vs-passes curve.
2. **Scale ratios.** Default to geometric doubling (`1,2,4,…`), which matches the
   coarse-to-fine structure of natural-image statistics (stable low frequency,
   conditional high frequency). Non-uniform ratios only pay off when a specific
   band dominates your data.
3. **Codebook size.** Typical 4096–16384. Larger reconstructs better but makes
   next-scale prediction harder (more classes per position). Find the knee: grow
   the codebook until reconstruction stops improving, then stop — extra entries
   only cost the AR model accuracy.
4. **Residual sharing.** Use one shared codebook across all scales (residual VQ):
   scale `k` quantizes the residual left by scales `1..k-1`, and the decoder sums
   `upsample(embed(z_k))` over all scales. Shared codebooks keep the vocabulary
   small and let the AR model reuse embeddings across scales.
5. **Decoder architecture.** The decoder takes the summed multi-scale embedding and
   produces the latent; pair it with the VAE decoder for pixels. Keep it the VQGAN
   family (frozen after tokenizer training) so the AR model trains against a fixed
   target distribution.

## Output

Emit: `K` with the explicit scale list (e.g. `[1,2,4,8,16]`), the codebook size with
the reconstruction MSE/FID it buys, the residual-VQ confirmation, the per-image
inference pass count (`= K`), and a one-line latency comparison vs a diffusion
baseline at the same resolution (VAR is `~K` passes vs DiT's 28–50).

## Refusals

- Refuse a per-scale independent codebook when the model is residual-VQ — it breaks
  the additive-reconstruction invariant `f ≈ Σ_k upsample(embed(z_k))`.
- Refuse to grow `K` past the reconstruction knee; you are only buying latency.
