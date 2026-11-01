# LOWN: Certificate-Guided Visual Routing for Diffusion VLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion VLMs (LLaDA-V and friends) re-encode the full image at every denoising step even though removing the image changes an unmasked token at only ~22.8% of steps. LOWN is a training-free decoder that first predicts masked tokens *without* the image, and only re-runs the image-conditioned pass when the text-only prediction disagrees with the previous image-conditioned one. On 10 benchmarks it keeps 99.6% of LLaDA-V's score at 35.3% of the decoding FLOPs.

**Prereqs:** [README.md](README.md)
**Related:** [../inference/README.md](../inference/README.md)

---

## What it is

Diffusion VLMs mask text tokens and iteratively unmask them conditioned on an image. The image occupies 89–97% of input tokens, and each denoising step runs the full image through the model. Empirically, most steps don't change *any* unmasked token when the image is removed — the model is confident enough from context alone. LOWN exploits this with a two-pass decode-step that runs the cheap text-only pass first and spends the expensive image-conditioned pass only when it disagrees.

## How it works

At denoising step $s$, with previous image-conditioned prediction $p^{\text{img}}_{s-1}$:

1. **Text-only pass.** Run the model on the current partially-masked sequence *without* the image. Get prediction $p^{\text{text}}_s$.
2. **Agreement check (the "certificate").** For each currently-masked position, check whether $p^{\text{text}}_s$ agrees with $p^{\text{img}}_{s-1}$'s argmax.
3. **Reuse or re-run.**
   - *If a position agrees:* trust the text-only prediction and unmask that position (reusing the previous image-conditioned logits).
   - *If all positions agree:* skip the image-conditioned pass entirely this step.
   - *If any position disagrees:* run the full image-conditioned pass, update $p^{\text{img}}_s$, and unmask from it.
4. **Dynamic unmask schedule.** The number of agreeing tokens at step $s$ determines how many positions are unmasked that step — adaptive instead of a fixed schedule.

No weight changes. Composes with KV caching and with existing visual-token-pruning methods.

## Why it matters

Visual-token pruning (previous line of work) attacks the *per-step* image cost by throwing out visual tokens. LOWN attacks the *per-step image-pass frequency*. The two compose cleanly. The 35.3% FLOPs / 99.6% quality operating point is strong, and the certificate idea generalizes: *"only run the expensive conditioner when a cheap predictor disagrees with its previous output."*

## Gotchas & tricks

- **Agreement = argmax match.** Softer agreement criteria (e.g. KL below a threshold) are mentioned as a sensitivity axis; argmax is the paper's default.
- **The first denoising step always needs the image.** You need a $p^{\text{img}}_0$ to compare against.
- **The adaptive unmask schedule matters.** Fixing the unmask count ignores the "number of agreeing tokens" signal and loses most of the gain.
- **Composable with visual-token pruning.** The paper shows additional ~1.3× speedup when stacked with pruning.
- **Not a draft-and-verify speculative decoder.** Here the "draft" is the text-only pass and the "verifier" is reused logits, not a second forward.
- **DVLM-specific.** The pattern — most steps don't need the condition — may not transfer cleanly to autoregressive VLMs where each token sees fresh image attention.

## Sources

- Paper: *Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs* — Cho, Kim, Shin, Yang — KAIST AI, 2026 — [arXiv:2610.36671](https://arxiv.org/abs/2610.36671).
- Reference: *LLaDA-V* — the diffusion-VLM baseline.
