# Certificate-Guided Visual Routing (LOWN)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Training-free inference-time trick for *diffusion* vision-language models (DVLMs). Image tokens are 89–97% of the input but, measured on LLaDA-V, image-conditioning changes a decoded token on only **22.8% of denoising steps on average** (as low as 7.9%). LOWN predicts masked tokens *without the image* at each step and only runs the full image-conditioned forward when the no-image prediction disagrees with the last image-conditioned one. Agreement acts as a *certificate* that the image is not needed. Retains **99.6% of LLaDA-V's score** across ten benchmarks at **35.3% of decoding FLOPs**; composes with KV caching and visual-token pruning.

**Prereqs:** none
**Related:** none

---

## What it is

Diffusion VLMs (LLaDA-V family) decode by iteratively unmasking tokens over many denoising steps. At each step the model processes the full input: image tokens + current partial sequence. For most of the sequence most of the time, the image doesn't actually change the step's output — but the model doesn't know that in advance, so it processes the image every step.

LOWN turns "do I need the image on this step?" into a cheap check that is self-verifying.

## How it works

Let $p_{\text{img}}(x_t)$ and $p_{\text{no-img}}(x_t)$ be the model's predicted distribution over masked token $x_t$ with and without the image, respectively. At each denoising step:

1. **No-image forward.** Compute $p_{\text{no-img}}$ for the masked tokens — the image path is skipped entirely.
2. **Compare to last cached image-conditioned prediction.** For each masked token, if $\arg\max p_{\text{no-img}} = \arg\max p_{\text{img}}^{\text{cached}}$, the token is *certified*: the image was not changing its prediction and we can reuse the cached distribution.
3. **Image-conditioned refresh only if disagreement.** If any token disagrees, run the full image-conditioned forward to update the cache.
4. **Unmasking budget scaled by certification.** The number of tokens unmasked at this step is proportional to how many were certified — more certification allows more parallel unmasking.

Agreement certifies the earlier prediction for reuse; disagreement triggers the costly refresh.

- **Training-free.** No fine-tuning, no new parameters, no learned threshold.
- **Threshold-free.** The exact-match agreement check is the gate; there is no $\tau$ to tune.
- **Composes with KV caching and visual token pruning.** The no-image path doesn't invalidate the KV cache for the image tokens; the two optimizations stack.

## Why it matters

Diffusion-VLM inference has been *so* expensive that it's hard to deploy. 35.3% of FLOPs at 99.6% quality is immediately useful. Three characteristics are important:

- **No retraining.** You can drop it in on LLaDA-V and its descendants today.
- **No tunable threshold.** The exact-match agreement check is simple and robust; the paper does not report a bad behavior-region on it.
- **Compositional.** Visual-token pruning (reduce the number of image tokens the model sees at all) and LOWN (reduce *how often* the image is attended to) are orthogonal and stack.

Also instructive as a general pattern: when a conditional input dominates compute cost, a cheap check for "is the condition changing the output?" can gate the expensive path.

## Gotchas & tricks

- **Only diffusion VLMs.** The certificate check depends on an iterative denoising structure. AR VLMs produce one token at a time against the whole input; the same gating would need rethinking.
- **Image-conditioning disagreement rate is model-dependent.** 22.8% on average for LLaDA-V. Models whose image-conditioning changes outputs more often will see less speedup.
- **Agreement = exact token match.** More permissive agreement checks (top-$k$ overlap, KL threshold) would need tuning and might lose the training-free property.
- **Certification is per-token, not per-step.** The paper's granularity allows partial image refresh (some tokens certified, some not); hybrid behavior keeps the gating pessimistic and safe.
- **At 1-step inference this gain disappears.** The method's leverage is in multi-step diffusion decoding where re-processing the image many times is wasteful.

## Sources

- Paper: *Look Only When Needed: Certificate-Guided Visual Routing for Efficient Diffusion VLMs* — D. Cho, J. Kim, S. Shin, E. Yang, KAIST AI, 2026 — arXiv:2610.36671.
- Project: https://hyeon9698.github.io/LOWN/
- Code: https://github.com/hyeon9698/LOWN
