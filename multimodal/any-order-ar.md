# Any-Order Generation from Pretrained AR Image Models
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Pretrained raster-scan autoregressive (AR) image models already know the joint distribution over image tokens — they just lack *non-sequential conditional pathways*, which is why they can't inpaint, outpaint, or edit. A two-part recipe adapts them to any-order generation without training from scratch: (1) **future-aware positional embeddings** that resolve a target-position identifiability problem at negligible overhead, and (2) an **ordering curriculum** that walks from the pretrained raster solution to a good any-order basin. Costs 13–33% extra compute beyond pretraining (vs. discrete diffusion's ~3×), preserves KV caching and vLLM compatibility, and beats MaskGIT on inpainting (ImageNet FID 6.21 vs 10.44).

**Prereqs:** [bpe](../fundamentals/bpe.md)
**Related:** [rope](../fundamentals/rope.md), [_positional-encoding](../fundamentals/_positional-encoding.md)

---

## What it is

Discrete-diffusion image generators (MaskGIT, MAR, RAR) train from scratch on a mixture of token orderings, paying ~3× standard raster-scan AR compute. They also drop AR's inference-stack advantages — KV caching, vLLM-style batched decoding — because the per-step conditioning changes shape.

The authors' claim: *the knowledge is already there*. A pretrained AR model knows $p(x)$ for images; what's missing is the ability to express $p(x_T \mid x_S)$ for arbitrary subsets $S$ and target positions $T$. Add the two conditional-pathway ingredients and the model unlocks.

## How it works

### 1. Future-aware positional embeddings

Classical raster-scan AR encodes "I'm predicting the next token in scan order." For any-order prediction the model needs to encode "I'm predicting *this specific position*, conditional on *those other positions already being filled*." The target-position identifiability problem: without explicit position-of-target information, the model cannot distinguish between predicting pixel $(10, 7)$ and pixel $(10, 8)$ given the same context.

Future-aware PEs supply the target position as an additional embedding. "Negligible overhead" because it's a small additive component at the right layer, not a parameter blowup.

### 2. Ordering curriculum

An ordering is a permutation over image positions; raster scan is one specific permutation. A curriculum traces a stable optimization path:

- **Start:** pretrained raster-scan basin.
- **Intermediate:** mixtures of orderings close to raster but with small perturbations.
- **End:** fully arbitrary (any-order) training.

The curriculum keeps the optimizer in a well-behaved region near the pretrained solution rather than throwing the model into any-order from scratch. Compute: 13–33% beyond pretraining — about one tenth of training from scratch.

### Inference preserved

Because the model is still an AR transformer with a modified PE, the standard AR inference stack works unchanged: KV caching, vLLM, batched decoding, speculative decoding where applicable.

## Why it matters

- **Any-order AR at a fraction of the cost.** Discrete-diffusion from scratch is ~3× pretraining compute; this recipe is +13–33% on top of existing AR weights.
- **AR inference stack is preserved.** The one-line "pay 3× training to lose KV cache" tradeoff of discrete diffusion is avoided.
- **Numbers hold up.** On LlamaGen-L/XL and RAR-L/XL: ImageNet inpainting FID **6.21** vs MaskGIT 10.44; competitive with MAR-L; class-conditional FID increases by 1.2–1.9 (the price of flexibility).
- **Likely to become a default post-pretraining step** for AR image generators, in the same way instruction-tuning became a default post-training step for AR LMs.

## Gotchas & tricks

- **Future-aware PE must be injected at the right layer.** Too early and it mixes with content embeddings; too late and the attention path can't use it. The paper reports where it lands.
- **Curriculum pacing matters.** Jumping to full any-order mid-run destabilizes training; the paper's reported budget (13–33% extra) assumes a measured ramp.
- **Trades flexibility for a small class-conditional FID loss.** 1.2–1.9 FID increase on class-conditional generation is not free — the model becomes more general at a small cost.
- **Not continuous-token models.** The recipe is for *discrete-token* AR image generators (LlamaGen, RAR). Continuous-token AR (MAR) is a different beast.
- **ImageNet-scale, not frontier-model-scale.** The paper demonstrates on LlamaGen / RAR; scaling to larger models and more diverse datasets is plausible but not demonstrated.

## Sources

- Paper: *Unlocking Any-Order Generation in Pretrained Autoregressive Image Models* — R. Pramanik, M. Pedersoli, Z. Yin, Stony Brook · ILLS · ÉTS Montréal, 2026 — arXiv:2610.39293.
- Code: https://github.com/Ordered-AR/unlockedadaptation
