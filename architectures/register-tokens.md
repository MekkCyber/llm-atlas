# Register Tokens
*Depth — extra learned tokens appended to a transformer's input that soak up "junk" activations and clean up feature maps.*

**TL;DR:** Vision Transformers (ViTs) develop **high-norm outlier patch tokens** during training — patches whose activations dominate feature maps and degrade downstream use (segmentation, feature matching, detection). **Register tokens** (Darcet et al., 2023) are a simple fix: prepend a small number of learned tokens to the input sequence that the model can dump those high-norm activations into, freeing the real patch tokens to carry clean features. Registers helped ViTs and are now shown to also help **Diffusion Transformers (DiTs)** — with an interesting twist: DiTs don't exhibit the ViT-style patch-token outliers, yet they still benefit, and *more so in pixel-space DiTs than in latent-space DiTs*. **Register Guidance** amplifies the register contribution at inference to further improve visual structure.

**Prereqs:** [transformer-block](transformer-block.md)
**Related:** [multi-head-attention](multi-head-attention.md), [qk-norm](qk-norm.md)

---

## What it is

Register tokens are `N` (typically 4–16) additional learned embeddings prepended (or appended) to a transformer's input sequence. They receive full attention like ordinary tokens but are stripped from the output before downstream heads see it. Functionally, they are a "scratch space" the model uses to store information that doesn't belong to any real input position.

The original ViT observation (Darcet et al.): as ViTs train, a small number of patch tokens end up with disproportionately large activation norms. These outliers correlate with degraded feature quality on dense-prediction downstream tasks. Adding register tokens gives the model somewhere else to put that information; the outliers move to the registers and the real patches stay clean.

## How it works

**Mechanically trivial.** Extend the input sequence:

```
input = [reg_1, reg_2, ..., reg_N, patch_1, patch_2, ..., patch_P]
```

where `reg_i` are learned parameters. Standard attention across the whole sequence. Output: discard the register positions, use the patch outputs downstream. Nothing else changes.

Empirically what happens during training:
- The high-norm outlier activations that previously landed on patch tokens instead concentrate on the register tokens.
- Patch-token feature maps become smoother and higher quality for dense prediction.
- Attention maps of the register tokens display more "global" patterns — they aggregate information across the image.

### The DiT twist

Starodubcev et al. (2026, [arXiv 2605.16147](https://arxiv.org/abs/2605.16147)) examine registers in Diffusion Transformers:

- **DiTs don't exhibit the ViT patch-token outlier pattern**, so the original *motivation* for registers doesn't apply here.
- **But registers still help DiTs.** And more in *pixel-space* DiTs than in *latent-space* DiTs.
- Analysis of intermediate representations shows that register tokens in DiTs produce **cleaner feature maps at high noise levels** — the regime where pixel-space DiTs are most fragile.
- Recent pixel-space DiT architectures already contain implicit register-like structures (e.g., dedicated global-info tokens), which the authors argue partially explains their strong empirical performance.

They propose **Register Guidance**: at inference time, amplify the contribution of the register-token activations responsible for global visual structure — analogous to how classifier-free guidance amplifies conditional over unconditional predictions. Improves coherence and structure of generated samples.

## Why it matters

- **Free feature-map cleanup.** A handful of extra tokens costs almost nothing and consistently improves downstream feature quality across ViT-like backbones.
- **Cheap architectural add-on for DiTs.** Slots into existing DiT stacks without recipe changes and helps most where DiTs are weakest (pixel-space, high noise levels).
- **Register Guidance opens a new inference knob.** DiT sampling has had a limited set of steerable levers (CFG, scheduler); Register Guidance adds another that trades off differently.
- **Explains implicit design patterns.** Several pixel-space DiT architectures include tokens that turn out to be de facto registers — naming the pattern makes it deliberate rather than accidental.

## Gotchas & tricks

- **Number of registers is a hyperparameter.** ViT literature uses 4–16; more is not obviously better. Ablate on the target task.
- **Position matters mildly.** Prepending vs. appending gives similar results in most setups; consistency across training and inference is more important than the choice.
- **Don't feed register outputs to downstream heads.** They're internal scratch space; passing them through classification/generation heads adds noise.
- **DiT registers work differently from ViT registers.** In ViTs they mop up outliers; in DiTs they produce cleaner high-noise feature maps. Don't assume the mechanism transfers.
- **Register Guidance interacts with CFG.** Both guidance schemes affect the sampling trajectory; tune them jointly rather than in isolation.

## Sources

- Paper: *Vision Transformers Need Registers* — Darcet, Oquab, Mairal, Bojanowski, Meta AI, 2023 — arXiv 2309.16588. Original register-tokens paper for ViTs.
- Paper: *Registers Matter for Pixel-Space Diffusion Transformers* — Starodubcev, Sudakov, Drobyshevskiy, Babenko, Baranchuk, Yandex Research, 2026 — [arXiv 2605.16147](https://arxiv.org/abs/2605.16147). Extends registers to DiTs and introduces Register Guidance.
