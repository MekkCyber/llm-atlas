# Distribution Matching Distillation (DMD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A score-based distillation technique that turns a multi-step diffusion teacher into a one- or few-step generator by minimizing the **reverse KL** between the generator's output distribution and the teacher's. Rather than matching per-sample outputs (as in classical knowledge distillation), DMD matches the *score fields* — the gradients of log-density — via a difference of two scores (real and fake) estimated by a small network. The 2026 paper **IDM** (Intrinsic Distribution Matching) extends this: it decomposes the reverse-KL field into a Gaussian-moment component and a non-Gaussian residual, and adds an explicit moment-matching loss using the generator's EMA features.

**Prereqs:** none
**Related:** [rejection-sampling](rejection-sampling.md)

---

## What it is

Diffusion models generate by iteratively denoising, taking tens to hundreds of network evaluations. **Distillation** compresses this to one or a few evaluations.

DMD's trick is to match **distributions** instead of **samples**. The reverse-KL between the student's output distribution $p_\theta$ and the teacher's $p_{\text{teacher}}$ has gradient

$$
\nabla_\theta \mathrm{KL}(p_\theta \,\|\, p_{\text{teacher}}) = \mathbb{E}_{x \sim p_\theta} \!\left[\, (s_\theta(x) - s_{\text{teacher}}(x)) \cdot \nabla_\theta x \,\right]
$$

where $s(x) = \nabla_x \log p(x)$ is the score. The score difference is estimated by a small "fake-score" network trained in parallel. The entire pipeline matches the generator's distribution against the teacher's without ever producing paired (one-step, many-step) samples.

## How it works

### Classical DMD (Yin et al. 2023)

- A **fake-score network** $s_\phi(x)$ is trained to approximate the generator's current output distribution's score.
- A frozen **teacher score** $s_{\text{teacher}}(x)$ (from the full diffusion model) represents the target.
- Generator gradient uses the score *difference* $s_\phi(x) - s_{\text{teacher}}(x)$.

### IDM (2026) — the current frontier

IDM's contribution is a decomposition and an added loss term:

1. **Decompose the reverse-KL field** into (a) a Gaussian-moment component (first and second moments of a chosen feature) and (b) a non-Gaussian residual.
2. **Match the moment component explicitly.** Add a loss that forces the student's feature mean and per-coordinate standard deviation to equal the teacher's.
3. **Use the student's own EMA copy as the feature extractor.** No external DINOv2 or CLIP needed. The EMA features are competitive with DINOv2 across the paper's five metrics.
4. **Keep the full DMD correction on top.** The moment loss supplements the score-difference signal; it does not replace it.

### Training loop

```
for step:
    noise z, sample from student x = G_θ(z)
    update fake-score s_φ on x (one step)
    L_dmd = <(s_φ(x) - s_teacher(x)), x>   # score-difference term
    L_moment = ||mean(f(x)) - mean(f(x_teacher))||² + ||std(f(x)) - std(f(x_teacher))||²
               with f = student_EMA_features
    L = L_dmd + λ · L_moment
    update G_θ
```

## Why it matters

Classical score-based distillation collapses on text-to-image models that are supposed to render specific details — the score difference supervises direction but not the explicit moments of features (text fidelity, color balance). IDM shows that explicitly supervising those moments adds signal cleanly without extra model residency. On Z-Image-Turbo 6B and Qwen-Image 20B at 2 NFE:

- **94.25% / 90.20%** on Chinese/English LongText-Bench — best among compared methods at that budget.
- Ablations isolate the moment loss contribution.
- EMA features competitive with DINOv2 across all five metrics tested.

## Gotchas & tricks

- **Fake-score network has to track the generator.** If it lags, the score difference becomes stale and training stalls or diverges.
- **Reverse-KL mode-covering vs mode-seeking.** DMD uses reverse KL, which is mode-seeking — the student locks onto the teacher's dominant modes and may miss rare ones. The literature has forward-KL variants that trade this off.
- **Moment matching uses *generator* features**, not teacher features: self-anchored. This avoids needing a frozen external feature extractor but ties quality to how discriminative the student EMA's features are.
- **LoRA fine-tuning of the teacher works.** IDM is evaluated with LoRA on both backbones, so you don't have to hold the full teacher resident.
- **Few-step, not one-step is the easier regime.** At 2 NFE you have some iteration budget; at 1 NFE every method's gap to the teacher widens.

## Sources

- Paper: *Intrinsic Distribution Matching* — D. Liu, Y. Han, K. Zheng et al., Inclusion AI · Westlake · Tsinghua · Zhejiang · SII · MBZUAI, 2026 — arXiv:2610.39633 — the moment-decomposition extension.
- Paper: *One-step Diffusion with Distribution Matching Distillation* — Yin, Gharbi, Pan et al., 2023 — the original DMD (image generation).
- Code: https://github.com/inclusionAI/IDM
