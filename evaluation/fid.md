# FID (Fréchet Inception Distance)
*Depth — the de facto generative-image evaluation metric, and its hidden randomness.*

**TL;DR:** **FID** measures distributional distance between generated and real images by fitting Gaussians to **InceptionV3-pool3 features** of each set and computing the Fréchet distance between the two Gaussians. Heusel et al., 2017. Despite being the field's default for almost a decade, FID is **noisier than the way it's reported**: the FID Lottery paper (Dufour, Efros, Pérez, 2026) shows that **retraining a generative model with a different seed moves FID 3.2× more** (in Inception feature space) than redrawing samples from a fixed network — so single-seed comparisons in most papers fall inside training-seed noise.

**Prereqs:** *(none)*
**Related:** [../multimodal/README.md](../multimodal/README.md)

---

## What it is

For a real set $R$ and generated set $G$:

1. Run each image through **InceptionV3** trained on ImageNet, extract the 2048-dim pool3 features.
2. Fit a Gaussian $\mathcal{N}(\mu_R, \Sigma_R)$ to real features and $\mathcal{N}(\mu_G, \Sigma_G)$ to generated features.
3. **FID** is the Fréchet (Wasserstein-2) distance between the two Gaussians:

$$
\text{FID} = \lVert \mu_R - \mu_G \rVert_2^2 + \mathrm{tr}(\Sigma_R + \Sigma_G - 2(\Sigma_R \Sigma_G)^{1/2})
$$

Lower is better. Reported on a fixed reference set (ImageNet 256×256 train, FFHQ, COCO, etc.).

Conventions for class-conditional ImageNet: 50k generated samples vs 50k real samples, fixed split.

## How it works

### Why an Inception backbone

The metric is **representation-relative**: distances are measured in a perception-friendly feature space, not pixel space. InceptionV3-pool3 was a reasonable choice in 2017 (ImageNet-pretrained, semantically rich) and froze as the standard. Modern variants (FID-CLIP, FID-DINO) exist but have not displaced the original.

### Variants in active use

| Variant | Backbone | Use |
|---|---|---|
| FID | InceptionV3 (pool3, ImageNet) | the original; reported by ~every generative-image paper |
| sFID | InceptionV3 (spatial features) | reported alongside FID for finer-grained accuracy |
| FID-CLIP | CLIP ViT-B/32 | less sensitive to ImageNet bias |
| FID-DINO | DINO ViT | newer, less Inception-biased |
| FVD | I3D video backbone | video-generation analog |

## Why it matters

FID is the **single number** the whole generative-image field has been comparing against. Reproducibility of FID directly underwrites reproducibility of the field's progress claims.

The **FID Lottery** paper treats FID as a random variable on a two-axis panel of **training seed × generation seed**, measured across several hundred SiT networks trained on class-conditional ImageNet 256×256:

- **Generation-seed variance.** Re-sampling from one fixed network gives a small, controllable FID range.
- **Training-seed variance.** Retraining the same recipe with a different seed moves FID **3.2× more** (in Inception feature space) than re-sampling.

Implication: a paper reporting a 0.3-point FID improvement from a single training seed is almost certainly **inside training-seed noise** — i.e. not significant. Multi-seed FID is the only honest way to compare recipes.

## Gotchas & tricks

- **Use the same reference set.** "ImageNet 256×256 FID" requires the standard train split with the standard preprocessing — variations in resize / center-crop / clip mode shift FID by points, not fractions.
- **Number of samples matters.** Below ~10k generated samples, FID is biased high (small-sample Gaussian fit). Pin to 50k for ImageNet, 10k–50k for smaller datasets, and never compare across different sample counts.
- **InceptionV3 is ImageNet-biased.** Generated samples that don't look like ImageNet classes (e.g. text-to-image) get systematic FID penalties not tied to perceived quality.
- **FID is **not** a likelihood.** It's a distance, not a density. It can't be compared across resolutions or modalities without a different backbone.
- **Single-seed reports are likely noise.** Per the FID Lottery: always report at least 3-seed mean + std, or quote a confidence interval. **Do not** compare a single-seed `A=2.0` against a single-seed `B=2.2` and conclude A wins.
- **Training-seed dominance.** Most papers vary the *sampling* seed when assessing variance — but the FID Lottery shows training-seed variance dominates by 3.2×. Sampling-seed error bars **understate** real uncertainty by that factor.

## Sources

- Paper: *GANs Trained by a Two Time-Scale Update Rule Converge to a Local Nash Equilibrium* — Heusel, Ramsauer, Unterthiner, Nessler, Hochreiter, NeurIPS 2017 — original FID definition.
- Paper: *The FID Lottery: Quantifying Hidden Randomness in Generative-Model Evaluation* — Dufour, Efros, Pérez (Kyutai, UC Berkeley), 2026, arXiv 2606.20536 — the seed-variance characterization driving this page.
- Reference: [LiveCodeBench](livecodebench.md) for an example of contamination-resistant eval; analogous methodology gap for image-generation evals.
