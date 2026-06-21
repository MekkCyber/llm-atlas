# Taylor-Calibrate
*Depth — principled initialization for converting softmax attention layers into linear-attention layers via Taylor expansion of the softmax kernel.*

**TL;DR:** Hybrid linear-attention models (some linear-attention layers + a few retained softmax layers) are an appealing path to faster long-context inference, and the practical way to get them is to *convert* a pretrained Transformer — not to pretrain from scratch. Conversion is brittle because the linear-attention init is heuristic and starts far from the softmax teacher. **Taylor-Calibrate** (Zhou et al., U Sydney + Together AI + UC Berkeley + UT Austin + Microsoft, arXiv 2606.16429) derives the init by Taylor-expanding the softmax kernel around the teacher's operating point and matching coefficients with the linear feature map. Up to **88× improvement** in zero-shot student quality; matched-recovery training targets reached with **4.9–9.2× fewer tokens**.

**Prereqs:** [multi-head-attention](multi-head-attention.md), [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [mla](mla.md), [../inference/README.md](../inference/README.md)

---

## What it is

A closed-form initialization scheme for the feature map of a linear-attention layer being **distilled from a pretrained softmax-attention teacher**. Instead of initializing the linear-attention parameters randomly or with a hand-tuned heuristic, derive them by:

1. Identifying the teacher's typical operating point — the distribution of $Q \cdot K^\top$ values across a calibration batch.
2. Taylor-expanding the softmax kernel $\mathrm{softmax}(QK^\top / \sqrt{d})$ around that operating point.
3. Solving for the linear-attention feature map $\phi(\cdot)$ whose inner product matches the Taylor-expanded softmax kernel to first and second order.

The student then *starts* with attention outputs that approximate the teacher's, instead of starting from a random or scaled-Identity init that the distillation loss has to drag toward the teacher from a long distance.

## How it works

### The softmax-to-linear conversion problem

Softmax attention computes:

$$
\mathrm{Attn}(Q, K, V) = \mathrm{softmax}\!\left(\frac{Q K^\top}{\sqrt{d}}\right) V
$$

Linear attention replaces the softmax kernel with an inner product of feature maps $\phi(Q)$ and $\phi(K)$:

$$
\mathrm{LinAttn}(Q, K, V) = \phi(Q) \cdot (\phi(K)^\top V) / Z
$$

with $Z$ a normalization. The two are equal only when $\phi$ is chosen so that $\phi(q) \cdot \phi(k) \approx \exp(q \cdot k / \sqrt{d})$ over the relevant input regime.

### The Taylor expansion

Around an operating point $z_0$ representative of typical $q \cdot k / \sqrt{d}$:

$$
\exp(z) \approx \exp(z_0) \cdot (1 + (z - z_0) + \tfrac{1}{2}(z - z_0)^2 + \dots)
$$

Taylor-Calibrate's contribution is to **measure $z_0$ on the teacher's actual inference distribution** (rather than assuming $z_0 = 0$ as prior work does) and to solve for the feature map coefficients that match the expansion to second order over the empirical operating-point spread.

### Why prior init is brittle

Naive init schemes either (a) set $\phi(x) = x$ (identity — works only when $z \approx 0$), (b) use an ELU+1 nonlinearity (works for some distributions, fails for others), or (c) initialize randomly. None of these encode the teacher's actual operating-point statistics, so the student starts with attention outputs that may differ by orders of magnitude from the teacher — and the distillation loss has to fix the error from scratch.

## Why it matters

- **Hybrid linear-attention is the deployable long-context path.** Pretraining a new architecture from scratch is expensive; converting an existing strong Transformer is the only realistic route to a 70B+ hybrid model.
- **Up to 88× zero-shot improvement** in a representative ablation means the converted model is usable before any distillation training, opening the door to even cheaper recovery schedules.
- **4.9–9.2× fewer tokens** to hit a target quality reduces post-conversion fine-tuning cost by nearly an order of magnitude.
- **Math-grounded.** The init derivation isn't a heuristic — it's a closed-form solution to the kernel-matching problem.

## Gotchas & tricks

- **Calibration-batch choice matters.** The operating point $z_0$ is measured on a calibration batch; using a batch that doesn't reflect production inputs gives the wrong $z_0$.
- **Per-layer (and per-head) operating points differ.** The original paper computes layer/head-wise $z_0$ values; using a global mean throws away most of the benefit.
- **Doesn't replace distillation training.** Taylor-Calibrate is an *init*; downstream training still needs to happen — it just gets to start much closer to the teacher.
- **Hybrid choice (which layers stay softmax) is orthogonal.** Taylor-Calibrate calibrates the linear-attention layers; the decision of which layers to convert vs keep softmax is a separate one.
- **Sensitive to the operating-point regime.** If inference inputs push attention into a very different regime than calibration, the Taylor approximation degrades. For very long contexts this matters more.

## Sources

- Paper: *Taylor-Calibrate: Principled Initialization for Hybrid Linear Attention Distillation* — Zhongzhu Zhou, Qingyang Wu, Junxiong Wang, Mayank Mishra, Shuaiwen Leon Song, Ben Athiwaratkun, Chenfeng Xu, U Sydney + Together AI + UC Berkeley + UT Austin + Microsoft, 2026, arXiv 2606.16429.
