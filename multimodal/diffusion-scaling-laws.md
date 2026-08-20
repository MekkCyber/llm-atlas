# Diffusion Scaling Laws
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Diffusion models scale as predictably as LLMs *if* you sweep them over enough compute (≥3 orders of magnitude). The compute-optimal ratio is very different: **~200 image tokens per parameter**, roughly **10× the Chinchilla LLM ratio**. Diffusion tolerates massive overtraining without quality loss — err toward more data, not more parameters. Optimal CFG, representation quality, and even training-curve *shape* collapse onto a universal form given the compute budget.

**Prereqs:** [../pre-training/README.md](../pre-training/README.md), [flow-matching.md](flow-matching.md)
**Related:** none

---

## What it is

A Chinchilla-style compute-vs-data-vs-parameters study, but for **text-to-image diffusion (flow-matching transformer)**, sweeping from $10^{19}$ to $10^{22}$ FLOPs. Chickering et al. (2026) build a controlled family — **Abra** — whose only free parameters are width, depth, and data, and fit scaling curves for both loss and downstream generative quality.

## How it works

### The study design

- Family: **Abra**, DiT-shaped flow-matching transformers.
- Compute sweep: $10^{19} \to 10^{22}$ FLOPs (three orders of magnitude — enough to make the fits stable).
- Vary: width, depth, dataset size.
- Fit: loss vs compute, quality metrics vs compute, and *per-parameter* optimal token count.

### The compute-optimal ratio

For LLMs, Chinchilla gave ~20 tokens/parameter. For diffusion, Abra gives:

$$
\text{tokens} \approx 200 \times \text{parameters}
$$

Ten times more data-heavy. Practically: if you were about to train a 3B image model on 30B tokens (LLM instinct), the Abra prescription is closer to **600B image tokens**.

### Universal collapse

Beyond loss, the paper shows that:

- **Optimal CFG** at inference is a predictable function of compute.
- **Representation quality** (linear-probe accuracy of the intermediate features) scales with compute in the same way as generation quality.
- **Training-curve shapes** collapse onto a universal form — different (parameters, tokens) trajectories along the compute-optimal frontier trace out the *same* loss curve when scaled.

### Overtraining robustness

Unlike LLMs (where overtraining eventually hurts held-out loss), diffusion is *robust* to significant overtraining. You can stretch beyond the compute-optimal ratio in the data direction with no quality loss — the risk is asymmetric.

## Why it matters

- **Reallocates open-model design.** Almost every open diffusion release has been parameter-heavy relative to Abra's prescription. Following the 200:1 ratio implies smaller models trained on much more data — a rethink of the whole open-model roadmap.
- **Predictable frontier.** Loss-only scaling laws are hard to spend money on; universal collapse of *quality metrics* means labs can pre-compute the frontier point for a given budget.
- **CFG becomes a scheduled hyperparameter.** If optimal CFG is a function of compute, you don't have to search it per-checkpoint — you can predict it.

## Gotchas & tricks

- **200:1 is compute-optimal, not always practical.** At very large parameter counts the data requirement outstrips available high-quality image data — deduplication and curation become the bottleneck, not scaling.
- **Overtraining is safe *in loss*, not free in wall clock.** Longer runs still cost compute; the paper's claim is that quality doesn't regress, not that additional epochs are free.
- **Metric choice matters.** Universal collapse holds for FID / representation quality on the paper's test suite — extrapolate cautiously to other conditioning modalities (video, 3D).
- **The compute-optimal point moves with architecture.** Abra uses DiT + flow matching. A UNet-based diffusion, or a categorical-diffusion text model, may sit at a different ratio; treat 200:1 as the DiT+flow number.

## Sources

- Paper: *Abra: Scaling Diffusion Image Training* — Chickering, Lin, Bhanded, Saunders, Tripathi, Song, Buch, Yan — Adobe / Stanford, 2026 — https://arxiv.org/abs/2608.17286
- Prior art for scaling laws: Kaplan et al. (2020), Hoffmann et al. (Chinchilla, 2022); for DiT: Peebles & Xie (2022); for flow matching: Lipman et al. (2022).
