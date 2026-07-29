# Model-Intrinsic Reward for Compositional Diffusion (TILT)
*Depth — a training-free test-time alignment method that steers diffusion sampling using a reward defined entirely from the base model's own conditional distributions.*

**TL;DR:** Improve compositional prompt-following ("a red cube on top of a blue cylinder next to a green sphere") in text-to-image diffusion **without** an external classifier or reward model. TILT interprets compositional failures as overlap-mode failures between joint and single-concept distributions of the base model, defines a reward from those internal distributions, and steers sampling toward a **KL-constrained tilted target** with closed-form guidance updates.

**Prereqs:** [classifier-free-guidance.md](classifier-free-guidance.md)
**Related:** [../post-training/_rewards.md](../post-training/_rewards.md)

---

## What it is

Compositional prompts break diffusion models: samples pick a favorite concept and drop the rest. Existing fixes use an external classifier, a fine-tuned reward model, or attention-manipulation heuristics. TILT observes that the base model *already contains* enough signal — the joint conditional (all concepts together) and the marginal conditionals (each concept alone) differ in an informative way — and turns that difference into a reward with no external supervision.

## How it works

Define a reward `r(x)` that is high when all requested concepts are jointly present, computed from the base model's own likelihoods (log-density ratios of joint vs single-concept conditionals). Then form a KL-constrained tilted target distribution:

```
p_tilt(x) ∝ p_base(x | c) · exp(r(x) / β)
```

This has a closed-form guiding-step update at diffusion sampling time (a modification to the CFG-composed velocity). The interaction between concept distributions produces two natural guidance strategies (attractive-toward-joint and repulsive-from-single-concept modes); TILT combines them in a hybrid that empirically wins.

Everything happens at inference — the base model is frozen, no reward model is trained, no fine-tuning occurs.

## Why it matters

Test-time compositional alignment has needed either a classifier or a training run. Showing the base model contains the reward signal is cheaper, less brittle (no reward model to overfit), and applicable to any diffusion model that supports classifier-free guidance. On T2I-CompBench, TILT improves compositional alignment while preserving image quality vs prior training-free compositional-guidance baselines.

## Gotchas & tricks

- The reward uses log-density *ratios*, so it works even when the base model's absolute likelihoods are miscalibrated — as long as the relative comparison is meaningful.
- `β` (the KL temperature) is the main knob: too small and samples collapse to the mode of the tilted distribution; too large and the tilt does nothing.
- The hybrid guidance combines attractive-toward-joint and repulsive-from-single-concept — running only one degrades on different failure modes.
- Cost is roughly `(1 + K) ×` the plain CFG cost, where K is the number of single-concept conditionals evaluated (usually small — 2–4 concepts).

## Sources

- Paper: *TILT: Improving Compositional Generation in Diffusion Models with a Model-Intrinsic Reward* — 2026 — [arXiv:2607.21606](https://arxiv.org/abs/2607.21606)
