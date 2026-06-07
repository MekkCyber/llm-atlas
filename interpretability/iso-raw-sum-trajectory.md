# Iso-Raw-Sum Trajectory (IRST) and Noisy Quantization for Arithmetic
*Depth — a geometric account of why LLMs are fragile at addition, and a usable inference-time correction derived from the same geometry.*

**TL;DR:** LLMs are paradoxically bad at arithmetic — fluent symbol manipulators that miss carries. Looking at the residual stream during multi-operand addition, the *Shape of Addition* paper (RL-MIND, 2026) identifies the **Iso-Raw-Sum Trajectory (IRST)**: representations are *anchored* by digit semantics and *modulated* by a continuous "carry fiber." Arithmetic errors are then explained as **Geometric Slippages** — internal neural noise pushes a continuous *Carry Potential* across a quantization threshold. The same geometry justifies a lightweight inference-time check that detects and corrects these failures, and explains why probes can simultaneously read out "ground truth" and "hallucination" from a single activation vector.

**Prereqs:** [interpretability/README.md](./README.md)
**Related:** [evaluation/math500.md](../evaluation/math500.md), [evaluation/aime.md](../evaluation/aime.md)

---

## What it is

Consider multi-operand addition $\sum_i a_i = s$. The paper finds that the residual stream, projected onto a learned 2D plane, traces out a trajectory parameterised by:

- a **raw-sum coordinate** — a discrete, semantic anchor corresponding to each digit's contribution;
- a **carry fiber** — a continuous coordinate that modulates the anchor according to how much carry is "pending."

All inputs that share the same digit-wise raw sum fall on the same trajectory regardless of operand order — hence *iso-raw-sum*. The carry fiber is continuous because the model maintains carry as a soft signal that gets quantised only when the answer must be emitted.

## How it works

### The Noisy Quantization Model

Let $c \in \mathbb{R}$ be the carry potential at a digit position. The correct answer digit requires thresholding:

$$
d = \mathrm{round}(\,\text{raw-sum} + \mathbb{1}[c > \theta]\,)
$$

If neural noise $\epsilon$ pushes $c$ past the threshold $\theta$ (or pulls it back), the rounded answer flips by 1. The error pattern is geometric, not symbolic: nearby inputs (small carry-potential perturbations) produce identical answers; inputs close to $\theta$ flip.

### Probe versatility

Because *both* ground truth and a hallucinated alternative live on the same fiber (different sides of $\theta$), a linear probe can decode either depending on which projection direction it picks. This explains the otherwise puzzling result that the same activation supports probes reading out different "facts" — they're slices of the same continuous structure.

### Geometric consistency check

At inference, look at the carry potential. If $c$ is suspiciously close to $\theta$ for any digit position, flag the prediction. The paper proposes a lightweight check — derive a confidence score from distance-to-threshold — that detects and corrects quantization failures with no extra training and minimal overhead.

## Why it matters

- **Closes the loop on a common mech-interp story.** Most mech-interp papers stop at "and now we understand this." Here the geometry yields a concrete intervention with measurable accuracy gains.
- **Reframes arithmetic errors.** They aren't symbolic mistakes; they're continuous slips across quantization thresholds. That changes how you'd try to fix them — better calibration, not better tokenisation.
- **Probe versatility has policy implications.** If a single activation supports multiple readouts, "the model knows the answer is X" depends on which probe you use. Activation-based interp claims need a geometric framing to be falsifiable.

## Gotchas & tricks

- **Carry potential is per-position.** Different digit positions have different thresholds; treat them as a set, not a single number.
- **Calibration of $\theta$ is dataset-dependent.** Pulling a threshold from one operand-magnitude regime and applying it to another fails — calibrate per regime.
- **The check is a confidence bound, not a fix.** It tells you when the model is likely wrong; it doesn't give you the right answer. Pair with a fallback (re-prompt, external calculator, or argmax over both quantizations).
- **Generalises beyond addition with care.** The same fiber structure likely holds for other modular-arithmetic-style tasks; multiplication and division have richer fiber structure and probably need different anchors.
- **Open code.** Available at github.com/RL-MIND/Shape-of-Addition for the geometric check and probe experiments.

## Sources

- Paper: *The Shape of Addition: Geometric Structures of Arithmetic in Large Language Models* — RL-MIND group, 2026 — [arXiv:2606.03645](https://arxiv.org/abs/2606.03645).
- Code: https://github.com/RL-MIND/Shape-of-Addition
