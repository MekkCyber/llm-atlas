# VR-GRPO
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for training reasoning in *visual space* — no text CoT, no image-text pairs. VR-GRPO decomposes the reward into a **global** signal (whole-trajectory logical coherence) and a **step-level** signal (per-step physical consistency), and optimizes both under the standard GRPO group-baseline update. The two terms are complementary: global-only leaves per-step drift unpenalized; step-only can't enforce cross-step logic.

**Prereqs:** [../grpo.md](../grpo.md)
**Related:** [prm.md](./prm.md) · [orm.md](./orm.md) · [long-cot-rl.md](./long-cot-rl.md)

---

## What it is

Most multimodal reasoning training uses text-CoT as the reasoning substrate. VR-GRPO learns reasoning *directly from visual demonstrations* — the model reasons in visual tokens (predicted frames, spatial states) without a language intermediate. Introduced with UniVR (2026), which uses VR-GRPO to learn complex reasoning, fine-grained physical dynamics, and long-horizon planning from purely visual data.

Two coupled rewards structure the training:

- **Global logical-coherence reward** — trajectory-level judgement of whether the whole predicted sequence tells a coherent story. Classic ORM-style outcome signal, but computed over visual trajectories.
- **Step-level physical-consistency reward** — per-step judgement that each predicted frame follows physically from the previous. Classic PRM-style process signal, but grounded in physical plausibility rather than reasoning correctness.

## How it works

For each visual prompt (a starting frame or scene description), sample $G$ trajectories from the current policy. Compute two rewards per trajectory:

$$
r^{\text{global}}_i = R_{\text{coherence}}(\tau_i), \qquad r^{\text{step}}_i = \tfrac{1}{T}\sum_t R_{\text{physical}}(\tau_i^{(t-1)}, \tau_i^{(t)})
$$

Combine into a scalar per trajectory:

$$
r_i = \lambda \, r^{\text{global}}_i + (1 - \lambda) \, r^{\text{step}}_i
$$

Then apply the standard GRPO update: subtract the group mean, normalize by the group std, broadcast the advantage to every token in the trajectory, PPO-clipped policy loss with a KL penalty to a reference model. $\lambda$ balances outcome vs. process; UniVR reports the two terms are approximately equally weighted.

The rewards themselves are learned models trained on VR-X's 16-source annotated corpus (long-horizon manipulation, spatial puzzles, physical reasoning) — no hand-crafted heuristics.

## Why it matters

- **Language-free reasoning training.** Removes the assumption that reasoning must be verbalized before it's supervised. Opens a training substrate for tasks where language is a lossy encoding of what the model needs to represent (physical dynamics, spatial planning).
- **Complementary reward decomposition.** Neither reward alone gets close to the joint. Global-only accepts trajectories with per-frame physics violations; step-only accepts locally-plausible but incoherent stories. The joint captures both.
- **Transfer to language benchmarks.** UniVR's visual-only training improves standard multimodal-understanding benchmarks too — evidence that visual-space reasoning is a real capability lever, not just a curiosity.
- **Up to 25% on VR-X.** The gain over GRPO with a single reward is large and reproduces across VR-X's 16 source domains.

## Gotchas & tricks

- **Both reward models must be honest.** If the physical-consistency reward is too permissive, step-level signal degenerates to noise; if too strict, only near-copy trajectories pass. Calibrate both against held-out human judgment before running RL.
- **$\lambda$ isn't free.** Extreme $\lambda \to 0$ or $\lambda \to 1$ collapses to ORM-only or PRM-only, and neither ablation reproduces the joint result. Middle range is where it lives.
- **Length asymmetry.** Because the step reward is averaged over $T$, long trajectories look "smoother" than short ones under the step signal. Normalize or penalize length explicitly, as elsewhere in reasoning RL.
- **Visual reward hacking.** Watch for policies that generate stylistically-generic frames that always score highly under the coherence RM — same failure mode as text CoT reward hacking, in visual costume.
- **Data curation matters more than reward tuning.** VR-X's 16-source coverage is what lets the trained policy generalize; a narrower training suite over-fits to the specific reward-model biases.

## Sources

- Paper: *UniVR: Thinking in Visual Space for Unified Visual Reasoning* — Wei et al., BJTU / ByteDance, 2026 — introduces VR-GRPO and the VR-X benchmark.
- See also: [../grpo.md](../grpo.md) for the base algorithm, and [prm.md](./prm.md) / [orm.md](./orm.md) for the process- vs outcome-reward distinction VR-GRPO fuses.
