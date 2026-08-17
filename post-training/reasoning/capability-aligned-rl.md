# Capability-aligned RL (CaRL)

*Depth — RL that rewards refusal over confidently-wrong long reasoning, with hindsight augmentation turning past failures into refusal supervision.*

**TL;DR:** Long-CoT RL currently gives models an incentive to fabricate: any long, plausible-sounding trace that lands on *some* answer gets a nonzero probability of matching the verifier, so the policy converges toward always producing an answer even on tasks it cannot solve. **Capability-aligned Reinforcement Learning (CaRL)** reshapes the reward so refusal *beats* futile reasoning on out-of-capability problems, and introduces **hindsight refusal augmentation** that converts observed wrong answers into training signal for "you should have refused." The result is a model that learns its own capability boundary and stops fabricating past it.

**Prereqs:** [long-cot-rl](long-cot-rl.md), [rlvr](../rlvr.md), [_rewards](../_rewards.md)
**Related:** [../../safety/futile-reasoning.md](../../safety/futile-reasoning.md), [../../safety/sandbagging.md](../../safety/sandbagging.md)

---

## What it is

A verifier-based RL post-training recipe with a **three-outcome reward** instead of the classical binary reward:

- **Correct answer** — high reward.
- **Explicit refusal** — high reward (only when the problem is genuinely out of capability, as determined by a capability proxy).
- **Confidently wrong long derivation** — low or negative reward, penalized more the longer the trace.

Plus a **data-side augmentation**: after each round of rollouts, examples where the model generated confidently-wrong long CoTs are automatically relabeled with a target of the form "acknowledge the problem is out of my capability and refuse," and fed back into the next round as refusal-supervision.

## How it works

**Reward shape.** For each rollout $(q, o)$ with verifier $r_{\text{verify}}(q, o) \in \{0, 1\}$ and refusal detector $r_{\text{refuse}}(o) \in \{0, 1\}$:

$$
R(q, o) = \begin{cases}
+1 & \text{if } r_{\text{verify}} = 1 \\
+\alpha & \text{if } r_{\text{refuse}} = 1 \text{ and } q \notin \mathcal{C}(\text{model}) \\
-\beta \cdot \text{len}(o) & \text{if } r_{\text{verify}} = 0 \text{ and } r_{\text{refuse}} = 0
\end{cases}
$$

$\mathcal{C}(\text{model})$ is a capability estimate — e.g., a proxy classifier trained on pass@$K$ of the base, or a simple threshold on rollout consensus. $\alpha < 1$ ensures the model still prefers *solving* to *refusing* when it can. $\beta$ scales the length penalty so long fabrications hurt more than short ones.

**Hindsight refusal augmentation.** After each rollout batch, collect the $(q, o)$ pairs where the verifier failed but the model produced a long confident-looking answer. Rewrite $o$ into a refusal target ("This problem is beyond my current capability. I decline to answer.") and add $(q, o_{\text{refuse}})$ to the next round's dataset as supervised data (or as high-reward rollouts).

**Loop.** Standard [GRPO](../grpo.md) / RLVR loop with the reshaped reward + augmented data.

## Why it matters

- **Directly attacks confidently-wrong CoT.** The most user-harming failure mode of reasoning models is not that they refuse — it's that they produce long, plausible, wrong derivations. CaRL is a first-principles fix.
- **Refusal as a first-class training target.** Prior work treats refusal as a safety alignment concern; CaRL treats it as a *reasoning* concern — the model learns to distinguish its own capability boundary.
- **Composable with existing reasoning RL.** Drops into GRPO-style loops; requires only a refusal detector and a capability proxy.
- **Aligns training incentive with user trust.** A model that refuses when it doesn't know is more useful than one that always answers, once outputs are audited.

## Gotchas & tricks

- **Capability proxy is delicate.** If $\mathcal{C}(\text{model})$ is too permissive, the model over-refuses. If too restrictive, it under-refuses. Calibrate against a held-out set of solvable/unsolvable problems.
- **Length penalty tuning.** Too aggressive → model cuts short even useful reasoning; too weak → fabrications remain profitable. Track avg length and pass@1 jointly.
- **Refusal detector games.** The model can learn to *say* "I can't do this" while still fabricating; harden the detector against templated pseudo-refusals.
- **Related but distinct from sandbagging.** [Sandbagging](../../safety/sandbagging.md) is *deliberate* under-performance despite capability; CaRL targets *involuntary* fabrication despite lack of capability. Same visible symptom (wrong answer), opposite cause.
- **Watch out for a hard boundary.** A step-function capability estimate creates a cliff between "answer confidently" and "refuse" — smooth it with a calibrated probability.

## Sources

- Paper: *Knowing When to Quit: Diagnosing and Training LLMs to Abort Futile Reasoning* — Guan, Zeng, Xin, Lu, Lin, Han, Sun, Meng (CASIA / Tencent), 2026, [arXiv:2607.29211](https://arxiv.org/abs/2607.29211) — introduces CaRL and hindsight refusal augmentation.
