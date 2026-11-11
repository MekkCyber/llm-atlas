# Ground-Drop GRPO (GD-GRPO)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A GRPO variant for visual tool-using agents that forces the policy to *rely on tool-returned observations* rather than memorizing shortcuts from the input image. During RL rollouts, the original visual input is stochastically removed and the policy only gets reward when it recovers the answer via tool calls (crop, zoom, OCR, detector, etc.). Introduced in **Reframe** (2026) as the first stage of a two-stage agentic post-training recipe.

**Prereqs:** [./grpo.md](./grpo.md), [./_rl.md](./_rl.md)
**Related:** [./og-mopd.md](./og-mopd.md), [./rlvr.md](./rlvr.md), [../multimodal/README.md](../multimodal/README.md)

---

## What it is

Standard GRPO for a visual agent samples $G$ trajectories per prompt, assigns each an outcome reward, and normalizes against the group mean. The problem for tool-using visual agents: the base model often answers correctly without calling any tool — it memorizes the input image too well. The resulting policy *looks* right at eval but does not learn to ground its answers in retrieved evidence, so it fails under distribution shift.

GD-GRPO plugs an **observation-dependence prior** into the GRPO loop by dropping the original image with probability $p$ during rollout, so the policy must emit tool calls whose returned observations carry the evidence.

## How it works

Per prompt $q$ (image + instruction), sample $G$ trajectories with a modified rollout:

1. With probability $p$, replace the original image in $q$ with a null / masked placeholder before the policy sees it. The policy's only route to the real image is now via tool calls (crop with coords, zoom, OCR, object detection, etc.).
2. Compute an **evidence-aware reward**: $r_i = r_\text{outcome}(o_i) + \lambda \cdot r_\text{grounded}(o_i)$, where $r_\text{grounded}$ rewards trajectories whose final answer cites observations returned by tool calls. $\lambda$ small (~0.1–0.3).
3. Group-normalize as usual: $A_i = (r_i - \bar r) / \sigma_r$, broadcast to every token in $o_i$.
4. Standard PPO-clipped policy update with KL to the reference, as in GRPO.

The drop probability $p$ is a curriculum knob — often ramped from 0 at warmstart to ~0.5 late in training.

## Why it matters

- Separates **capability** (can the policy answer) from **groundedness** (does it answer by looking). Outcome-only GRPO conflates them.
- Measurable "observation dependence" and "tool gain" ablations go up relative to outcome-only GRPO. On the Reframe benchmark suite (14 image + video tasks), GD-GRPO + OG-MOPD beats both Mix-RL and vanilla MOPD.
- The trick generalizes beyond vision: any tool-using agent whose tools return rich observations can be trained with an "input drop" to enforce tool reliance.

## Gotchas & tricks

- **Too-aggressive drop collapses reward.** If $p$ is high before the policy knows how to use tools, no trajectory gets reward and the group variance zeroes. Curriculum $p$ from 0 upward.
- **Ground-truth attribution is fuzzy.** $r_\text{grounded}$ needs to detect "did the answer actually use the tool output". A cheap detector (string overlap between tool output and final rationale) works; richer models help.
- **Interacts with KL.** Dropping the image shifts the policy distribution hard; a loose KL to the reference lets the policy drift into unhelpful tool-only prompting. Keep $\beta$ in the normal GRPO range (0.01–0.1).
- **Doesn't need task-specific SFT cold start** — one of the paper's headline findings. Standard GRPO visual-agent recipes do.

## Sources

- Paper: *Reframe: Observation-Aligned Multi-Teacher On-Policy Distillation for Agentic Visual Reasoning* — Li et al. (CASIA), 2026 — [arXiv:2611.12886](https://arxiv.org/abs/2611.12886)
