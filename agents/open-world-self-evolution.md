# Open-world self-evolution (OpenSkill)
*Depth — bootstrap an agent's skills and its own verifier from open-world artefacts, with no target-task supervision.*

**TL;DR:** Existing self-evolving-agent pipelines assume either curated skills, successful trajectories, or a usable verifier. OpenSkill removes all three: given only a task prompt, the agent mines docs / repos / web pages for *grounded knowledge anchors*, synthesises them into transferable skills, and invents virtual practice tasks whose answers are derivable from the anchors — supplying both training material and a supervision-free verifier proxy. Target-task labels are reserved for final evaluation.

**Prereqs:** [rlvr](../post-training/rlvr.md), [grpo](../post-training/grpo.md)
**Related:** [self-improving-harness-and-weights](self-improving-harness-and-weights.md), [trace-derived-skills](trace-derived-skills.md), [world-model-tool-use](world-model-tool-use.md)

---

## What it is

A framework for *open-world self-evolution* — improving an agent post-deployment when the only signal available is "here is the task description". Most prior self-evolving systems quietly require: a skill library written by humans, gold trajectories to imitate, or a verifier function. In open-world deployments (build me a tool that talks to this new API; debug this codebase) none of these exist. OpenSkill replaces them with information mined from the open web.

## How it works

Three passes that compose into a closed loop:

1. **Anchor mining.** Crawl documentation, repos, and the web for facts grounded enough to act as ground-truth (API signatures, function tests, mathematical identities). Anchors are *not* about the target task; they are about the world the target task sits in.
2. **Skill synthesis.** Use the anchors as raw material to draft reusable skill descriptors: short procedures plus the anchors they touch. Skills are model-agnostic — designed so a different base model can pick them up unchanged.
3. **Virtual-task practice.** Synthesise self-contained tasks whose answers are checkable against the anchors (not against the target task). The agent practises on these; the anchor-derived check stands in for a verifier. Skills are refined based on virtual-task outcomes.

Final-task supervision is held out and used only for evaluation, satisfying the "no target-task supervision" constraint.

## Why it matters

- Removes the verifier-in-the-loop assumption that quietly underlies most "self-improving agent" claims.
- Skills transfer across models without per-model adaptation, which makes the practice phase a one-time cost.
- The anchor-mining + virtual-task pattern generalises beyond OpenSkill — it's a recipe for synthesising supervision wherever raw docs exist.

## Gotchas & tricks

- **Anchor quality is the bottleneck.** Wrong anchors propagate everywhere downstream; the system needs an anchor-trust score, not just an anchor.
- **Self-built verifier ≠ target-task verifier.** Reported alignment is empirical, not guaranteed; gaps will show up on adversarial target tasks.
- **Practice tasks can be over-fit to the anchor distribution.** Diversity in anchor sources matters as much as quantity.

## Sources

- Paper: *OpenSkill: Open-World Self-Evolution for LLM Agents* — Song, Zhang, Liang et al. — 2026 — [arXiv:2606.06741](https://arxiv.org/abs/2606.06741)
