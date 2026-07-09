# Skill optimization (SkillOpt-Lite)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Improve an LLM agent without updating weights: keep its execution trajectory (or its full harness) as an **editable text artifact**, treat performance improvement as **zeroth-order optimization** over that artifact, and let the agent edit it between rounds. SkillOpt-Lite distills earlier SkillOpt into a minimal loop grounded in PAC-learning design principles, and extends the same idea to full-harness optimization (HarnessOpt). Ships as a VSCode extension activated by one line.

**Prereqs:** *(none)*
**Related:** [reflexive-video-agent.md](./reflexive-video-agent.md), [../post-training/policy-aware-prompt-adaptation.md](../post-training/policy-aware-prompt-adaptation.md)

---

## What it is

Agent "self-improvement" without weight updates has been a live thread — Reflexion (self-reflection into a text buffer), Voyager (skill library growth), and the original SkillOpt. The core pattern: keep an evolvable text artifact (a trajectory, a skill library, a harness) that captures what the agent has learned, and let the agent edit it in response to new outcomes.

SkillOpt-Lite formalizes this as **zeroth-order optimization** over the text artifact — the agent's next edit is a discrete update chosen without access to gradients, guided by observed reward/success signals. PAC-learning theory gives design principles for what the artifact should look like: rich enough to represent the target skill, structured enough that random-ish edits explore a useful neighborhood, compact enough that convergence is fast.

## How it works

**The artifact.** An execution trajectory (or full harness — prompts + tools + workflow config), stored as an editable file. Each successful run appends / rewrites into the artifact; each failure suggests a targeted edit.

**The loop.**

1. Run the agent with the current artifact on a task.
2. Score the outcome.
3. Ask the LLM (as its own optimizer) to propose a **minimal edit** to the artifact — add a lesson, delete a wrong assumption, refine a step.
4. Apply. Repeat.

**Minimalism as a design principle.** SkillOpt-Lite drops most of the earlier SkillOpt scaffolding (multi-tier memory, complex evaluation harness) in favor of the smallest viable loop. Faster convergence and higher final performance vs SkillOpt v1 on the same benchmarks.

**HarnessOpt** — same idea, larger artifact. Instead of just a trajectory, optimize the full agent harness (system prompt, tool definitions, workflow steps). Requires wider edit scope but same ZO-optimization framing.

**Packaging.** VSCode extension exposes the loop as a single command; agents evolve inside the developer's editor with minimal setup.

## Why it matters

- **RL-free agent evolution.** No RL rollouts, no reward model, no weight updates — a compellingly cheap alternative when RL is infeasible.
- **Theoretical grounding.** Zeroth-order optimization + PAC-learning design principles → less "prompt-eng magic," more principled iteration.
- **Faster convergence than earlier SkillOpt.** Minimal loop wins over more-complex predecessors on the benchmarks reported.
- **Deployment simplicity.** One-line VSCode activation is a real usability contribution — most self-improvement systems live in research code.

## Gotchas & tricks

- **Artifact drift.** Repeated edits without constraints can bloat or corrupt the artifact. The paper's minimal loop implicitly prunes; production users need explicit rollback / clean-up.
- **ZO optimization is sample-hungry.** No gradients means every edit requires an outcome; converging on a hard task can take many rounds. Not free.
- **Depends on the LLM's editing skill.** If the LLM can't propose good minimal edits, the loop stalls. A weaker LLM with SkillOpt-Lite plateaus faster than a stronger one.
- **Not a substitute for post-training.** For skills the base model can't do at all, no amount of trajectory editing recovers the missing capability. SkillOpt evolves *how* the agent uses its abilities, not what those abilities are.
- **Composes with LLM-as-a-Tutor** and other prompt-side adaptations — different axes of the same "improve without weight updates" strategy.

## Sources

- Paper: *SkillOpt-Lite: Better and Faster Agent Self-evolution via One Line of Vibe* — Shen, Li, Zhang, LMMs-Lab / NTU MMLab / Microsoft, 2026 — [arXiv:2607.03451](https://arxiv.org/abs/2607.03451).
