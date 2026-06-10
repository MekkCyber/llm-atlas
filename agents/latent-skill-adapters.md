# Latent-Skill Adapters (LatentSkill)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A scheme for **converting textual procedural skills into modular adapter weights** for LLM agents. A **hypernetwork** ingests the natural-language description of a skill and emits LoRA-style adapter parameters; the base LM is frozen, and the agent **loads the right skill's adapter** instead of re-injecting the skill text into context. Composition is parameter-space, injection strength is a scalar knob. **+21.4 ALFWorld points with 64.1% fewer prefill tokens** in the introducing paper (LatentSkill, Yu et al. 2026, arXiv 2606.06087).

**Prereqs:** [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [README.md](README.md) · [bayesian-harness.md](bayesian-harness.md)

---

## What it is

Agent systems accumulate procedural skills: "how to disambiguate the user's intent before calling Tool X", "the canonical sequence for handling a refund", and so on. The default storage is **the prompt** — every step ships skill text in the context. Costs grow with the number of skills (context bloat) and the per-step prefill bill is dominated by repeated skill text.

Latent-skill adapters move skills to **weight space**:
- Each skill has a natural-language description (the "skill manifest").
- A **hypernetwork** $H_\phi$ maps the manifest to a set of LoRA-style adapter weights for the base LM.
- At runtime, the agent picks which skill is relevant, loads $H_\phi(\text{skill manifest})$ as a transient adapter, and runs the base model with adapter active.

Skills are no longer in the context; they're in the weights, for as long as they're loaded.

## How it works

- **Hypernetwork training.** Train $H_\phi$ end-to-end on triples (manifest, task, optimal trajectory). The hypernetwork output is plugged into the base LM as a LoRA-style adapter; loss is the standard SFT / OPD loss on the trajectory. $H_\phi$'s weights are trained; base LM stays frozen.
- **Inference loop.** Agent receives task → picks relevant skill(s) → calls $H_\phi$ to materialize the adapter → forward passes through (base + adapter) → emits actions. No skill text in prompt.
- **Composition.** Multiple skills' adapters add in parameter space ($A_1 + A_2$), with each weighted by an **injection strength** $\alpha_i$ — the agent (or a controller) can dial how much each skill contributes.
- **Skill update.** Adding a new skill = adding a manifest and (optionally) fine-tuning $H_\phi$ on its data. No prompt-engineering pass.

## Why it matters

- **Context-bloat fix.** Prefill cost no longer scales with the number of skills the agent might use. Long-running agents with many skills get cheaper, not more expensive, per step.
- **Modularity.** Skills can be added, removed, replaced, version-controlled — without rewriting the agent's prompt.
- **Controllable injection.** Scalar $\alpha_i$ on each skill is a *real* knob, not a "please use skill X" suggestion in the prompt that the model can ignore.
- **Pairs with frozen-weights paradigms.** Agents that use closed-weight base models can still benefit if $H_\phi$ is trained on a smaller open model and the adapters are then transferred (with caveats).

## Gotchas & tricks

- **Skill collisions.** Adding LoRAs in parameter space isn't free — composing two skills can produce conflicting weight changes. Some implementations re-normalize after summation; others train compositionally from the start.
- **Manifest format is part of the design.** $H_\phi$ is a function of the manifest text; small wording changes can produce different adapters. Maintain manifest discipline like you'd maintain a config schema.
- **Catastrophic forgetting through the hypernetwork.** Continual training of $H_\phi$ on new skills can degrade emitted adapters for old ones. Use replay or per-skill adapter caching.
- **Adapter-loading latency.** Materializing a fresh LoRA per skill call has overhead. Cache hot skills' adapters; re-materialize only for cold skills.
- **Base-model coupling.** $H_\phi$ is trained against a specific base LM. Swapping the base requires re-training the hypernetwork.

## Sources

- Paper: *LatentSkill: From In-Context Textual Skills to In-Weight Latent Skills for LLM Agents* — Yu, Zhou, Xu, Guo, Shan, Fu, Wang, Liu, Yu, Zhang, Lin — SJTU / Sun Yat-Sen / Shanghai Innovation Institute / OPPO, 2026 — arXiv 2606.06087.
