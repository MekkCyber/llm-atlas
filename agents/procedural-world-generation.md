# Procedural World Generation for Agent Training
*Depth — program-driven simulators plus a shared neural renderer as an open-ended environment supply for agent RL.*

**TL;DR:** Hand-authored agent environments bottleneck RL. Procedural world generation pairs **program-driven simulators or game engines** — which define layouts, dynamics, and task structure algorithmically — with a **shared neural renderer** that turns their state into a visual observation stream. Agents interact only with the rendered stream; the engines provide ground-truth dynamics for free. Introduced by AgentGarten (2026) along with *Adversarial Forcing* for distilling the renderer so autoregressive rollouts stay temporally consistent.

**Prereqs:** [../post-training/_rl.md](../post-training/_rl.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)
**Related:** [reflective-rulebooks.md](reflective-rulebooks.md) · [../multimodal/README.md](../multimodal/README.md)

---

## What it is

A supply-side answer to the agent-benchmark bottleneck. Instead of curating environments by hand, write engines that *generate* them: randomised room layouts, physics parameters, object sets, task graphs. Each generated world is a self-consistent simulation, with the engine's code acting as ground-truth dynamics.

Agents never see engine source — they see **pixels** rendered by a neural model shared across engines. The renderer is where the generality lives; the engines are where the task diversity lives.

## How it works

Three components:

1. **Engines** — procedural game/sim engines (platformers, grid worlds, 3D physics sandboxes) each parameterised to produce many instances. The engine is the source of truth for state, action semantics, and reward.
2. **Shared neural renderer** — a conditional generative model that maps engine state → visual observation. Trained to be *temporally consistent* under long rollouts.
3. **Playbooks** — natural-language experience logs each agent writes while exploring; later agents consult them, building a compounding knowledge base.

The renderer is distilled with **Adversarial Forcing**: a GAN-style discriminator critiques the renderer's own long rollouts, forcing it to stay on-manifold even without teacher-forcing shortcuts. This is what lets a single renderer serve many engines.

## Why it matters

Benchmarks run out of problems; procedural engines don't. The decoupling of *mechanism* (engines) from *appearance* (one renderer) is the right factorisation: engine diversity scales environment supply linearly, while a single renderer amortises the hard visual-realism training cost across all of them. Playbooks make agents compound rather than reset each run.

Also a template for VLA / computer-use / web-agent training without hand-authoring websites or applications: write the engine, let the renderer handle the pixels.

## Gotchas & tricks

- A single shared renderer must stay temporally consistent — regular distillation drifts under long autoregressive rollouts. Adversarial Forcing is the fix the AgentGarten paper proposes; cheaper alternatives (teacher-forcing-only, step distillation) tend to accumulate error.
- Procedurally generated worlds can be *trivially solvable* or *impossibly hard* if generation parameters aren't curated — tie the parameter distribution to the current agent's success rate (automatic curriculum).
- Playbooks are easy to poison: an early agent's wrong hypothesis propagates. Keep provenance and expire stale entries.
- Not a replacement for grounded real-world environments; its gains are in *supply*, not fidelity to any single target domain.

## Sources

- Paper: *AgentGarten: Code Worlds for Evolving Agents* — Chi et al., MirroS / Tsinghua / PKU, 2026 — [arXiv:2610.12374](https://arxiv.org/abs/2610.12374).
