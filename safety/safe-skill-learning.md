# Safe Skill Learning for Computer-Use Agents
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Computer-use agents that "learn skills from successful trajectories" implicitly assume a **safe, static environment**. **Safe skill learning** drops that assumption: a *skill boundary* abstraction filters unsafe skill candidates using multi-source supervision, and a *selective skill reuse* policy decomposes tasks so that only the safe subset of skills runs in the current context. Introduced as SkillHarness (2026); reduces unsafe-skill rate by 57.1% vs. unconstrained skill-learning baselines.

**Prereqs:** [_attacks.md](_attacks.md)
**Related:** [_jailbreaks.md](_jailbreaks.md), [../agents/README.md](../agents/README.md)

---

## What it is

Skill-learning frameworks for computer-use agents (CUAs) typically promote *any* successful trajectory into a reusable skill. In a dynamic environment — pop-ups, prompt injection in the rendered page, deceptive UI — a "successful" trajectory may actually contain unsafe steps (clicking a malicious link, sending data to the wrong endpoint).

Safe skill learning adds a constraint layer on top of skill acquisition: skills are *only internalised when the boundary is provably safe*, and skill reuse at runtime is *conditional on the current context*.

## How it works

Two components:

1. **Skill boundary.** Each candidate skill (extracted from a trajectory) is scored against multi-source supervision signals:
   - **Outcome signal** — did the trajectory satisfy the user goal without side-effects?
   - **Environment signal** — were there adversarial elements (pop-ups, suspicious DOM) in the trajectory?
   - **Self-evaluation signal** — does the agent flag any step as risky in reflection?
   Skills below threshold are dropped from the skill library entirely.

2. **Selective skill reuse.** At runtime, tasks are decomposed and each sub-step queries the skill library *conditioned on the current context*. Only skills whose boundary covers the current context are activated. A skill safe in one app may not activate in another.

The skill library is updated continuously: as the agent encounters new boundary-violating contexts, it pushes safety constraints back into the library.

## Why it matters

- **Safety is part of skill acquisition, not a post-hoc filter.** Most CUA-safety work has been runtime guardrails; this brings the safety constraint into the *learning* loop.
- **Necessary for production CUAs.** As CUAs deploy into real browsers and OS shells, environments are adversarial by default — pop-ups, malvertising, prompt-injection in rendered content.
- **Frames skill reuse as a safety problem.** A skill that worked once in a safe context can be catastrophic in a hostile one. Conditional activation is the right primitive.

## Gotchas & tricks

- The supervision signals **trade off recall vs. safety**. Tight thresholds drop too many useful skills; loose thresholds let unsafe ones through.
- **Skill boundary annotation** is a real labelling cost — the multi-source signals need to be defined per environment class.
- **Adversarial drift.** Once the safety filter is known, attackers can craft trajectories that pass it. Periodic boundary re-evaluation against new attack patterns is part of the lifecycle.

## Sources

- Paper: *SkillHarness: Harnessing Safe Skills for Computer-Use Agents* — anonymous, 2026 — [arXiv:2606.20636](https://arxiv.org/abs/2606.20636).
