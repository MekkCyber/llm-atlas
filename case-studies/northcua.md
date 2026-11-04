# Case Study: NorthCUA

*A family of open-source computer-use agents (Nano 9B / Mini 35B / Pro 397B, built on Qwen 3.5) trained by scaling **experience**: a Flywheel generates executable tasks, a shared Harness runs them, a milestone-guided recursive loop retries the hard ones, and a three-stage pipeline (mid-training → long-horizon SFT → joint desktop–web RL) turns the resulting trajectories into models.*

**Related concepts:** [experience-flywheel](../agents/experience-flywheel.md) · [milestone-guided-search](../agents/milestone-guided-search.md) · [grpo](../post-training/grpo.md) · [mid-training](../pre-training/mid-training.md) · [rl-prompt-curation](../post-training/rl-prompt-curation.md) · [_rl](../post-training/_rl.md)

---

## What this is

**NorthCUA**, released November 2026 by the NorthCUA Team at Shanghai Qiji Zhifeng. A tech report describing a full computer-use agent (CUA) stack:

- Three models at different scales — **Nano (9B)**, **Mini (35B)**, **Pro (397B)** — all based on Qwen 3.5, plus a low-latency variant **Ultrafast**.
- Four system components — **Experience Flywheel**, **Harness**, **recursive self-improvement**, **staged training**.
- Open-source releases of both the Harness and the model family.

The central claim is that CUA progress is bottlenecked by **experience** — verified end-to-end trajectories with real environments and real tool outcomes — not by model capacity. The Flywheel is the apparatus for manufacturing that experience at scale; the training pipeline is what converts it into a model.

Benchmarks claimed: SOTA among open-source CUAs on OSWorld Verified, OSWorld 2.0, WeaveBench, Agents' Last Exam, and WebArena-Verified; Pro competitive with proprietary frontier CUAs.

---

## System overview

```
                 Experience Flywheel
                 ─────────────────── (scaling intents, environments, verifiers)
                         │
                         ▼
                 Executable tasks                        ◀──────────┐
                         │                                          │
                         ▼                                          │
                 NorthCUA Harness                                    │
                 ─────────────── (sandbox + Agent Kit + judge)       │
                         │                                          │
                         ▼                                          │
       Verified trajectories                                        │
                         │                                          │
              ┌──────────┴──────────┐                               │
              │                     │                               │
              ▼                     ▼                               │
     Training pipeline       Unsolved tasks ─────▶ Recursive self-improvement
     (3 stages)                                   (milestone-guided search)
              │
              ▼
     NorthCUA models
```

---

## The Experience Flywheel

Three "scaling" sub-components produce executable tasks that can be run and scored:

- **Scaling Intents.** Derive task requests from the inventory of installed software and locally available content. Instead of asking humans for "what would a user do," enumerate what the environment *supports* and generate requests over that surface.
- **Scaling Environments.** Build workspaces for each task — or reconstruct external websites as replayable mirrors — so tasks run in controlled, verifiable environments.
- **Scaling Verifiers.** Adapt the request, the initial state, and the verifier *together*. A task is only valid if someone can tell whether it's done; the verifier evolves alongside the task.

Grounding data (how UI elements map to pixels) is produced by screen parsing (OmniParser) and, for specialized apps, retrieval over the app's documentation.

For more depth, see [experience-flywheel.md](../agents/experience-flywheel.md).

---

## The Harness

- **Agent Kit runtime.** The common agent loop (perceive → think → act) sits above platform adapters that abstract away OS/browser differences. Same agent, different platforms.
- **Long-horizon context.** Three mechanisms kick in once trajectories get long:
  - **Sliding window of recent screenshots** with pre-action visual overlays, so the model sees *what it just did*.
  - **Task-preserving context compaction** that drops old visual tokens but keeps the trajectory's semantic skeleton.
  - **Working-memory notes** written by the agent as it works.
- **Agent-as-a-judge.** A separate evaluator agent inspects both the recorded trajectory and the live environment state, producing a multidimensional success report (task done, errors made, efficiency).
- **Batch Rollout Service.** Isolated E2B sandboxes under a central controller, with preparation / execution overlap and automatic recovery on sandbox failure.

---

## Recursive self-improvement

For tasks the model can't solve at SFT-time quality, two complementary mechanisms:

- **Skill-guided exploration.** Previously-verified procedures are cataloged as skills; the agent reuses them as starting points on structurally similar unsolved tasks. Cuts the search space dramatically.
- **Milestone-guided search.** Verified intermediate states (software + agent context) become checkpoints that later attempts branch from. Instead of restarting hard tasks from scratch, a new attempt continues from the last known-good milestone.

The combination lets the Flywheel's output set grow even on tasks the model initially fails: an unsolvable task today becomes solvable tomorrow once skills and milestones have accumulated.

For more depth, see [milestone-guided-search.md](../agents/milestone-guided-search.md).

---

## Staged training pipeline

```
Qwen 3.5 base
     │
     ▼
Stage 1 — Broad GUI mid-training
     │     grounding + judged-successful trajectories
     │     reported to span 4M+ steps across dozens of applications
     ▼
Stage 2 — Verified long-horizon SFT
     │     complete end-to-end workflows with verified outcomes
     ▼
Stage 3 — Joint desktop–web RL (GRPO)
     │     tasks pre-filtered for outcome variation across sampled attempts
     ▼
NorthCUA Nano / Mini / Pro
```

- **Stage 1 (mid-training).** Not a pretraining-scale corpus and not SFT; a wide-exposure stage on grounding data and judge-approved trajectories — the model learns to "look at a screen and understand what is actionable."
- **Stage 2 (SFT).** Full-workflow training on completed, verified trajectories.
- **Stage 3 (RL).** GRPO on tasks selected because sampled attempts give varied outcomes — the only tasks where RL has signal. Explicitly joint: desktop and web environments in one batch, one policy.

See [mid-training](../pre-training/mid-training.md) for the broader mid-training concept; [grpo](../post-training/grpo.md) for the RL algorithm.

---

## The Ultrafast variant

A latency-optimized variant built on **pre-RL Nano**. Replaces the generation head with a **Jev-inspired choice head** that scores a finite action set directly; open-ended fields (text input, URLs) still generate as normal. The Head + LoRA configuration is reported as 2.3× faster than full generation on matched successful WebArena-Verified tasks, trading a modest quality drop for interactive latency.

---

## Key results (as reported in the tech report)

- **Pro (397B)** outperforms Qwen 3.8-Max and Kimi-K3 on grounding benchmarks, OSWorld-Verified, and WebArena-Verified.
- **Mini (35B)** surpasses Kimi-K3 on OSWorld-Verified and Qwen 3.8-Max on WebArena-Verified.
- The family leads open-source CUAs across grounding, desktop, and web task categories.
- **Ultrafast** — 2.3× faster than full-generation Nano on matched WebArena-Verified successes.

Exact numeric scores on WeaveBench and Agents' Last Exam are in the report's appendices but not reproduced here.

---

## Why this matters

- **Experience-first scaling thesis.** NorthCUA, CUA-Master, and parallel efforts (OpenCUA, etc.) all argue that CUA progress is unlocked by the quality and quantity of verified trajectories, not by model capacity alone. NorthCUA is the clearest realization of that thesis with an open-source release.
- **Reusable Harness.** The Harness being released separately is the single most reusable artifact — most CUA research is blocked by the infrastructure for safely running thousands of long-horizon agent sessions, not by the model.
- **Pattern for other agent domains.** The four-component split (task generation → execution → recursive retry → staged training) is transferable to any long-horizon agent task where outcomes are checkable.

---

## Caveats

- Numbers reported in a tech report, with limited independent reproduction at time of writing.
- Scaling claims (4M+ mid-training steps, 397B model) are expensive to verify.
- OSWorld 2.0 and some long-horizon benchmarks only partially discussed in the visible report sections.

---

## Sources

- Paper: *NorthCUA Technical Report: Turbocharging Computer-Use Agents through Experience at Scale* — NorthCUA Team, Shanghai Qiji Zhifeng, 2026 — [arXiv:2611.02102](https://arxiv.org/abs/2611.02102).
- Code / models: project pages linked from the arXiv HTML.
- Related: *CUA-Master* ([arXiv:2611.03126](https://arxiv.org/abs/2611.03126)) — parallel work on nested double-loop recursive-improvement for CUAs.
