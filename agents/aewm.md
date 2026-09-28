# Agent-Editing World Model (AEWM)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A "world model" for LLM agents that models **how the agent's own reasoning and actions edit future task progress**, not what tools will return. Combines an *Action Judge* (Critical / Exploratory / Noisy) with a *State Revision* module that rewrites noisy reasoning-plus-action segments in place. Wired into real execution as **EditAct**, it changes the state under later decisions rather than merely critiquing them. A rejection-sampling fine-tune on verified EditAct trajectories (**AEWM-RFT**) distills the behavior for use without online guidance.

**Prereqs:** [README.md](README.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/_post-training.md](../post-training/_post-training.md), [../safety/cot-monitoring.md](../safety/cot-monitoring.md)

---

## What it is

Long-horizon LLM agents suffer from **task-state contamination**: unsupported assumptions and outdated plans persist in history and warp later decisions. Simulating tool outputs (as in prior "language world models") does not help when the real tool is available. AEWM's insight: the useful thing to model is the *agent's trajectory* — which decisions matter, which are noise, and how to overwrite the noisy ones so the running state stays clean.

## How it works

Two modules:

1. **Action Judge.** For each reasoning+action step, output one of three labels:
   - **Critical** — load-bearing, don't touch.
   - **Exploratory** — a probe, keep as evidence.
   - **Noisy** — an unsupported assumption / outdated plan / hallucinated tool-result assertion; a candidate for rewrite.

2. **State Revision.** Given the same observed history and a noisy segment, generate a corrected reasoning+action continuation that would have been consistent with what the environment actually returned.

**EditAct** uses both online: whenever the Judge flags a step Noisy, State Revision rewrites the trajectory and the agent proceeds from the revised state. **AEWM-RFT** takes verified EditAct trajectories (final task succeeded) and does rejection-sampling fine-tuning on them, distilling the behavior into the base policy so no online AEWM guidance is needed at inference.

Training uses mid-training + SFT across Search, Terminal, and Software Engineering environments.

## Why it matters

- Names a specific, common failure of long-horizon agents (task-state contamination) and gives a concrete supervision interface for it.
- Sidesteps the awkward simulation problem: instead of modelling tool responses (high-entropy, low-value), it models the *agent's own trajectory* — where actual leverage exists.
- The distilled form (AEWM-RFT) is deployable without a live judge, an important property for latency-bound agents.

## Gotchas & tricks

- Judge label balance is a training-data issue: pure Critical/Exploratory data doesn't teach the model how to *edit*.
- State Revision must respect the environment's real feedback — rewriting reality is a failure mode; the paper's constraint is that rewrites are conditioned on the *observed* history, not on preferred outcomes.
- Rejection-sampling fine-tuning risks distribution collapse when the verifier is weak; the paper uses task success as the verifier, keeping the loop tight.

## Sources

- Paper: *Agent-Editing World Model: Rethinking World Modeling for LLM Agents* — Sun, Chen, Meng, Deng, Song, Jiang, Zhao, Xu, Wen — 2026 — [arXiv:2609.28416](https://arxiv.org/abs/2609.28416).
