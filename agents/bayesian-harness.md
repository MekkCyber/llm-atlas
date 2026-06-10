# Bayesian Agent Harness
*Depth — posterior-guided skill evolution for frozen-weights LLM agents.*

**TL;DR:** Treat reusable agent **skills and SOPs as Bayesian hypotheses** about whether the frozen base model will succeed under a given prompt / context / harness environment. Record verified trajectory evidence, maintain a **feature-conditioned categorical posterior** per skill, then map posterior state into a small set of inspectable **harness actions** — *patch, split, compress, retire, explore*. Introduced as **Bayesian-Agent** by Wu et al. (IDEA Research / HKUST(GZ) / DataArcTech), 2026 (arXiv 2606.08348). With DeepSeek-V4-Flash frozen: **SOP-Bench 80→95%**, **Lifelong AgentBench 90→100%**, **RealFin-Bench 45→65%**.

**Prereqs:** [latent-skill-adapters.md](latent-skill-adapters.md)
**Related:** [README.md](README.md) · [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

Most agent systems keep reusable skills as text — bullet lists of SOPs, recipes, lessons-learned. They get updated by **heuristic reflection**: append what worked, drop what failed. Counts-of-occurrence get treated as reliable belief about success rates, which they aren't.

Bayesian-Agent replaces that with a calibrated posterior:
- Each skill has a **categorical posterior** over success states, conditioned on contextual features (prompt template, tool subset, harness backend, …).
- Every verified trajectory is a piece of **evidence** that updates the posterior.
- Posterior summaries (mean, variance, low-data flags) drive a **decision layer** that selects from five harness actions.

The contract: weights are frozen; what evolves is the harness's belief state and the prompt-side artifacts it controls (skill text, guardrails, SOPs).

## How it works

- **Evidence collection.** When the agent runs a task, log the trajectory plus a structured set of features: which skill(s) were applied, which tools fired, which backend ran, what the verifier said.
- **Posterior update.** For each skill, given the feature vector of this trajectory, update a feature-conditioned categorical over outcomes (succeeded / failed / partial / timed-out, possibly more).
- **Decision actions.** Given posterior state, the harness can:
  - **Patch** — emit a small text edit to the skill to address a high-posterior failure mode.
  - **Split** — when the posterior is multimodal under different features, split the skill into specialized variants.
  - **Compress** — when the posterior is confidently good, shorten the skill text to its essentials.
  - **Retire** — when the posterior shows persistent low success, drop the skill.
  - **Explore** — sample a deliberately varied harness to gather evidence in under-explored feature regions.
- **Model-facing artifacts.** Patches become guardrails in the prompt; posterior summaries stay available as an audit trail (calibrated, not "this worked once").

## Why it matters

- **Calibrated belief beats heuristic reflection.** Reflection that "remembers" only recent successes overweights noise. A posterior with explicit uncertainty doesn't.
- **Audit trail.** Each skill's posterior is inspectable: "we kept this skill because in 47 trajectories of class X it succeeded 42 times; we retired this one because…"
- **Frozen-weights friendly.** Useful when the base model is closed or expensive to fine-tune — Bayesian-Agent's improvements all come from harness state, not weight updates. Headline numbers are on DeepSeek-V4-Flash with weights frozen.
- **Backend-portable.** The same posterior layer is shown working on top of native, GenericAgent, mini-swe-agent, and Claude Code backends — the harness logic is base-model-agnostic.

## Gotchas & tricks

- **Feature engineering still matters.** A categorical conditioned on poorly-chosen features is just an uncalibrated marginal. Pick features the posterior can meaningfully discriminate on.
- **Cold-start.** Few trajectories means high-variance posteriors. The *explore* action exists for this reason; without explicit exploration the posterior gets stuck early.
- **Patch quality.** "Patch" actions are still text edits to skills — they're as good as the LLM emitting them. Posterior-driven *when* to patch is the contribution; *what* to write is still LLM-quality.
- **Composability with weight updates.** Bayesian-Agent and weight-update fine-tuning are orthogonal — you can do both. But updating weights invalidates accumulated posteriors (they were conditioned on the old model).
- **Skill churn.** Aggressive retirement on a noisy posterior can drop useful-but-rarely-used skills. Use confidence-weighted retirement.

## Sources

- Paper: *Bayesian-Agent: Posterior-Guided Skill Evolution for LLM Agent Harnesses* — Wu, Yang, Liu, Lin, Zhang, Shi, Jiang, Xu, Li, Guo — IDEA Research / HKUST(GZ) / DataArcTech, 2026 — arXiv 2606.08348.
- Code: https://github.com/DataArcTech/Bayesian-Agent.
