# Feedback-Enriched Environments (FEE)
*Depth — environment-side scaffolding schedule that switches from action guidance to observation enrichment as agent RL progresses.*

**TL;DR:** Long-horizon agent RL breaks on reward sparsity. Instead of paying for more SFT warmup, reshape the environment: early in training add explicit action guidance; later, remove guidance and instead enrich the observation channel. The transition shifts pressure from imitation to grounded exploration and stabilises RL algorithms like GRPO / GSPO / DAPO across Qwen3 scales on SciWorld and BFCL.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [procedural-memory.md](procedural-memory.md), [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md)

---

## What it is

An environment-design pattern for long-horizon agent RL. Standard training either pretrains with SFT to bootstrap policies (expensive, data-hungry, limits exploration) or trains from scratch with sparse task rewards (unstable). FEE is a middle path: keep pure RL, but let the *environment* provide extra signal that phases from prescriptive to descriptive.

## How it works

Two dimensions of scheduling:

- **Intra-episode:** during a single rollout, the environment's feedback shape changes with step index — early steps see action guidance, later steps see enriched observations.
- **Inter-episode:** across the training run, the balance of guidance ↔ enrichment slides toward enrichment.

Two feedback shapes:

- **Action guidance** — the environment suggests or biases toward a valid next action given the current state. Speeds up early rollouts, but if kept on it becomes an inference-time crutch that the policy never internalises.
- **Observation enrichment** — the environment expands its observation channel: additional state, hints, tool schema clarifications, causal explanations of prior effects. The action space is unchanged; the model must still choose, but on richer evidence.

The scheduling rule (from the paper's pilot) is: reformulate environments to shift from action guidance early to observation enrichment late, both within episodes and across the training run.

## Why it matters

- **Reward sparsity fix without SFT.** No SFT dataset, no reference policy — just a modified environment.
- **Composes with any on-policy RL.** Verified with GRPO, GSPO, DAPO across Qwen3 sizes.
- **Internalisation, not test-time crutch.** Policy weights absorb environmental guidance rather than depending on it at inference. The paper isolates this with ablations.
- **Stabilises training.** Reduced entropy volatility during training; better proactive state-space exploration on hard tasks.

## Gotchas & tricks

- **Schedule matters.** Constant action guidance leads to inference-time dependence; constant observation enrichment does not solve early-training sparsity. The transition is the key.
- **Intra-group feedback consistency** is a stability boundary — feedback that varies too much across the parallel rollouts in a GRPO group destabilises the advantage estimate.
- **Requires environment writeability.** FEE assumes you can modify the environment's feedback channel; it's directly applicable to synthetic / simulator environments (SciWorld, WebArena) and to environments with a designed observation format (BFCL), less so to real APIs where observations are fixed.
- **Not a reward-shaping replacement.** FEE enriches *what the agent sees*; task reward remains the learning signal. Combine with verifier / RLVR-style rewards, not replace.

## Sources

- Yuan, Jin, Cao — *Environments as Scaffold: Enriching Feedback to Bootstrap Self-Evolving Agents in Long-Horizon Tasks* — 2026 — https://arxiv.org/abs/2609.08404
