# Progress Advantage
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** For any RL-post-trained policy, the per-step log-ratio between the trained policy and its reference (`log π_RL − log π_ref`) *exactly recovers* the optimal advantage function under the RL objective. So step-level (process-style) scoring for agents falls out of post-training for free — no separately-trained PRM required.

**Prereqs:** [_rl.md](_rl.md), [grpo.md](grpo.md), [ppo.md](ppo.md), [dpo.md](dpo.md)
**Related:** [reasoning/prm.md](reasoning/prm.md), [_rewards.md](_rewards.md)

---

## What it is

Process Reward Models (PRMs) score *intermediate steps* in a trajectory, which is exactly what agent training wants for credit assignment. Building PRMs for real agents is brutal: long horizons, irreversible actions, stochastic environment feedback, and prohibitive annotation cost.

Progress Advantage observes that an RL-trained policy already encodes the per-step advantage as a log-ratio against the reference. Under a general stochastic MDP with the standard KL-regularized RL objective (the one PPO, GRPO, and DPO all implicitly optimize), the optimal advantage is:

$$
A^\star(s,a) \;=\; \beta \cdot \log \frac{\pi_{\text{RL}}(a \mid s)}{\pi_{\text{ref}}(a \mid s)}
$$

— a derivation parallel to the DPO log-ratio identity, but generalized to a stepwise MDP. So you can score any step of an *unseen* trajectory by querying both policies.

## How it works

- Take an RL-trained checkpoint π_RL and its reference π_ref (the SFT/base checkpoint used as the KL anchor during RL).
- For a candidate agent trajectory, compute per-action log-probabilities under both policies.
- The signed log-ratio is the per-step *progress advantage* — positive when the RL update made this action more likely (i.e. a good step), negative otherwise.
- Use as a step-level scorer: rerank candidate steps in search, gate tool calls, or feed back as a PRM-style dense reward for a downstream RL run.

## Why it matters

- **Zero extra training cost.** No PRM annotation, no Monte-Carlo labelling. Just keep the reference checkpoint around (which RL already requires).
- **Generalizes the DPO insight.** DPO showed log-ratios *are* implicit rewards for preference learning; Progress Advantage extends the same identity to stepwise advantages in general stochastic MDPs.
- **Kills a whole pipeline.** PRM construction for agents has been the bottleneck to step-level RL on long-horizon tasks; this paper proposes a free alternative that matches or beats explicit PRMs on agentic benchmarks.

## Gotchas & tricks

- The identity assumes the *correct* KL coefficient β was used during RL; if β was tuned or annealed, the scale of the advantage isn't comparable across checkpoints.
- π_ref must be the *exact* reference used during RL — replacing it with a different base post hoc breaks the identity.
- The signal weakens for actions the RL run never explored; in distribution-shifted situations the log-ratio is a worse advantage estimator than a freshly-trained PRM.

## Sources

- Paper: *Neglected Free Lunch from Post-training: Progress Advantage for LLM Agents* — Oh, Li, Park, Yeh, Mallick, Li, Wisconsin / Argonne, 2026 — arXiv:2606.26080.
- Related: *Direct Preference Optimization* — Rafailov et al., 2023 — establishes the log-ratio-as-implicit-reward identity for the preference case.
