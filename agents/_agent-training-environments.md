# Agent Training Environments

*Taxonomy — the executable substrates that produce verifiable trajectories and rewards for agent post-training.*

**TL;DR:** Agent RL/SFT needs *environments*, not just trajectories — a single frozen trajectory is one demo, while an environment can be re-queried into many verifiable tasks and provides execution feedback. Environments are scarce, so a small family of engineering recipes has emerged: synthesize from scratch (LLM writes the spec), recover from existing trajectories, or grow difficulty over training generations. In 2026 the dominant open-source stack pairs trajectory recovery with off-policy environment evolution.

**Related taxonomies:** [_rl](../post-training/_rl.md), [_rewards](../post-training/_rewards.md)
**Depth files covered here:** [trajectory-to-environment](trajectory-to-environment.md) · [environment-evolution](environment-evolution.md)

---

## The problem

Agent training loops consume environments per rollout, and each rollout needs a workspace whose state changes deterministically enough that a reward predicate can grade the result. Real production environments (developer machines, CI systems, browsers) are hard to snapshot and expensive to run. Trajectories from those environments are common — they are logged transcripts — but a trajectory is a frozen demo, not a substrate.

Without executable environments, RLVR has nothing to verify against, agent SFT sees only imitation targets, and evaluation is stuck on curated benchmarks that quickly saturate.

## The shared pattern

Every method here produces a triple:

```
(workspace, task, verifier)
```

Where **workspace** is bootable state, **task** is a query the policy must satisfy, and **verifier** is a predicate — usually a re-execution check — that grades the resulting rollout. Differences are in *where the triple comes from*: hand-authored, LLM-synthesized, trajectory-recovered, or evolved.

## Variants

| Approach | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Hand-curated (SWE-bench Verified, WebArena) | Human experts author workspaces and gold verifiers | Highest quality, lowest scale | Evaluation benchmarks. |
| Synthesized from scratch | LLM writes env spec + task from templates | Cheap and scalable, quality drift | Warm-start when no trajectories exist. |
| [trajectory-to-environment](trajectory-to-environment.md) | Replay file ops in a frozen trajectory to reconstruct its workspace, then synthesize new tasks on it | Requires trajectory corpus; realistic; verifiable | Terminal / code-execution agents with logged sessions. |
| Co-evolved (on-policy) | Generate envs targeting the current policy's on-policy weaknesses | Signal tightens fast, generalizes poorly, stalls with model | Small internal loops for one policy. |
| [environment-evolution](environment-evolution.md) | Off-policy difficulty growth along RL-objective-derived directions, scheduled by generation | Sustained signal, generalizes across policies | Long-horizon RL on top of an existing env pool. |

## How to choose

Start with **hand-curated** environments for evaluation — nothing else offers the label quality. For training, if you have a trajectory corpus (agent logs, competition submissions, public bash histories), begin with **trajectory-to-environment** — you inherit realism for free. Layer **environment-evolution** on top once the base pool saturates; the two compose (breadth then depth). **Synthesized-from-scratch** is a warm-start when no trajectories exist. Avoid pure **co-evolution** unless you specifically want a one-policy loop — it does not transfer.

## Adjacent but distinct

- [rl-prompt-curation](../post-training/rl-prompt-curation.md) — curates prompts against a fixed env; this taxonomy is about the env itself.
- [rejection-sampling](../post-training/rejection-sampling.md) — selects among model outputs, not among environments.

## Sources

- Paper: *Terminal-Universe* — Wu et al., 2026 — trajectory recovery.
- Paper: *Environment Evolution for Terminal Agents* — Fan et al., 2026 — off-policy evolution.
- Benchmark: SWE-bench Verified / Pro — hand-curated coding-agent envs.
