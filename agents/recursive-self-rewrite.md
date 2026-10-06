# Recursive Self-Rewrite (RSR)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A trajectory-synthesis recipe for agent SFT: run one base model under several *harnesses* that give it scaffolding the deployed agent won't have (verifier access, extra tools, longer scratchpads), capture successful trajectories, then recursively rewrite them as "runbooks" that execute cleanly under a plain deployment harness. Introduced in *Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite* (2026, arXiv:2610.02826); lifts Terminal-Bench 2 pass@3 from 57.0% to 74.2% on Qwen-3.8-27B.

**Prereqs:** [../post-training/_post-training.md](../post-training/_post-training.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [README.md](README.md)

---

## What it is

Agents trained from traces generated under harness-rich conditions (verifier calls baked in, hidden state visible, longer context) learn to depend on affordances the deployed harness doesn't offer. RSR is a decontamination layer between harness-boosted exploration and SFT data: successful but "cheated" trajectories are mechanically rewritten into trajectories executable without the extra scaffolding.

## How it works

Three agents operate on each successful rollout:

1. **Planner.** Extracts the solved task's procedure into a *runbook* — a sequence of self-contained steps that reference only standard tool calls.
2. **Critic.** Screens the runbook for leakage: verifier calls, hidden-state references, non-standard tool arguments. Flagged runbooks are recursively revised; this is the "recursive" step.
3. **Executor.** Re-runs qualified runbooks in a fresh sandbox under the general (deployment-shaped) harness. Only runbooks that re-complete successfully are kept as SFT trajectories.

The recursion continues until the runbook is both leakage-free and executes end-to-end under the general harness, or it's dropped.

```
successful rollout under harness H₁
       │
       ▼ Planner → runbook
       │
       ▼ Critic → leakage? ─ yes ─▶ revise, loop
       │                  no
       ▼ Executor under H_general → succeeds? ─ no ─▶ drop
       │                            yes
       ▼ keep as SFT trajectory
```

Expansion ratio is non-trivial: 2,001 original successful trajectories produce 11,094 clean rewrites by exploring alternate runbook phrasings during the recursive-revision step.

## Why it matters

Harness-assisted exploration (verifier-in-the-loop sampling, tool-augmented search) is a cheap way to find solutions to tasks the base model otherwise can't solve. But the resulting trajectories train the model to rely on the harness. RSR separates *finding* a solution from *expressing* it in the trace that gets distilled — closer in spirit to procedural distillation than to pure rejection sampling.

Across ~3K terminal tasks, three harnesses jointly solved 759 — 34.3% more than the strongest single harness. SFT on RSR data beat both the base and SFT on raw harness-boosted trajectories on all four Terminal-Bench variants.

## Gotchas & tricks

- **The critic is load-bearing.** A permissive critic misses leakage; a strict one drops everything. The paper iterates critic prompts against known-good runbooks.
- **Recursion depth matters.** Hard runbooks need multiple revise passes; a cap is needed to avoid infinite loops on uncompilable procedures.
- **Harness diversity > harness strength.** The 34.3% coverage lift came from using *different* harnesses, not a single very strong one — different scaffolding reveals different task families.
- **Doesn't replace RL.** RSR produces clean SFT data. For hard long-horizon tasks it's a complement to on-policy RL, not a substitute.

## Sources

- Paper: *Scaling Trajectories for Complex Tasks through Recursive Self-Rewrite* — Li et al., 2026 — [arXiv:2610.02826](https://arxiv.org/abs/2610.02826).
