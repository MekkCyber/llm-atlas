# Environment Reshaping for Agent RL
*Depth — programmable plugins that mutate a static agent environment to target the current policy's weaknesses.*

**TL;DR:** Agent RL environments are usually static: same tasks, same failure modes, quickly saturated. *Environment reshaping* wraps a static benchmark with a programmable plugin layer that mutates initial state, injects distractors, or alters the observation surface — without touching the underlying task semantics or verifier. A companion diagnosis loop watches rollouts, infers the policy's current failure modes, and synthesizes new plugins to attack them. EnvHarness reports up to **+9.0 points** on held-out instances and **-9.8%** completion steps across five benchmarks.

**Prereqs:** [README.md](README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/rlvr.md](../post-training/rlvr.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

A pattern in agent RL where the environment is treated as a *co-evolving opponent* rather than a fixed distribution. Formally, given a base environment `E = (S, A, T, R, verifier)`, reshaping introduces a plugin layer `P` that produces a family `E_P = (φ_P(S), A, ψ_P(T), R, verifier)`: initial state, transitions, and observations are re-parameterized, but reward and verifier stay untouched so the training signal remains sound.

Plugins are small programs (rule sets, distractor generators, observation filters) either hand-written or synthesized by a diagnostic model from rollout traces.

## How it works

The loop, following the EnvHarness / EnvRigger pattern:

```
E_0 = base_env
π_0 = initial policy
for round r = 1..N:
    trajectories = rollout(π_{r-1}, E_{r-1}, N_rollouts)
    diagnoses = analyze_failures(trajectories)   # what does π_{r-1} get wrong?
    new_plugins = rigger(diagnoses)              # synthesize plugins targeting those
    E_r = apply_plugins(E_{r-1}, new_plugins)
    π_r = rl_update(π_{r-1}, E_r)
```

Two design invariants that make it sound:

1. **Verifier preservation.** The plugin must not change what "success" means. If it did, the reward signal would drift and the policy's improvements would become artifacts of the reshape.
2. **Semantic invariance of the task.** A reshape that turns "book a flight" into "read a wiki page" is a different benchmark, not a training aid. Reshapes vary state, distractors, and observations *within* the task's intent.

The diagnosis step is typically an LLM prompted with a batch of failed trajectories, asked to name the recurring failure mode; the rigger step is an LLM prompted with that name plus the plugin API, asked to emit code.

## Why it matters

Agent RL is bottlenecked on *task diversity*, not compute — Terminal-Bench, SWE-bench, WebArena all have a small fixed distribution and saturate quickly. Environment reshaping is a route past the wall without commissioning new benchmarks: the *plugin* is the new training signal, and the plugin bank grows monotonically with training. Compatible with any pipeline that owns its own rollout loop.

## Gotchas & tricks

- **Reshape budgets.** Applying too many plugins at once creates OOD-only distributions that don't transfer back to the base benchmark. Keep a fraction of rollouts on unreshaped `E_0` as ground.
- **Verifier drift is silent.** If a plugin accidentally alters state the verifier reads, reward will look great and quality will collapse on the base benchmark. Test each plugin against `E_0`'s verifier before enabling.
- **Diagnosis quality dominates.** A weak diagnostic LLM produces plugins that attack random features. Prefer a stronger model than the one being trained.
- **Plugin overfitting.** The policy can learn to ignore common distractor shapes without addressing the underlying deficit. Rotate plugin families across rounds so no single family becomes the only signal.
- **Original benchmark still runs.** After training, evaluate on the frozen base benchmark, not the reshaped distribution — that's what production-shaped tests measure.

## Sources

- Paper: *EnvHarness: Awakening Static Worlds for Agent Learning* — Huang et al., Google Research, 2026 — [arXiv:2608.19880](https://arxiv.org/abs/2608.19880).
- Code: <https://github.com/google-research/envharness>
- Related: *Curriculum for Agents* (open literature) — humans-in-the-loop version of the same reshape-to-target-weakness idea.
