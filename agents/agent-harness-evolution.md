# Agent Harness Evolution
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Population-based search over an agent's **harness** (prompts, tools, skills, control flow) with the base model frozen. A preserve-and-extend contract admits only variants that extend benchmark coverage without regressing any task; an archive keeps alternative lineages so recombination isn't stuck in a single line of descent; per-benchmark verifiers provide fitness with no gold solutions. Introduced by DarwinX (Salesforce, 2026).

**Prereqs:** [../agents/README.md](README.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md)

---

## What it is

An LLM agent's capability comes from two places: the model's weights and the harness that wraps it (system prompt, tool set, saved skills, control-flow scaffolding). Traditional self-improvement loops edit the harness along a single lineage — one variant per step, keep-or-reject — which is path-dependent, prone to local optima, and often ships fixes that quietly regress other tasks.

Harness evolution reframes this as **population-based selection** with the model frozen. Multiple harness variants live at once; each is scored by task-verifier fitness; only variants that *extend coverage* without regressing any preserved task survive; an archive keeps alternative lineages available for future recombination.

## How it works

The loop keeps four pieces:

1. **Population** — a set of harness variants, each fully executable against benchmarks.
2. **Edit sources** — proposed changes come from three channels: failure evidence (traces of failed rollouts), teacher evidence (a stronger model's demonstrations), and self-evidence (the agent's own reflections). All three plug into one edit interface.
3. **Preserve-and-extend contract** — a variant is admitted only if it (a) newly passes at least one task no ancestor did *and* (b) does not regress any task the ancestor already passed. Local wins that break other things are rejected outright.
4. **Archive of lineages** — alternative branches are kept even if temporarily dominated, enabling recombination when a later edit unlocks their strengths.

Fitness is whatever the benchmark provides — unit tests, verifiers, task-specific scorers. No gold solutions or hand-picked winners are needed; the contract enforces monotonicity on a growing basket of tasks.

## Why it matters

- **Turns eval compute into durable capability.** Rollouts you'd already run to score a harness now become the search signal — the "eval budget" trains the harness.
- **Avoids single-lineage regression.** The preserve-and-extend contract makes forgetting structurally impossible within the search.
- **Model-agnostic.** DarwinX reports a Terminal-Bench-evolved harness transferring unchanged to SWE-bench Verified and to a *different* base model — evidence that what's evolved is general agent competence, not benchmark-specific patches.
- **Frontier without training.** Reaches the verified frontier on Terminal-Bench 2.1 (84.7%) with the model weights untouched.

## Gotchas & tricks

- **Contract only works if the preserved-task set is honest.** If you silently drop a regressing task, "preserve-and-extend" is a lie. Explicit regression sets are essential.
- **Archive size grows.** Keeping every alternative lineage forever isn't free. In practice you cap by diversity metrics and cull dominated lineages that haven't contributed to a merge in $N$ rounds.
- **Verifier gaming.** Any strong-enough optimizer over a verifier eventually exploits its quirks — DarwinX partially mitigates by evolving on one split and testing on a held-out split (TerminalWorld), but this remains the standard verifier-hacking risk.
- **Compute scales with population × benchmark size.** Each variant re-runs every preserved task. Expect $O(P \cdot T)$ rollouts per generation; small populations (10–20) are more practical than large evolutionary populations.

## Sources

- Paper: *DarwinX: Evolving Agent Harnesses Through Natural Selection* — Zhang et al., Salesforce AI Research, 2026 — [arXiv:2608.07545](https://arxiv.org/abs/2608.07545).
- Related: sibling optimizer-driven approach in AutoDesign (see [meta-harness-optimizer.md](meta-harness-optimizer.md)).
