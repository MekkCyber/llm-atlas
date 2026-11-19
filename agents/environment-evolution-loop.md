# Environment Evolution Loop
*Depth — turn real code repositories into verifiable training environments, then auto-filter them for training value.*

**TL;DR:** Agent RL is bottlenecked by *environments*, not data. RepoFactory wraps real GitHub repositories as sandboxed, verifier-backed environments (NL spec + tests + sandbox), and an **Environment Evolution Loop** cheaply filters them for training signal before paying the cost of real RL rollouts.

**Prereqs:** [../post-training/rlvr](../post-training/rlvr.md)
**Related:** [../post-training/rl-prompt-curation](../post-training/rl-prompt-curation.md), [../data/quality-filtering](../data/quality-filtering.md)

---

## What it is

A **data-engineering pipeline** that treats training environments the same way quality-filtering treats pretraining data: start with a large messy pool, apply cheap proxy signals to drop the useless ones, keep only what carries training value.

Each **environment** is one repository packaged as:

- A **natural-language specification** of the project.
- A **sandbox** (Docker image, dependencies, entrypoints).
- A **verifier** (the repo's own test suite, static checks, optional format checks).

The agent task is: "build this project from the spec until the verifier passes."

## How it works

Two stages:

**1. Repo → environment synthesis.** Scrape, filter by test coverage, extract NL spec from README + docstrings, containerize, verify that the ground-truth repo passes its own tests inside the sandbox. Drop any repo whose tests are flaky or trivially pass on an empty scaffold.

**2. Evolution loop.** For each candidate environment:

```
while not stable:
    rollout a WEAK model on the env (cheap proxy)
    measure: can it make any progress?    (too-hard filter)
    measure: do tests constrain behavior? (too-easy filter)
    measure: does a stronger-weak model do strictly better? (signal check)
    mutate: trim test suite, rephrase spec, add hints, or discard
```

The proxy rollouts use a small/cheap model; only environments that pass all three checks graduate to the real RL training pool. The mutation step can trim over-specified tests (that would reward brittle agents) or expand under-specified ones.

Training on 10,159 evolved environments:

| Model | Env count | NL2Repo-Bench | avg(5 other benchmarks) |
|---|---|---|---|
| Qwen3.5-35B-A3B (no envs) | 0 | base | base |
| + raw envs (no loop) | ~10k | small gain | small gain |
| + evolution loop | 10,159 | substantially higher | substantially higher |

The loop gives the bulk of the gain for small environment counts — filtering matters more than volume.

## Why it matters

Coding-agent RL has been gated by whoever can afford to curate environments. RepoFactory is the first large-scale answer with a **reusable automated loop**: the environment-side analog of RLVR for reasoning. Expect it to become default infrastructure for any lab running coding-agent RL at scale, especially where buying commercial environments (SWE-Bench Pro, etc.) isn't an option.

## Gotchas & tricks

- **Test suites are leaky.** Many real-world tests pass with placeholder code; the loop's "too-easy" check catches most but not all of these.
- **Sandbox escape risk.** Real repos can include install scripts that reach the network. Pin a no-network sandbox and allowlist dependencies.
- **Verifier gaming.** Agents learn to game Python's import-failure-as-success loopholes; verify that the repo's `__main__` actually runs end-to-end.
- **Weak-model proxy is noisy.** The "does a stronger weak model do strictly better" signal needs enough rollouts to be statistically meaningful — the paper uses pass@k over tens of rollouts.
- **License hygiene.** Scraped repos carry their own licenses; keep an allowlist or you can't release the environment set.

## Sources

- Paper: *From Repositories to Training Environments: Scaling Long-Horizon Coding Agents with Environment Evolution Loop* — Tian, Zheng, Yang, Du, Yang, Zhao, Yang, Dai (Beihang / SJTU / IQuest), 2026 — [arXiv:2611.21152](https://arxiv.org/abs/2611.21152)
