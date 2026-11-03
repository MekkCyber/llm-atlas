# Agent Experience Curation (GEPA-Curator)
*Depth — a curation layer on top of reflective prompt optimizers: skip tasks that teach the wrong lesson, retrieve prior successes as context.*

**TL;DR:** Reflective prompt optimizers (GEPA, ACE) learn from failed rollouts, but a "failed" rollout can mean many things — bad prompt, flaky environment, noisy judge, harness bug — and the optimizer can be taught the wrong lesson. GEPA-Curator wraps any such optimizer with two components: **task curation** (skip tasks whose signal is likely unreliable) and **prior-success retrieval** (feed past successful trajectories for the same task back in as context). On GoBrowse / ALFWorld / AppWorld / τ-bench with Gemini 3.1 Pro and Claude Opus 5 as optimizers, matches or beats both GEPA and ACE, with gains up to 20.7% / 21.2% on the noisiest benchmarks.

**Prereqs:** [../post-training/rl-prompt-curation.md](../post-training/rl-prompt-curation.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/rejection-sampling.md](../post-training/rejection-sampling.md) · [../post-training/_post-training.md](../post-training/_post-training.md) · [README.md](README.md)

---

## What it is

Reflective prompt optimization iterates: run the agent, observe the trajectory, let an LLM "reflector" rewrite the prompt / policy, repeat. GEPA (Opsahl-Ong et al.) and ACE (Agentic Context Engineering) are leading instances.

The problem GEPA-Curator fixes: not every rollout is a *learnable* data point. A rollout can fail because:
- The agent genuinely picked a bad action (useful signal).
- The judge model mis-scored the trajectory (noise).
- The environment was flaky or non-deterministic (noise).
- The harness has a bug the agent can't influence (noise).

Reflecting on noise teaches the wrong lesson. GEPA-Curator curates *which rollouts the optimizer gets to see*, and augments the context with past successes.

## How it works

### Two components

**1. Task curation.** Before invoking the optimizer, each candidate task is scored for *signal reliability*:
- Trajectory structure (does the final step look like a judge-visible outcome?).
- Environment stability (is the task known to be deterministic?).
- Prior behavior of this task across runs (high-variance tasks get down-weighted).

Low-reliability tasks are skipped — the optimizer does not get a reflection on them.

**2. Prior-success retrieval.** For each surviving task, the harness looks up previously *successful* trajectories on the same or similar task from an experience store. These are injected as reference context when the agent runs the current task — giving the policy an exemplar to anchor on.

### Why the pair matters

Each component addresses a different failure mode:
- Curation keeps the optimizer from updating on noise.
- Retrieval raises the baseline success rate so the few reflections that remain are on *genuine* policy failures.

Together they tighten the feedback loop: fewer, cleaner updates.

### Measured results

Across four benchmarks (GoBrowse, ALFWorld, AppWorld, τ-bench) and two optimizers (Gemini 3.1 Pro, Claude Opus 5):
- Mean improvement **+20.7% over GEPA** at its best, **+21.2% over ACE** at its best.
- Largest gains on noisy / long-horizon benchmarks; small or zero on short, deterministic ones (consistent with the mechanism).

## Why it matters

- **Model-agnostic plug-in.** Works on top of any reflective optimizer; no retraining.
- **Mirrors the "better data > more data" shift in pretraining.** Agent-training is catching up to the lesson that reward-signal quality matters more than quantity.
- **Operationally cheap.** Curation and retrieval are both cheap compared to the optimizer loop they sit above.

## Gotchas & tricks

- **The experience store needs maintenance.** A stale success store will suggest exemplars from an outdated policy / environment and *hurt* performance.
- **Reliability scoring is heuristic.** The paper's rules are task-domain-specific; naïve transfer to a new agent domain requires rewriting the reliability features.
- **Doesn't help if the base optimizer's reflections are the limit.** If the reflector LLM can't propose a better prompt even on clean data, curation doesn't help — Garbage In, Garbage Out at the reflector step.
- **Risk of overfitting to the retrieval index.** Over-reliance on retrieved exemplars can collapse policy diversity. The paper reports monitoring this; mitigation details are limited.
- **Benchmark-specific gains.** The headline +20% numbers are on the noisiest benchmarks; expect smaller gains on well-conditioned ones.

## Sources

- Paper: *Curated Learning for Self Improving Agents (GEPA-Curator)* — Zack Yu, Nicholas Lee, Minjae Lee, Lakshya Agarwal, Luca Manolache, Wonjun Kang, Matei Zaharia, Michael W. Mahoney, Kurt Keutzer, Sehoon Kim, Amir Gholami — UC Berkeley / ICSI / LBNL / Furiosa AI / KAIST, 2026 — arXiv 2610.37651.
- Related: *GEPA* (reflective prompt optimizer) — the base optimizer being wrapped.
- Related: *ACE: Agentic Context Engineering* — comparison system in the paper.
- Related: [rl-prompt-curation.md](../post-training/rl-prompt-curation.md) — the Kimi-k1.5 ancestor on the RL-prompt side.
