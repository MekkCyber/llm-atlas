# Metacognitive agent architecture
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A scientific-discovery agent factored into three cooperating roles — a **knowledge compiler** that mines accumulated science for priors, an **assessor** that scores the current status of evidence, assumptions, and open questions at every step, and a **regulator** that converts those scores into the next research goal and the reasoning tactic to use. The assessor's output is a persisted, interrogable artifact — not an internal monologue — so the agent can later revise its own beliefs.

**Prereqs:** [../post-training/reasoning/prm.md](../post-training/reasoning/prm.md), [../post-training/reasoning/mcts.md](../post-training/reasoning/mcts.md)
**Related:** [../safety/cot-monitoring.md](../safety/cot-monitoring.md), [../post-training/reasoning/long-cot-rl.md](../post-training/reasoning/long-cot-rl.md)

---

## What it is

A reference architecture for long-horizon research / ideation agents. Rather than one monolithic planner → executor stack, the agent splits "reasoning about the problem" from "reasoning about its own reasoning". The latter — monitoring, re-assessing, choosing tactics — is what the metacognition literature calls *regulation*.

The payoff is two-fold: the belief state becomes a *first-class object* (readable, writable, loggable), and the regulator can swap tactics instead of always drilling deeper with CoT.

## How it works

The three roles run as explicit stages per step, over a persistent `BeliefState`:

- **Knowledge compiler** — retrieves and distills priors from the scientific literature into a `BeliefState` structured as `{evidence, assumptions, open_questions}`. Runs lazily (on-demand per goal) rather than once upfront.
- **Assessor** — a scoring pass over the current `BeliefState` that labels each entry's confidence, missing-evidence gaps, and "stuck" flags. Output is structured, not freeform.
- **Regulator** — reads the assessor's scores and either (a) refines the active research goal, (b) picks a reasoning method from a reusable library (Bayesian update, counterfactual, ablation sweep, literature search, wet-lab proposal), or (c) halts.

Each step appends to the `BeliefState`; nothing is overwritten silently. The regulator's tactic choice is logged alongside its justification, which is what makes the loop *process-legible*.

## Why it matters

Pushes scientific agents past "better retrieval + LLM-as-planner" toward architectures where *the agent's beliefs* are a first-class artifact. Two downstream benefits:

- **Oversight surface.** The `BeliefState` is a scratchpad legible to humans; overseers can read and edit it without having to parse free-form CoT.
- **Tactic diversity.** The regulator can prefer methods other than "think harder", avoiding the well-known failure mode where reasoning models over-exploit CoT for problems that need a different tool.

On Taiyi-Bench (265 ideation tasks, 10 disciplines), Sage-Taiyi scores 54.0 overall vs 47.9 for AI-Scientist-v2. Five expert-supervised collaborations produced validated conditions: cobalt-catalyzed carbonylation yield from trace to 84 %, photocatalytic allylic amination from 13 % to 69 %.

## Gotchas & tricks

- **Assessor drift.** If the assessor is the same base model as the executor, its scores correlate with the mistakes the executor is making. Needs either a distilled scorer or periodic external calibration.
- **BeliefState bloat.** Everything-appended logs grow fast; the regulator must have a summarization step or the context window tapers the whole loop.
- **Tactic library is the ceiling.** The regulator can only pick tactics it knows about. Benchmarks that reward tactics outside the library stall the agent silently.
- **Open benchmarks still sparse.** Only Taiyi-Bench + two external ideation benchmarks measure this directly; most agent evals still score outcomes, not metacognitive quality.

## Sources

- Paper: *Sage-Taiyi: A Metacognitive Architecture for Scientific Discovery* — 2026 — [arXiv:2611.12762](https://arxiv.org/abs/2611.12762)
