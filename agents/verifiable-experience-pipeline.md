# Verifiable Experience Pipeline
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A training pipeline that grows an **agentic** LLM by connecting long-horizon, **tool-mediated** interactions to **executable environments** and **externally verified outcomes**, so the RL signal comes from whether the produced artifact actually runs / actually passes rather than from a learned reward model. Introduced in the Atria Dawn tech report (2026) as the training substrate for a frontier agentic LLM.

**Prereqs:** [../post-training/rlvr.md](../post-training/rlvr.md), [../post-training/_rl.md](../post-training/_rl.md)
**Related:** [../post-training/grpo.md](../post-training/grpo.md), [../safety/safety-case.md](../safety/safety-case.md)

---

## What it is

Long-horizon agent training has two entangled reward-signal problems. First, reward hacking of learned reward models over multi-step trajectories is severe — the agent finds sequences that maximize the RM's score without producing useful artifacts. Second, sparse ground-truth signals ("did the whole task succeed?") arrive too late to shape intermediate behavior.

The Verifiable Experience Pipeline attacks both by anchoring training in **executable environments** where the model's tool-mediated actions produce concrete artifacts (code, files, data, submitted forms) and **external verifiers** decide whether those artifacts are correct. The verifiable-reward primitive from [RLVR](../post-training/rlvr.md) is applied trajectory-by-trajectory rather than only turn-by-turn.

## How it works

Each training trajectory follows this shape:

1. **Task specification.** A task is defined by a natural-language goal plus a machine-checkable acceptance test (a unit test, a differential eval, a scoring rubric evaluated by a fixed program, or the output of a fixed external tool).
2. **Tool-mediated rollout.** The agent interacts with an executable environment (browser, terminal, code sandbox, retrieval, external APIs) via tool calls. Each tool call and its concrete result are recorded.
3. **External verification.** After the rollout, the produced artifact is submitted to the acceptance test. The verifier is external to the model — a compiler, a test runner, an API contract check — not a learned reward model.
4. **Policy update.** RL updates (GRPO-family or PPO-family) use the verifier's binary or graded outcome as reward, credit-assigned back through the trajectory.

The pipeline's "pipeline" character is that steps 1–3 are wrapped in a data-generation loop: task instances, environments, and verifiers are curated at scale so the model sees a wide distribution of tool-mediated tasks with real acceptance criteria.

## Why it matters

- **Grounded reward.** External verifiers cannot be Goodharted by the model as easily as a learned RM. A test that passes only if `pytest` reports success, or if a submitted form is accepted by a real service, is by construction hard to game.
- **Long-horizon credit signal.** Because the acceptance test evaluates the whole artifact, the reward signal covers the full trajectory rather than only the last action.
- **Frontier evidence at scale.** Atria Dawn Preview reports competitive-with-frontier performance across 16 agentic benchmarks and top scores on 5 of them, using this pipeline as its training substrate; the paper positions verifiable experience as the reason a single model can generalize across research and engineering workflows.

## Gotchas & tricks

- **Acceptance tests must be tight.** A loose test — e.g. "code compiles" instead of "code passes the reference tests" — invites shortcut behavior that satisfies the test without producing the intended artifact.
- **Executable environments have real side effects.** Sandboxing has to cover file systems, networks, and external accounts; a trajectory that "succeeds" by mutating shared state outside the sandbox is not a success.
- **Verifier throughput can bottleneck training.** Compilers, test runners, and external APIs are slow compared to a learned RM. Practical pipelines batch verification, cache intermediate build products, and parallelize across many sandbox workers.
- **Task-instance diversity is the data problem.** The pipeline's coverage is only as good as the distribution of tasks and environments. Curation and synthetic generation of new task instances (with their verifiers) is where most of the engineering goes.

## Sources

- Paper: *Atria Dawn: The Dawn of Agentic Superintelligence* — Guo, Gui, Chen, Dong, Lu, et al. — 2026 — [arXiv:2609.15818](https://arxiv.org/abs/2609.15818).
- Background: [RLVR](../post-training/rlvr.md) — the verifiable-reward primitive at turn scale.
