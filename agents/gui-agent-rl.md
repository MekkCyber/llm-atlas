# GUI Agent RL (with Mocked Environments)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** RL post-training for GUI agents (mobile, desktop, web) using **synthesized resettable mock apps** as the training environment. The bottleneck is cheap, deterministic, high-fidelity environments — not the RL algorithm. MAI-Gym synthesizes mocks by cloning real app UIs into deterministic replicas with execution-validated verifiers (did the booking actually get placed?), scales to 20,000 parallel environment instances, and demonstrates sim-to-real transfer: RL in a mocked train-ticketing app lifts real-phone task success from 28.8% → 48.1%.

**Prereqs:** [../post-training/grpo.md](../post-training/grpo.md), [../post-training/rlvr.md](../post-training/rlvr.md)
**Related:** [../post-training/_rl.md](../post-training/_rl.md), [../systems/partial-rollouts.md](../systems/partial-rollouts.md)

---

## What it is

A GUI agent is a model that controls a user interface — taps, swipes, keyboard inputs, API calls — to accomplish a user task (book a ticket, fill a form, navigate a site). RL post-training requires three things:

1. An environment the agent can act in.
2. A reward signal from a verifier that confirms task completion.
3. The ability to run many parallel rollouts cheaply.

Real apps are too slow, too flaky, and too fragile for RL at scale (rate limits, non-determinism, side effects on real user accounts). **Mocked apps** are the solution: software replicas with the same observable UI and the same action API, but deterministic, resettable, and side-effect-free.

---

## How it works

### Mock synthesis

For each real app, build a deterministic mock that:

- Mirrors the real UI layout (so the agent sees the same screens).
- Accepts the same actions (tap at coordinate, type text).
- Has a small internal state machine that advances in response to actions.
- Exposes an **execution verifier** — a function from state to {task-complete, task-failed, pending} that doesn't depend on human judgment.

Mock synthesis in MAI-Gym is scripted / semi-automated; the paper also supports cross-app tasks that compose state from multiple mocks.

### Reward from execution verifier

$R = 1$ if the verifier confirms completion within the step budget, else 0 (with optional shaping for partial progress). This is RLVR at the environment level.

### RL loop

Use GRPO (or any critic-free group-relative method) with the mocked env as the rollout substrate:

1. Spawn 20k parallel mock instances across machines.
2. Each instance runs an independent agent rollout against a sampled task.
3. Collect $(prompt, action-trace, verifier\_reward)$ tuples.
4. Group-relative advantage; standard PPO clip; KL anchor to SFT.

### Sim-to-real check

Evaluate on real devices to confirm that policies trained in mocks transfer. MAI-Gym reports the 28.8% → 48.1% real-phone success delta as the headline sim-to-real result.

---

## Why it matters

- **Makes GUI agent RL reproducible.** Previous computer-use RL was gated by access to proprietary environments (OSWorld etc.) and compute at frontier labs. Mocked apps + the open MAI-Gym platform lower the barrier dramatically.
- **Analogous to robotics sim-to-real.** Both start from "the real environment is too expensive for RL" and converge on "build cheap simulators with execution verifiers, prove sim-to-real empirically." The GUI case is arguably easier because UI replication is a scripting problem, not a physics problem.
- **Supports open-ended tasks.** Cross-app tasks (verified via composite execution checks) extend the recipe past single-screen flows to realistic workflows.
- **Scales.** 20k parallel environments is enough to feed modern GRPO rollouts without becoming the bottleneck.

---

## Gotchas & tricks

- **Sim fidelity gap.** Visual rendering differences, timing differences, missing error states — mock-trained policies can brittle on real devices. The paper handles this with high-fidelity screenshot cloning; cross-app verifiers in the mock catch many issues.
- **Reward hacking in mocks.** Agents may discover mock-only exploits (e.g., tap a UI element in a way that only the mock permits). Keep a real-device evaluation gate.
- **Verifier coverage drives reward quality.** If the execution verifier only checks "task completed" and not "no side effects," the agent will shortcut. Compose verifiers (did you book a ticket? did you also not open settings?) for stricter signals.
- **Mock maintenance cost.** When real apps update, mocks drift. The paper's recipe includes a mock-regeneration step; still, this is operational cost.
- **Compose with computer-use agent recipes.** OpenGCUA-style harness-to-model transfer runs on top of this environment stack; MAI-Gym is the training layer, OpenGCUA is the deployed-agent layer.

---

## Sources

- Paper: *MAI-Gym: An Open All-in-One Platform for GUI Agent Evaluation and RL Training across Mobile and Computer Environments* — Tong et al., Alibaba Tongyi-MAI, 2026 — arXiv [2610.20939](https://arxiv.org/abs/2610.20939).
- Related: *OpenGCUA* — Zhou et al., 2026 — arXiv [2610.22200](https://arxiv.org/abs/2610.22200).
- Baseline: *GRPO* — [../post-training/grpo.md](../post-training/grpo.md).
