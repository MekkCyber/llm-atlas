# Verification Horizon

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A position-plus-experiments framework that argues, for coding agents, **verifying a candidate solution is now harder than generating one**. Any reward signal sits on three axes — *scalability*, *faithfulness*, *robustness* — that cannot be maximized simultaneously. As policy capability grows, every fixed verifier eventually breaks (reward hacking, signal saturation). The fix is to **co-evolve verifier and generator** and to design *targeted* verifiers per task type rather than picking one generic reward.

**Prereqs:** [_rewards](_rewards.md), [rlvr](rlvr.md), [_rl](_rl.md)
**Related:** [cot-reward-model](cot-reward-model.md), [reward-model-discretization](reward-model-discretization.md), [reasoning/orm](reasoning/orm.md), [reasoning/prm](reasoning/prm.md)

---

## What it is

A unifying lens on coding-agent RL: every reward function is *only a proxy* for human intent, and the difficulty of building one is twofold:

1. **Intent is underspecified by nature** — even the user can't always state what counts as success.
2. **Optimization widens the proxy-intent gap** — once the policy has enough capacity, it learns to satisfy the proxy in ways the proxy didn't anticipate (reward hacking) or pushes the proxy outside the region it was calibrated for (signal saturation).

The "Verification Horizon" is the capability frontier past which a given fixed verifier flips from *driving improvement* to *being gamed*. Different tasks have different horizons; you can't tell which verifier you need without naming the task.

## How it works

### The three axes

| Axis | Meaning | High-axis examples | Low-axis examples |
| --- | --- | --- | --- |
| **Scalability** | Cost per verification across many rollouts | Unit-test verifier (free) | Human verifier (slow) |
| **Faithfulness** | How closely the verifier tracks real intent | Human verifier (truth itself) | Format regex (proxy) |
| **Robustness** | How well the verifier resists adversarial policy behavior | Hidden-test verifier | Visible-test verifier |

No verifier is high on all three simultaneously. RLVR is high on scalability + robustness but limited in faithfulness (only works where correctness is mechanically checkable). Human verification is faithful but unscalable.

### The four concrete reward constructions studied

| Verifier | Task type | Wins on | Fails on |
| --- | --- | --- | --- |
| **Unit-test verifier** | General coding | Scalability, robustness | Faithfulness — tests under-specify intent |
| **Rubric verifier** | Frontend tasks | Faithfulness (rubrics encode taste) | Scalability — rubrics are hand-written |
| **User-as-verifier** | Real-world agent tasks | Maximum faithfulness | Scalability — humans don't scale |
| **Automated agent verifier** | Long-horizon tasks | Scalable + somewhat faithful | Robustness — agent verifier is itself hackable |

### Co-evolution

The proposed pattern: pair the generator with a verifier that *also* gets updated as the policy improves. When the verifier is a model (rubric LLM, agent verifier), retrain or update its judging rubric/policy whenever the generator clears the current Verification Horizon. The horizon is not a constant — it's a moving target the verifier must chase.

## Why it matters

- **Names a phase change** in agentic RL that frontier labs have been quietly observing. The bottleneck has shifted from "can we generate?" to "can we score?"
- **Replaces "pick the best verifier" with a design framework.** Different points on the policy-capability curve and different task types require different verifier choices.
- **Concrete recipes.** Within each task family, the paper shows targeted verifier design *measurably suppresses reward hacking* and improves task completion across internal and public coding benchmarks.
- **Pairs naturally with [reward-model-discretization](reward-model-discretization.md):** discretization addresses oversensitivity inside a single verifier; co-evolution addresses faithfulness drift across capability levels.

## Gotchas & tricks

- **No single verifier transfers across task families.** Frontend tasks need rubrics; backend code needs tests; long-horizon agent tasks need agent verifiers. Pretending one verifier does all three is the failure mode.
- **Visible vs hidden tests matters more than people admit.** A policy that can see the test set during rollout will fit to the tests verbatim. Hidden test sets are necessary for robustness.
- **Agent verifiers stack hackability.** Using an LLM agent as a verifier shifts the reward-hacking surface from "trick the regex" to "trick another LLM." Strongly recommend pairing with rubrics or rule checks rather than using alone.
- **Capability-tracked rotation.** As the policy improves, the relative cost of each verifier shifts. Schedule re-evaluations of which verifier is dominant for which subset of prompts.

## Sources

- Paper: *The Verification Horizon: No Silver Bullet for Coding Agent Rewards* — Liu, Wang, et al., 2026 — Qwen Team / Alibaba — [arXiv:2606.26300](https://arxiv.org/abs/2606.26300).
- Background: *DeepSeek-R1* — DeepSeek, 2025 — explicit choice to stay on rule-verifier RLVR to avoid reward-hacking risk.
- Background: *Discretizing Reward Models* — Viswanathan et al., 2026 — complementary fix on the within-verifier oversensitivity axis.
