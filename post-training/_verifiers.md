# Verifiers for Agent RL
*Taxonomy — the four families of verification signals used to score agent rollouts, and how to choose.*

**TL;DR:** For frontier coding agents the generate↔verify asymmetry has flipped: generating plausible candidates is cheap, *reliably checking them* is the bottleneck. Verifiers fall into four families — **test verifier**, **rubric verifier**, **user-as-verifier**, **agent verifier** — and they trade off on three axes: **scalability** (cheap to apply at training scale), **faithfulness** (proxy for human intent), and **robustness** (survives reward hacking and signal saturation). No fixed verifier nails all three; **the verifier must co-evolve with the generator** as policy capability grows.

**Related taxonomies:** [_rewards](_rewards.md) · [_rl](_rl.md) · [_post-training](_post-training.md)
**Depth files covered here:** [rlvr](rlvr.md) · [cot-reward-model](cot-reward-model.md)

---

## The problem

Verifiable rewards work brilliantly for math (extracted answer match) and code (unit tests pass). The problem starts when tasks become **long-horizon**, **rubric-defined**, or **intent-dependent**:

- Tests can't capture frontend UX, code style, or maintainability.
- Rubrics drift as the model's capability changes — a rubric calibrated for a 50% model is too coarse at 90%.
- Users provide only sparse, noisy verification signal at production scale.
- Agent verifiers (LM-as-judge for long horizons) inherit the policy's own blind spots.

Every verifier is a proxy for human intent; the gap between proxy and intent is what reward hacking and signal saturation exploit.

## The shared pattern

A verifier maps `(prompt, trajectory) → scalar quality signal`. The four families differ on **who produces the signal** and **what intent they encode**:

| Family | Signal source | Where intent lives |
| --- | --- | --- |
| Test verifier | Deterministic tests (unit, integration) | The tests themselves — pre-specified |
| Rubric verifier | Scoring rubric applied by code or LLM | The rubric design — pre-specified |
| User-as-verifier | Human end-user feedback | Production usage — emergent |
| Agent verifier | A scoring LLM agent (often the same family as the generator) | The judge agent's training — implicit |

## Variants

| Verifier | Scalability | Faithfulness | Robustness | When it wins |
| --- | --- | --- | --- | --- |
| **Test verifier** ([RLVR](rlvr.md)) | Best — free per check | Narrow — only what tests cover | High — code can't be fooled | General coding with test coverage |
| **Rubric verifier** | Good — LLM judge | Higher than tests on UX/style | Drifts with policy capability | Frontend, formatting, style |
| **User-as-verifier** | Poor — sparse, slow | Highest — actual user intent | High — humans are not easily fooled | Real-world agent tasks |
| **Agent verifier** ([CoT RM](cot-reward-model.md)) | Moderate — one LM forward | High inside training distribution | Co-degrades with the policy | Long-horizon tasks no test/rubric covers |

## How to choose

- **Coding task with strong test coverage.** Default to test verifier. Cheapest, hardest to hack.
- **Frontend / UX / style.** Tests won't measure what matters. Use rubric verifier with periodic recalibration as the policy improves.
- **Real-world long-horizon agent tasks.** Tests don't apply, rubrics drift fast. User-as-verifier where you can afford the latency; agent verifier where you can't.
- **Long-horizon reasoning.** Agent verifier (CoT-style) is the most general option but the most vulnerable to co-evolution failure with the policy.

The recurring lesson: **no fixed verifier survives a growing policy**. Plan to re-tune or re-train the verifier at every major capability jump.

## Adjacent but distinct

- [_rewards](_rewards.md) — the broader taxonomy. Verifiers are the *signal-production* side of rewards; this file zooms in on the agent/coding case where the signal *is* the bottleneck.
- [reasoning/prm](reasoning/prm.md), [reasoning/orm](reasoning/orm.md) — outcome- and process-reward variants for reasoning. Same family as agent verifier; different label structure.
- [progress-advantage](progress-advantage.md) — the *implicit* verifier that falls out of RL training, distinct from any of the four explicit families above.

## Sources

- Paper: *The Verification Horizon: No Silver Bullet for Coding Agent Rewards* — Wang, Zhang, Liu, Zhang et al., 2026 — [arXiv:2606.26300](https://arxiv.org/abs/2606.26300). Alibaba Qwen.
- Paper: *Tülu 3* — AI2, 2024 — origin of the RLVR formulation underlying the test-verifier family.
- Paper: *Kimi k1.5* — Moonshot AI, 2025 — CoT reward model as the agent-verifier instance.
