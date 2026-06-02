# Rubric Reward

*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A process-style reward for long-context reasoning RL that scores responses by *which gold entities appear along the reasoning chain*, at entity granularity, and is applied **only on responses that already have the correct final answer**. The positive-only gating breaks the standard PRM reward-hacking failure mode: the policy cannot game the rubric on wrong answers, only refine the reasoning quality among correct ones.

**Prereqs:** [../rlvr](../rlvr.md), [../grpo](../grpo.md), [../_rewards](../_rewards.md)
**Related:** [prm](prm.md) · [orm](orm.md) · [long-cot-rl](long-cot-rl.md) · [../_rl](../_rl.md)

---

## What it is

[Process reward models (PRMs)](prm.md) score reasoning steps individually, in principle giving fine-grained credit to the parts of a CoT that contributed to the answer. In practice they are brittle: the policy learns to insert step text that earns process credit without actually progressing toward the answer, and reward hacking dominates whatever signal the PRM was supposed to provide. DeepSeek-R1 famously tried and abandoned PRMs as RL rewards for this reason.

Rubric rewards are a more constrained design point. They define process supervision *as a checklist* — specifically, the set of gold entities (names, key facts, intermediate quantities) that a correct reasoning chain should mention. Each entity present in the chain adds a small reward. Crucially, the rubric reward is applied **only when the final answer is correct**: wrong-answer responses receive zero rubric credit regardless of how many entities they cite.

---

## How it works

### Constructing the rubric

For each training prompt, derive the set of gold entities the reasoning chain *should* surface — for multi-hop QA, this is the chain of entities along the ground-truth retrieval path; for math, it could be the named intermediate quantities. The source paper builds these for long-context QA by tracing knowledge-graph walks used to construct the questions.

### The reward signal

For a response o on prompt q:

```
r_final(q, o)   = 1 if final answer matches gold, else 0
r_rubric(q, o)  = (# rubric entities cited in o) / |rubric|   if r_final == 1
                = 0                                            if r_final == 0
R(q, o)         = r_final + λ · r_rubric
```

λ is small (the rubric is a refinement on top of correctness, not a substitute). The positive-only gate is the load-bearing design choice.

### Why positive-only gating defeats reward hacking

If r_rubric is paid on incorrect responses, the policy can score by stuffing entity names into the response while still answering incorrectly. The rubric becomes an attack surface. With r_rubric gated on r_final = 1, the policy must *first* answer correctly to earn rubric credit. That removes the hacking path — entity stuffing on a wrong answer earns nothing.

### Integration with GRPO

Used as the per-response reward in [GRPO](../grpo.md) (or any group-relative algorithm). The group-relative advantage normalizes within a prompt, so the rubric component contributes to the gradient only when the group contains a mix of correct responses with different rubric scores — exactly the regime where it should differentiate.

---

## Why it matters

- **A workable form of process supervision.** PRMs failed as RL rewards because they were attackable; the positive-only rubric reward is structurally hack-resistant by removing the wrong-answer reward path.
- **Strong long-context-reasoning gains.** Consistent improvements across three reasoning LLMs (4B–30B) on five long-context benchmarks vs RLVR baselines, in the source paper.
- **Composable with RLVR.** The rule-based final-answer reward stays the primary signal. The rubric reward is a refinement that fights "lucky" correct answers without actually citing the right evidence.

---

## Gotchas & tricks

- **Rubric quality bounds reward quality.** Bad rubrics (missing relevant entities, including irrelevant ones) propagate directly into the policy. Treat rubric construction as data work, not a one-shot script.
- **λ should be small.** A rubric reward that dominates final-answer reward turns the policy back into a reward hacker — it learns to maximize the rubric on responses that are barely correct.
- **Positive-only is the whole point.** A naive implementation that drops the gate and pays rubric credit on wrong answers reproduces the PRM failure mode. The gate is non-negotiable.
- **Entity-overlap is one rubric form among many.** The general pattern — "checklist of items a good answer should cite, paid only on correct answers" — extends to math (must mention these intermediate quantities), code (must call these APIs), etc. The same gating logic applies.

---

## Sources

- Paper: *LongTraceRL: Learning Long-Context Reasoning from Search Agent Trajectories with Rubric Rewards* — Lin, Zhang, Hou, Li, 2026 — Tsinghua KEG. Introduces the entity-rubric reward with positive-only gating, paired with tiered distractors mined from search-agent trajectories.
- Code/data: https://github.com/THU-KEG/LongTraceRL
