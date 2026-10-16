# Action-Level Distillation (ActionOPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** Distillation at the **action/decision** level rather than the token level. At states the student visits, the teacher is sampled for its action; the student is rewarded for semantic agreement. Requires no teacher logits, no shared tokenizer, and no teacher-side instrumentation — black-box API access is enough.

**Prereqs:** [on-policy-distillation.md](./on-policy-distillation.md), [grpo.md](./grpo.md), [rlvr.md](./rlvr.md)
**Related:** [_distillation.md](./_distillation.md) · [rejection-sampling.md](./rejection-sampling.md)

---

## What it is

Token-level distillation needs white-box access to the teacher — logits or at minimum a shared tokenizer so sequences line up. For API-only frontier teachers and heterogeneous student architectures, that access does not exist.

ActionOPD retargets supervision to the unit that *does* travel across tokenizers: the action. In a tool-use agent, an action is "call `search` with query X" or "write to `file.py` the content Y." The student's action matches the teacher's whenever they'd do the same thing — regardless of how the action is tokenized or verbalized.

## How it works

Agent setting. Let the state at step $t$ be $s_t$ (observation history + goal), the student's action $a^S_t \sim \pi_\theta(\cdot \mid s_t)$, and the teacher's action $a^T_t \sim \pi_T(\cdot \mid s_t)$.

1. **Rollout.** The student takes a trajectory in the environment, producing states and actions $\{(s_t, a^S_t)\}$.
2. **Teacher action sampling.** At each state $s_t$ visited by the student, query the teacher for $a^T_t$ (one query per state — no logit extraction).
3. **Semantic agreement reward.** Define a reward $r_t = \mathrm{match}(a^S_t, a^T_t)$. "Match" is action-type and argument equivalence (same tool, equivalent arguments), not textual string equality — a cheap semantic check or an LLM judge.
4. **GRPO update.** Treat $r_t$ as a per-step reward and run GRPO over the full trajectory. Advantages are group-relative across $G$ student rollouts per prompt.

**Self-ActionOPD** is the same recipe with the same model as both student and teacher, with the teacher given privileged context (e.g., a system prompt revealing the goal structure). The student learns to reach the privileged-context behavior without the privilege.

## Why it matters

- **Black-box teachers work.** The frontier closed models that make the best teachers are often API-only. Token-level methods can't use them; ActionOPD can.
- **Tokenizer-independent.** Teacher and student can share no vocabulary at all. Important for cross-family distillation (Llama ← GPT, Qwen ← Claude, etc.).
- **Natural fit for agents.** Agent episodes are already structured as action sequences; the action-level unit matches the task's native granularity.
- **Measured gains.** 3.3–6.9 point averages on four out-of-distribution tool-use benchmarks over strong baselines, on Qwen3-4B / 8B.

## Gotchas & tricks

- **"Match" is a classifier.** Semantic action equivalence is itself a judgment call — the reward is only as reliable as the matcher. An LLM judge is common; a rule-based matcher is cheaper but brittle.
- **Teacher is called at every state the student visits.** For a $k$-step rollout and $G$-way GRPO, that's $k \cdot G$ teacher calls per prompt — budget this carefully.
- **State-coverage mismatch.** The teacher is queried only at student-visited states. If the student never visits a state where the teacher would do something importantly different, that teaching signal is unreachable.
- **No guarantee of correctness.** The teacher's action might be wrong; the student is being trained to imitate the teacher, not to be optimal. Combine with task-level verifier reward for correctness guarantees.
- **Argument equivalence is where things break.** Tool arguments (file paths, query strings, SQL predicates) often have many surface forms for the same meaning; the matcher must handle this or ActionOPD will mis-penalize valid student actions.

## Sources

- Paper: *ActionOPD: Distilling Decisions, Not Tokens* — 2026 — [arXiv:2610.19820](https://arxiv.org/abs/2610.19820). Results on Qwen3-4B / Qwen3-8B with four tool-use benchmarks; also introduces Self-ActionOPD.
