# Hybrid Rubric + Rule Reward
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A composite RL reward that sums two complementary signals — a *rubric* score (an LLM-judged multi-dimensional quality rating) and a *rule* score (deterministic checks like format compliance or execution success). Combines RLHF-style nuance with RLVR-style robustness. Introduced at scale in Qwen-AgentWorld (2026) to sharpen language-world-model simulation fidelity along five judged dimensions while pinning down structural correctness with rules.

**Prereqs:** [_rewards.md](_rewards.md), [rlvr.md](rlvr.md)
**Related:** [grpo.md](grpo.md) · [_rl.md](_rl.md) · [reasoning/orm.md](reasoning/orm.md) · [../case-studies/qwen-agentworld.md](../case-studies/qwen-agentworld.md)

---

## What it is

Pure rule-based rewards (the RLVR pattern) are robust to gaming but blind to nuance: they catch "this is the wrong answer" but not "this is the right answer but poorly structured." Pure learned-reward models (preference RMs) are sensitive to nuance but easy to game, especially over long rollouts. A hybrid reward sums the two with explicit weights, giving the policy a signal that combines the strengths of both.

The hybrid form for one response `o`:

```
R(o) = Σ_i w_rubric_i · rubric_i(o) + Σ_j w_rule_j · rule_j(o)
```

Where `rubric_i` is an LLM-judged scalar along one quality dimension (faithfulness, completeness, plausibility, …), and `rule_j` is a deterministic boolean or scalar check (format match, value within range, expected tool called, …).

## How it works

Two implementation pieces matter:

**Rubric judging.** A judge LLM is prompted with the response and a fixed rubric per dimension ("rate the predicted state's faithfulness to expected behavior on a 0-5 scale, considering …"). The rubric is published with the system. Multiple dimensions are typical — Qwen-AgentWorld grades on 5 dimensions covering correctness, completeness, formatting, plausibility, and execution-realism.

**Rule checks.** Cheap, deterministic — regex match on output format, exact-match on a key field, exit code from a script, presence of a required substring. Rules anchor the reward against grossly malformed outputs that the rubric judge might still rate generously.

The two are weighted into a single scalar per response and fed into the standard GRPO / PPO update.

## Why it matters

- Bridges the gap between RLVR (works only on cleanly verifiable tasks) and RLHF-style preference learning (more flexible but reward-hackable). Hybrid rewards work in the messy middle — tasks like simulation fidelity, document generation, or agent traces where some aspects are checkable and others are not.
- Rubric dimensions are *named*, *published*, and *auditable*. Easier to diagnose reward hacks than with an opaque preference RM ("rubric dimension 2 collapsed — formatting reward dominates").
- Compositionally robust: adding a new dimension or rule does not require retraining a reward model, just a new prompt or check.

## Gotchas & tricks

- Weight tuning is the operational pain. A too-heavy rubric dominates and invites judge-hacking; a too-heavy rule term ignores quality. Sweep early and re-tune after policy drift.
- The judge LLM is itself a model and shares its training biases. Multi-judge consensus or rotating judges is a known mitigation.
- Rubrics inflate over training (the policy learns to write outputs that score 5/5 even when they don't deserve it). Pair with periodic rubric audits or rotate judge models.
- Cost: rubric judging is an LLM call per response, multiplying RL rollout cost. Practical setups batch judging or use a smaller judge model for the rubric and a larger one for spot checks.
- Distinct from preference-RM PPO: hybrid rewards score outputs directly along named axes; preference RMs score relative pairs. Both can be combined (rubric + rule + preference RM) for the most flexible setup at the cost of more knobs.

## Sources

- Paper: *Language World Models for General Agents (Qwen-AgentWorld)* — Qwen Team, 2026 — [arXiv:2606.24597](https://arxiv.org/abs/2606.24597) — hybrid rubric+rule used to sharpen simulation fidelity.
- Reference: *Constitutional AI* — Bai et al., Anthropic, 2022 — rubric-style scoring lineage.
- Reference: *DeepSeek-R1* — DeepSeek, 2025 — direct-sum composite rewards (rule + format + language consistency) — same family.
