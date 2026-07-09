# Policy-aware prompt adaptation (LLM-as-a-Tutor)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** In non-verifiable RL (open-ended instruction following judged by an LLM), training prompts are usually static while the policy improves — so once the policy exceeds prompt difficulty, all rollouts look equally good, the judge can't discriminate, and reward signal collapses. LLM-as-a-Tutor uses **a single LLM as both examiner and generator**: pairwise-compare rollouts to detect non-challenging prompts, then **append atomic constraints** to raise difficulty. Append-only design monotonically increases difficulty in step with the policy — self-calibrating, no external schedule.

**Prereqs:** [rl-prompt-curation.md](./rl-prompt-curation.md), [_rewards.md](./_rewards.md), [grpo.md](./grpo.md)
**Related:** [_rl.md](./_rl.md), [rlvr.md](./rlvr.md)

---

## What it is

In verifiable RL (RLVR: math, code) the reward comes from a rule check — always discriminative, always well-scaled. In non-verifiable RL (instruction following, open-ended generation) the reward comes from an **LLM judge** applied against a per-prompt rubric. This has a critical failure mode: once the policy is good enough that most rollouts satisfy the rubric, the judge collapses to giving all rollouts similar scores. **Group-relative advantages** (GRPO, mirror descent) then become tiny — the gradient signal vanishes.

The usual fix is to make the *rubric* adapt (harder rubric criteria over training). LLM-as-a-Tutor argues the *prompt itself* is the missing axis: rewrite the prompt to make the task genuinely harder for the current policy.

## How it works

**One LLM, two roles per training step:**

1. **Examiner role.** After sampling $G$ rollouts per prompt, the LLM performs pairwise comparisons among them. If the rollouts are hard to differentiate (or the LLM judges them roughly equal), the prompt is flagged **non-challenging** for the current policy.

2. **Generator role.** For each non-challenging prompt, the LLM appends an **atomic constraint** — a single additional requirement to the existing prompt (e.g., "and use only words with 5 letters or fewer", "and format the answer as a haiku", "and reference at least two historical figures"). Never rewrites; only appends.

The append-only design has two properties:

- **Monotonic difficulty.** Constraints can only make the task harder. The prompt gets harder as long as the policy keeps outrunning it.
- **Comparability preserved.** Old-prompt rubric criteria still apply; the new constraint is just an *additional* rubric axis.

The examiner and generator are the **same LLM** (typically the same one used as the judge), and the whole loop runs online during RL. No external difficulty schedule, no separate curriculum learner.

## Why it matters

- **Fixes a known failure mode of long-horizon non-verifiable RL** — reward signal collapse under a static prompt pool. Straightforward once named.
- **Beats prior policy-adaptive baselines** (rubric-adaptive, prompt-rewriting) on three complex instruction-following benchmarks.
- **Self-calibrating.** The loop uses the LLM's own comparison ability to decide when to escalate difficulty — no hand-set schedule.
- **Prompt adaptation as a missing axis of policy-awareness.** Rubric-adaptive methods (rubric evolves) and prompt-rewriting methods (prompt paraphrased) were the prior state; append-only prompt adaptation is a distinct axis worth naming.

## Gotchas & tricks

- **Constraint atomicity matters.** If the generator appends broad constraints (e.g., "make it philosophical"), the prompt loses discriminating power differently — grade collapse in the other direction. Atomic = one verifiable, checkable requirement per step.
- **The same LLM as judge, examiner, and generator** — potentially self-reinforcing biases. Some ablation would want to vary the roles.
- **Not for verifiable RL.** In RLVR, prompt difficulty is fixed by the ground-truth answer; there's no room for a tutor. Applies specifically to non-verifiable / rubric-based RL.
- **Prompt drift over training.** After many rounds, the prompt is heavily annotated with appended constraints; it starts to look unlike natural user queries. This is a train-vs-test distribution problem — periodic reset or a natural-prompt tail might be needed.
- **Composes with GRPO.** LLM-as-a-Tutor doesn't touch the RL update itself; it just manages the prompt pool. Any group-relative algorithm can use it.

## Sources

- Paper: *LLM-as-a-Tutor: Policy-Aware Prompt Adaptation for Non-Verifiable RL* — Kim, Ho, Hwang, et al., KAIST AI, 2026 — [arXiv:2607.04412](https://arxiv.org/abs/2607.04412).
