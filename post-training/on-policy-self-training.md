# On-Policy Self-Training (OPST)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A data-free self-improvement loop for LLMs: given only coarse topic labels (e.g. "Abstract Algebra"), the model writes its own questions, rationales, and candidate answers, then distills them under an **EMA copy of itself as teacher**. The teacher sees the rationale and answer as *privileged context* while the student sees only the question. Pseudo-verification and a consensus gate keep training near the capability frontier, so the loop can be run for several rounds with no human-curated data.

**Prereqs:** [grpo.md](grpo.md), [rejection-sampling.md](rejection-sampling.md), [rlvr.md](rlvr.md)
**Related:** [function-space-distillation.md](function-space-distillation.md), [reasoning/long-cot-rl.md](reasoning/long-cot-rl.md), [_rewards.md](_rewards.md)

---

## What it is

Self-training normally needs curated questions and reference answers. OPST removes both. The only external input is a list of topic descriptors. From these, the model generates questions, generates rationale+answer pairs, filters them, and distills them back into itself — all on-policy, so the training data stays close to what the student is currently able to produce.

## How it works

The loop has four stages per round:

1. **Generate.** For each topic, the student samples a batch of questions, then a rationale+answer per question.
2. **Pseudo-verify.** Keep a (question, answer) only if the model reproduces the same answer in a fresh context and on a paraphrase of the question. Prevents trivially inconsistent self-reinforcement.
3. **Consensus-gate the curriculum.** Drop questions the student already answers consistently. The remaining questions sit at the frontier of the student's capability.
4. **Distill.** An **EMA copy of the student** acts as the teacher. The teacher is conditioned on the privileged (rationale, answer) context; the student is conditioned only on the question. The student matches the teacher's token distributions via a **skewed Jensen-Shannon divergence (α ≈ 0.8)**. Because the teacher is always a strictly stronger view of the same model, targets are stable.

Repeat for several rounds; the student and the EMA teacher co-improve.

## Why it matters

- Loosens the "data wall" for post-training: a model's own on-policy exploration becomes a credible training distribution.
- Separates *what the teacher sees* from *what the student sees* — a cleaner way to think about teacher-student distillation in general. The ablation that the privileged rationale matters more than the privileged answer is a reusable insight.
- Reported gains on **Gemma 4 E2B** without human data: **+10.2 MMLU, +4.0 GPQA, +7.5 HellaSwag**. Beats SFT and GRPO on the same self-generated data, and beats curated-question baselines (u-OPSD, R-Zero). Visual OPST on unlabeled images adds **+6.9 VQAv2** at 2B.

## Gotchas & tricks

- **Self-consistency is not correctness.** The pseudo-verifier only checks reproducibility, so a model that is consistently wrong will reinforce the error. OPST can amplify what the model knows, not fix what it doesn't.
- **Multi-domain transfer is uneven.** The paper reports a 4.9-point regression on MATH-500 in a multi-domain run; mixing topics is not free.
- **Topic descriptors are still hand-chosen.** The loop is data-free but not goal-free; the choice of topic set shapes the eventual competence profile.
- **EMA, not frozen snapshot.** The teacher must trail the student slowly (α small) so targets are smooth; a stale frozen teacher collapses to SFT-on-own-outputs.

## Sources

- Paper: *On-Policy Self-Training: Data-Free Self-Improvement in Large Language Models* — 2026 — https://arxiv.org/abs/2610.32770
