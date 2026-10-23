# Bridged Self-Distillation (BSD)
*Depth — on-policy distillation that uses the teacher only for the trace, with a "self-teacher" (student with privileged context) scoring the student's own rollouts.*

**TL;DR:** Classical distillation needs shared tokenizers (match logits) or hidden-state access (match internals). BSD instead has the student do the scoring: a copy of the student is given the teacher's trace as **privileged context** (the "self-teacher"), and it scores the student's own rollouts token-by-token. A verifier picks the direction of each update; the self-teacher's token-level guidance sets the weight. Works for shared-tokenizer, cross-tokenizer, and *closed-API* teachers in one unified method. Beats baselines tailored to each setting across 12 teacher-student pairs on 3 math benchmarks.

**Prereqs:** [_rl](_rl.md), [grpo](grpo.md), [rejection-sampling](rejection-sampling.md)
**Related:** [rlvr](rlvr.md), [reasoning/traceforge](reasoning/traceforge.md)

---

## What it is

Distilling reasoning from a stronger teacher into a smaller student usually assumes one of:

- **Logit match** — requires shared tokenizer.
- **Hidden-state match** — requires access to teacher internals (open weights).
- **Trace imitation (SFT on teacher traces)** — works everywhere, but is purely offline and style-coupled.

**BSD** is on-policy (student rollouts) and only needs the teacher's trace *as text*. The student plays two roles: the policy being trained, and a frozen "self-teacher" copy that is given the teacher's trace as privileged context. Because the self-teacher is the same architecture as the student, it can score the student's rollouts token-by-token without any tokenizer or logit alignment with the teacher.

---

## How it works

For each prompt:

1. **Student rolls out.** Standard on-policy sample, no teacher context.
2. **Self-teacher scores.** A frozen copy of the student is run with the teacher's trace prepended as privileged context (hidden from the student at rollout time). It assigns a per-token "preferred vs. not" signal to the student's rollout.
3. **Verifier decides direction.** For each prompt, a deterministic verifier (math checker, test suite) decides whether the student's final answer is correct.
4. **Update.** On successful rollouts, reinforce the trajectory with the self-teacher's per-token guidance as weight. On unsuccessful rollouts, penalize using the same per-token weights to focus the penalty where the self-teacher thought the student went wrong.

The self-teacher never needs the teacher's hidden states — only the trace text. The teacher's role is to produce reasoning content the self-teacher can condition on; the student's role is to produce rollouts whose on-policy correction uses the self-teacher.

---

## Why it matters

Three operational wins:

1. **Works with closed APIs.** You can distill from GPT-5-class teachers that only return text, with no logit access.
2. **Cross-tokenizer.** No need to align subword inventories between teacher and student.
3. **On-policy signal.** Unlike SFT on teacher traces, BSD updates the student on its *own* rollouts, which avoids the usual SFT-to-RL mismatch.

Experimentally, one unified method beats three families of per-setting distillation tricks across 12 teacher-student pairs on three AIME-class math benchmarks.

---

## Gotchas & tricks

- **Self-teacher must be frozen.** If the self-teacher drifts along with the student, the signal degrades (both are now wrong together). Freeze at the pre-distillation checkpoint.
- **Privileged context length.** Long teacher traces inflate self-teacher inference cost; cache them, or truncate to the parts the verifier cares about.
- **Verifier is the ceiling.** No verifier → no direction signal → BSD degrades toward SFT. Reserve BSD for domains where correctness is machine-verifiable.
- **Not for pretraining.** BSD is a post-training technique; it does not replace SFT, it layers on top.

---

## Sources

- Paper: *Bridged Self-Distillation for On-Policy Learning* — Runsheng Chen, Yihong Liu, Raoyuan Zhao, Michael A. Hedderich, Hinrich Schütze, Matthias Grabmair — MCML / TUM / LMU, 2026 — [arXiv:2610.26397](https://arxiv.org/abs/2610.26397).
- Background: on-policy distillation (Hinton et al., 2015, generalizations); DPO and GRPO as neighbors in the on-policy RL space.
