# d-OPSD — On-policy Self-distillation for Diffusion LLMs
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy self-distillation, adapted to diffusion LLMs. Two changes from AR-OPSD: the self-teacher conditions on the student's own *suffix* (the generated answer) rather than a left-to-right prefix, and supervision is at the **step level** of the denoising trajectory instead of the token level. Outperforms RLVR and SFT on reasoning while using ~10% of RLVR's optimization steps.

**Prereqs:** [rlvr](rlvr.md), [_rl](_rl.md)
**Related:** [grpo](grpo.md), [long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

On-policy self-distillation (OPSD) is the LLM post-training trick where a student model is supervised against a "self-teacher" — typically the same model conditioned on extra information the student doesn't get. For autoregressive LLMs, the privileged information is a teacher-forced **prefix** of the correct answer, and supervision is on the next-token distribution.

This doesn't translate to **diffusion LLMs** (dLLMs), which generate in arbitrary order via iterative denoising. There is no canonical left-to-right "prefix" and token-level KL doesn't match the step-wise structure of denoising. d-OPSD rebuilds the recipe to respect both facts.

---

## How it works

### Self-future suffix conditioning

The student generates a full candidate answer (the **self-future**). That answer is then fed back as a **suffix** condition to construct a self-teacher: the teacher denoises the same prompt with the self-future in scope, which gives it privileged information about where the trajectory is going. The student is supervised against the teacher's denoising distributions at each step.

Because dLLMs are bidirectional and order-agnostic, the suffix conditioning is meaningful in a way that prefix conditioning wouldn't be — and the privileged information is exactly the "ending" the student is aiming for, rather than a leaked head.

### Step-level supervision

Standard OPSD takes a token-level KL between student and teacher logits at every position. d-OPSD instead computes divergence at the **denoising-step level**: at each step $t$ of the iterative denoising schedule, the student's predicted clean tokens are aligned with the teacher's, and the loss is summed across steps.

This matches the actual computation graph of a dLLM (each step produces a full-sequence prediction; the next step refines it) and gives gradient pressure at every refinement, not just the final layer.

### On-policy loop

Each training step:
1. Sample a prompt and run the student to produce a self-future.
2. Build the teacher by conditioning the same model on prompt + self-future suffix.
3. Run both through the denoising schedule; compute step-level divergence as the loss.
4. Update the student.

The teacher is the student itself with extra context — no external teacher, no separate model.

---

## Why it matters

- **First OPSD recipe tailored for dLLMs.** Existing AR-OPSD recipes silently bake in left-to-right ordering; d-OPSD removes that assumption.
- **Sample-efficient relative to RLVR.** ~10× fewer optimization steps to reach RLVR's reasoning-benchmark scores. The self-teacher supplies a denser supervision signal than verifiable rewards alone.
- **Opens dLLM post-training.** Diffusion LLMs have lagged AR LLMs on reasoning benchmarks partly because the post-training stack didn't exist. d-OPSD is one of the first techniques that respects the architecture rather than retrofitting AR recipes.

---

## Gotchas & tricks

- **Suffix conditioning only works for dLLMs.** Applying suffix conditioning to an AR model would leak the answer.
- **Step alignment matters.** If the student and teacher are at different points on the denoising schedule, the step-level loss is comparing apples to oranges. Implementations couple their schedules tightly.
- **Self-future quality bounds learning.** The teacher is only as good as the student's own generation. Early-training students produce noisy self-futures; warmup on SFT before switching to d-OPSD is recommended.
- **Reasoning benchmarks were the test bed.** Code released — open question whether the technique transfers to non-reasoning dLLM domains.

---

## Sources

- Paper: *Learning from the Self-future: On-policy Self-distillation for dLLMs* — Yifu Luo et al., Tsinghua / TUM / NTU et al., 2026 — [arXiv:2606.18195](https://arxiv.org/abs/2606.18195).
- Code: [github.com/xingzhejun/d-OPSD](https://github.com/xingzhejun/d-OPSD).
