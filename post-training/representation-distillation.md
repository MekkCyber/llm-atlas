# Representation Distillation (Hidden-State KD)
*Depth — supervise the student's intermediate hidden states against the teacher's, instead of (or alongside) the LM-head logits.*

**TL;DR:** Standard on-policy distillation matches the student's next-token distribution to the teacher's via KL on logits. **Representation distillation** moves the supervision upstream: on the same on-policy rollouts, regress the student's residual stream at chosen layers onto the teacher's. The gradient bypasses the LM head, which is both the dominant transient-memory cost (logits over $|V|$) and the dominant source of variance in the KD signal. OPRD (Yang et al., 2026) is the canonical instance and reports **1.44× faster training, up to 54% less actor-update transient memory**, and a closed student–teacher gap on math reasoning.

**Prereqs:** [grpo](grpo.md), [_post-training](_post-training.md)
**Related:** [rlvr](rlvr.md), [long-cot-rl](reasoning/long-cot-rl.md), [_rewards](_rewards.md)

---

## What it is

A distillation objective for LLMs where the loss is computed in **hidden-state space**, not output space. Pick a set of layers in the student (often matched 1-1 with the teacher); on every on-policy rollout, push the student's residual stream at those layers toward the teacher's via an L2 or cosine objective. The LM-head KL term can stay as a co-loss or be dropped entirely.

## How it works

```
for prompt, response in on_policy_rollouts:
    h_teacher = teacher.forward(response, return_hidden=L)   # selected layers L
    h_student = student.forward(response, return_hidden=L)
    loss_rep = sum(F.mse_loss(h_student[l], h_teacher[l]) for l in L)
    loss_lm  = (optional) KL(student_logits, teacher_logits)
    (loss_rep + α * loss_lm).backward()
```

Two key choices:

- **Which layers to align.** Aligning every layer is wasteful; aligning only the final pre-LM-head layer is fragile. Mid-network selection (e.g. every 4th layer) is a common compromise.
- **Whether to project.** If the teacher and student have different hidden sizes, you need a small learned projection (a linear layer per aligned site, trained jointly). For same-size pairs no projection is needed.

The theoretical case (OPRD): the conditional variance of the gradient when the target is the teacher's hidden state is *zero* (the target is deterministic given the rollout), whereas KL-on-logits is unbiased but high-variance. Empirically, this turns into faster convergence and tighter final gaps.

## Why it matters

- **Removes the LM-head bottleneck.** The actor-update memory footprint for on-policy KD is dominated by materializing logits over the full vocabulary at every position. Hidden-state targets skip that — directly reflected in the 54% transient-memory drop OPRD reports.
- **Lower-variance updates.** Especially in long-rollout regimes (reasoning RL), hidden-state targets give a denser, more deterministic supervision signal than logit-space KL.
- **Composes with on-policy RL.** Same rollouts that drive GRPO can carry the representation-KD signal. No second pass over the dataset.

## Gotchas & tricks

- **Layer-pair selection matters more than the loss form.** Pairing layers that aren't representationally aligned (e.g. early student layers with late teacher layers) destabilizes training.
- **Don't drop the LM-head term too early.** Hidden-state loss alone can let the student drift in directions the LM head can't read. A small co-loss (α ≈ 0.1) on logits acts as a regularizer.
- **Tokenizer/embedding mismatch breaks everything.** Representation distillation across different tokenizers requires careful re-tokenization plus alignment of positions — easier to use a teacher with the student's tokenizer.
- **Watch for representation collapse on padding.** Aligning hidden states at padding positions silently drags the average loss. Mask explicitly.

## Sources

- Paper: *OPRD: On-Policy Representation Distillation* — Yang et al., Zhejiang U. / Ant Group, 2026 — [arXiv:2606.06021](https://arxiv.org/abs/2606.06021) — names the technique and reports the training-speed and memory wins on math-reasoning benchmarks.
- Related: classical "hint" distillation (Romero et al., 2014) is the spiritual ancestor — the LLM/on-policy formulation is what's new.
