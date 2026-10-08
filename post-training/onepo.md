# OnePO — One-Stage Policy Optimization

*Depth — RL-only domain adaptation that skips SFT by fixing the two failure modes of mixed-policy RL (Gradient Starvation, Teacher-Distribution Anchoring).*

**TL;DR:** The received recipe for domain adaptation is **SFT then RL** — SFT teaches the vocabulary, RL teaches the policy. OnePO argues the SFT stage is a crutch: if you fix the specific reasons mixed-policy RL fails from a base model, you can drop SFT and reach higher final quality with less data. The two fixes are **Adaptive Objective Evolution** (reweight the loss as the policy moves) and **Teacher Retirement** (stop using the teacher once it stops helping).

**Prereqs:** [grpo.md](./grpo.md), [rlvr.md](./rlvr.md), [_rl.md](./_rl.md)
**Related:** [training-inference-mismatch.md](./training-inference-mismatch.md), [rejection-sampling.md](./rejection-sampling.md), [fine-tuning/on-policy-distillation.md](./fine-tuning/on-policy-distillation.md)

---

## What it is

OnePO is a mixed-policy RL algorithm — the policy generates most rollouts, but a stronger teacher model occasionally provides rollouts the student imitates. HuatuoGPT-3 uses it to adapt a *base* LLM (no SFT) into a medical specialist in a single RL stage.

The paper names two failure modes that have made SFT-less RL brittle:

- **Gradient Starvation.** Teacher rollouts contain informative but low-probability-under-the-base-policy tokens. The policy-gradient update is probability-weighted, so these high-information tokens get vanishingly small learning signal.
- **Teacher-Distribution Anchoring.** As the student improves, the teacher's rollouts become stale, but the training signal still pulls the student toward the teacher's distribution. The policy gets anchored to a less-fit distribution.

OnePO targets both.

## How it works

### Adaptive Objective Evolution

The loss function dynamically reweights teacher-sampled tokens by their current *under-policy* probability:

```
w_t = f(π_student(y_t | ...), threshold)
L_teacher_t = w_t · logprob_loss(student, teacher_token_t)
```

As `π_student(y_t)` rises (student has learned the token), `w_t` falls — the loss stops wasting gradient budget there. As `π_student(y_t)` stays low (gradient starvation territory), `w_t` is boosted to recover signal.

### Teacher Retirement

Keep running statistics on each teacher: how often does its sampled trajectory still outperform the student? Once the ratio drops below a threshold, the teacher is retired — subsequent batches sample purely on-policy.

Retirement can be per-prompt-class or global. In practice the global version is simpler and triggers around epoch 2 for HuatuoGPT-3's medical adaptation setup.

### Pipeline

1. Base LLM, no SFT.
2. Mix on-policy student rollouts with teacher rollouts (teacher = a stronger domain-expert model).
3. Compute GRPO/RLVR-style advantages; apply the Adaptive Objective Evolution weights to teacher tokens.
4. Monitor teacher usefulness; retire when below threshold.
5. Continue purely on-policy RL to convergence.

## Why it matters

- **Removes the SFT stage.** No preference-labeled or demonstration-labeled training set required; a weaker base LLM + a stronger teacher + a verifier is the full ingredient list.
- **HuatuoGPT-3 reports 67.2 on HealthBench (Total) with 20K training samples**, beating SFT+RL and pure RL baselines in the paper's comparison.
- **A cleaner diagnosis.** Even where SFT+RL stays the practical choice, naming Gradient Starvation and Teacher-Distribution Anchoring gives a mental model for *why* previous RL-only attempts collapsed. Both are specific, measurable, and fixable.

## Gotchas & tricks

- **Teacher quality matters more than teacher coverage.** OnePO assumes the teacher is strictly better than the base student on the target distribution. A weak teacher triggers Teacher Retirement almost immediately — in that case you're doing pure RL from base, which usually fails.
- **Verifier matters.** OnePO is RLVR-flavored: it needs a correctness signal on every rollout to drive advantages. For medicine the paper uses rule-based + LLM-judge. Without that, Teacher Retirement cannot be scheduled by evidence.
- **Not a replacement for RLHF-style preference work.** OnePO is about capability adaptation to a *verifiable* target domain. Alignment / helpfulness / harmlessness still need the usual RLHF stack.
- **Related to standard mixed-policy RL in RLHF**, but the specific contributions are the diagnosed failure modes and the two fixes, not the mixed-policy framing itself.

## Sources

- Paper: *HuatuoGPT-3: RL-Only Domain Adaptation from Base Models* — Xie et al., CUHK-Shenzhen, 2026 — [arXiv:2610.05966](https://arxiv.org/abs/2610.05966).
- Related: *DeepSeek-R1-Zero* — DeepSeek, 2025 — GRPO from base with verifiable rewards, no SFT.
- Related: standard RLHF mixed-policy references — Christiano et al., 2017; Stiennon et al., 2020.
