# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation recipe in which the *student* generates its own rollouts and a *teacher* (usually a stronger post-trained model, sometimes the student's own thinking mode) provides token-level supervision on those rollouts. Unlike offline distillation, the training distribution matches deployment; unlike RL, the learning signal is teacher-defined rather than reward-defined. Now the workhorse for turning weak instruct-tuned or reasoning-hybrid models into strong ones (Qwen3, Llama 3.x, Gemma 3.x).

**Prereqs:** [rejection-sampling.md](rejection-sampling.md), [_post-training.md](_post-training.md), [grpo.md](grpo.md)
**Related:** [dpo.md](dpo.md), [long-cot-rl.md](reasoning/long-cot-rl.md), [rlvr.md](rlvr.md)

---

## What it is

Two knobs describe any distillation setup: *who samples* (offline dataset vs on-policy student) and *what teacher provides* (final answers, reasoning traces, token logits, or preferences). On-policy distillation fixes **student sampling** and **teacher supervision on those samples** — the student rolls out its own trajectory and the teacher scores or corrects it token by token.

The setup is a middle ground between SFT and RL. SFT trains on fixed teacher-generated data, drifting from the student's own distribution. RL uses reward signals, which are sparse and noisy at token scale. OPD uses the teacher's full token distribution over the student's own trajectories — every step is supervised, and the trajectories are what the student would actually produce at inference.

## How it works

1. **Sample** a batch of prompts. The *student* generates responses on-policy (with temperature > 0).
2. **Teacher forward pass** over each student response, producing per-token distributions or per-token scores.
3. **Loss** — typical choices are token-level KL from teacher to student, or a weighted cross-entropy against the teacher's top-k. Some recipes add a length or KL-to-reference regularizer.
4. **Optimizer step** on the student only; the teacher is frozen.

Key variants: **reference-free OPD** (teacher is the student's own thinking mode, no external labels), **privileged-information OPD** (teacher sees a worked solution the student doesn't), and **self-retiring OPD** (per-skill plateau detection halts distillation on that skill).

## Why it matters

- **Distribution match.** Training on student-generated trajectories closes the offline-SFT distribution gap without paying RL's variance.
- **Dense signal.** Every token is supervised — no reward sparsity.
- **Cheap compared to RL.** No critic, no reward model to train, no PPO clip/KL to hyperparameter-tune.

OPD has become the default second stage after SFT in many open-weight recipes, and it's what makes "long thinking" mode transferable back into direct-response mode via cross-mode parameter transfer.

## Gotchas & tricks

- **EOS-token mismatch** between base student and teacher inflates output length. Treat functionally equivalent EOS variants as a single stopping action rather than distinct tokens.
- **Teacher-optimal ≠ student-optimal** — a teacher chasing an unreachable target can degrade the student. Add a KL-to-reference term or use per-skill retirement.
- Privileged information (a worked solution shown only to the teacher) often helps *not* because it leaks answers but because it enables cross-mode transfer between direct-response and thinking-enabled inference.
- Rollouts are the dominant cost — reuse student generations across multiple teachers or across a training window when possible.

## Sources

- "When EOS Tokens Disagree: Understanding Length Inflation in On-Policy Distillation" — Yang et al., 2026 — [arXiv:2609.20511](https://arxiv.org/abs/2609.20511).
- "RetireOPD: Self-Retiring On-Policy Distillation for Agentic Reinforcement Learning" — Yu et al., 2026 — [arXiv:2609.20784](https://arxiv.org/abs/2609.20784).
- "What Does Privileged Information Add to On-Policy Self-Distillation?" — Zhang et al., 2026 — [arXiv:2609.20612](https://arxiv.org/abs/2609.20612).
