# LatentGRM
*Depth — latent-trajectory generative reward modeling.*

**TL;DR:** Generative reward models that write long textual judgements are expensive to serve inside RL loops. LatentGRM replaces the discrete reasoning tokens with a short *latent* trajectory — a sequence of continuous hidden vectors — trained to preserve the semantic content of the textual judgement under a reconstruction loss. A separate interpreter decodes criterion-level evaluations from a saved latent trajectory for offline inspection. Accuracy competitive with SFT text-judge baselines at 4B and 8B scales, with large reductions in trajectory length and inference time.

**Prereqs:** [_rewards](_rewards.md), [cot-reward-model](cot-reward-model.md)
**Related:** [_distillation](_distillation.md), [grpo](grpo.md), [ppo](ppo.md)

---

## What it is

A reward-model architecture that performs *multi-criterion evaluation* (helpfulness, correctness, safety, …) using a compact continuous latent trajectory instead of a long textual chain of thought. Serves two needs simultaneously: fast scoring inside RL training loops, and offline interpretability through a learned decoder.

## How it works

1. **Teacher judge.** An explicit SFT'd text judge writes out a long multi-criterion evaluation and a final scalar score for each (prompt, response) pair.
2. **Latent trajectory encoder.** A smaller student model is trained to produce a *short sequence* of continuous latent vectors from the same input, from which both the score and (via the interpreter) the criterion-level text can be reconstructed.
3. **Semantics-preserving compression loss.** Combines (a) a direct score-matching loss to the teacher, (b) a reconstruction loss where the interpreter must decode criterion-level text from the latent trajectory, (c) typically a contrastive term so trajectories encoding similar judgements stay close.

At inference, only the latent-trajectory student runs — the interpreter is only needed when a human wants to see *why* the model gave that score.

## Why it matters

Reward / judge models are increasingly in the inner loop of RL training. Cutting judge inference time 5–10× while keeping an interpretability back door directly accelerates every GRPO / PPO run that uses a generative RM. The pattern — compress reasoning into a latent *and* decode it on demand — may generalize beyond reward modeling.

## Gotchas & tricks

- **Interpreter drift.** If the interpreter and the student aren't trained jointly, the decoded explanation can diverge from the actual latent content. Train end-to-end.
- **Scalar score first, text second.** The score is the primary training signal; the reconstructed text is an interpretability affordance, not a replacement for a text judge where a human needs real explanations in the loop.
- **Short trajectories only help where discrete-token CoT was long.** On simple preference comparisons, the compression saves little.

## Sources

- Paper: *Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression* — Liang, Yang, Chen, Zhang, Wang, Wang, Li, 2026 — [arXiv:2610.09788](https://arxiv.org/abs/2610.09788)
