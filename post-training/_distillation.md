# Distillation for LLMs

*Taxonomy — training a smaller / student model to match the behavior of a larger / teacher model.*

**TL;DR:** The classic recipe is to minimize a cross-entropy or KL divergence between student and teacher outputs on some set of prompts. The two axes that matter in modern LLM distillation are *who generated the trajectory the loss is computed on* (teacher, student, or hybrid) and *what representation is matched* (token distribution, hidden states, judgements). The field is converging on a surprising result: fully on-policy distillation isn't always best — Semi-OPD (regenerate student rollouts once, reuse) often wins outright.

**Related taxonomies:** [_rl](_rl.md), [_rewards](_rewards.md)
**Depth files covered here:** [semi-opd](semi-opd.md), [cgpo](cgpo.md), [latent-grm](latent-grm.md)

---

## The problem

A capable but expensive teacher exists; you want to train a cheaper student to behave similarly. Straight SFT on teacher-written answers (offline teacher-rollout distillation) is easy but has a classic failure: the student is trained only on the teacher's own trajectories, so at inference it encounters its own distribution of context that was never in training. That causes degradation on long trajectories.

The RL-shaped fix is to roll out the *student*, label with the teacher, and distill — i.e. on-policy distillation (OPD). This matches training and inference distributions but is expensive (constant regeneration) and sometimes drifts.

## The shared pattern

All modern LLM distillation setups fit:

1. **Pick a trajectory source.** Teacher rollouts, student rollouts, or hybrid.
2. **Pick a target.** Teacher next-token probabilities (KL), teacher hidden states (feature distillation), teacher scalar judgement (reward distillation).
3. **Pick an update frequency.** Fresh student rollouts every step (OPD), fresh once (Semi-OPD), or static (offline SFT distillation).

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Offline SFT distillation (no depth file yet) | Teacher rolls out, student is SFT'd on those tokens | Train/inference distribution mismatch | Short-context tasks, strong teacher |
| On-policy distillation (OPD) (no depth file yet) | Regenerate student rollouts every step, KL to teacher logits | Expensive; drifts when teacher–student poorly aligned | Teacher–student top-k overlap is already high |
| [semi-opd](semi-opd.md) | Generate student rollouts *once*, distill against them repeatedly | Not fully on-policy; needs enough initial coverage | Misaligned pairs; cost-sensitive runs (wins in 14/17 pairs) |
| [cgpo](cgpo.md) | Per-sample routing: distill where student is weak, verifier-RL where competent | Needs a per-sample competence estimator | Mixed-difficulty data with a verifier available |
| [latent-grm](latent-grm.md) | Distill a reward model's *reasoning* trajectory into a short latent sequence | Latent reasoning is less interpretable without the decoder | Reward-model inference is in a tight loop |
| On-policy distillation in RL (DAgger-flavored) (no depth file yet) | Mix student and teacher rollouts; teacher relabels student-reached states | Still needs frequent teacher queries | Classical imitation learning benchmarks |

## How to choose

- **Default to Semi-OPD** unless the teacher and student are already close enough that OPD doesn't drift. The alignment check (top-k output-token overlap) is a cheap diagnostic — compute it before committing.
- **Use CGPO when you have both a teacher and a verifier**, and the data mixes easy and hard tasks.
- **Use LatentGRM for reward models**, not for policies — it's a very specific use of the pattern (compress a judge's CoT into a latent trajectory).
- Offline SFT distillation remains fine for short-context, high-alignment tasks and is the right starting point to establish a baseline.

## Adjacent but distinct

- **Pure SFT on teacher-generated data (synthetic data distillation)** — overlaps with offline SFT distillation but typically uses the teacher as a *dataset generator* rather than as a target distribution to KL against.
- **Knowledge distillation on hidden states (feature distillation)** — targets internal representations rather than output distributions; standard in CV, less common in LLM post-training.
- **Model souping / weight merging** — not distillation; merges weights of similar-sized models.

## Sources

- Paper: *When Do We Need On-Policy Distillation? Distilling on Offline Student Rollouts Is Often Better* — Fu, Jiang, Liu et al., 2026 — [arXiv:2610.11291](https://arxiv.org/abs/2610.11291)
- Paper: *Judging in Latent Space: Efficient Generative Reward Modeling via Semantics-Preserving Compression* — 2026 — [arXiv:2610.09788](https://arxiv.org/abs/2610.09788)
- Classic: *Distilling the Knowledge in a Neural Network* — Hinton, Vinyals, Dean, 2015.
