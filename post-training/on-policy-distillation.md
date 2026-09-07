# On-Policy Distillation
*Depth — training a student on its own rollouts with dense token-level supervision from a teacher.*

**TL;DR:** On-policy distillation (OPD) samples completions from the student, then trains it to match a teacher's next-token distributions on those completions. Unlike rejection-sampling distillation (train on teacher outputs) or classic offline KD (train on a fixed corpus), OPD's supervision is dense and covers exactly the states the student visits. Tsinghua's 2026 "OPD II" analysis shows this makes it striking data-inefficient: **one query** trains the student for hundreds of steps almost as well as a full dataset, and 16 queries close the gap entirely — OPD is *data-overfed but algorithm-starved*.

**Prereqs:** [rlvr](rlvr.md), [rejection-sampling](rejection-sampling.md)
**Related:** [rl-prompt-curation](rl-prompt-curation.md), [ppo](ppo.md), [long-cot-rl](reasoning/long-cot-rl.md)

---

## What it is

An on-policy fine-tuning method for the case where you have (a) a policy $\pi_\theta$ to train and (b) a teacher $\pi_T$ (usually a larger or better-trained model). At each step:

1. Sample a rollout $y \sim \pi_\theta(\cdot | x)$ for some query $x$.
2. For each token $y_t$, query the teacher for its distribution $\pi_T(\cdot | x, y_{<t})$.
3. Update $\theta$ to minimize a divergence (typically forward or reverse KL, or the token-level cross-entropy with the teacher's top-$k$) between $\pi_\theta(\cdot | x, y_{<t})$ and $\pi_T(\cdot | x, y_{<t})$.

The rollout is on-policy; the supervision is dense (per-token from the teacher). Recent open-source recipes use OPD as the last stage of post-training pipelines (e.g. distilling reasoning from a stronger teacher into a smaller model).

## How it works

The interesting analytic frame from OPD II is **state coverage**: the fraction of states — (prefix, next-token) pairs — visited by full-data OPD that a smaller query set's rollouts also visit.

- One query already covers **71.5%** of full-data OPD's states, most of it within the first 100 training steps: the student's own decoding entropy makes each additional rollout explore a wide state space even from a single prompt.
- **16 semantically distinct queries** reach 98.9% coverage and match full-data training.
- Alignment (student-teacher KL) drops at a similar rate whether the training corpus is 1 query or 100k — so the bottleneck is *how fast the student absorbs supervision*, not *how much supervision is exposed*.

Because task content and induced state coverage decouple, even content-light templates ("solve this: ...") and off-domain queries from WildChat approach the same performance.

## Why it matters

- **Data-minimal post-training.** For labs distilling a frontier teacher into a smaller model, OPD reduces the "distillation dataset" from millions of prompts to tens.
- **Rethinks recent post-training results.** Some published gains attributed to careful prompt corpora may in fact be produced by state coverage alone; the corpus barely matters.
- **Refocuses future work.** The bottleneck named — the student's absorption rate — is an algorithm-side problem, pointing at higher-effective-learning-rate variants (better loss shape, better teacher-query sampling, longer effective context per teacher query).

## Gotchas & tricks

- **Diverse queries beat repeat queries.** One query works; 16 *distinct* queries work best. Repeating a single query fewer than 100 times leaves coverage on the table.
- **Content-light templates suffice.** If diversity is what matters, curated in-domain queries are not required — templates and off-domain queries perform comparably.
- **Interacts with SFT and RL.** OPD is often used *after* SFT and *before* an RLVR stage; the state coverage argument suggests SFT's role may be to bootstrap the student's decoder rather than to teach content per se.
- **Teacher-side cost dominates.** With one query, teacher-side cost per step is fixed and dwarfs student-side cost; batch teacher queries aggressively.
- **On-policy ≠ RL.** OPD does not use rewards — it uses teacher log-probs. It is a supervised method that happens to consume on-policy states.

## Sources

- Paper: *Rethinking On-Policy Distillation of Large Language Models II: One Training Example* — Fu et al., Tsinghua, 2026 — the one-query and state-coverage results — https://arxiv.org/abs/2609.04172
- Paper: *On-Policy Distillation of Language Models* — Agarwal et al., 2023 — early OPD framing.
- Paper: *Distilling Step-by-Step* — Hsieh et al., 2023 — rationale-guided distillation, off-policy sibling.
