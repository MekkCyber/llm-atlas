# On-Policy Distillation (OPD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** On-policy distillation samples trajectories from the *student* and matches a teacher distribution on those trajectories. The KL direction (reverse vs forward) determines what transfers: **reverse-KL** copies compositional *skill* but almost no new factual knowledge; **forward-KL** restores factual transfer at some cost to skill-generalisation. The gains OPD routinely shows on math/reasoning come from skill, not from expanding parametric memory.

**Prereqs:** [fine-tuning/README.md](./fine-tuning/README.md), [_post-training.md](./_post-training.md)
**Related:** [grpo.md](./grpo.md) · [rejection-sampling.md](./rejection-sampling.md) · [cot-reward-model.md](./cot-reward-model.md)

---

## What it is

Classic knowledge distillation trains the student on teacher-generated sequences. **On-policy** distillation flips this: the student generates, and the teacher provides supervision at every student state. The student is minimising a KL divergence between its distribution and the teacher's, averaged over the *student's* trajectory distribution. The direction of that KL turns out to encode what the method transfers.

## How it works

Let $\pi_\theta$ be the student and $\pi_T$ the teacher. OPD rolls out $\tau \sim \pi_\theta$ and at each token applies a KL loss:

- **Reverse-KL:** $\mathrm{KL}(\pi_\theta \,\|\, \pi_T)$ — mode-seeking. Mass concentrates where the student *already* assigns probability. Learns teacher *behaviour on the student's own high-probability paths*.
- **Forward-KL:** $\mathrm{KL}(\pi_T \,\|\, \pi_\theta)$ — mean-seeking. Student must cover every teacher mode, including long-tail factual branches.

The paper runs both on a controlled synthetic suite where factual facts and compositional reasoning structure are independently varied, then measures transfer to *held-out* compositions and *held-out* facts.

## Why it matters

- **Reverse-KL OPD** reliably transfers compositional skill across unseen reasoning structures but installs almost no new facts. Math-competition gains on strong students show up *without* expanding parametric knowledge.
- **Forward-KL OPD** installs new facts but transfers compositional skill less reliably — the mean-seeking objective forces the student to spread mass over branches it hadn't been reaching.
- Practical rule: use reverse-KL OPD when the teacher owns a *procedure* you want to copy; use forward-KL (or continued pre-training / retrieval) when you need the student to know new *facts*.

## Gotchas & tricks

- "On-policy distillation improves reasoning" ≠ "the student learned anything factually new." Measure both skill transfer and factual transfer separately before claiming a knowledge win.
- Reverse-KL is a cheaper SFT signal than it looks: because sampling is on-policy, you only ever need teacher logits for states the student actually visits, not for every teacher trajectory.
- Mixing directions (small forward-KL weight on top of reverse-KL) is a reasonable middle ground; the paper does not fully characterise this mixture.
- Compositional generalisation claims live or die on how adversarial the held-out compositions are — build the eval before trusting the training signal.

## Sources

- Paper: *On-Policy Distillation Teaches New Skills but Not New Knowledge* — Yixuan Tang, Yi Yang, 2026 — [arXiv:2610.09639](https://arxiv.org/abs/2610.09639)
