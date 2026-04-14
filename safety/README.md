# Safety & Alignment

*How models are made helpful, harmless, and honest — and how they're stress-tested for failure modes.*

---

## What This Is

Safety work spans training-time techniques (what you reward, what you penalize), evaluation (how you measure harm), and adversarial probing (how you find failures before users do). It overlaps with post-training but has its own literature, threat models, and methods.

---

## What Belongs Here

- **Alignment techniques** — RLHF for helpfulness/harmlessness, constitutional AI, self-critique, rule-based rewards.
- **Refusal training** — how models learn to decline, over-refusal, calibration.
- **Red-teaming** — manual and automated adversarial probing, attack taxonomies.
- **Jailbreaks** — prompt-based attacks, suffix attacks, multi-turn attacks, defenses.
- **Sycophancy & honesty** — reward hacking toward user approval, honesty training.
- **Dangerous capability evaluation** — CBRN, cyber, autonomy evals.
- **Monitoring & classifiers** — input/output filters, moderation models.
- **Policy & deployment** — responsible scaling, model cards, release decisions.

## Reading Order

1. RLHF for safety vs. helpfulness tradeoffs
2. Constitutional AI & self-critique
3. Red-teaming methodology
4. Jailbreaks & defenses
5. Dangerous capability evals
6. Deployment & monitoring

---

## Related

- [post-training/](../post-training/) — safety training is a post-training technique
- [evaluation/](../evaluation/) — safety evals are a subset of evaluation
- [interpretability/](../interpretability/) — understanding *why* a model behaves a certain way
