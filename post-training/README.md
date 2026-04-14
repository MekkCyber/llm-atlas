# Post-Training

*How pretrained models are refined — through supervised fine-tuning, reinforcement learning, and preference learning — with the algorithms, reward signals, and rollout infrastructure that turn a base model into something useful.*

---

## What This Is

A pretrained base model knows a lot but is not directly useful. Post-training is the set of techniques that shape its behavior: SFT to teach format and style, preference learning (RLHF, DPO, GRPO) to align it with what people actually want, and RL with verifiable rewards to produce capabilities like math and code. Fine-tuning and reasoning are important enough to have their own subfolders.

---

## What Belongs Here

- **Supervised fine-tuning (SFT)** — instruction tuning, data design, format conventions.
- **Reward modeling** — how preference data is collected and a reward model trained.
- **Policy optimization** — PPO, GRPO, REINFORCE variants, KL penalties.
- **Preference-based objectives** — DPO, IPO, KTO, and why they avoid explicit reward models.
- **Rollout infrastructure** — generating samples at scale, batching, off-policy correction.
- **Constitutional AI & RLAIF** — using models to generate training signal.
- **Post-training recipes** — full pipelines combining SFT + preference learning + RL.

## Reading Order

1. SFT & instruction tuning
2. Reward modeling
3. PPO and the RLHF pipeline
4. DPO and preference objectives
5. GRPO and RL for reasoning
6. Rollout infrastructure

---

## Subfolders

- [fine-tuning/](fine-tuning/) — SFT, LoRA, adapter methods, model merging.
- [reasoning/](reasoning/) — CoT, PRMs, test-time compute, RL for reasoning.

---

## Related

- [pre-training/](../pre-training/) — where the base model comes from.
- [systems/](../systems/) — the rollout and orchestration infrastructure post-training relies on.
- [evaluation/](../evaluation/) — reward models are themselves evaluated here.
- [safety/](../safety/) — safety is a post-training technique.
