# Fine-Tuning

*Adapting a pretrained model to a new task, domain, or behavior — without retraining from scratch.*

---

## What This Is

Fine-tuning is the bridge between a base model and a useful one. It covers full-parameter SFT, parameter-efficient methods (LoRA, QLoRA, DoRA, adapters), and the recipes for making them work in practice.

Distinct from RL post-training (which lives one level up in [post-training/](../)) — fine-tuning here means supervised adaptation.

---

## What Belongs Here

- **Full-parameter SFT** — recipes, data formats, hyperparameters, catastrophic forgetting.
- **LoRA & variants** — LoRA, QLoRA, DoRA, LoRA+, rank selection.
- **Adapters & prefix methods** — adapter layers, prefix tuning, prompt tuning.
- **PEFT libraries** — what's in the toolkit and when to use each method.
- **Merging** — model merging, LoRA merging, task arithmetic.
- **Instruction tuning** — dataset design, format conventions, multi-task mixtures.

## Reading Order

1. Full-parameter SFT basics
2. Instruction tuning data design
3. LoRA (the idea)
4. QLoRA & quantized fine-tuning
5. Adapter / prefix methods
6. Model merging

---

## Related

- [../](../) — the broader post-training context (RL, preference learning)
- [../../quantization/](../../quantization/) — QLoRA sits at this intersection
- [../../data/](../../data/) — SFT is only as good as its data
