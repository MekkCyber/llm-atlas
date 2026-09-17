# Privileged Distillation
*Depth — teacher trained with access to hidden state; student trained on its outputs without that access.*

**TL;DR:** Privileged distillation is a two-stage synthetic-data pipeline: (1) build a **teacher** that has access to some "privileged" information the deployed student will never see; (2) generate its outputs as training data for the student, who learns to imitate them from the ordinary inputs alone. Mind2Dialogue (Wang et al., 2026) uses this to train "human-aware" chat assistants: the teacher sees simulated user mental states; the student sees only the user's messages.

**Prereqs:** [_data-curation](_data-curation.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [quality-filtering](quality-filtering.md), [../post-training/_post-training.md](../post-training/_post-training.md)

---

## What it is

A supervision design pattern where the teacher has strictly more information than the student, and the student learns from teacher outputs (not teacher internals). The technique is old in structured learning ("learning using privileged information", Vapnik) but has become newly useful for LLM SFT because:

- Simulators can supply hidden state (user mental state, ground truth reasoning, oracle answers).
- Modern LLMs can act as strong conditional-imitation teachers.
- Deployed assistants have no way to observe hidden state, so training data must bridge the observation gap.

## How it works

1. **Define the privileged variable `z`.** In Mind2Dialogue, `z` is the user's beliefs / goals / mental state, produced by a psychology-guided simulator.
2. **Teacher generation.** For each simulated dialogue turn `x`, the teacher LM sees `(x, z)` and produces a response `y = T(x, z)`. The teacher's outputs implicitly encode `z`.
3. **Student training.** The student LM is fine-tuned on `(x → y)` pairs — never seeing `z` explicitly. The gradient forces the student to develop internal proxies for `z` from just `x`.
4. **Optional distillation losses.** In addition to cross-entropy on `y`, one can use KL between teacher and student output distributions for finer signal.

For dialogue assistants, `x` is the user's message history, `z` is the simulated user mental state, `y` is a "well-informed" response.

## Why it matters

Most SFT/RLHF data implicitly assumes the assistant can only observe surface text — so it never learns to model latent user state well. Privileged distillation gives a cheap way to inject theory-of-mind-like sensitivity into an assistant *without* running a mental-state model at inference time. Mind2Dialogue reports **+26.6 to +40.9 percentage-point** improvements in preference-following generation across Qwen, Llama, and OLMo baselines.

More broadly, the same pattern applies wherever a simulator or oracle can supply privileged info: ground-truth reasoning traces (for reasoning distillation), true retrieval passages (for RAG distillation), or task-completion metadata (for agent distillation).

## Gotchas & tricks

- **`z` must be *inferable* from `x`.** If `z` is completely independent of what the student sees, the student can't learn it and just averages responses.
- **Simulator bias transfers to the student.** Whatever biases the mental-state simulator has, the student inherits. Validate the simulator like any other data source.
- **Not the same as chain-of-thought distillation.** In CoT distillation, the "privileged" info is the reasoning trace generated *by* the teacher. In privileged distillation, it's supplied *to* the teacher.
- **Diverse `z` samples.** Multiple mental-state samples per user context yield better coverage than a single one.
- **Beware label leakage.** If the privileged variable trivially appears in the teacher's output text, the student learns a shortcut rather than an internal model.

## Sources

- Paper: *Mind2Dialogue: Training Human-Aware Language Models by Simulating User Mental States* — Wang, Zhou, … Wang — 2026 — [arXiv:2609.15972](https://arxiv.org/abs/2609.15972).
- Origin (structured learning): *Learning Using Privileged Information* — Vapnik & Vashist, 2009.
