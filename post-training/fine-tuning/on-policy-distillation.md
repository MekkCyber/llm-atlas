# On-Policy Distillation

*Depth — distill a teacher LLM into a student by matching the teacher's logits on sequences the **student itself** generated.*

**TL;DR:** Classical distillation trains the student on the teacher's logits over the *teacher's* samples. On-policy distillation (OPD) trains it on the teacher's logits over the *student's* samples — same loss, different data distribution. Because the student is scored on its own output distribution, OPD gives dense token-level supervision at every point the student actually visits in sequence space, closing the imitation-vs-generation gap that off-policy distillation leaves.

**Prereqs:** [../_post-training.md](../_post-training.md), [../rejection-sampling.md](../rejection-sampling.md)
**Related:** [../rlvr.md](../rlvr.md), [../grpo.md](../grpo.md)

---

## What it is

Three distillation data distributions you can train on:

| Setup | Where sequences come from | Where logits come from |
|---|---|---|
| **Off-policy distillation** | teacher samples | teacher logits |
| **Student rollout + rejection** | student samples, filtered correct | teacher logits (or ground truth) |
| **On-policy distillation** | student samples (unfiltered) | teacher logits at every student position |

OPD scores *every* student rollout — right or wrong — by the teacher's token-wise distribution. The loss is typically forward KL `KL(π_student ‖ π_teacher)` or reverse KL at each token position:

```
L_OPD = −E_{s ~ π_student}  Σ_t  π_teacher(s_t | s_<t) · log π_student(s_t | s_<t)
```

## How it works

### The pipeline

1. Sample a batch of rollouts from the **current** student policy: `o_1, …, o_G ~ π_student(·|q)`.
2. Run the teacher over each `(q, o_i)` prefix-by-prefix to get `π_teacher(· | q, o_<t)` at every position.
3. Compute a token-wise divergence between `π_student` and `π_teacher` and backprop.

Unlike RL, there is no sparse reward; every token has gradient. Unlike SFT on teacher samples, the student is never asked to imitate strings it would never produce — the supervision lives at points on the student's own trajectory.

### Vocabulary alignment

OPD assumes a shared vocabulary. Different tokenizers break the per-position alignment. Common workarounds:

- **Strict matching:** supervise only token positions where student and teacher tokenizations coincide (cross-tokenizer OPD, 2026).
- **MSE on mismatched spans:** map both to byte-level and apply a continuous loss — attempted and shown to *degrade* accuracy (Alibaba 2026) because the mismatched-span gradient fights the strict gradient.
- **Shared top-k vocab subset:** supervise only the top-k shared tokens the student would pick — reportedly matches full-vocab OPD with far less engineering (Alibaba 2026).

### Teacher sees student sequence

The teacher must support arbitrary prefixes — it is run in *scoring* mode, not generation. For a frontier teacher this is one forward pass per rollout, O(G · T · teacher_cost) per batch.

## Why it matters

- **Denser supervision than RL for the same rollout budget.** Each student token gets a gradient; RL only grades the response outcome. For reasoning and code, where many tokens are "load-bearing", OPD trains faster than RLVR per rollout.
- **No reward hacking.** The teacher is the oracle, not a learned reward model, so there is no gameable reward surface.
- **Combines naturally with RL for the hardest problems.** OPD gives signal where RL has none (the all-failure group); RL gives signal where OPD can't (outcome correctness on long trajectories). See [../diffgate.md](../diffgate.md) for a difficulty-gated combination.

## Gotchas & tricks

- **Teacher cost is the dominant RL-ish cost.** For a frontier teacher, the forward passes to score student rollouts are the budget. Can be reduced with teacher distillation to a smaller "proxy teacher" first.
- **Student exploration is still your problem.** OPD only supervises what the student explores. For reasoning, a student that always collapses to a short answer never gets long-CoT signal, no matter how good the teacher is. Pair with reasoning-aware prompts or rationale scaffolding (see [../reasoning/rgpo.md](../reasoning/rgpo.md)).
- **Forward vs reverse KL** matters. Forward KL `KL(π_teacher ‖ π_student)` is mean-seeking (student spreads mass over teacher's modes); reverse KL is mode-seeking (student picks the teacher's best mode and sharpens there). Reasoning typically wants reverse KL.
- **Logit remapping** across tokenizers is a solved research problem but a load-bearing engineering problem. See *cross-tokenizer OPD* (Alibaba 2026) for the current best heuristic.

## Sources

- Paper: *Rethinking Cross-Tokenizer On-Policy Distillation: From Alignment Coverage to Supervision Reliability* — Hou et al., Alibaba Cloud, 2026 — [arXiv:2610.08448](https://arxiv.org/abs/2610.08448).
- Paper: *DiffGate: Difficulty-Gated Teacher Guidance for On-Policy Distillation* — 2026 — [arXiv:2610.04596](https://arxiv.org/abs/2610.04596).
- Earlier: *On-Policy Distillation of Language Models: Learning from Self-Generated Mistakes* — Agarwal et al., 2024 — the OPD framing in the pre-reasoning-RL era.
- Related: *Distilling the Knowledge in a Neural Network* — Hinton et al., 2015 — the off-policy ancestor.
