# Cross-Paradigm On-Policy Distillation
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A distillation recipe that trains a single student model (typically a discrete diffusion LLM) using teachers from **different generative paradigms** — an autoregressive vision-language model for understanding and a continuous flow model for image generation. On-policy samples from the student are scored against each teacher's preferences, so the student learns one unified behavior that matches both teachers in their respective strengths without inheriting their specific inference recipes.

**Prereqs:** [unified-diffusion-llm.md](unified-diffusion-llm.md), [../post-training/fine-tuning/README.md](../post-training/fine-tuning/README.md)
**Related:** [unified-diffusion-llm.md](unified-diffusion-llm.md)

---

## What it is

Classical distillation trains the student to match a teacher that uses the **same** paradigm (AR → AR, flow → flow). Cross-paradigm distillation forces a student in one paradigm to match teachers in *other* paradigms:

- **AR VLM teacher** → emits strong understanding behavior for VQA, grounding, reasoning.
- **Continuous flow teacher** → emits strong image samples.
- **Discrete diffusion student** → learns to do both, in its own inference modality.

The twist is **on-policy**: samples come from the current student, not from the teachers. Each teacher scores the student's own outputs, and the student is updated to shift probability mass toward higher-scoring samples. This avoids the usual "teacher sample ≠ student distribution" mismatch, and lets the student learn where its *actual* failure modes are.

## How it works

Each training step:

1. **Student generation.** Sample a batch from the current student policy $\pi_\theta$ on prompts that span both modalities.
2. **Teacher scoring.** For an understanding sample, the AR VLM scores the student's token sequence (log-likelihood, or a preference judgment). For a generation sample, the flow teacher scores the student's image tokens (e.g., via rendering + a continuous model's likelihood under the shared tokenizer latent).
3. **Loss.** A direct-preference-style or policy-gradient loss that pulls $\pi_\theta$ toward teacher-preferred completions. (NLD-Uni uses an OPD objective; the exact form is on-policy and preference-based.)
4. **Modality routing in the batch.** Prompts are tagged by modality so the right teacher is used for each sample.

```
prompt ──▶ student π_θ sample ──▶  teacher_AR or teacher_flow scores it
                 │                            │
                 └──── on-policy loss  ◀──────┘
```

Student parameters are shared across modalities; the only modality branch is which teacher scores the sample.

## Why it matters

- **Paradigm independence.** Any teacher capable of producing a score — not just same-paradigm teachers — can be a distillation source. Unlocks cross-paradigm transfer that offline distillation can't.
- **On-policy fixes coverage.** Offline distillation drowns in off-distribution teacher samples. On-policy keeps the training signal relevant to what the student actually does.
- **Primary training path for [unified diffusion LLMs](unified-diffusion-llm.md).** Enables one discrete diffusion model to match AR understanding and flow-model generation simultaneously.
- **Scales naturally with teacher capability.** Replacing a teacher with a better same-paradigm model is a drop-in upgrade.

## Gotchas & tricks

- **Score calibration across teachers.** AR log-likelihood and flow-model likelihood live on different scales. The paper normalizes within each teacher's score distribution before combining.
- **Teacher cost.** Scoring every student sample through two large teachers is expensive. Reuse of KV caches and batched scoring help.
- **Reward hacking across paradigms.** The student may find outputs that score high on both teachers but are generally bad. KL anchoring to a stable reference and periodic human spot-checks matter here, same as RLHF.
- **Not the same as model souping or merging.** Souping averages weights of same-architecture models; this trains a different-architecture student against them.
- **Related to RLAIF.** Teachers playing the role of AI judges is a familiar pattern; the novelty is specifically crossing paradigm boundaries.

## Sources

- Paper: *Nemotron-Labs-Diffusion-Uni: Elastic Self-Refining Unified Diffusion LLM with Cross-Paradigm On-Policy Distillation* — Li, Ye, Heinrich, Fu, Zhang, Liu, Grover, Molchanov, 2026 — [arXiv:2610.36466](https://arxiv.org/abs/2610.36466)
