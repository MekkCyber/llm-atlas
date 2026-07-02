# Distillation for LLM Post-Training

*Taxonomy — teacher-to-student capability transfer as a post-training move, complementary to RL and rejection-sampling SFT.*

**TL;DR:** Distillation compresses a stronger teacher into a smaller/weaker student. In LLM post-training three axes matter: (a) *what supervises* (teacher logits vs teacher completions), (b) *whose trajectories* the loss is computed on (teacher's or student's — off-policy vs on-policy), and (c) *how many teachers*. The 2026 default for capability integration is **on-policy distillation from one or more specialized RL teachers**: it removes exposure bias while providing dense token-level signal.

**Related taxonomies:** [_post-training](_post-training.md) · [_rl](_rl.md) · [_rewards](_rewards.md)
**Depth files covered here:** [on-policy-distillation](on-policy-distillation.md) · [multi-teacher-on-policy-distillation](multi-teacher-on-policy-distillation.md)

---

## The problem

RL post-training pushes specific capabilities (math, code, tool use), but *integrating* several such capabilities into one model tends to degrade each one (Mix-RL, cross-domain interference, reward-scale mismatch). A common workaround is to train a big teacher (or many domain teachers) and then compress that capability into a deployable student. The design space is: how to transfer *without* losing what RL bought, and *without* running RL again.

## The shared pattern

Every variant fits `teacher(s) → student` with three choices:

1. **Signal form.** Hard-label completions (SFT-style) or full teacher token distributions (KD-style).
2. **Trajectory source.** *Off-policy* (teacher rollouts): frozen dataset, easy but exposure-biased. *On-policy* (student rollouts): student generates, teacher supervises the student's own tokens — dense, exposure-bias-free, but requires the teacher in the training loop.
3. **Teacher count.** One general teacher, one privileged teacher, or an ensemble of domain teachers.

## Variants

| Technique | Key idea | Main tradeoff | When it wins |
| --- | --- | --- | --- |
| Off-policy KD / SFT distillation | Train student on teacher completions or logits | Exposure bias; student never sees its own errors during training | Cheap; teacher available only as an API |
| [on-policy-distillation](on-policy-distillation.md) | Student rolls out, teacher supervises token-level | Dense signal, no exposure bias | Capability transfer within one domain |
| DOPD (dual on-policy) | Privileged teacher + privileged student, advantage-aware routing | Complexity; two forward heads | Grounded / retrieval settings where privilege illusion hides |
| [multi-teacher-on-policy-distillation](multi-teacher-on-policy-distillation.md) | Per-domain RL teachers, one on-policy student | Requires domain-teacher fleet | Frontier post-training with decoupled domain teams |
| Rejection-sampling SFT | Sample from teacher, keep passing, SFT on completions | Coarser signal; only "correct" trajectories | When only outputs (not logits) are available |
| Param-merge / soup | Merge domain weights directly | No training; often lossy | Quick integration of loosely-related models |

## How to choose

- Building a **deployable dense checkpoint from an RL teacher** in one domain → on-policy distillation.
- **Integrating multiple RL specialists** into one student → multi-teacher on-policy distillation (MOPD).
- Teacher has access to **privileged inputs** (RAG, tools, gold answers) → dual on-policy (DOPD) to avoid *privilege illusion*.
- **Only teacher API** available (no logits, no in-training access) → rejection-sampling SFT.
- **Loosely-related fine-tunes** to merge cheaply → parameter merging / model souping — see [../pre-training/model-souping.md](../pre-training/model-souping.md).

## Adjacent but distinct

- **RLVR / GRPO** ([rlvr](rlvr.md), [grpo](grpo.md)) — train the student directly against verifiable rewards. Complements distillation; often used *inside* the teacher.
- **[rejection-sampling](rejection-sampling.md)** — a data-generation step; can be viewed as the coarsest form of off-policy distillation.
- **Model souping** ([../pre-training/model-souping.md](../pre-training/model-souping.md)) — parameter-space combination, no student rollouts involved.

## Sources

- Paper: *DOPD: Dual On-policy Distillation* — Li et al., 2026 — privilege illusion and advantage-aware routing.
- Paper: *MOPD: Multi-Teacher On-Policy Distillation for Capability Integration in LLM Post-Training* — Ma et al., 2026 — deployed in MiMo-V2-Flash.
- Blog / paper: on-policy distillation lineage — Agarwal et al. ("On-Policy Distillation of Language Models") and Google's on-policy KD papers.
