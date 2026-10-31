# OmniSchema-Aware On-Policy Distillation (OSAD)
*Depth — one specific technique, grounded in its source paper(s).*

**TL;DR:** A self-distillation training procedure for multi-task multimodal parsing. In each round, the model is trained on its *own* rollouts against token-level targets produced by a reference-conditioned copy of itself; after the round, the reference is refreshed with the trained weights. Parser and supervision co-evolve. Combined with a unified output *schema* across all tasks (OmniSchema), OSAD gives a single token-role-based objective that supervises all ten parsing tasks the authors tackle. Powers Youtu-Parsing-Omni (5B) to 96.96 Overall on OmniDocBench v1.6 and 75.08 Avg. on OmniParsingBench, best among open-weight models.

**Prereqs:** [rejection-sampling](../post-training/rejection-sampling.md)
**Related:** [_rl](../post-training/_rl.md)

---

## What it is

Classical knowledge distillation needs a *frozen stronger teacher*. On-policy self-distillation needs a *trusted self-teacher at each step*. OSAD takes the self-distillation route and defines the self-teacher as a **reference-conditioned copy** of the current model: the same weights, but conditioned on a reference / ground-truth-shaped input that makes the next-token distribution target-aligned.

The procedure is driven by a shared output schema — OmniSchema — that unifies the outputs of multiple parsing tasks (document, chart, OCR, audio, video, chemical, music). Because every task produces structured records in the same schema, a token-role-based objective (e.g., "this token is a key, this is a value, this is a span-endpoint") applies to all tasks at once.

## How it works

### Per-round loop

1. **Rollout.** The current model $\pi_\theta$ generates completions on training prompts.
2. **Reference-conditioned target.** A copy of $\pi_\theta$, conditioned on the task's reference answer or structured ground truth, generates a *targeted* distribution over the next token. This acts as the teacher distribution.
3. **Token-role-based loss.** Each token in the output is tagged by its role in OmniSchema (key, value, span, delimiter, etc.). The loss is a role-weighted combination that respects the structure.
4. **Weight update.** $\pi_\theta$ is updated against the reference-conditioned target.
5. **Refresh.** The reference copy is updated to the newly trained $\pi_\theta$. Parser and supervision co-evolve.

### Key properties

- **No external teacher.** The reference copy is the same model; the only asymmetry is that it's conditioned on more structured input.
- **Shared objective across tasks.** OmniSchema's token roles make the loss modality- and task-agnostic.
- **On-policy.** The model learns from its own rollouts, not pre-collected demonstrations.
- **Co-evolution avoids drift.** Refreshing the reference each round prevents the teacher from becoming stale relative to the student.

## Why it matters

Multimodal parsers typically train each task separately (OCR, chart, document, audio transcription). OSAD + OmniSchema shows that *one* shared on-policy loss can supervise all of them simultaneously, with a single 5B checkpoint:

- **96.96 Overall on OmniDocBench v1.6**
- **75.08 Avg. on OmniParsingBench** — best among open-weight models, with gap concentrated in audio and video
- **95.02** chart-parsing score
- Highest exact chemical-fingerprint agreement
- Matches a dedicated optical-music model in sheet-music transcription

The modality-agnostic, task-agnostic objective is the generalization story: as long as you can describe a task's output in OmniSchema, OSAD trains it.

## Gotchas & tricks

- **Reference conditioning design is the whole trick.** If the "reference-conditioned copy" doesn't actually produce better next-token predictions than the student, OSAD degenerates into self-training on its own rollouts.
- **Schema coverage limits task coverage.** A task whose output doesn't fit OmniSchema can't be trained by this objective. The paper's schema covers ten parsing tasks; broader tasks (e.g., open-ended VQA) would need extension.
- **Co-evolution can drift.** The reference tracking the student means both can walk together into a bad basin. The paper reports strong results but doesn't ablate the drift risk in detail from the abstract.
- **Not pure RL.** The signal is a distillation target, not a scalar reward, so this is closer to self-distillation than to RLHF or RLVR. The co-evolution structure resembles reference-model refresh in iterated DPO.
- **5B is the demonstrated scale.** Larger scales plausibly work but aren't reported.

## Sources

- Paper: *Youtu-Parsing-Omni: One Encoder, One Schema, All Modalities* — Y. Shen, Z. Chen, X. Li et al. (30 authors), Tencent, 2026 — arXiv:2610.39301.
- Code / model: https://github.com/TencentCloudADP/youtu-parsing — Youtu-Parsing-Omni and the OSAD training code.
- Model weights: huggingface.co/tencent/Youtu-Parsing-Omni
