# Instruction Distillation from Few-Shot Examples

*Depth — replacing concatenated few-shot ICL examples with a compact, human-editable task instruction distilled from those examples.*

**TL;DR:** Standard few-shot in-context learning (ICL) packs N demonstration examples into the prompt. This works up to a context-length limit, but accuracy degrades as N grows (noise accumulation, distraction), inference cost balloons, and the resulting prompt is opaque. **Instruction distillation** replaces the example list with a single *distilled task instruction* — a structured description of the classification criteria + an explicit task definition, derived offline from the same demonstrations. Result on the Call Playbook B2B benchmark (Rotman et al., 2026): **−99% tokens, +7% macro AUC, robustness preserved as context grows** (vs. −9 F1 points for token-compression baselines).

**Prereqs:** [_post-training](_post-training.md)
**Related:** [fine-tuning](fine-tuning/README.md), [rl-prompt-curation](rl-prompt-curation.md)

---

## What it is

An offline pipeline that converts a few-shot demonstration set into a portable artifact:

```
Demonstration set D = { (xᵢ, yᵢ) }
        ↓ (offline LLM-based distillation)
Distilled task instruction I = {
    structured classification criteria,
    explicit task description,
    edge-case notes
}
```

At inference time the prompt is `I + query`, not `D + query`. The distilled instruction is small (a few hundred tokens), human-readable, and *editable* — a domain expert can tweak the criteria directly without re-running distillation.

## How it works

Distillation pipeline:

1. **Extract structured criteria.** Run an LLM over `D` with a meta-prompt that asks for the classification criteria in a structured form (e.g. "for each class, list the cues that distinguish it from the others").
2. **Add a precise task description.** Frame the task in declarative terms, not as "here are examples; do this."
3. **Capture edge cases.** Note ambiguous patterns and how they should be resolved.
4. **Validate.** Run the distilled instruction over a held-out validation set; iterate if needed.

The output is a compact, interpretable prompt artifact that ships as the production prompt.

## Why it matters

Three properties make this a credible alternative to either large-N ICL or full SFT for domain-specialized classification:

- **Token efficiency.** Reported 99% reduction in tokens per query vs. traditional ICL — non-trivial at scale.
- **Robustness as context grows.** Token-compression baselines collapse (>9 F1 point drop) as the conversational context surrounding the query expands; the distilled instruction's compact form sidesteps this.
- **Auditability and editability.** Unlike fine-tuned weights, the distilled instruction is *inspectable*. A subject-matter expert can read the criteria, identify mistakes, and edit them directly. This matters for compliance-sensitive domains (sales, legal, support).

It's "prompt engineering" reframed as a learned-but-portable artifact — versionable, auditable, refinable without retraining infrastructure.

## Gotchas & tricks

- **Distillation quality is bounded by the meta-model.** A weaker distillation LLM produces vague criteria; a stronger one is needed for the offline step (you only pay this cost once).
- **Domain-specific terminology must be preserved.** A common failure mode is the distiller "summarizing away" the exact phrasing that distinguishes classes — explicit instruction to preserve terminology helps.
- **Not a substitute for SFT when behavior must be tightly controlled.** SFT bakes in patterns that survive prompt edits; instruction distillation lives at the prompt layer, so adversarial prompt injection still applies.
- **Composes with light SFT.** A SFT'd model + a distilled instruction can outperform either alone, since the instruction encodes the criteria and SFT encodes the formatting conventions.

## Sources

- Paper: *Distilling Examples into Task Instructions: Enhanced In-Context Learning for Real-World B2B Conversations* — Rotman, Kopilov, Berger Zalmanson, Allouche, 2026 — https://arxiv.org/abs/2606.15641
