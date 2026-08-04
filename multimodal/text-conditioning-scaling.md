# Text-Conditioning Scaling in Visual Generation
*Depth — scaling laws for the text side of diffusion, and the GPG/ED metrics that reveal them.*

**TL;DR:** Diffusion loss doesn't scale with the number of tokens in a prompt, which has historically hidden the scaling behavior of text conditioning. This paper finds that converged loss *does* scale with the amount of **structured language** in the prompt — linearly against a white-box likelihood metric (**GPG**) and by a power law against a black-box attribute-density metric (**ED**). Restructuring prompts to boost these metrics — and training a **prompt rewriter** to do it automatically — pushes open-weight T2I generation past the closed-weight frontier on compositional and reasoning benchmarks.

**Prereqs:** [_data-curation.md](../data/_data-curation.md)
**Related:** [flow-matching](./native-mesh-generation.md)

---

## What it is

Two complementary measurements of prompt "structuredness" that empirically correlate with converged diffusion loss, and the training-time interventions they suggest.

- **GPG (white-box likelihood metric).** Uses a language model's likelihood over the prompt to score how structured / non-generic the caption is. Higher GPG ⇒ more "structure per token."
- **ED (black-box attribute-density metric).** Counts concrete visual attributes explicitly mentioned per prompt.

## How it works

**Scaling law.** With controlled training runs (same architecture, same data volume, varied prompt distribution), the converged diffusion loss:

- decreases approximately **linearly** with GPG;
- follows a **power law** in ED.

**Interventions guided by the law.**

1. **Diffusability side (training data).** Construct structured training prompts with semantic and geometric annotations *derived from the images themselves* — automated re-captioning that raises the GPG/ED of every training pair.
2. **Promptability side (inference).** Train a **prompt rewriter** that turns a user's short prompt into a structured, high-GPG/ED prompt the model wants to see. Recipe: SFT on rewritten prompts → cold-start against the base model → **verifier-gated on-policy distillation** (only rewrites that verifiably improve outputs are distilled back).

## Why it matters

- Turns text-conditioning quality into an *optimization target* with concrete metrics, not a heuristic. Data curation for T2I now has a scaling curve to climb.
- Verifier-gated on-policy distillation for the rewriter is a reusable pattern for any pre-model text preprocessor.
- The resulting open-weight system beats or matches the strongest closed-weight T2I models on compositional, reasoning, and world-knowledge benchmarks with the same base diffusion architecture — evidence the ceiling was on the *text* side, not the *visual* side.

## Gotchas & tricks

- Automated re-captioning must actually add signal, not just adjectives. Structured annotations that repeat information the image already implies don't move GPG.
- The rewriter is trained to rewrite for *this* generator's structural preferences — transferring across generators is not automatic.
- ED counts attributes but doesn't check that they render correctly; keep a separate visual-consistency reward when using ED as a training signal.

## Sources

- Paper: *Scaling Properties of Text Conditioning in Visual Generation* — Chen et al. (ByteDance Seed), 2026 — [arXiv:2607.29679](https://arxiv.org/abs/2607.29679)
