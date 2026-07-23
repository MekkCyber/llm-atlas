# Attention registers in Diffusion Transformers
*Depth — structural template tokens as implicit state registers in text-to-image DiTs.*

**TL;DR:** In large text-to-image DiTs, the *structural* text tokens — the fixed template scaffolding around the prompt (delimiters, stop-tokens, boilerplate) — carry almost no prompt-specific content at the text encoder, yet act as dominant image-to-text attention sinks and **causally maintain object identity** inside the DiT. Ablating them breaks object identity even though they don't encode identity semantically. Named "implicit semantic registers" by analogy to attention-register phenomena in ViTs.

**Prereqs:** [../fundamentals/attention.md](../fundamentals/attention.md)
**Related:** [README.md](./README.md)

---

## What it is

Attention registers, in vision transformers, are tokens that don't carry a semantic role at input but end up serving as hidden state buffers the model routes through. This paper's contribution is showing the same phenomenon inside a large-scale diffusion transformer for text-to-image, sitting on the **template tokens of the text prompt**.

The tokens in question are the ones the text encoder essentially fills with nothing prompt-specific: delimiters, ends-of-sequence, template boilerplate. Empirically, they carry almost no prompt content at the text-encoder output — but they *cause* image quality and identity when you touch them.

## How it works

Method:

1. **Attention decomposition.** For a chosen layer, decompose attention into token-span, head, and layer contributions.
2. **Content vs structure partition.** Separate the prompt tokens (words the user wrote) from the structural tokens (template scaffolding).
3. **Causal interventions.** Ablate, resample, or replace structural tokens' contributions inside the DiT while leaving content tokens untouched.

Findings:

- Structural tokens carry **near-zero prompt-specific information at the text encoder** (probes fail).
- Inside the DiT, structural tokens are **dominant image-to-text attention sinks** — the image side of the network routes through them.
- **Ablating structural tokens causally destroys object identity** in generated images. Ablating content tokens changes appearance more than identity.

The reading: structural tokens act as **implicit semantic registers** — a store for identity-carrying state the DiT constructs during denoising.

## Why it matters

- **Bridges mech-interp to DiTs.** The attention-decomposition + causal-intervention toolkit built for LLMs transfers to production text-to-image models.
- **Compositional-editing lever.** Now you know *where* to intervene to preserve or transfer object identity — the template tokens, not the content ones.
- **Explains prompt-engineering rituals.** Delimiters, empty separators, template boilerplate — they were doing more than we thought, because DiTs found somewhere useful to route through.

## Gotchas & tricks

- The definition of "structural" vs "content" token depends on the specific text encoder's tokenization; a different tokenizer moves the register slots.
- Attention sinks are model-family-dependent — the finding is on a class of large DiTs, not all diffusion models.
- Causal test relies on the ability to ablate cleanly. Naïve zero-ablation can introduce artifacts; the paper uses resample-based controls.

## Sources

- Paper: *Text Template Tokens Are Implicit Semantic Registers in Diffusion Transformers* — Li et al., 2026 — [arXiv:2607.19139](https://arxiv.org/abs/2607.19139)
- Related prior work: "attention registers" in ViTs (Darcet et al., 2023).
